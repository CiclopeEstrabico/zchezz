import os, random, time, glob, subprocess, re, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# ═══════════════════════════════════════════════════════════
#  Pipeline Ultra-Otimizado — RAM-SAFE Streaming (Nodes: 5000)
# ═══════════════════════════════════════════════════════════

SF_PATH    = r"engine\stockfish_fast\stockfish\stockfish-windows-x86-64-avxvnni.exe"
INPUT_DIR  = r'data\extra_quiet_wdl'
OUT_DIR    = r'data\extra_quiet_sf_nodes5000_wdl'
PROGRESS_F = r'sf_analyze\sf_progress_nodes5000.txt'

NODES      = 5000
N_THREADS  = os.cpu_count() or 16
BATCH_SIZE = 50_000

os.makedirs(OUT_DIR, exist_ok=True)


def get_valid_existing_chunks():
    """
    Returns a set of chunk indices that have already been processed
    and whose output files are non-zero in size (i.e. not corrupt/incomplete).
    """
    valid = set()
    for path in glob.glob(os.path.join(OUT_DIR, "chunk_*.parquet")):
        if os.path.getsize(path) > 0:
            basename = os.path.basename(path)          # chunk_0042.parquet
            idx_str  = basename.replace("chunk_", "").replace(".parquet", "")
            try:
                valid.add(int(idx_str))
            except ValueError:
                pass
    return valid


def evaluate_batch_sf(fens, sf_path, nodes):
    """
    Comunicação direta UCI via subprocess para máxima velocidade.
    """
    results = []
    process = subprocess.Popen(
        [sf_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1
    )

    def write(msg):
        process.stdin.write(msg + "\n")
        process.stdin.flush()

    write("uci")
    write("setoption name Hash value 16")
    write("isready")

    while True:
        line = process.stdout.readline()
        if "readyok" in line:
            break

    re_score_cp   = re.compile(r"score cp (-?\d+)")
    re_score_mate = re.compile(r"score mate (-?\d+)")

    for fen in fens:
        # Extract side to move from FEN (second field)
        side = fen.split()[1]   # 'w' or 'b'

        write(f"position fen {fen}")
        write(f"go nodes {nodes}")

        last_cp = 0
        while True:
            line = process.stdout.readline()
            if not line:
                break
            if "info" in line and "score" in line:
                m_mate = re_score_mate.search(line)
                m_cp   = re_score_cp.search(line)
                if m_mate:
                    last_cp = 10000 if int(m_mate.group(1)) > 0 else -10000
                elif m_cp:
                    last_cp = int(m_cp.group(1))
            if "bestmove" in line:
                # ── THE FIX ──────────────────────────────────────────────
                # Stockfish score is side-to-move relative.
                # Negate for Black so stored cp is always White-relative.
                cp_white = last_cp if side == 'w' else -last_cp
                wdl = 1.0 / (1.0 + np.exp(-cp_white / 320.0))
                results.append({'fen': fen, 'cp': cp_white, 'wdl': wdl})
                break

    write("quit")
    process.terminate()
    return results


def worker_fn(args):
    fens, sf_path, nodes = args
    try:
        return evaluate_batch_sf(fens, sf_path, nodes)
    except Exception as e:
        return [f"ERRO: {str(e)}"]


def main():
    print(f"--- Ultra RAM-SAFE Pipeline (Nodes: {NODES}) ---")
    print(f"CPUs: {N_THREADS}  |  Nodes: {NODES}  |  Batch: {BATCH_SIZE:,}")

    if not os.path.exists(SF_PATH):
        print(f"ERRO: Executável do Stockfish não encontrado em: {SF_PATH}")
        print("Por favor, instale o Stockfish ou edite a variável SF_PATH no script.")
        return

    shards = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
    if not shards:
        print(f"ERRO: Nenhum arquivo .parquet encontrado em {INPUT_DIR}")
        return

    # ── Retomada robusta ──────────────────────────────────────────
    # Verifica quais chunks já existem e têm tamanho > 0 bytes.
    done_chunks = get_valid_existing_chunks()
    print(f"Chunks já processados (válidos): {len(done_chunks)}")
    if done_chunks:
        print(f"  Índices: {sorted(done_chunks)[:10]}{'...' if len(done_chunks) > 10 else ''}")

    # Conta o total de posições disponíveis para estimar o ETA global
    print(f"\nEscaneando {len(shards)} shards para contar posições...")
    total_positions = 0
    shard_sizes = {}
    for shard_path in shards:
        try:
            pf = pq.ParquetFile(shard_path)
            n  = pf.metadata.num_rows
            shard_sizes[shard_path] = n
            total_positions += n
        except Exception as e:
            print(f"  Aviso: não foi possível ler {shard_path}: {e}")

    total_chunks = (total_positions + BATCH_SIZE - 1) // BATCH_SIZE
    skipped      = len(done_chunks)
    remaining    = total_chunks - skipped
    print(f"Total de posições: {total_positions:,}  |  Chunks totais: {total_chunks}  |  Restantes: {remaining}")

    if remaining <= 0:
        print("✓ Todas as posições disponíveis já foram processadas!")
        return

    # ── Iteração por shards ───────────────────────────────────────
    t_session_start = time.time()
    chunk_counter   = 0          # índice global do chunk (usado para nomear arquivo)
    chunks_done_session = 0      # quantos chunks concluímos NESTA sessão (para ETA)
    elapsed_accumulator = 0.0   # soma de tempos dos chunks desta sessão

    for shard_path in shards:
        num_rows = shard_sizes.get(shard_path, 0)
        if num_rows == 0:
            continue

        print(f"\nShard: {os.path.basename(shard_path)} ({num_rows:,} posições)")

        for i in range(0, num_rows, BATCH_SIZE):
            chunk_idx = chunk_counter
            chunk_counter += 1

            out_name = os.path.join(OUT_DIR, f"chunk_{chunk_idx:04d}.parquet")

            # Pula chunks já processados com arquivo válido (> 0 bytes)
            if chunk_idx in done_chunks:
                print(f"  Chunk {chunk_idx:04d} — já processado, pulando.")
                continue

            # Remove arquivo corrompido (zero bytes) se existir
            if os.path.exists(out_name) and os.path.getsize(out_name) == 0:
                print(f"  Chunk {chunk_idx:04d} — arquivo zero bytes detectado, reprocessando.")
                os.remove(out_name)

            df_part = pd.read_parquet(shard_path).iloc[i : i + BATCH_SIZE]
            fens    = df_part['fen'].tolist()
            del df_part; gc.collect()

            print(f"  Processando Chunk {chunk_idx:04d} ({len(fens):,} posições)...")

            t0 = time.time()

            sub_size    = max(1, len(fens) // N_THREADS)
            sub_chunks  = [fens[j : j + sub_size] for j in range(0, len(fens), sub_size)]
            worker_args = [(sc, SF_PATH, NODES) for sc in sub_chunks if sc]

            results = []
            with ProcessPoolExecutor(max_workers=N_THREADS) as pool:
                futures = [pool.submit(worker_fn, a) for a in worker_args]
                for future in as_completed(futures):
                    res = future.result()
                    if res and isinstance(res[0], str) and res[0].startswith("ERRO"):
                        print(f"\n{res[0]}")
                        return
                    results.extend(res)

            if results:
                df_out = pd.DataFrame(results)
                df_out.to_parquet(out_name, index=False)
                del df_out; gc.collect()
            else:
                print(f"  Aviso: Chunk {chunk_idx:04d} retornou vazio, nenhum arquivo salvo.")

            # ── ETA ──────────────────────────────────────────────────
            elapsed = time.time() - t0
            elapsed_accumulator  += elapsed
            chunks_done_session  += 1
            pos_per_sec           = len(results) / elapsed if elapsed > 0 else 0
            avg_time_per_chunk    = elapsed_accumulator / chunks_done_session
            chunks_remaining      = remaining - chunks_done_session
            eta_sec               = avg_time_per_chunk * chunks_remaining
            eta_min               = eta_sec / 60

            print(
                f"    Chunk {chunk_idx:04d} — {len(results):,} pos | "
                f"{elapsed:.1f}s ({pos_per_sec:.1f} p/s) | "
                f"ETA: {eta_min:.1f} min ({chunks_remaining} chunks restantes)"
            )

            # Persiste progresso
            with open(PROGRESS_F, 'w') as f:
                f.write(f"{chunks_done_session + skipped}")

    total_elapsed = time.time() - t_session_start
    print(f"\n✓ Processo concluído com sucesso!")
    print(f"  Tempo total da sessão: {total_elapsed/60:.1f} min")
    print(f"  Arquivos salvos em: {OUT_DIR}")


if __name__ == "__main__":
    main()
