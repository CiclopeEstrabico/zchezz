
import os, chess, chess.engine, random, time, glob
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# ═══════════════════════════════════════════════════════════
#  Pipeline de Re-avaliação com Stockfish — Profundidade 6
# ═══════════════════════════════════════════════════════════

# AJUSTE ESTE CAMINHO PARA O SEU STOCKFISH LOCAL (Windows .exe)
# DOWNLOAD: https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-avxvnni.zip
SF_PATH    = r"engine\stockfish_fast\stockfish\stockfish-windows-x86-64-avxvnni.exe" 

INPUT_DIR  = r'data\extra_quiet_wdl'
OUT_DIR    = r'data\extra_quiet_sfdp8_wdl'
PROGRESS_F = r'sf_analyze\sf_progress_dp8.txt'
DEPTH      = 8

# Cuidado com estouro de memória: 
# Cada thread abre um Stockfish com 16MB de Hash.
N_THREADS  = os.cpu_count() or 16
TARGET     = 1_000_000_000 # Processa tudo o que encontrar
CHUNK_SIZE = 50_000

os.makedirs(OUT_DIR, exist_ok=True)

def evaluate_chunk(args):
    fens, depth, sf_path = args
    import chess, chess.engine
    results = []
    
    try:
        # Cada processo abre uma instância do Stockfish
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    except Exception as e:
        return [f"ERRO: Não foi possível abrir o Stockfish em {sf_path}. Erro: {str(e)}"]

    engine.configure({"Threads": 1, "Hash": 16})

    for fen in fens:
        try:
            board = chess.Board(fen)
            # Avaliação rápida no depth solicitado
            info  = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info['score'].white()

            if score.is_mate():
                cp = 10000 if score.mate() > 0 else -10000
            else:
                cp = score.score()

            # Converte centipawns → WDL via sigmoid (fator 320)
            wdl = 1.0 / (1.0 + np.exp(-cp / 320.0))
            results.append({'fen': fen, 'cp': cp, 'wdl': wdl})
        except Exception:
            pass

    engine.quit()
    return results

def main():
    print("--- Stockfish Quiet Analysis Pipeline (Depth 6) ---")
    
    if not os.path.exists(SF_PATH):
        print(f"ERRO: Executável do Stockfish não encontrado em: {SF_PATH}")
        print("Por favor, instale o Stockfish ou edite a variável SF_PATH no script.")
        return

    print(f"CPUs: {N_THREADS}  |  Stockfish Depth: {DEPTH}  |  Target: {TARGET:,} posições")

    # ── Coleta FENs dos parquets de entrada ──────────────────────
    parquet_shards = sorted(glob.glob(os.path.join(INPUT_DIR, "*.parquet")))
    if not parquet_shards:
        print(f"ERRO: Nenhum arquivo .parquet encontrado em {INPUT_DIR}")
        return

    print(f"Lendo FENs de {len(parquet_shards)} shards...")
    all_fens = []
    for fname in parquet_shards:
        try:
            df = pd.read_parquet(fname, columns=['fen'])
            all_fens.extend(df['fen'].tolist())
            del df; gc.collect()
        except Exception as e:
            print(f"Erro ao ler shard {fname}: {e}")

    total_avail = len(all_fens)
    print(f"Total disponível: {total_avail:,}")

    # ── Retomada — pula FENs já avaliadas ────────────────────────
    existing_out = sorted(glob.glob(os.path.join(OUT_DIR, "chunk_*_sf.parquet")))
    already_done = len(existing_out) * CHUNK_SIZE
    print(f"Progresso atual: {len(existing_out)} chunks salvos (~{already_done:,} posições)")

    if already_done >= total_avail:
        print("✓ Todas as posições disponíveis já foram processadas!")
        return

    # Shuffle determinístico para consistência entre retomadas
    random.seed(42)
    random.shuffle(all_fens)
    
    fens_to_eval = all_fens[already_done : already_done + TARGET]
    print(f"FENs a avaliar nesta sessão: {len(fens_to_eval):,}")
    del all_fens; gc.collect()

    # ── Loop de avaliação paralela por chunks ────────────────────
    chunks = [
        fens_to_eval[i:i+CHUNK_SIZE]
        for i in range(0, len(fens_to_eval), CHUNK_SIZE)
    ]
    print(f"Subdivindo em {len(chunks)} chunks de {CHUNK_SIZE:,}")

    t_start   = time.time()
    chunk_idx = len(existing_out)

    for i, chunk in enumerate(chunks):
        t0 = time.time()
        
        # Divide o chunk entre os processos
        sub_size   = max(1, len(chunk) // N_THREADS)
        sub_chunks = [chunk[j:j+sub_size] for j in range(0, len(chunk), sub_size)]
        worker_args = [(sc, DEPTH, SF_PATH) for sc in sub_chunks if sc]

        rows = []
        with ProcessPoolExecutor(max_workers=N_THREADS) as pool:
            futures = [pool.submit(evaluate_chunk, a) for a in worker_args]
            for future in as_completed(futures):
                res = future.result()
                if res and isinstance(res[0], str) and res[0].startswith("ERRO"):
                    print(f"\n{res[0]}")
                    return
                rows.extend(res)

        if not rows:
            print(f"Aviso: Chunk {chunk_idx} retornou vazio.")
            continue

        # Salva o chunk
        df_chunk = pd.DataFrame(rows)
        out_path = os.path.join(OUT_DIR, f'chunk_{chunk_idx:04d}_sf.parquet')
        df_chunk.to_parquet(out_path, index=False)

        # Estatísticas
        elapsed          = time.time() - t0
        total_session_done = (i + 1) * CHUNK_SIZE
        remaining_session = len(chunks) - i - 1
        eta_min           = (remaining_session * elapsed) / 60
        pos_per_sec       = len(rows) / elapsed

        print(f"  Chunk {i+1:03d}/{len(chunks)} — {len(rows):,} pos | {elapsed:.1f}s ({pos_per_sec:.1f} p/s) | ETA: {eta_min:.1f} min")

        with open(PROGRESS_F, 'w') as f:
            f.write(str(already_done + total_session_done))

        chunk_idx += 1
        del df_chunk, rows; gc.collect()

    print(f"\n✓ Processamento concluído!")
    print(f"  Tempo total da sessão: {(time.time()-t_start)/60:.1f} min")
    print(f"  Arquivos salvos em: {OUT_DIR}")

if __name__ == "__main__":
    main()
