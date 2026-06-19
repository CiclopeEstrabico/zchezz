import os, sys, json, subprocess, time, math, random, threading, datetime, tempfile, signal, io
from queue import Queue, Empty
import chess
import chess.pgn

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES
# ═══════════════════════════════════════════════════════════════════════════════

 # Executar limpeza
cmd = (
    'Get-CimInstance Win32_Process -Filter "Name = \'zchezz*\'" | '
    'Invoke-CimMethod -MethodName Terminate'
)
subprocess.run(["powershell", "-Command", cmd], capture_output=True)
print("[OK] Processos Anteriores encerrados.")

ENGINES_CFG = [
    {
        "path":     r"engine\c\zchezz_v305\zchezz.exe",
        "label":    "Zchezz-v305",
        "tc_mode":  "movetime",
        "tc_value": 200,
        "tc_inc":   0
    },
    {
        "path":     r"engine\c\zchezz_v305\zchezz.exe",
        "label":    "Zchezz-v305",
        "tc_mode":  "movetime",
        "tc_value": 200,
        "tc_inc":   0
    },
]

TOTAL_GAMES  = 20000
CONCURRENCY  = 16
MAX_PLIES    = 400
MOVE_TIMEOUT = 35.0
REPORT_EVERY = 10   # Imprime status a cada N jogos concluídos

SAVE_PGN            = False
SAVE_OPENING_IN_EPD = True
RESULTS_DIR         = r"tests\selfplay_results"

OPENING_MODE        = "book"  # "book", "random" ou "all" (N1 book + N2 random separados)
BOOK_PORTION        = 0.97    # Usado no modo "all" (ex: 0.5 = 50% dos jogos do 'all' serão de livro puramente)
OPENING_FOLDER      = r"openings"
RANDOM_PLIES        = 6
SAME_OPENING_TWICE  = True   # Se True, ambos jogos do par usam a mesma posição de abertura
COLOR_SWAP          = True    # Sempre joga cada abertura nos dois lados (Engine1 Brancas E Pretas)

# ═══════════════════════════════════════════════════════════════════════════════
# SAÍDA E LOGS
# ═══════════════════════════════════════════════════════════════════════════════
os.makedirs(RESULTS_DIR, exist_ok=True)
TIMESTAMP  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE   = os.path.join(RESULTS_DIR, f"selfplay_{TIMESTAMP}.log")
PGN_FILE   = os.path.join(RESULTS_DIR, f"selfplay_{TIMESTAMP}.pgn")
EPD_FILE   = os.path.join(RESULTS_DIR, f"selfplay_{TIMESTAMP}.epd")

MATE_SCORE = 99999999

_log_lock = threading.Lock()
def log(*args):
    msg = " ".join(map(str, args))
    with _log_lock:
        print(msg, flush=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE INSTANCE (PERSISTENTE)
# ═══════════════════════════════════════════════════════════════════════════════

class EngineInstance:
    def __init__(self, path: str, label: str, tc_mode="depth", tc_value=7, tc_inc=0):
        self.path      = os.path.abspath(path)
        self.label     = label
        self.tc_mode   = tc_mode
        self.tc_value  = tc_value
        self.tc_inc    = tc_inc
        self.process   = None
        self.queue     = Queue()
        self.suppress_nnue = False
        self.start_fen = None  # Set per-game; None means standard startpos

    def _reader(self):
        try:
            for line in iter(self.process.stdout.readline, ''):
                line = line.strip()
                if line: self.queue.put(line)
        except Exception: pass

    def _stderr_reader(self):
        try:
            for line in iter(self.process.stderr.readline, ''):
                line = line.strip()
                if not line: continue
                if self.suppress_nnue and any(x in line.lower() for x in ["nnue", "loading"]):
                    continue
                if any(x in line.lower() for x in ["error", "exception", "nnue", "loading"]):
                    log(f"  [Native/{self.label}] {line}")
        except Exception: pass

    def start(self):
        if self.process and self.process.poll() is None:
            return True # Já rodando

        engine_dir = os.path.dirname(self.path)
        try:
            args = [self.path]
            if os.path.exists(os.path.join(engine_dir, "nnue_weights.bin")):
                args += ["--nnue", "nnue_weights.bin"]
            
            # Global NNUE suppression logic
            global _nnue_logged
            with _nnue_logged_lock:
                if self.path in _nnue_logged: self.suppress_nnue = True
                else: _nnue_logged.add(self.path); self.suppress_nnue = False

            self.process = subprocess.Popen(
                args, cwd=engine_dir,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            threading.Thread(target=self._reader,        daemon=True).start()
            threading.Thread(target=self._stderr_reader, daemon=True).start()

            self._send("uci")
            t0 = time.time()
            while time.time() - t0 < 5:
                try:
                    if "uciok" in self.queue.get(timeout=0.1): break
                except Empty:
                    if self.process.poll() is not None: return False
            
            self._send("isready")
            t0 = time.time()
            while time.time() - t0 < 5:
                try:
                    if "readyok" in self.queue.get(timeout=0.1): break
                except Empty: pass
            
            self._send("ucinewgame")
            self._send("isready")
            return self.process.poll() is None
        except Exception as e:
            log(f"  [ERRO] Falha ao iniciar {self.label}: {e}")
            return False

    def reset_state(self):
        """Prepara o motor para uma nova partida sem reiniciar processo."""
        self._send("ucinewgame")
        self._send("isready")
        t0 = time.time()
        while time.time() - t0 < 2:
            try:
                if "readyok" in self.queue.get(timeout=0.1): break
            except Empty: pass

    def _send(self, cmd):
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(cmd + "\n")
                self.process.stdin.flush()
                return True
            except: pass
        return False

    def get_move(self, board, wtime=0, btime=0):
        if not self.process or self.process.poll() is not None:
            if not self.start(): return None, 0

        # Limpa queue
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except Empty: break

        m_list = " ".join([m.uci() for m in board.move_stack])
        if self.start_fen:
            pos_cmd = f"position fen {self.start_fen}" + (f" moves {m_list}" if m_list else "")
        else:
            pos_cmd = f"position startpos" + (f" moves {m_list}" if m_list else "")
        self._send(pos_cmd)

        if self.tc_mode == "depth":
            tc_cmd = f"go depth {self.tc_value}"
        elif self.tc_mode == "movetime":
            tc_cmd = f"go movetime {self.tc_value}"
        elif self.tc_mode == "nodes":
            tc_cmd = f"go nodes {self.tc_value}"
        else:
            tc_cmd = f"go wtime {wtime} btime {btime} winc {self.tc_inc} binc {self.tc_inc}"
        self._send(tc_cmd)
        
        move, score, nodes, depth = None, None, 0, 0
        t_start = time.time()
        while time.time() - t_start < MOVE_TIMEOUT:
            try:
                line = self.queue.get(timeout=0.1)
                if line.startswith("info "):
                    parts = line.split()
                    if "nodes" in parts:
                        try: nodes = int(parts[parts.index("nodes")+1])
                        except: pass
                    if "depth" in parts:
                        try: depth = int(parts[parts.index("depth")+1])
                        except: pass
                if "score cp" in line:
                    parts = line.split()
                    try: idx = parts.index("cp"); score = int(parts[idx+1])
                    except: pass
                elif "score mate" in line:
                    parts = line.split()
                    try:
                        m_val = int(parts[parts.index("mate")+1])
                        if board.turn == chess.WHITE:
                            score = MATE_SCORE if m_val > 0 else -MATE_SCORE
                        else:
                            score = MATE_SCORE if m_val < 0 else -MATE_SCORE
                    except: pass
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) > 1: move = parts[1]
                    break
            except Empty:
                if self.process.poll() is not None: break
                continue
        
        if not move: return "TIMEOUT" if self.process.poll() is None else None, 0, 0, 0
        
        # Convert eval score to white's perspective
        if board.turn == chess.BLACK and score is not None:
            score = -score
            
        return move, score, nodes, depth

    def stop(self):
        if self.process:
            try:
                self._send("quit")
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except: pass
            self.process = None

# Estatísticas globais
completed_games   = 0
_real_total_games = 0
start_time        = 0.0   # set in main() after opening indexing
_stats_lock       = threading.Lock()
_file_lock        = threading.Lock()
_nnue_logged      = set()
_nnue_logged_lock = threading.Lock()
_cfg_labels = [c["label"] for c in ENGINES_CFG]
stats = {n: {"w":0,"l":0,"d":0,"pts":0.0,"to":0,"err":0,"games":0,"total_ms":0,"total_moves":0,"total_nodes":0,"total_depth":0} for n in _cfg_labels}
h2h = {(n1,n2): {"w":0,"l":0,"d":0} for n1 in _cfg_labels for n2 in _cfg_labels if n1!=n2}

# ═══════════════════════════════════════════════════════════════════════════════
# AUXILIARES (UI e Elo)
# ═══════════════════════════════════════════════════════════════════════════════

class OpeningIndex:
    """
    Memory-efficient opening index.
    Scans files once at startup recording only byte offsets — no positions
    are kept in memory. A single seek() + readline() fetches any entry
    on demand, so 100 MB+ EPD files are handled without RAM issues.
    """
    def __init__(self):
        # Each entry: (filepath, byte_offset, kind)  kind = 'epd' | 'pgn'
        self._index = []

    def __len__(self):
        return len(self._index)

    def __bool__(self):
        return bool(self._index)

    def _build_epd(self, fpath):
        idx_path = fpath + ".idx"
        import struct
        if os.path.exists(idx_path) and os.path.getmtime(idx_path) >= os.path.getmtime(fpath):
            try:
                data = open(idx_path, "rb").read()
                offsets = list(struct.unpack_from(f"<{len(data)//8}Q", data))
                for off in offsets: self._index.append((fpath, off, "epd"))
                log(f"  [EPD] {os.path.basename(fpath)}: {len(offsets):,} pos (cached)")
                return
            except Exception: pass
        offsets = []
        with open(fpath, "rb") as f:
            while True:
                offset = f.tell()
                raw = f.readline()
                if not raw: break
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("#"): continue
                if len(line.split()) >= 4:
                    offsets.append(offset)
        try:
            import struct
            with open(idx_path, "wb") as f:
                f.write(struct.pack(f"<{len(offsets)}Q", *offsets))
        except Exception: pass
        for off in offsets: self._index.append((fpath, off, "epd"))
        log(f"  [EPD] {os.path.basename(fpath)}: {len(offsets):,} pos indexed")

    def _build_pgn(self, fpath):
        with open(fpath, "rb") as f:
            game_start = None
            while True:
                offset = f.tell()
                raw = f.readline()
                if not raw:
                    if game_start is not None:
                        self._index.append((fpath, game_start, "pgn"))
                    break
                line = raw.decode("utf-8", errors="ignore")
                if line.startswith("[Event "):
                    game_start = offset
                elif line.strip() == "" and game_start is not None:
                    self._index.append((fpath, game_start, "pgn"))
                    game_start = None

    def shuffle(self):
        random.shuffle(self._index)

    def fetch(self, idx):
        """Fetch one opening by index. Returns {"moves": [...], "fen": None|str}."""
        fpath, offset, kind = self._index[idx]
        if kind == "epd":
            with open(fpath, "rb") as f:
                f.seek(offset)
                raw = f.readline().decode("utf-8", errors="ignore").strip()
            parts = raw.split()
            fen = " ".join(parts[:4]) + " 0 1"
            try:
                board = chess.Board(fen)
                return {"moves": [], "fen": board.fen()}
            except Exception:
                return {"moves": [], "fen": None}
        else:  # pgn
            with open(fpath, "rb") as f:
                f.seek(offset)
                block = []
                blank_count = 0
                while True:
                    raw = f.readline()
                    if not raw: break
                    line = raw.decode("utf-8", errors="ignore")
                    block.append(line)
                    if line.strip() == "":
                        blank_count += 1
                        if blank_count >= 2: break
                    else:
                        blank_count = 0
            buf = io.StringIO("".join(block))
            game = chess.pgn.read_game(buf)
            if game is None: return {"moves": [], "fen": None}
            uci_seq, board = [], game.board()
            for move in game.mainline_moves():
                if move in board.legal_moves: uci_seq.append(move.uci()); board.push(move)
                else: break
            return {"moves": uci_seq, "fen": None}

    def random_pick(self):
        return self.fetch(random.randrange(len(self._index)))


def load_all_openings(folder_path):
    """
    Build an OpeningIndex from all .pgn and .epd files in folder_path.
    Only byte offsets are stored — no positions loaded into RAM.
    """
    idx = OpeningIndex()
    if not os.path.exists(folder_path): return idx
    for f_name in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, f_name)
        if f_name.endswith(".epd"):   idx._build_epd(fpath)
        elif f_name.endswith(".pgn"): idx._build_pgn(fpath)
    log(f"[Openings] Indexed {len(idx)} positions from '{folder_path}' (offsets only, no RAM load)")
    return idx

def random_opening(n_plies: int, start_board=None) -> dict:
    board = start_board.copy() if start_board else chess.Board()
    moves = []
    for _ in range(n_plies):
        legal = list(board.legal_moves)
        if not legal or board.is_game_over(): break
        move = random.choice(legal)
        moves.append(move.uci()); board.push(move)
    return {"moves": moves, "fen": None}

def elo_diff(wins, losses, draws):
    N = wins + losses + draws
    if N == 0: return "N/A"
    E = (wins + draws * 0.5) / N
    elo = -800.0 if E==0 else (800.0 if E==1 else -400 * math.log10(1/E - 1))
    return f"{'+' if elo>=0 else ''}{elo:.1f}"

def print_status(is_final: bool = False):
    global completed_games, _real_total_games, start_time, stats, h2h
    elapsed = time.time() - start_time
    gps_val = completed_games / elapsed if elapsed > 1 else 0
    gps     = f"{gps_val:.2f}" if elapsed > 1 else "—"

    if not is_final and gps_val > 0 and completed_games < _real_total_games:
        remaining = (_real_total_games - completed_games) / gps_val
        h, m, s   = int(remaining // 3600), int((remaining % 3600) // 60), int(remaining % 60)
        eta_str   = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
        eta_line  = f"ETA    : ~{eta_str} (a {datetime.datetime.now() + datetime.timedelta(seconds=remaining):%H:%M:%S})\n"
    else:
        eta_line  = ""

    title = "RESULTADO FINAL" if is_final else "STATUS"
    log(f"\n{'='*52}\n=== {title}\nJogos  : {completed_games}/{_real_total_games} | Vel: {gps} j/s\nTempo  : {elapsed:.1f}s\n{eta_line}{'-'*52}")
    for r, (name, s) in enumerate(sorted(stats.items(), key=lambda x: x[1]["pts"], reverse=True), 1):
        g = s["games"]
        mean_ms = int(s["total_ms"]/s["total_moves"]) if s["total_moves"] else 0
        pct = lambda n: f"{n/g*100:.1f}%" if g else "0.0%"
        alerts = f"  [!] T:{s['to']} Err:{s['err']}" if s["to"] or s["err"] else ""
        log(f"{r}. {name} | Pts:{s['pts']}/{g} ({pct(s['pts'])})\n   V:{s['w']}({pct(s['w'])}) E:{s['d']}({pct(s['d'])}) D:{s['l']}({pct(s['l'])}) T.med:{mean_ms}ms{alerts}")
    log(f"{'='*52}")

def format_eval_comment(score, ms):
    if score is None: return f"{{ {ms}ms }}"
    if abs(score) > 89000:
        mate = math.ceil((900000 - abs(score)) / 2) + 1
        return f"{{ #{'' if score>=0 else '-'}{mate} | {ms}ms }}"
    return f"{{ {'+' if score>=0 else ''}{score/100:.2f} | {ms}ms }}"

# ═══════════════════════════════════════════════════════════════════════════════
# GAME LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def play_game(game_id, w_engine, b_engine, opening, results_queue):
    start_t = time.time()
    # Reset state instead of starting process
    w_engine.reset_state(); b_engine.reset_state()

    # Opening is a dict: {"moves": [...], "fen": None|"<fen>"}
    opening_fen_base = opening.get("fen")  # None for PGN openings
    forced_moves     = opening.get("moves", [])

    # Set start_fen on engines so get_move sends the right position command
    w_engine.start_fen = opening_fen_base
    b_engine.start_fen = opening_fen_base

    board = chess.Board(opening_fen_base) if opening_fen_base else chess.Board()
    pgn_game = chess.pgn.Game()
    if opening_fen_base:
        pgn_game.headers["FEN"]      = opening_fen_base
        pgn_game.headers["SetUp"]    = "1"
    pgn_game.headers.update({"Event": "Selfplay EXE", "Round": str(game_id), "White": w_engine.label, "Black": b_engine.label})

    node, ply, applied = pgn_game, 0, []
    # Capture opening positions for EPD if requested
    opening_history_states = []
    for uci in forced_moves:
        try:
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                pre_fen = board.fen()
                board.push(move); node = node.add_main_variation(move); applied.append(uci); ply += 1
                if SAVE_OPENING_IN_EPD:
                    opening_history_states.append({"fen": pre_fen, "score": None, "evaluator": "book"})
            else: break
        except: break
    
    opening_fen = board.fen()
    winner, reason, to, err = None, "Normal", False, False
    timing = {w_engine.label: {"ms":0,"moves":0,"nodes":0,"depth":0}, b_engine.label: {"ms":0,"moves":0,"nodes":0,"depth":0}}
    
    wtime = w_engine.tc_value if w_engine.tc_mode == "fixedtime" else 0
    btime = b_engine.tc_value if b_engine.tc_mode == "fixedtime" else 0

    # Store each state to save as EPD later
    history_states = []

    try:
        while not board.is_game_over() and ply < MAX_PLIES:
            is_w = board.turn == chess.WHITE
            active = w_engine if is_w else b_engine
            t0 = time.time()
            
            # Captura o fen antes do motor realizar a jogada
            current_fen = board.fen()
            
            move_str, score, move_nodes, move_depth = active.get_move(board, wtime, btime)
            ms = int((time.time() - t0) * 1000)
            timing[active.label]["ms"]    += ms
            timing[active.label]["moves"] += 1
            timing[active.label]["nodes"] += move_nodes
            timing[active.label]["depth"] += move_depth

            if active.tc_mode == "fixedtime":
                if is_w: wtime = max(0, wtime - ms + w_engine.tc_inc)
                else:    btime = max(0, btime - ms + b_engine.tc_inc)

            if move_str == "TIMEOUT": to = True; reason = f"Timeout ({active.label})"; winner = "black" if is_w else "white"; break
            if not move_str: err = True; reason = f"Erro ({active.label})"; winner = "black" if is_w else "white"; break

            try:
                move = board.parse_uci(move_str)
                # Reject null moves (0000 / (none)) and moves outside the legal
                # set.  Pushing a null move corrupts the repetition counter and
                # can trigger a false draw before real 3-fold has occurred.
                if move == chess.Move.null() or move not in board.legal_moves:
                    raise ValueError(f"Null/illegal: {move_str}")
                board.push(move)
                node = node.add_main_variation(move, comment=format_eval_comment(score, ms))
                ply += 1
                history_states.append({"fen": current_fen, "score": score, "evaluator": active.label})
            except:
                err = True
                _is_null = move_str in ("0000", "(none)")
                reason = (
                    f"Vitoria por Lance Nulo ({active.label} jogou {move_str})"
                    if _is_null else
                    f"Lance Invalido ({move_str}) [{active.label}]"
                )
                winner = "black" if is_w else "white"
                # Annotate the last PGN node so the infraction is visible on replay
                node.comment = (
                    f"[RESULTADO: {active.label} jogou lance nulo/ilegal '{move_str}' — "
                    f"vitoria para {'Pretas' if is_w else 'Brancas'}]"
                )
                break

        if not to and not err:
            res = board.result()
            if res == "1-0": winner = "white"; reason = "Checkmate" if board.is_checkmate() else "Vitoria"
            elif res == "0-1": winner = "black"; reason = "Checkmate" if board.is_checkmate() else "Vitoria"
            else: winner = None; reason = "Empate"

        # Set PGN Result and Termination headers
        if   winner == "white": pgn_result = "1-0"
        elif winner == "black": pgn_result = "0-1"
        else:                   pgn_result = "1/2-1/2"
        pgn_game.headers["Result"]      = pgn_result
        pgn_game.headers["Termination"] = reason

        # Normalise scores: None→0, mate→±MATE_SCORE (matches tournament.py)
        def normalise_score(sc):
            if sc is None: return 0
            if abs(sc) >= MATE_SCORE // 2:
                return MATE_SCORE if sc > 0 else -MATE_SCORE
            return sc

        # Combine opening positions (if SAVE_OPENING_IN_EPD) + game positions
        all_states = opening_history_states + history_states

        epd_lines = []
        final_res = pgn_result
        for idx, st in enumerate(all_states):
            norm_score = normalise_score(st['score'])
            epd_lines.append(f"{st['fen']} c0 \"{final_res}\"; c1 \"{norm_score}\"; c2 \"{st['evaluator']}\"; c3 \"{idx}\";")

        # Empty string when no positions — main loop guards against writing blank lines
        final_epd_str = "\n".join(epd_lines)

        results_queue.put({
            "id": game_id, "white": w_engine.label, "black": b_engine.label, "winner": winner, "reason": reason,
            "pgn": str(pgn_game) if SAVE_PGN else None,
            "epd": final_epd_str,
            "timeout": to, "error": err, "timing": timing, "duration": time.time() - start_t
        })
    except Exception as e:
        log(f"  [ERRO JOGO {game_id}] Exception: {e}")

def main():
    global completed_games, _real_total_games, start_time, _stats_lock, _file_lock, stats, h2h
    all_openings = load_all_openings(OPENING_FOLDER) if OPENING_MODE in ["book", "all"] else OpeningIndex()

    # Shuffle the index once so book picks are non-repeating across the run
    all_openings.shuffle()

    # ── Opening count summary ──────────────────────────────────────────────
    n_book_positions = len(all_openings)
    log(f"[Openings] {n_book_positions:,} linhas de abertura indexadas de '{OPENING_FOLDER}'")

    N = len(ENGINES_CFG)
    pairs_idx      = [(i,j) for i in range(N) for j in range(i+1, N)]
    pair_multiplier = 2  # COLOR_SWAP é sempre ativo: cada abertura gera 2 jogos (um por cor)

    # ── Cap book games to available openings ──────────────────────────────
    # TOTAL_GAMES representa o total de posições de abertura alocadas no torneio.
    # Se SAME_OPENING_TWICE for True, o número real de jogos dobra.
    raw_iterations = max(1, TOTAL_GAMES // max(1, len(pairs_idx)))

    if OPENING_MODE == "book":
        raw_n_book, raw_n_rand = raw_iterations, 0
    elif OPENING_MODE == "random":
        raw_n_book, raw_n_rand = 0, raw_iterations
    else:  # "all"
        raw_n_book = int(raw_iterations * BOOK_PORTION)
        raw_n_rand = raw_iterations - raw_n_book

    # Cap: each pair can consume at most n_book_positions openings
    # (they cycle via modulo, so the real cap is the full index).
    # For a fair round-robin each pair should get the same number, so cap
    # globally: max book iterations per pair = n_book_positions // len(pairs)
    # (or len(pairs)==0 guard).
    if n_book_positions > 0 and raw_n_book > 0:
        max_book_iters = max(1, n_book_positions // max(1, len(pairs_idx)))
        if raw_n_book > max_book_iters:
            capped_n_book = max_book_iters
            log(f"[Openings] AVISO: {raw_n_book} iterações de livro por par solicitadas, "
                f"mas apenas {n_book_positions} posições disponíveis para {len(pairs_idx)} par(es). "
                f"Reduzindo para {capped_n_book} iterações de livro por par.")
            raw_n_book = capped_n_book
            raw_n_rand += (raw_iterations - raw_n_book - raw_n_rand)  # make up diff with random

    iterations     = raw_n_book + raw_n_rand
    n_book_f       = raw_n_book
    n_rand_f       = raw_n_rand
    games_per_pair = iterations * pair_multiplier
    _real_total_games = games_per_pair * len(pairs_idx)

    book_games_total = n_book_f * 2 * len(pairs_idx)
    rand_games_total = n_rand_f * 2 * len(pairs_idx)
    log(f"[Openings] Jogos com livro : {book_games_total:,}  |  Jogos aleatórios: {rand_games_total:,}  |  Total: {_real_total_games:,}  (color-swap sempre ativo)")

    schedule = []
    gid = 1

    # Global counter so book picks cycle through the shuffled index without repeating
    book_cursor = 0

    for (i, j) in pairs_idx:
        n_b = min(n_book_f, len(all_openings))
        n_r = n_rand_f + (n_book_f - n_b)

        sequence = []
        for k in range(n_b):
            sequence.append(("book", (book_cursor + k) % max(1, len(all_openings))))
        book_cursor += n_b
        for _ in range(n_r):
            sequence.append(("random", None))

        for s_idx, (kind, oidx) in enumerate(sequence):
            # COLOR_SWAP sempre ativo: joga a abertura nos dois sentidos
            schedule.append((gid, i, j, kind, oidx)); gid += 1
            if SAME_OPENING_TWICE:
                # Repete a MESMA abertura com as cores invertidas
                schedule.append((gid, j, i, kind, oidx)); gid += 1
            else:
                # Usa a próxima abertura da sequência para o jogo de cor invertida
                next_oidx = (oidx + 1) % max(1, len(all_openings)) if kind == "book" else None
                schedule.append((gid, j, i, kind, next_oidx)); gid += 1

    random.shuffle(schedule)

    # start_time set here, AFTER indexing, so ETA reflects actual game time
    start_time = time.time()
    log(f"\n=== SELFPLAY EXE (Motores Persistentes) ===\nJogos: {_real_total_games} | Paralelismo: {CONCURRENCY}\n")

    task_q, res_q = Queue(), Queue()
    for item in schedule: task_q.put(item)

    def worker_loop():
        # Initialize engine processes
        instances = []
        for cfg in ENGINES_CFG:
            instances.append(EngineInstance(**cfg))
        for e in instances: e.start()

        while True:
            try:
                gid, w_idx, b_idx, kind, oidx = task_q.get_nowait()
                # Resolve opening lazily — only one seek+readline per game
                if kind == "book":
                    opening = all_openings.fetch(oidx)
                else:
                    opening = random_opening(RANDOM_PLIES)
                play_game(gid, instances[w_idx], instances[b_idx], opening, res_q)
                task_q.task_done()
            except Empty: break

        for e in instances: e.stop()

    threads = [threading.Thread(target=worker_loop, daemon=True) for _ in range(CONCURRENCY)]
    for t in threads: t.start()

    fe = open(EPD_FILE, "w", encoding="utf-8")
    fp = open(PGN_FILE, "w", encoding="utf-8") if SAVE_PGN else None

    try:
        while completed_games < _real_total_games:
            try: res = res_q.get(timeout=1.0)
            except Empty:
                if all(not t.is_alive() for t in threads): break
                continue
            
            completed_games += 1
            w, b = res["white"], res["black"]
            with _stats_lock:
                stats[w]["games"] += 1; stats[b]["games"] += 1
                stats[w]["total_ms"]    += res["timing"][w]["ms"];    stats[w]["total_moves"] += res["timing"][w]["moves"]
                stats[w]["total_nodes"] += res["timing"][w]["nodes"]; stats[w]["total_depth"] += res["timing"][w]["depth"]
                stats[b]["total_ms"]    += res["timing"][b]["ms"];    stats[b]["total_moves"] += res["timing"][b]["moves"]
                stats[b]["total_nodes"] += res["timing"][b]["nodes"]; stats[b]["total_depth"] += res["timing"][b]["depth"]
                if res["timeout"]: stats[w]["to"] += 1; stats[b]["to"] += 1
                if res["error"]:   stats[w]["err"] += 1; stats[b]["err"] += 1
                if res["winner"] == "white": stats[w]["w"]+=1; stats[w]["pts"]+=1; stats[b]["l"]+=1; h2h[(w,b)]["w"]+=1; h2h[(b,w)]["l"]+=1
                elif res["winner"] == "black": stats[b]["w"]+=1; stats[b]["pts"]+=1; stats[w]["l"]+=1; h2h[(w,b)]["l"]+=1; h2h[(b,w)]["w"]+=1
                else: stats[w]["d"]+=1; stats[w]["pts"]+=0.5; stats[b]["d"]+=1; stats[b]["pts"]+=0.5; h2h[(w,b)]["d"]+=1; h2h[(b,w)]["d"]+=1
            
            with _file_lock:
                # Fix: only write EPD if there are positions to write (avoids blank lines)
                if res["epd"]:
                    fe.write(res["epd"] + "\n"); fe.flush()
                if fp and res["pgn"]: fp.write(res["pgn"] + "\n\n"); fp.flush()
            
            if completed_games % REPORT_EVERY == 0: print_status()

    finally:
        fe.close()
        if fp: fp.close()

    log("\nTORNEIO FINALIZADO!")
    print_status(True)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s,f: sys.exit(0))
    main()
