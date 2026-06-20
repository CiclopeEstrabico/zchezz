import os, json, subprocess, time, math, random, threading, datetime, tempfile, signal, sys, io, glob
from queue import Queue, Empty
import chess
import chess.pgn
import struct as _struct

# Import shared ELO calculator (trinomial model, same as cutechess-cli)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo_calc import elo_difference, estimated_elo, elo_wdl_summary

# ── AUTO-DETECT LATEST ENGINE ─────────────────────────────────────────────────
def find_latest_engine():
    """Find the latest zchezz version directory by version number."""
    base = os.path.join(os.path.dirname(__file__), "..", "engine", "c")
    pattern = os.path.join(base, "zchezz_v*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError("No zchezz engine directories found")
    latest = dirs[-1]  # highest version
    name = os.path.basename(latest)
    exe = os.path.join(latest, "zchezz.exe")
    nnue = os.path.join(latest, "nnue_weights.bin")
    return exe, name, nnue

_exe, _name, _nnue = find_latest_engine()

# Engine under test (auto-detected, override if needed)
MY_ENGINE = (_exe, f"Zchezz-{_name.replace('zchezz_', '')}")
MY_ENGINE_OPTIONS = {"NNUE": os.path.abspath(_nnue)}

# Anchor engines
ANCHORS = [
    {
        "path": r"engine\stockfish\stockfish.exe",
        "label": "SF-2800",
        "elo": 2800,
        "options": {"UCI_LimitStrength": "true", "UCI_Elo": "2800"},
    },
    {
        "path": r"engine\stockfish\stockfish.exe",
        "label": "SF-2900",
        "elo": 2900,
        "options": {"UCI_LimitStrength": "true", "UCI_Elo": "2900"},
    },
]

# TOURNAMENT PARAMETERS
GAMES_PER_ANCHOR = 300
CONCURRENCY      = 8
MAX_PLIES        = 400
MOVE_TIMEOUT_MAX = 35.0

# SEARCH CONTROL
TC_MODE      = "movetime"       # depth, movetime or fixedtime
TC_VALUE     = 200
TC_WINC      = 200

# OPENINGS
OPENING_FOLDER = r"openings"

# OUTPUT
SAVE_PGN     = True
RESULTS_DIR  = r"tests\elo_results"
TIMESTAMP    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE     = os.path.join(RESULTS_DIR, f"elo_test_{TIMESTAMP}.log")
ALL_PGNS     = os.path.join(RESULTS_DIR, f"elo_test_{TIMESTAMP}.pgn")

os.makedirs(RESULTS_DIR, exist_ok=True)

_log_lock = threading.Lock()
def log(*args):
    msg = " ".join(map(str, args))
    with _log_lock:
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode('ascii', 'replace').decode('ascii'), flush=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

# ── OPENING LOADER ────────────────────────────────────────────────────────────

class OpeningIndex:
    def __init__(self):
        self._index = []

    def add_epd(self, fpath):
        with open(fpath, "rb") as f:
            while True:
                offset = f.tell()
                raw = f.readline()
                if not raw: break
                if raw.strip() and not raw.startswith(b"#"):
                    self._index.append((fpath, offset, "epd"))

    def add_pgn(self, fpath):
        with open(fpath, "rb") as f:
            game_start = None
            while True:
                offset = f.tell()
                raw = f.readline()
                if not raw:
                    if game_start is not None: self._index.append((fpath, game_start, "pgn"))
                    break
                line = raw.decode("utf-8", errors="ignore")
                if line.startswith("[Event "):
                    if game_start is not None: self._index.append((fpath, game_start, "pgn"))
                    game_start = offset

    def fetch(self, idx) -> dict:
        fpath, offset, kind = self._index[idx]
        if kind == "epd":
            with open(fpath, "rb") as f:
                f.seek(offset)
                raw = f.readline().decode("utf-8", errors="ignore").strip()
            parts = raw.split()
            fen = " ".join(parts[:4]) + " 0 1"
            return {"moves": [], "fen": fen}
        else:
            with open(fpath, "rb") as f:
                f.seek(offset)
                game = chess.pgn.read_game(io.StringIO(f.read().decode("utf-8", errors="ignore")))
            if not game: return None
            uci_seq, board = [], game.board()
            for move in game.mainline_moves():
                uci_seq.append(move.uci()); board.push(move)
            return {"moves": uci_seq, "fen": None}

    def random_pick(self):
        if not self._index: return None
        return self.fetch(random.randrange(len(self._index)))

def load_all_openings(folder):
    idx = OpeningIndex()
    if not os.path.exists(folder): return idx
    for f in os.listdir(folder):
        fpath = os.path.join(folder, f)
        if f.endswith(".epd"): idx.add_epd(fpath)
        elif f.endswith(".pgn"): idx.add_pgn(fpath)
    return idx

# ── ENGINE INSTANCE ───────────────────────────────────────────────────────────

class EngineInstance:
    def __init__(self, path, label, options=None, go_cmd_override=None, **kwargs):
        self.path    = os.path.abspath(path)
        self.label   = label
        self.options = options or {}
        self.go_cmd_override = go_cmd_override
        self.process = None
        self.queue   = Queue()
        self.start_fen = None
        self.history = []

    def _reader(self):
        try:
            for line in iter(self.process.stdout.readline, ''):
                line = line.strip()
                if line:
                    self.queue.put(line)
        except: pass

    def start(self):
        engine_dir = os.path.dirname(self.path)
        try:
            self.process = subprocess.Popen(
                [self.path],
                cwd=engine_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            threading.Thread(target=lambda: self.process.stderr.read(), daemon=True).start()
            threading.Thread(target=self._reader, daemon=True).start()
            
            self._send("uci")
            t0 = time.time()
            has_uciok = False
            while time.time() - t0 < 10:
                try:
                    line = self.queue.get(timeout=0.1).strip()
                    if "uciok" in line.lower(): 
                        has_uciok = True
                        break
                except Empty:
                    if self.process.poll() is not None: break
            if not has_uciok: return False

            for name, val in self.options.items():
                self._send(f"setoption name {name} value {val}")

            self._send("isready")
            t0 = time.time()
            while time.time() - t0 < 15:
                try:
                    if "readyok" in self.queue.get(timeout=0.1).lower(): break
                except Empty: pass
            
            self._send("ucinewgame")
            self._send("isready")
            t0 = time.time()
            while time.time() - t0 < 5:
                try:
                    if "readyok" in self.queue.get(timeout=0.1).lower(): break
                except Empty: pass
            return True
        except: return False

    def _send(self, cmd):
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(cmd + "\n")
                self.process.stdin.flush()
                return True
            except: pass
        return False

    def get_move(self, board, wtime, btime, winc, binc):
        if not self.process: self.start()
        
        # BULLETPROOF SYNC: stop any running search, drain its bestmove, then isready
        self._send("stop")
        # Drain until we see bestmove or timeout
        t0 = time.time()
        while time.time() - t0 < 0.5:
            try:
                line = self.queue.get(timeout=0.05)
                if line.startswith("bestmove"):
                    break  # consumed the stale bestmove
            except Empty:
                break  # nothing left, stop wasn't needed
        
        # Now isready as a hard sync barrier
        self._send("isready")
        t0 = time.time()
        while time.time() - t0 < 10:
            try:
                line = self.queue.get(timeout=0.1)
                if "readyok" in line:
                    break
            except Empty:
                pass
        
        # Final flush
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except Empty: break

        # Use proper position command with full move list (preserves history for repetition detection)
        m_list = " ".join([m.uci() for m in board.move_stack])
        if self.start_fen:
            pos_cmd = f"position fen {self.start_fen}" + (f" moves {m_list}" if m_list else "")
        else:
            pos_cmd = f"position startpos" + (f" moves {m_list}" if m_list else "")
        self._send(pos_cmd)
        
        if self.go_cmd_override:
            go_cmd = f"go {self.go_cmd_override}"
        elif TC_MODE == "movetime":
            go_cmd = f"go movetime {TC_VALUE}"
        elif TC_MODE == "depth":
            go_cmd = f"go depth {TC_VALUE}"
        else:
            go_cmd = f"go wtime {wtime} btime {btime} winc {winc} binc {binc}"

        if not self._send(go_cmd): return None
        move = None
        t0 = time.time()
        while time.time() - t0 < MOVE_TIMEOUT_MAX:
            try:
                line = self.queue.get(timeout=0.1)
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) > 1: move = parts[1]
                    break
            except Empty: pass
        return move

    def stop(self):
        if self.process:
            try: self._send("quit")
            except: pass
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except: pass
            self.process = None

# ── ELO ESTIMATOR ─────────────────────────────────────────────────────────────

class EloEstimator:
    """Track W/D/L per anchor and estimate ELO using trinomial model."""
    def __init__(self):
        self.per_anchor = {}  # label → {w, d, l, elo}

    def add(self, anchor_label, anchor_elo, score):
        if anchor_label not in self.per_anchor:
            self.per_anchor[anchor_label] = {"w": 0, "d": 0, "l": 0, "elo": anchor_elo}
        s = self.per_anchor[anchor_label]
        if score == 1.0: s["w"] += 1
        elif score == 0.0: s["l"] += 1
        else: s["d"] += 1

    def estimate(self):
        """Weighted average of per-anchor ELO estimates."""
        if not self.per_anchor:
            return 0.0, float('inf')

        total_w = sum(1.0 / max(ci, 1) ** 2
                      for label, s in self.per_anchor.items()
                      for est, ci in [estimated_elo(s["w"], s["d"], s["l"], s["elo"])])
        weighted_elo = 0.0
        combined_var = 0.0

        for label, s in self.per_anchor.items():
            est, ci = estimated_elo(s["w"], s["d"], s["l"], s["elo"])
            se = ci / 1.96 if ci < float('inf') else float('inf')
            if se == 0 or se == float('inf'):
                continue
            weight = 1.0 / se ** 2
            weighted_elo += est * weight
            combined_var += 1.0 / (se ** 2)

        if combined_var > 0:
            final_elo = weighted_elo / combined_var
            final_se = 1.0 / math.sqrt(combined_var)
            return final_elo, final_se * 1.96
        return 0.0, float('inf')

    def estimate_mle(self):
        """Backward-compatible interface."""
        return self.estimate()

# ── TORNEIO ───────────────────────────────────────────────────────────────────

def play_game(game_id, pair_cfg, results_queue, opening=None):
    white_cfg, black_cfg = pair_cfg
    w_eng = EngineInstance(**white_cfg)
    b_eng = EngineInstance(**black_cfg)
    
    opening_fen = opening.get("fen") if opening else None
    forced_moves = opening.get("moves") if opening else []
    
    w_eng.start_fen = opening_fen
    b_eng.start_fen = opening_fen
    
    if not w_eng.start() or not b_eng.start():
        results_queue.put({"id": game_id, "error": True})
        w_eng.stop(); b_eng.stop()
        return

    board = chess.Board(opening_fen) if opening_fen else chess.Board()
    pgn_game = chess.pgn.Game()
    pgn_game.headers["White"], pgn_game.headers["Black"] = w_eng.label, b_eng.label
    if opening_fen:
        pgn_game.headers["FEN"] = opening_fen
        pgn_game.headers["SetUp"] = "1"
    
    node = pgn_game
    for uci in (forced_moves or []):
        try:
            m = board.parse_uci(uci)
            board.push(m); node = node.add_main_variation(m)
        except: break

    wtime, btime = TC_VALUE if TC_MODE == "fixedtime" else 0, TC_VALUE if TC_MODE == "fixedtime" else 0
    winc, binc = TC_WINC if TC_MODE == "fixedtime" else 0, TC_WINC if TC_MODE == "fixedtime" else 0
    
    ply = 0
    last_mover_is_white = None
    broke_early = False
    while not board.is_game_over(claim_draw=True) and ply < MAX_PLIES:
        is_white = board.turn == chess.WHITE
        active = w_eng if is_white else b_eng
        t0 = time.time()
        move_str = active.get_move(board, wtime, btime, winc, binc)
        elapsed_ms = int((time.time() - t0) * 1000)
        
        if is_white: wtime = max(0, wtime - elapsed_ms + winc)
        else: btime = max(0, btime - elapsed_ms + binc)

        if not move_str:
            # Engine failed to return a move — treat as loss for that engine
            broke_early = True
            last_mover_is_white = is_white
            break
        try:
            move = board.parse_uci(move_str)
            if move in board.legal_moves:
                board.push(move); node = node.add_main_variation(move); ply += 1
            else:
                broke_early = True
                last_mover_is_white = is_white
                break
        except:
            broke_early = True
            last_mover_is_white = is_white
            break

    # Determine result
    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
    elif broke_early and last_mover_is_white is not None:
        # Engine that failed to move loses
        result = "0-1" if last_mover_is_white else "1-0"
    else:
        result = "1/2-1/2"  # hit MAX_PLIES, count as draw

    results_queue.put({
        "id": game_id, "white": w_eng.label, "black": b_eng.label,
        "result": result, "pgn": str(pgn_game)
    })
    w_eng.stop(); b_eng.stop()

def main():
    log(f"Iniciando Torneio de Elo - Zchezz (Modo Pares + Aberturas)")
    log(f"Testando: {MY_ENGINE[1]} | Config: {TC_MODE}={TC_VALUE}")
    
    op_idx = load_all_openings(OPENING_FOLDER)
    stats = {a["label"]: {"w": 0, "d": 0, "l": 0, "elo": a["elo"]} for a in ANCHORS}
    estimator = EloEstimator()
    results_queue = Queue()
    task_queue = Queue()
    
    schedule = []
    game_id = 1
    for anchor in ANCHORS:
        for _ in range(GAMES_PER_ANCHOR // 2):
            opening = op_idx.random_pick() if op_idx else {}
            # Jogo 1: Zchezz Brancas
            schedule.append((game_id, ({"path": MY_ENGINE[0], "label": MY_ENGINE[1], "options": MY_ENGINE_OPTIONS}, anchor), opening))
            game_id += 1
            # Jogo 2: Zchezz Pretas (Mesma Abertura)
            schedule.append((game_id, (anchor, {"path": MY_ENGINE[0], "label": MY_ENGINE[1], "options": MY_ENGINE_OPTIONS}), opening))
            game_id += 1

    random.shuffle(schedule)
    for s in schedule: task_queue.put(s)
    
    def runner():
        while True:
            try:
                gid, pair, op = task_queue.get_nowait()
                play_game(gid, pair, results_queue, op)
                task_queue.task_done()
            except Empty: break
            except Exception: break

    threads = [threading.Thread(target=runner, daemon=True) for _ in range(CONCURRENCY)]
    for t in threads: t.start()

    completed = 0
    total_games = len(schedule)
    while completed < total_games:
        try:
            res = results_queue.get(timeout=1.0)
            completed += 1
            if "error" in res: continue
            
            an_label = res["black"] if res["white"] == MY_ENGINE[1] else res["white"]
            res_val = res["result"]
            score = 1.0 if (res_val == "1-0" and res["white"] == MY_ENGINE[1]) or (res_val == "0-1" and res["black"] == MY_ENGINE[1]) else (0.0 if (res_val == "1-0" and res["black"] == MY_ENGINE[1]) or (res_val == "0-1" and res["white"] == MY_ENGINE[1]) else 0.5)
            
            estimator.add(an_label, stats[an_label]["elo"], score)
            if score == 1.0: stats[an_label]["w"] += 1
            elif score == 0.0: stats[an_label]["l"] += 1
            else: stats[an_label]["d"] += 1
            
            if SAVE_PGN:
                with open(ALL_PGNS, "a", encoding="utf-8") as f: f.write(res["pgn"] + "\n\n")
            
            elo, margin = estimator.estimate_mle()
            log(f"[{completed:3}/{total_games}] {res['white']:15} vs {res['black']:15} | Res: {res_val:7} | ELO: {elo:7.1f} +/- {margin:4.1f}")
        except Empty:
            if all(not t.is_alive() for t in threads): break
        except KeyboardInterrupt:
            log("Torneio interrompido!")
            break

    elo, margin = estimator.estimate_mle()
    log("\n" + "="*50 + f"\n* ELO ESTIMADO: {elo:.1f} +/- {margin:.1f} *\n" + "="*50)

if __name__ == "__main__":
    main()
