import os, json, subprocess, time, math, random, threading, datetime, tempfile, signal, sys, io
from queue import Queue, Empty
import chess
import chess.pgn
import struct as _struct

# ── CONFIGURAÇÕES DO USUÁRIO ──────────────────────────────────────────────────
# Engine Zchezz que será testado
MY_ENGINE = (r"engine\c\zchezz_v305\zchezz.exe", "Zchezz-v305")
MY_ENGINE_OPTIONS = {"NNUE": os.path.abspath(r"engine\c\zchezz_v305\nnue_weights.bin"), "SyzygyPath": os.path.abspath(r"tablebases")}

# Motores Âncora
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

# PARÂMETROS DO TORNEIO
GAMES_PER_ANCHOR = 300          
CONCURRENCY      = 8           
MAX_PLIES        = 400         
MOVE_TIMEOUT_MAX = 35.0        

# CONTROLE DE BUSCA
TC_MODE      = "movetime"       #depth, movetime or fixedtime
TC_VALUE     = 200            
TC_WINC      = 200          

# GESTÃO DE ABERTURAS
OPENING_FOLDER = r"openings"

# OPÇÕES DE SAÍDA
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
    def __init__(self):
        self.results = []

    def add(self, anchor_elo, score):
        self.results.append((anchor_elo, score))

    def estimate_mle(self):
        if not self.results: return 0.0, 0.0
        
        # Regularization: add a small "draw" to avoid infinite Elo on 100% winrate
        smoothing = [(r[0], 0.5) for r in self.results[:2]] if len(self.results) > 10 else []
        temp_results = self.results + smoothing

        total_p = sum(r[1] for r in temp_results) / len(temp_results)
        elo_diff = 0
        if total_p > 0 and total_p < 1:
            elo_diff = -400 * math.log10(1/total_p - 1)
        E = sum(r[0] for r in temp_results) / len(temp_results) + elo_diff
        
        hess = 0
        for _ in range(30):
            grad, hess = 0, 0
            for A_i, score_i in temp_results:
                exp_term = 10**((A_i - E) / 400)
                p_i = 1 / (1 + exp_term) if exp_term < 1e15 else 0.0
                grad += (score_i - p_i)
                hess += -p_i * (max(1e-15, 1 - p_i)) * (math.log(10) / 400)
            if abs(grad) < 0.0001 or abs(hess) < 1e-18: break
            E = E - grad / (hess if abs(hess) != 0 else -1e-18)
        
        se = 1 / math.sqrt(abs(hess)) if abs(hess) > 0 else 0
        return E, se * 1.96

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
            
            estimator.add(stats[an_label]["elo"], score)
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
