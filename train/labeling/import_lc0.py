"""
train/labeling/import_lc0.py — Leela Chess Zero training-data importer

Downloads LC0 self-play `.tar` archives (the closest public equivalent to
AlphaZero data — DeepMind never released AlphaZero's own games) and converts
them into Zchezz's standard training format: parquet shards with columns
`fen` + `result` (CLAUDE.md rule 10), ready for train/train_nnue.py as a
RESULTS-ONLY source (`lam = 1.0`: target = real game outcome). The dataset
deliberately carries NO `cp` column, so blend_target() falls into its
"result only" path — the TD(1)/AlphaZero-style signal.

LC0 chunk format (lczero.org/dev/wiki/training-data-format-versions;
structs are PACKED, little-endian):

  V5TrainingData  (8308 bytes)
  V6TrainingData  (8356 bytes)   <- what modern runs emit

Shared head layout:

    offset  size  field
    ------  ----  --------------------------------------------------
         0     4  uint32  version            (3..6)
         4     4  uint32  input_format       (2 = classical, 3 = + e.p.)
         8  7432  float   probabilities[1858]      (policy, ignored here)
      7440   832  uint64  planes[104]        13 groups x 8 history steps
      8272     1  uint8   castling_us_ooo
      8273     1  uint8   castling_us_oo
      8274     1  uint8   castling_them_ooo
      8275     1  uint8   castling_them_oo
      8276     1  uint8   side_to_move_or_enpassant
      8277     1  uint8   rule50_count
      8278     1  uint8   invariance_info    (bit7 = stm in input fmt 3)
      8279     1  int8    result             (V5 ONLY; V6 renamed dummy)
      8280     4  float   root_q
      8284     4  float   best_q
      8288     4  float   root_d
      8292     4  float   best_d
      8296     4  float   root_m
      8300     4  float   best_m
      8304     4  float   plies_left
  V6 only:
      8308     4  float   result_q           <- game outcome [-1,1], STM-rel
      8312     4  float   result_d
      8316    12  float   played_q, played_d, played_m
      8328    12  float   orig_q, orig_d, orig_m       (value repair)
      8340     4  uint32  visits
      8344     4  uint16  played_idx, best_idx
      8348     4  float   policy_kld
      8352     4  uint32  reserved           (record ends at 8356)

PLANE SEMANTICS — input_format=1, EMPIRICALLY VERIFIED on test91 data
(the wiki documents formats 2/3; the actively-fed runs emit format 1,
whose layout differs — every claim below was confirmed against real
chunks via castling-rights anchors, see tests/debug_lc0_*.py):

  * planes[0..11] hold the CURRENT position, colour-GROUPED:
        t 0..5  = US   P N B R Q K     ("us" = side to move)
        t 6..11 = THEM P N B R Q K
    Planes 12..103 carry history/rule50 buckets (ignored here; they are
    all zero at ply 0, which is how game-start frames were identified).
  * Bit b of a plane maps to a square as follows:
        rank-from-the-side-to-move's-bottom = b // 8
        actual file                         = 7 - (b % 8)   <-- H-FIRST!
    i.e. within each row the bits run h,g,f,e,d,c,b,a. Verified anchors:
    e1 -> bit 3 (97% dominance among short-castling frames), h1 -> bit 0,
    a1 -> bit 7.
  * VERTICAL MIRROR: when BLACK is to move the whole position is mirrored
    before writing, so decoding un-mirrors: actual rank-from-white-bottom
    = 7 - (b // 8) iff black moves. Anchor: black's king on e8 with short
    rights also lands on bit 3.
  * Castling bytes are us/them-relative like everything else.
  * En passant is NOT stored in format 1 ("-").
  * stm byte (side_to_move_or_enpassant @8276): 0 = WHITE, 1 = BLACK.
    Polarity settled by 400/400 ply-0 game-start frames carrying byte 0
    (games always begin with white to move; alternation rate 1.000).

RESULT CONVENTION (why lam=1 works cleanly on this data):
  * V5: int8 `result` ∈ {-1, 0, +1}, SIDE TO MOVE's perspective.
  * V6: float `result_q` ∈ [-1, 1], SAME perspective (soft values possible
    via resignation-correction / adjudication).
  Converted to WHITE-relative probability p = (z_white + 1) / 2 with
  z_white = z (white to move) or -z (black to move), then snapped onto the
  exact grid {0.0, 0.5, 1.0}. Rows further than RESULT_SNAP_TOL from every
  grid point are DROPPED and counted — they would violate the project-wide
  `result` contract (rule 10).

USAGE
    python train/labeling/import_lc0.py --self-test      # decode unit test
    python train/labeling/import_lc0.py                  # config-block run (small)
    python train/labeling/import_lc0.py --tars URL1 URL2 --max-total-rows 500000

BIG RUN (--big) — bulk AlphaZero-style dataset, plan-first:
    python train/labeling/import_lc0.py --big            # PLAN ONLY: lists the date window's
                                                         # tars, sizes, ETA and row estimates
    python train/labeling/import_lc0.py --big --go       # download + convert into BIG_OUT_DIR
    knobs: --big-run / --big-date-from / --big-date-to /
           --big-budget-gb / --big-out-dir / --sample-every
    then filter to quiet .bin (the ONE position pipe):
        python train/labeling/process_positions.py \
            --in data/lc0_t91_big_res --out data/lc0_t91_big_q \
            --filters quiet --out-format bin
    then train (BIN_DATASETS entry or CLI):
        python train/train_nnue.py \
            --source kind=bin,path=data/lc0_t91_big_q/*.bin,k=1.0,name=lc0_t91_big \
            --checkpoint-source new --dataset-name lc0_t91_big

OUTPUT
    <out-dir>/part_NNNNN.parquet  shards with columns
        fen, result, cp, visits, root_q, tar
    (`cp` is LC0's own search evaluation in WHITE-relative centipawns,
    derived from root_q — see ADD_CP_FROM_ROOT_Q; it is NOT an outcome.)
    Feed to the trainer:
    python train/train_nnue.py \
        --source kind=parquet,path=data/lc0_test91_res,target_col=result,lam=1.0 \
        --checkpoint-source new --dataset-name lc0_smoke --epochs 2

QUIET FILTER + .BIN (the usual next step, reuses the ONE position pipe):
    python train/labeling/process_positions.py \
        --in data/lc0_test91_res --out data/lc0_test91_res_q \
        --filters quiet --out-format bin
    then train with:  --source kind=bin,path=data/lc0_test91_res_q/*.bin,k=1.0
"""

import argparse
import gzip
import os
import re
import tarfile
import time
import urllib.request

import numpy as np
import pandas as pd

# ═══════════════════════════ CONFIGURATION ═══════════════════════════
# Everything reachable from the CLI defaults to these constants (rule 8).
LC0_BASE_URL = "https://storage.lczero.org/files/training_data"
# test91 (run2) is LCZ's actively-fed run as of 2026-08. test80 stopped
# receiving real data after 2024-10 (recent tars there are 10 KB stubs).
DEFAULT_TARS = [
    "test91/training-run2-test91-20260820-0017.tar",
    "test91/training-run2-test91-20260820-0117.tar",
    "test91/training-run2-test91-20260820-0217.tar",
    "test91/training-run2-test91-20260820-0317.tar",
]
RAW_DIR = "C:/Zchezz/data_raw/lc0"                # downloaded .tar cache (gitignored)
OUT_DIR = "C:/Zchezz/data/lc0_test91_res"         # trainer-visible DATASETS dir
SAMPLE_EVERY = 10                                 # keep every Nth record; positions inside a
                                                  # game are highly correlated, N=10 keeps ~10%
MAX_TOTAL_ROWS = 1_000_000                        # stop converting once this many rows are kept
ROWS_PER_SHARD = 250_000                          # parquet shard size (trainer streams per file)
RESULT_SNAP_TOL = 0.05                            # max |soft_result − grid| to accept when
                                                  # snapping to {0,.5,1}; beyond → row dropped
MIN_VISITS = 0                                    # drop V6 records with fewer search visits
BUF_RECORDS = 16_384                              # decode-batch size (≈137 MB transient)
ADD_CP_FROM_ROOT_Q = True                         # store LC0's own search eval as a WHITE-relative
                                                  # `cp` column: cp = 320*logit((±root_q+1)/2). This is
                                                  # the labelling engine's evaluation (rule 10), NOT an
                                                  # outcome; results-only training (k=1.0 / lam=1.0)
                                                  # ignores it, but its presence makes the dataset
                                                  # reusable at any blend without relabelling.
CP_CLAMP = 12_000                                 # |cp| clamp; sigmoid(12000/320)≈1 so the exact cap
                                                  # is unidentifiable anyway, keeps int16 comfortable

# ── BIG-RUN PLANNER (--big) — ≥20 GB of fresh AlphaZero-style data ─────
# Discovers tar archives on the LCZ server within a date window, reports
# the plan (sizes, estimated rows, disk/training impact) WITHOUT touching
# the network beyond the directory listing, and downloads + converts only
# when --go is passed. Measured conversion yield on test91 (2026-08):
# ~4,800 records per compressed MB, ~73% survive process_positions' quiet
# filter, 75 B per .bin record.
BIG_RUN          = "test91"
BIG_DATE_FROM    = "20260801"                     # inclusive YYYYMMDD window
BIG_DATE_TO      = "20260831"                     # inclusive
BIG_BUDGET_GB    = 20.0                           # compressed tar volume to pull
BIG_SAMPLE_EVERY = 10                             # keep every Nth record
BIG_OUT_DIR      = "C:/Zchezz/data/lc0_t91_big_res"
BIG_GO           = False                          # False = plan only; --go flips this
DL_WORKERS       = 6                              # parallel tar downloads (server allows it;
                                                  # measured ~2 MB/s per connection)
# ══════════════════════════════════════════════════════════════════════

V5_RECORD = 8308
V6_RECORD = 8356
FILES = "abcdefgh"
STARTPOS_W = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
STARTPOS_B = STARTPOS_W.replace(" w ", " b ")


def _record_dtype(version: int) -> np.dtype:
    """Structured dtype with EXPLICIT offsets matching the packed C structs.

    Only consumed fields are named; `itemsize` pins the full stride so
    slicing stays exact. Offsets follow the wiki table in the docstring."""
    spec = [
        ("version",      np.uint32,   0),
        ("input_format", np.uint32,   4),
        ("planes",       (np.uint64, 104), 7440),
        ("cast_us_ooo",  np.uint8, 8272),
        ("cast_us_oo",   np.uint8, 8273),
        ("cast_them_ooo", np.uint8, 8274),
        ("cast_them_oo", np.uint8, 8275),
        ("stm_or_ep",    np.uint8, 8276),
        ("rule50",       np.uint8, 8277),
        ("invariance",   np.uint8, 8278),
        ("root_q",       np.float32, 8280),
        ("best_q",       np.float32, 8284),
        ("plies_left",   np.float32, 8304),
    ]
    itemsize = V5_RECORD
    if version == 5:
        spec.append(("result_i8", np.int8, 8279))
    else:
        itemsize = V6_RECORD
        spec += [
            ("result_q", np.float32, 8308),
            ("result_d", np.float32, 8312),
            ("visits",   np.uint32, 8340),
        ]
    return np.dtype({
        "names":    [s[0] for s in spec],
        "formats":  [s[1] for s in spec],
        "offsets":  [s[2] for s in spec],
        "itemsize": itemsize,
    })


def _dtype_for_size(size: int):
    """Detect the chunk-record version from the decompressed stream stride."""
    if size % V6_RECORD == 0:
        return 6, _record_dtype(6)
    if size % V5_RECORD == 0:
        return 5, _record_dtype(5)
    return None


# ────────────────────────────────────────────────────────────────────────
#  Plane decoding → FEN
# ────────────────────────────────────────────────────────────────────────

def side_to_move_is_black(rec: np.ndarray) -> np.ndarray:
    """Per-record side-to-move flag.

    Format 1 (test91 data): the dedicated byte, EMPIRICALLY 0=White/1=Black
    (settled via ply-0 game-start frames — see module docstring).
    Format 3 would carry stm in invariance_info bit 7 instead."""
    fmt = rec["input_format"]
    from_fmt3 = (rec["invariance"] >> 7) & 1
    return np.where(fmt == 3, from_fmt3, rec["stm_or_ep"]).astype(bool)


def en_passant_files(rec: np.ndarray) -> np.ndarray:
    """E.p. file (0..7) per record, or -1 when absent/unstored."""
    m = rec["stm_or_ep"].astype(np.int64)
    ep = np.where(m & (m - 1) == 0, np.log2(np.maximum(m, 1)).round(), -1).astype(np.int64)
    return np.where(rec["input_format"] == 3, ep, -1)


def decode_fens(rec: np.ndarray) -> tuple[list[str], np.ndarray, int]:
    """Vectorized planes → absolute-FEN decode for a batch of records.

    Layout contract (empirically verified, see module docstring):
      * planes[0..11]: t 0..5 = US P,N,B,R,Q K; t 6..11 = THEM (same order)
      * bit b: r_stm = b//8 counts ranks from the STM's bottom; within the
        row bits run h-first so actual file = 7 - b%8
      * black to move => vertical mirror (un-mirrored here)

    Emits white-POV FENs (Zchezz/python-chess convention: sq 0 = a8).
    Returns (fens, stms_black, n_startpos)."""
    n = len(rec)
    cur = rec["planes"][:, :12]                              # (n,12) t direct

    # Piece code per (record, square-bit): Σ (t+1)·bit_t(b); ≤1 piece/square.
    shifts = np.arange(64, dtype=np.uint64)
    bits = (cur[:, :, None] >> shifts[None, None, :]) & np.uint64(1)   # (n,12,64)
    weights = np.arange(1, 13, dtype=np.uint64)[None, :, None]
    codes = (bits * weights).sum(axis=1).astype(np.int32)              # (n,64)

    stm_black = side_to_move_is_black(rec)
    ep_files = en_passant_files(rec)

    codes_l = codes.tolist()
    black_l = stm_black.tolist()
    ep_l = ep_files.tolist()
    cu_ooo = rec["cast_us_ooo"].astype(bool).tolist()
    cu_oo = rec["cast_us_oo"].astype(bool).tolist()
    ct_ooo = rec["cast_them_ooo"].astype(bool).tolist()
    ct_oo = rec["cast_them_oo"].astype(bool).tolist()
    r50 = rec["rule50"].tolist()

    fens: list[str] = []
    n_startpos = 0
    for i in range(n):
        black = black_l[i]
        rows = [None] * 8                       # indexed by ABSOLUTE rank-top (0 = rank 8)
        for r_stm in range(8):
            rank_top = r_stm if black else 7 - r_stm
            row, empty = [], 0
            codes_row = codes_l[i]
            for f in range(8):                  # FEN order: files a .. h
                code = codes_row[r_stm * 8 + (7 - f)]   # LC0 packs h-first!
                if code == 0:
                    empty += 1
                    continue
                if empty:
                    row.append(str(empty))
                    empty = 0
                g = code - 1                     # 0..11
                is_us = g < 6                    # t 0..5 us block, 6..11 them
                letter = "PNBRQK"[g % 6]
                is_white = (not black) if is_us else black
                row.append(letter.upper() if is_white else letter.lower())
            if empty:
                row.append(str(empty))
            rows[rank_top] = "".join(row)

        if black:   # us = black, them = white
            w_k, w_q, b_k, b_q = ct_oo[i], ct_ooo[i], cu_oo[i], cu_ooo[i]
        else:       # us = white, them = black
            w_k, w_q, b_k, b_q = cu_oo[i], cu_ooo[i], ct_oo[i], ct_ooo[i]
        cast = (("K" if w_k else "") + ("Q" if w_q else "")
                + ("k" if b_k else "") + ("q" if b_q else "")) or "-"
        ep = "-" if ep_l[i] < 0 else FILES[ep_l[i]]
        fen = ("/".join(rows) + f" {'b' if black else 'w'} {cast} {ep} "
               + str(r50[i]) + " 1")
        if fen in (STARTPOS_W, STARTPOS_B):
            n_startpos += 1
        fens.append(fen)
    return fens, stm_black, n_startpos


def results_to_white_prob(rec: np.ndarray, version: int,
                          stm_black: np.ndarray) -> np.ndarray:
    """STM-relative outcome → WHITE-relative probability in [0,1].

    V5: int8 {-1,0,+1}; V6: float result_q in [-1,1]. Both are from the
    side to move's perspective, so negate first when black moves."""
    if version == 5:
        z = rec["result_i8"].astype(np.float64)
    else:
        z = rec["result_q"].astype(np.float64)
    z = np.where(stm_black, -z, z)               # now WHITE-relative
    return (z + 1.0) / 2.0


def snap_results(p: np.ndarray, tol: float):
    """Snap probabilities onto {0.0, 0.5, 1.0}.

    Returns (snapped, keep): rows containing NaN or farther than `tol`
    from every grid point are marked not-keepable."""
    grid = np.array([0.0, 0.5, 1.0])
    finite = np.isfinite(p)
    q = np.where(finite, p, 0.5)
    idx = np.abs(q[:, None] - grid[None, :]).argmin(axis=1)
    snapped = grid[idx]
    return snapped, finite & (np.abs(snapped - p) <= tol)


def _pawn_home_sanity(us_pawn_bits: np.ndarray) -> float:
    """Fraction of records where the side to move has ≥4 pawns inside its
    own first two relative ranks. Openings-heavy data should score high;
    a systematically low value means our orientation assumption is
    inverted — a cheap tripwire against silent plane bugs."""
    return float((us_pawn_bits[:, :16].sum(axis=1) >= 4).mean())


# ────────────────────────────────────────────────────────────────────────
#  Download + stream conversion
# ────────────────────────────────────────────────────────────────────────

def download_tar(rel_url: str, raw_dir: str) -> str:
    os.makedirs(raw_dir, exist_ok=True)
    dest = os.path.join(raw_dir, rel_url.replace("/", "_"))
    if os.path.exists(dest) and os.path.getsize(dest) > 10240:  # 10 KB = stub
        print(f"  cached: {dest} ({os.path.getsize(dest):,} B)")
        return dest
    url = f"{LC0_BASE_URL}/{rel_url}"
    print(f"  downloading {url}")
    t0 = time.time()
    # The storage server 403s Python-urllib's default User-Agent.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Zchezz-importer"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
            total = int(resp.headers.get("Content-Length", 0))
            done, last = 0, 0.0
            while True:
                block = resp.read(1 << 22)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                now = time.time()
                if now - last > 15:
                    last = now
                    pct = f"{done / total:.0%}" if total else "?"
                    mbps = done / max(now - t0, 1e-9) / 1e6
                    print(f"    {done:,} B ({pct}) {mbps:.1f} MB/s")
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)          # never leave a truncated tar behind
        raise
    print(f"  saved: {dest} ({done:,} B in {time.time() - t0:.0f}s)")
    return dest


def iter_records(paths, sample_every: int):
    """Yield (version, copied_record, tar_name) for every Nth chunk record.

    Records are COPIED out of the member buffer so multi-MB gzip buffers
    can be freed while batching (the structured rows are tiny)."""
    counter = 0
    for path in paths:
        tar_name = os.path.basename(path)
        with tarfile.open(path, "r|") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".gz"):
                    continue
                try:
                    raw = gzip.decompress(tf.extractfile(member).read())
                except (OSError, EOFError):
                    continue
                det = _dtype_for_size(len(raw))
                if det is None:
                    continue
                version, dt = det
                arr = np.frombuffer(raw, dtype=dt, count=len(raw) // dt.itemsize)
                arr = arr[arr["version"] == version]
                for row in arr:
                    counter += 1
                    if (counter - 1) % sample_every == 0:
                        yield version, row.copy(), tar_name


def convert(paths, out_dir: str, args) -> None:
    """Stream-decode every tar into parquet shards + a printed summary."""
    os.makedirs(out_dir, exist_ok=True)
    stats = {"read": 0, "thinned": 0, "kept": 0, "dropped_snap": 0, "dropped_visits": 0}
    versions = {5: 0, 6: 0}
    res_hist = {0.0: 0, 0.5: 0, 1.0: 0}
    shard_idx, shard_rows, startpos_total = 0, [], 0
    sanity_vals: list[float] = []
    t0 = time.time()

    def flush_shard():
        nonlocal shard_idx, shard_rows
        if not shard_rows:
            return
        df = pd.DataFrame(shard_rows,
                          columns=["fen", "result", "cp", "visits", "root_q", "tar"])
        path = os.path.join(out_dir, f"part_{shard_idx:05d}.parquet")
        df.to_parquet(path, index=False)
        print(f"  wrote {path} ({len(df):,} rows)")
        shard_idx += 1
        shard_rows = []

    def process_batch(buf):
        """Decode one same-version batch; append survivors to the shard."""
        nonlocal startpos_total
        if not buf:
            return
        version = buf[0][0]
        dt = _record_dtype(version)
        recs = np.array([b[1] for b in buf], dtype=dt)   # copies again: compact batch
        tars_of = [b[2] for b in buf]

        fens, stm_black, n_sp = decode_fens(recs)
        startpos_total += n_sp
        probs = results_to_white_prob(recs, version, stm_black)
        snapped, keep_snap = snap_results(probs, RESULT_SNAP_TOL)

        # LC0's own evaluation as WHITE-relative cp (see ADD_CP_FROM_ROOT_Q):
        # root_q is STM-relative expected score in [-1,1]; flip by stm, map
        # through the project's cp<->wdl convention (sigmoid(cp/320)).
        if ADD_CP_FROM_ROOT_Q and version == 6:
            q_stm = recs["root_q"].astype(np.float64)
            z_w = np.where(stm_black, -q_stm, q_stm)
            p_w = np.clip((z_w + 1.0) / 2.0, 1e-6, 1.0 - 1e-6)
            cp_white = np.clip(320.0 * np.log(p_w / (1.0 - p_w)),
                               -CP_CLAMP, CP_CLAMP).astype(np.int32)
        else:
            cp_white = None

        if version == 6:
            visits_ok = recs["visits"] >= args.min_visits
        else:
            visits_ok = np.ones(len(recs), bool)
        keep = keep_snap & visits_ok

        stats["dropped_snap"] += int((~keep_snap).sum())
        stats["dropped_visits"] += int((~visits_ok).sum())
        for s in snapped[keep]:
            res_hist[float(s)] += 1
        stats["kept"] += int(keep.sum())

        shifts = np.arange(64, dtype=np.uint64)
        us_pawn_bits = ((recs["planes"][:, 0][:, None] >> shifts[None, :])
                        & np.uint64(1)).astype(np.int32)      # group 0 = P us
        sanity_vals.append(_pawn_home_sanity(us_pawn_bits))

        for j in np.nonzero(keep)[0]:
            shard_rows.append({
                "fen": fens[j],
                "result": float(snapped[j]),
                "cp": (int(cp_white[j]) if cp_white is not None else None),
                "visits": int(recs["visits"][j]) if version == 6 else 0,
                "root_q": float(recs["root_q"][j]),
                "tar": tars_of[j],
            })
        if len(shard_rows) >= ROWS_PER_SHARD:
            flush_shard()

    buf: list[tuple[int, object, str]] = []
    pending_version = None
    for version, row, tar_name in iter_records(paths, args.sample_every):
        stats["thinned"] += 1
        versions[version] += 1
        if pending_version is not None and version != pending_version:
            process_batch(buf)
            buf = []
        pending_version = version
        buf.append((version, row, tar_name))
        stats["read"] += 1
        if len(buf) >= BUF_RECORDS:
            process_batch(buf)
            buf = []
            el = time.time() - t0
            print(f"  ... {stats['read']:,} sampled records, {stats['kept']:,} kept "
                  f"({el:.0f}s)", flush=True)
        if stats["kept"] >= args.max_total_rows:
            break
    process_batch(buf)
    flush_shard()

    total_kept = sum(res_hist.values())
    print("\n=== import summary ===")
    print(f"  records scanned:      {stats['read']:,} sampled "
          f"(v5={versions[5]:,}, v6={versions[6]:,})")
    print(f"  rows kept:            {stats['kept']:,}")
    print(f"  dropped unsnappable:  {stats['dropped_snap']:,}")
    print(f"  dropped <{args.min_visits} visits:   {stats['dropped_visits']:,}")
    print(f"  startpos FENs seen:   {startpos_total:,}  (orientation tripwire; "
          f"0 means decode is broken)")
    if sanity_vals:
        print(f"  pawn-home sanity:     {float(np.mean(sanity_vals)):.1%} "
              f"(STM pawns on own 2 ranks; near-0 means orientation inverted)")
    for k in sorted(res_hist):
        share = res_hist[k] / total_kept if total_kept else 0
        print(f"    result={k:.1f}: {res_hist[k]:,} ({share:.1%})")
    print(f"  shards written to {out_dir}: {shard_idx}")


# ────────────────────────────────────────────────────────────────────────
#  Self-test: synthesize startpos records (both sides to move) and verify
#  the decode round-trips exactly. Guards orientation/group/ordering bugs.
# ────────────────────────────────────────────────────────────────────────

def synthetic_record(stm_black: bool) -> np.ndarray:
    """Build a V6 record holding the startpos, written exactly the way LC0
    format-1 writers do: colour-grouped planes t 0..5 us / 6..11 them,
    bits packed h-first within each STM-bottom-up row, vertical mirror
    when black moves."""
    rec = np.zeros(1, dtype=_record_dtype(6))
    rec["version"] = 6
    rec["input_format"] = 1
    placement = [  # absolute squares a8=0 .. h1=63 → piece letter ('.'=empty)
        *(c for c in "rnbqkbnr"),          # rank 8
        *(c for c in "pppppppp"),          # rank 7
        *("." for _ in range(32)),         # ranks 6..3
        *(c for c in "PPPPPPPP"),          # rank 2
        *(c for c in "RNBQKBNR"),          # rank 1
    ]
    TYPE = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}
    for sq, ch in enumerate(placement):
        if ch == ".":
            continue
        is_white = ch.isupper()
        us = (is_white != stm_black)       # "us" owns it iff stm owns it
        t = TYPE[ch.upper()] + (0 if us else 6)
        file_, rank_top = sq % 8, sq // 8
        r_stm = rank_top if stm_black else (7 - rank_top)   # mirror on black
        rec["planes"][0][t] |= np.uint64(1) << np.uint64(r_stm * 8 + (7 - file_))
    rec["stm_or_ep"] = 1 if stm_black else 0
    rec["cast_us_oo"] = 1
    rec["cast_us_ooo"] = 1
    rec["cast_them_oo"] = 1
    rec["cast_them_ooo"] = 1
    rec["result_q"] = 0.0
    return rec


def self_test() -> None:
    for stm_black, expected in ((False, STARTPOS_W), (True, STARTPOS_B)):
        fens, bl, _ = decode_fens(synthetic_record(stm_black))
        got = fens[0]
        ok = got == expected and bool(bl[0]) == stm_black
        print(f"  [{'OK  ' if ok else 'FAIL'}] stm={'black' if stm_black else 'white'}: {got}")
        if not ok:
            print(f"         expected: {expected}")
            raise SystemExit(1)
    print("  self-test passed")


# ────────────────────────────────────────────────────────────────────────
#  Big-run planner: discover a date window of tars, plan or execute
# ────────────────────────────────────────────────────────────────────────

# The directory index is APACHE-STYLE HTML (<a href="NAME.tar">NAME.tar</a>
# followed by a human-readable size like "161M"), so sizes advertised on
# the page cannot be trusted numerically — each picked tar gets one cheap
# HEAD request for its exact Content-Length instead.
_HREF_TAR = re.compile(r'href="([^"]+\.tar)"', re.IGNORECASE)
_TAR_DATE = re.compile(r"-(\d{8})-\d{4}\.tar$")


def discover_tars(run: str, date_from: str, date_to: str, budget_gb: float,
                  raw_dir: str | None = None):
    """List the run's directory index and pick tars until the byte budget
    is reached. Returns (picked, skipped_stubs) where picked is a list of
    (relative_name, size_bytes) in chronological order.

    When raw_dir is given, tars already present in the local cache are
    SKIPPED and do not count against the byte budget — this makes a
    budget-capped window resumable: re-running with the same --raw-dir
    downloads only what is still missing."""
    url = f"{LC0_BASE_URL}/{run}/"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Zchezz-importer"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", "replace")

    names = sorted({m.group(1) for m in _HREF_TAR.finditer(html)})
    picked, total, stubs = [], 0, 0
    budget = int(budget_gb * 1e9)
    for name in names:
        dm = _TAR_DATE.search(name)
        if not dm:
            continue
        day = dm.group(1)
        if date_from and day < date_from:
            continue
        if date_to and day > date_to:
            continue
        rel = f"{run}/{name}"
        if raw_dir is not None:
            dest = os.path.join(raw_dir, rel.replace("/", "_"))
            if os.path.exists(dest) and os.path.getsize(dest) > 10240:
                continue                    # cached: free, not counted
        head = urllib.request.Request(
            f"{LC0_BASE_URL}/{run}/{name}", method="HEAD",
            headers={"User-Agent": "Mozilla/5.0 Zchezz-importer"})
        try:
            with urllib.request.urlopen(head, timeout=30) as resp:
                size = int(resp.headers.get("Content-Length", 0))
        except OSError:
            continue
        if size <= 10240:                        # server writes 10 KB stubs for empty hours
            stubs += 1
            continue
        picked.append((rel, size))
        total += size
        if total >= budget:
            break
    return picked, total, stubs


def big_run(args) -> None:
    """Plan (and optionally execute) the ≥20 GB AlphaZero-data run."""
    picked, total, stubs = discover_tars(args.big_run, args.big_date_from,
                                         args.big_date_to, args.big_budget_gb,
                                         args.raw_dir)
    if not picked:
        raise SystemExit(f"no tars found for {args.big_run} in "
                         f"[{args.big_date_from}..{args.big_date_to}] — widen the window")
    n_days = len({_TAR_DATE.search(os.path.basename(rel)).group(1) for rel, _ in picked})
    est_records = total / 1e6 * 4800             # measured yield, see config comment
    est_kept = est_records / max(args.sample_every, 1)
    est_bin = est_kept * 0.73 * 75               # ~73% quiet-filter pass, 75 B/record
    print(f"=== BIG-RUN PLAN {'(EXECUTE)' if args.go else '(plan only - add --go to run)'} ===")
    print(f"  run/window : {args.big_run} [{args.big_date_from}..{args.big_date_to}]")
    print(f"  tars       : {len(picked)} across {n_days} days "
          f"({stubs} empty-hour stubs skipped)")
    print(f"  download   : {total / 1e9:.2f} GB compressed "
          f"(~{(total / 1e6) / 20.0 / 60:.1f} h at 20 MB/s)")
    print(f"  records    : ~{est_records / 1e6:.0f}M scanned -> "
          f"~{est_kept / 1e6:.1f}M sampled rows")
    print(f"  after quiet: ~{est_kept * 0.73 / 1e6:.1f}M positions -> "
          f"~{est_bin / 1e9:.2f} GB of .bin (+ parquet ~{est_kept * 160e-9:.1f} GB)")
    print(f"  output dir : {args.out_dir}")
    print(f"  next steps : process_positions --filters quiet --out-format bin;"
          f" then train_nnue with kind=bin,k=1.0")
    for rel, size in picked[:3]:
        print(f"    e.g. {rel} ({size / 1e6:.0f} MB)")
    if len(picked) > 3:
        print(f"    ... and {len(picked) - 3} more")

    if not args.go:
        return

    # Parallel downloads: each tar goes to its own file, so plain threads
    # are enough (measured ~2 MB/s per connection on this server).
    from concurrent.futures import ThreadPoolExecutor
    print(f"downloading {len(picked)} tar(s) with {args.dl_workers} workers...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.dl_workers) as ex:
        paths = list(ex.map(lambda rel: download_tar(rel, args.raw_dir),
                            [rel for rel, _ in picked]))
    print(f"all tars cached ({sum(os.path.getsize(p) for p in paths) / 1e9:.2f} GB "
          f"in {time.time() - t0:.0f}s)")

    # Untouched --max-total-rows default would strangle a 20 GB run: lift it
    # to cover the whole budget-derived estimate (+20% headroom).
    if args.max_total_rows == MAX_TOTAL_ROWS:
        est = total / 1e6 * 4800 / max(args.sample_every, 1)
        args.max_total_rows = int(est * 1.2)
        print(f"[big] --max-total-rows auto -> {args.max_total_rows:,}")
    convert(paths, args.out_dir, args)


# ────────────────────────────────────────────────────────────────────────

def main() -> None:
    global SAMPLE_EVERY, MAX_TOTAL_ROWS, OUT_DIR
    p = argparse.ArgumentParser(description="LC0 tar → Zchezz parquet/bin (fen+result) importer")
    p.add_argument("--self-test", action="store_true",
                   help="run the decode unit test and exit")
    p.add_argument("--tars", nargs="*", default=DEFAULT_TARS,
                   help="tar paths RELATIVE to the LC0 training_data base URL")
    p.add_argument("--raw-dir", default=RAW_DIR, help="where downloaded tars are cached")
    p.add_argument("--out-dir", default=OUT_DIR, help="parquet shard output dir")
    p.add_argument("--sample-every", type=int, default=SAMPLE_EVERY,
                   help="keep every Nth chunk record")
    p.add_argument("--max-total-rows", type=int, default=MAX_TOTAL_ROWS,
                   help="stop converting after this many kept rows")
    p.add_argument("--min-visits", type=int, default=MIN_VISITS,
                   help="drop V6 records with fewer search visits")

    g = p.add_argument_group("big run (>=20 GB AlphaZero-style data)")
    p.add_argument("--big", action="store_true",
                   help="plan (or with --go, execute) a date-window bulk import")
    g.add_argument("--go", dest="go", action="store_true", default=BIG_GO,
                   help="--big only: actually download and convert (default: plan only)")
    g.add_argument("--big-run", default=BIG_RUN, help=f"(default: {BIG_RUN})")
    g.add_argument("--big-date-from", default=BIG_DATE_FROM, help=f"(default: {BIG_DATE_FROM})")
    g.add_argument("--big-date-to", default=BIG_DATE_TO, help=f"(default: {BIG_DATE_TO})")
    g.add_argument("--big-budget-gb", type=float, default=BIG_BUDGET_GB,
                   help=f"(default: {BIG_BUDGET_GB})")
    g.add_argument("--big-out-dir", default=BIG_OUT_DIR,
                   help="output dir used by --big when --out-dir is left at its default")
    g.add_argument("--dl-workers", type=int, default=DL_WORKERS,
                   help=f"parallel tar downloads (default: {DL_WORKERS})")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    if args.big:
        if args.out_dir == OUT_DIR:
            args.out_dir = args.big_out_dir     # --big defaults to its own output tree
        big_run(args)
        return

    paths = [download_tar(t, args.raw_dir) for t in args.tars]
    print(f"\nconverting {len(paths)} tar(s) -> {args.out_dir} "
          f"(sample every {args.sample_every}, cap {args.max_total_rows:,} rows)")
    convert(paths, args.out_dir, args)


if __name__ == "__main__":
    main()
