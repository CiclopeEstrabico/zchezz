# Zchezz — Termux / Android Quick Reference

> Copy-paste these commands in Termux to compile, test, and benchmark the engine.
> Set `VER` once at the top — all commands use it automatically.

```bash
# ╔══════════════════════════════════════════╗
# ║  SET THIS ONCE — your engine version     ║
# ╚══════════════════════════════════════════╝
VER=v313
```

---

## 0. First-Time Termux Setup

```bash
# Grant storage access (required once)
termux-setup-storage

# Install required packages
pkg update && pkg upgrade -y
pkg install clang make git nodejs python -y

# Optional: WASM builds
pkg install emscripten -y
```

---

## 1. Copy from Phone or GitHub

```bash
# ── Option A: Copy from phone Downloads ──
cp -r ~/storage/downloads/zchezz_${VER} ~/

# ── Option B: Clone from GitHub ──
git clone https://github.com/gitzambrano/zchezz.git
cp -r ~/zchezz/engine/c/zchezz_${VER} ~/
```

---

## 2. Compile (Native + WASM + Bundle)

```bash
cd ~/zchezz_${VER}

# Clean previous builds
rm -f zchezz zchezz_wasm.js zchezz_wasm.wasm zchezz_bundle.html

# ── Native (Termux/ARM) ──
make termux
ls -la zchezz && echo "Native build OK" || echo "Native build FAILED"

# ── WASM (requires emscripten) ──
make wasm
ls -la zchezz_wasm.wasm && echo "WASM build OK" || echo "WASM build FAILED"

# ── Bundle (single HTML file) ──
make wasm-bundle
ls -la zchezz_bundle.html && echo "Bundle build OK" || echo "Bundle build FAILED"
```

---

## 3. Test — UCI Protocol

```bash
cd ~/zchezz_${VER}

# Basic handshake (should print "id name Zchezz..." then "uciok")
echo -e "uci\nquit" | ./zchezz --nnue nnue_weights.bin

# Full handshake + readyok
echo -e "uci\nisready\nquit" | ./zchezz --nnue nnue_weights.bin

# List all UCI options
echo -e "uci\nquit" | ./zchezz --nnue nnue_weights.bin | grep "option name"
```

## 3b. Test — Search

```bash
cd ~/zchezz_${VER}

# Search startpos to depth 8 (should take < 1 second)
echo -e "uci\nisready\nposition startpos\ngo depth 8\nquit" \
  | ./zchezz --nnue nnue_weights.bin

# Search Kiwipete (complex middlegame)
echo -e "uci\nisready\nposition fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1\ngo depth 10\nquit" \
  | ./zchezz --nnue nnue_weights.bin

# Search with movetime (200ms per move)
echo -e "uci\nisready\nposition startpos moves e2e4 e7e5\ngo movetime 200\nquit" \
  | ./zchezz --nnue nnue_weights.bin

# MultiPV test (should report 4 separate lines)
echo -e "uci\nsetoption name MultiPV value 4\nisready\nposition startpos\ngo depth 10\nquit" \
  | ./zchezz --nnue nnue_weights.bin
```

## 3c. Test — Perft (Move Generation Correctness)

```bash
cd ~/zchezz_${VER}

# Startpos perft 5 → expected: 4,865,609
echo -e "uci\nisready\nposition startpos\ngo perft 5\nquit" \
  | ./zchezz --nnue nnue_weights.bin

# Kiwipete perft 4 → expected: 4,085,603
echo -e "uci\nisready\nposition fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1\ngo perft 4\nquit" \
  | ./zchezz --nnue nnue_weights.bin

# Promotion-heavy perft 5 → expected: 15,833,292
echo -e "uci\nisready\nposition fen rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8\ngo perft 5\nquit" \
  | ./zchezz --nnue nnue_weights.bin
```

## 3d. Test — Eval Sanity

```bash
cd ~/zchezz_${VER}

# Startpos — should show score ≈ 0cp (±50)
echo -e "uci\nisready\nposition startpos\ngo depth 12\nquit" \
  | ./zchezz --nnue nnue_weights.bin 2>&1 | grep "info depth 12 "

# KQK — should show score > +1000cp or mate (trivial win)
echo -e "uci\nisready\nposition fen 8/8/4k3/8/8/8/8/4K2Q w - - 0 1\ngo depth 10\nquit" \
  | ./zchezz --nnue nnue_weights.bin 2>&1 | grep "info depth 10"

# KvK — should show score = 0cp (insufficient material)
echo -e "uci\nisready\nposition fen 8/8/4k3/8/8/8/8/4K3 w - - 0 1\ngo depth 10\nquit" \
  | ./zchezz --nnue nnue_weights.bin 2>&1 | grep "info depth 10"

# KBvK — should show score = 0cp (insufficient material)
echo -e "uci\nisready\nposition fen 8/8/4k3/8/8/8/8/4KB2 w - - 0 1\ngo depth 10\nquit" \
  | ./zchezz --nnue nnue_weights.bin 2>&1 | grep "info depth 10"
```

## 3e. Test — NPS Benchmark

```bash
cd ~/zchezz_${VER}

# Depth 14 from startpos — record NPS
echo -e "uci\nisready\nposition startpos\ngo depth 14\nquit" \
  | ./zchezz --nnue nnue_weights.bin 2>&1 | grep "info depth 14 "

# Compare NPS across positions
for FEN in \
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" \
  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" \
  "8/8/4k3/8/8/8/8/4K2Q w - - 0 1"; do
  echo "=== $FEN ==="
  echo -e "uci\nisready\nposition fen $FEN\ngo depth 12\nquit" \
    | ./zchezz --nnue nnue_weights.bin 2>&1 | tail -3
done
```

## 3f. Test — Threads

```bash
cd ~/zchezz_${VER}

# Single thread
echo -e "uci\nsetoption name Threads value 1\nisready\nposition startpos\ngo depth 12\nquit" \
  | ./zchezz --nnue nnue_weights.bin 2>&1 | grep "info depth 12 "

# Multi-thread (adjust to your device's CPU cores)
echo -e "uci\nsetoption name Threads value 4\nisready\nposition startpos\ngo depth 12\nquit" \
  | ./zchezz --nnue nnue_weights.bin 2>&1 | grep "info depth 12 "
```

---

## 4. Copy Back to Phone

```bash
# Copy compiled binary + bundle to Downloads
cp ~/zchezz_${VER}/ ~/storage/downloads/
cp ~/zchezz_${VER}/zchezz_bundle.html ~/storage/downloads/zchezz_${VER}/

# Or copy entire folder
cp -r ~/zchezz_${VER} ~/storage/downloads/
```

---

## 5. Run Tournaments

```bash
# ── Copy tournament scripts from phone ──
mkdir -p ~/chess_test
cp ~/storage/downloads/chess_test/tournament.js ~/chess_test/
cp ~/storage/downloads/chess_test/playitself.js ~/chess_test/

# ── Run tournament ──
cd ~/chess_test
node tournament.js

# ── Run self-play ──
node playitself.js

# ── Copy results back to phone ──
cp ~/chess_test/*.pgn ~/storage/downloads/chess_test/
cp ~/chess_test/*.log ~/storage/downloads/chess_test/
```

---

## Quick One-Liners

```bash
# Full compile + quick test
cd ~/zchezz_${VER} && make termux && echo -e "uci\nisready\nposition startpos\ngo depth 8\nquit" | ./zchezz --nnue nnue_weights.bin

# Check engine version
echo -e "uci\nquit" | ~/zchezz_${VER}/zchezz --nnue ~/zchezz_${VER}/nnue_weights.bin | head -2

# Run automated test script
chmod +x RunZchezzTermux.sh && ./RunZchezzTermux.sh ${VER}

# Run bench if available
cd ~/zchezz_${VER} && ./bench nnue_weights.bin
```
