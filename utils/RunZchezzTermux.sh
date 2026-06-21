#!/data/data/com.termux/files/usr/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  RunZchezzTermux.sh — Zchezz Engine Test Suite for Termux/Android
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Usage:
#    chmod +x RunZchezzTermux.sh
#    ./RunZchezzTermux.sh              # runs all tests with default version
#    ./RunZchezzTermux.sh v314         # override version number
#    ./RunZchezzTermux.sh v314 quick   # only run quick tests (UCI + search)
#
# ═══════════════════════════════════════════════════════════════════════════════

# ── Configuration ─────────────────────────────────────────────────────────────
VERSION="${1:-v313}"                          # engine version (override via arg)
MODE="${2:-full}"                             # "quick" or "full"
ENGINE_DIR="$HOME/zchezz_${VERSION}"
ENGINE="${ENGINE_DIR}/zchezz"
NNUE="${ENGINE_DIR}/nnue_weights.bin"
DOWNLOADS="$HOME/storage/downloads"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

pass=0; fail=0; total=0

# ── Helper Functions ──────────────────────────────────────────────────────────
banner()  { echo -e "\n${CYAN}═══════════════════════════════════════════${NC}"; echo -e "${BOLD}  $1${NC}"; echo -e "${CYAN}═══════════════════════════════════════════${NC}"; }
section() { echo -e "\n${YELLOW}── $1 ──${NC}"; }
ok()      { ((pass++)); ((total++)); echo -e "  ${GREEN}✓${NC} $1"; }
ko()      { ((fail++)); ((total++)); echo -e "  ${RED}✗${NC} $1"; }
info()    { echo -e "  ${CYAN}ℹ${NC} $1"; }

uci_cmd() {
    # Run a UCI command sequence and return stdout
    echo -e "$1" | "$ENGINE" --nnue "$NNUE" 2>/dev/null
}

uci_search() {
    # Run a search and extract the last "info depth" line + bestmove
    local cmds="uci\nisready\n$1"
    local out
    out=$(echo -e "$cmds\nquit" | "$ENGINE" --nnue "$NNUE" 2>/dev/null)
    echo "$out"
}

# ═══════════════════════════════════════════════════════════════════════════════
banner "Zchezz ${VERSION} — Termux Test Suite"
# ═══════════════════════════════════════════════════════════════════════════════

echo -e "  Engine dir : ${ENGINE_DIR}"
echo -e "  Mode       : ${MODE}"
echo -e "  Date       : $(date '+%Y-%m-%d %H:%M:%S')"

# ── Check engine exists ──────────────────────────────────────────────────────
if [ ! -f "$ENGINE" ]; then
    echo -e "\n${RED}ERROR: Engine not found at ${ENGINE}${NC}"
    echo -e "  Compile first:  cd ${ENGINE_DIR} && make termux"
    echo -e "  Or copy from:   cp -r ${DOWNLOADS}/zchezz_${VERSION} ~/"
    exit 1
fi
if [ ! -f "$NNUE" ]; then
    echo -e "\n${RED}ERROR: NNUE weights not found at ${NNUE}${NC}"
    exit 1
fi

ok "Engine found: $(ls -la "$ENGINE" | awk '{print $5}') bytes"

# ═══════════════════════════════════════════════════════════════════════════════
section "1. UCI Handshake"
# ═══════════════════════════════════════════════════════════════════════════════

OUT=$(uci_cmd "uci\nquit")
if echo "$OUT" | grep -q "uciok"; then
    ok "UCI handshake: uciok received"
    # Extract engine name
    NAME=$(echo "$OUT" | grep "^id name" | head -1)
    info "$NAME"
else
    ko "UCI handshake FAILED — no uciok"
fi

OUT=$(uci_cmd "uci\nisready\nquit")
if echo "$OUT" | grep -q "readyok"; then
    ok "isready: readyok received"
else
    ko "isready FAILED — no readyok"
fi

# ── Check NNUE loaded ────────────────────────────────────────────────────────
OUT=$(uci_cmd "uci\nquit")
if echo "$OUT" | grep -qi "nnue\|weights\|network"; then
    ok "NNUE: weights loaded"
else
    info "NNUE: could not confirm load (check engine output)"
fi

# ── Check UCI options ─────────────────────────────────────────────────────────
OPTS=$(echo "$OUT" | grep -c "^option name")
info "UCI options reported: ${OPTS}"

# ═══════════════════════════════════════════════════════════════════════════════
section "2. Quick Search Test"
# ═══════════════════════════════════════════════════════════════════════════════

OUT=$(uci_search "position startpos\ngo depth 8")
BM=$(echo "$OUT" | grep "^bestmove" | head -1)
if [ -n "$BM" ]; then
    ok "Startpos depth 8: $BM"
    # Extract NPS from last info line
    NPS=$(echo "$OUT" | grep "info depth 8 " | grep -oP 'nps \K[0-9]+' | tail -1)
    [ -n "$NPS" ] && info "NPS: ${NPS}"
else
    ko "Startpos depth 8: no bestmove received"
fi

# ═══════════════════════════════════════════════════════════════════════════════
section "3. Eval Sanity"
# ═══════════════════════════════════════════════════════════════════════════════

# Startpos should be near 0cp
OUT=$(uci_search "position startpos\ngo depth 10")
SCORE=$(echo "$OUT" | grep "info depth 10 " | grep -oP 'score cp \K-?[0-9]+' | tail -1)
if [ -n "$SCORE" ]; then
    ABS=${SCORE#-}
    if [ "$ABS" -le 60 ]; then
        ok "Startpos eval: ${SCORE}cp (expected ≈0)"
    else
        ko "Startpos eval: ${SCORE}cp (expected ≈0, got >60)"
    fi
else
    info "Startpos eval: could not parse score"
fi

# KQK should be > +1000cp
OUT=$(uci_search "position fen 8/8/4k3/8/8/8/8/4K2Q w - - 0 1\ngo depth 10")
SCORE=$(echo "$OUT" | grep "info depth" | tail -1)
if echo "$SCORE" | grep -q "score mate"; then
    ok "KQK eval: mate (correct — trivial win)"
elif echo "$SCORE" | grep -qP "score cp [0-9]{4,}"; then
    ok "KQK eval: >1000cp (correct — trivial win)"
else
    ko "KQK eval: unexpected — $SCORE"
fi

# KvK should be 0cp
OUT=$(uci_search "position fen 8/8/4k3/8/8/8/8/4K3 w - - 0 1\ngo depth 10")
SCORE=$(echo "$OUT" | grep "info depth" | tail -1 | grep -oP 'score cp \K-?[0-9]+')
if [ "$SCORE" = "0" ]; then
    ok "KvK eval: 0cp (correct — insufficient material)"
else
    ko "KvK eval: ${SCORE}cp (expected 0)"
fi

# KBvK should be 0cp
OUT=$(uci_search "position fen 8/8/4k3/8/8/8/8/4KB2 w - - 0 1\ngo depth 10")
SCORE=$(echo "$OUT" | grep "info depth" | tail -1 | grep -oP 'score cp \K-?[0-9]+')
if [ "$SCORE" = "0" ]; then
    ok "KBvK eval: 0cp (correct — insufficient material)"
else
    ko "KBvK eval: ${SCORE}cp (expected 0)"
fi

# ═══════════════════════════════════════════════════════════════════════════════
section "4. MultiPV Test"
# ═══════════════════════════════════════════════════════════════════════════════

OUT=$(uci_search "setoption name MultiPV value 4\nposition startpos\ngo depth 10")
LINES=$(echo "$OUT" | grep "info depth 10 " | grep -c "multipv")
if [ "$LINES" -ge 3 ]; then
    ok "MultiPV: ${LINES} lines at depth 10"
else
    ko "MultiPV: only ${LINES} lines (expected ≥3)"
fi

if [ "$MODE" = "quick" ]; then
    banner "Quick Test Complete: ${pass}/${total} passed"
    [ "$fail" -gt 0 ] && exit 1 || exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════════
section "5. Perft — Move Generation Correctness"
# ═══════════════════════════════════════════════════════════════════════════════

check_perft() {
    local name="$1" fen="$2" depth="$3" expected="$4"
    local out
    if [ "$fen" = "startpos" ]; then
        out=$(uci_search "position startpos\ngo perft $depth")
    else
        out=$(uci_search "position fen $fen\ngo perft $depth")
    fi
    local count
    count=$(echo "$out" | grep -ioP '(?:nodes searched|total).*?(\d+)' | grep -oP '\d+' | tail -1)
    if [ -z "$count" ]; then
        count=$(echo "$out" | grep -oP '^\d+$' | tail -1)
    fi
    if [ "$count" = "$expected" ]; then
        ok "Perft $name: $count ✓"
    else
        ko "Perft $name: got ${count:-???}, expected $expected"
    fi
}

check_perft "startpos d5" "startpos" 5 "4865609"
check_perft "kiwipete d4" "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" 4 "4085603"
check_perft "promo d5" "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8" 5 "15833292"

# ═══════════════════════════════════════════════════════════════════════════════
section "6. NPS Benchmark"
# ═══════════════════════════════════════════════════════════════════════════════

FENS=(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1|Startpos"
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1|Kiwipete"
    "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2|1.e4 Nc6"
    "8/8/4k3/8/8/8/8/4K2Q w - - 0 1|KQK endgame"
)

for entry in "${FENS[@]}"; do
    FEN="${entry%%|*}"
    NAME="${entry##*|}"
    OUT=$(uci_search "position fen $FEN\ngo depth 12")
    NPS=$(echo "$OUT" | grep "info depth 12 " | grep -oP 'nps \K[0-9]+' | tail -1)
    NODES=$(echo "$OUT" | grep "info depth 12 " | grep -oP 'nodes \K[0-9]+' | tail -1)
    DEPTH=$(echo "$OUT" | grep "info depth" | tail -1 | grep -oP 'depth \K[0-9]+')
    info "${NAME}: NPS=${NPS:-?}  nodes=${NODES:-?}  depth=${DEPTH:-?}"
done

# ═══════════════════════════════════════════════════════════════════════════════
section "7. Movetime Search"
# ═══════════════════════════════════════════════════════════════════════════════

OUT=$(uci_search "position startpos moves e2e4 e7e5 g1f3\ngo movetime 200")
BM=$(echo "$OUT" | grep "^bestmove" | head -1)
if [ -n "$BM" ]; then
    ok "Movetime 200ms: $BM"
else
    ko "Movetime 200ms: no bestmove"
fi

# ═══════════════════════════════════════════════════════════════════════════════
section "8. Thread Test"
# ═══════════════════════════════════════════════════════════════════════════════

for T in 1 2 4; do
    OUT=$(uci_search "setoption name Threads value $T\nposition startpos\ngo depth 10")
    BM=$(echo "$OUT" | grep "^bestmove" | head -1 | awk '{print $2}')
    NPS=$(echo "$OUT" | grep "info depth 10 " | grep -oP 'nps \K[0-9]+' | tail -1)
    if [ -n "$BM" ]; then
        ok "Threads=${T}: bestmove=${BM}  NPS=${NPS:-?}"
    else
        ko "Threads=${T}: no bestmove (crash/hang?)"
    fi
done

# ═══════════════════════════════════════════════════════════════════════════════
banner "Results: ${pass}/${total} passed, ${fail} failed"
# ═══════════════════════════════════════════════════════════════════════════════

if [ "$fail" -gt 0 ]; then
    echo -e "${RED}  ⚠ Some tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}  All tests passed!${NC}"
    exit 0
fi
