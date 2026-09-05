#!/usr/bin/env bash
# Reproducible v3.21 PGO+LTO build.
# Default is AVX2 for portability. On a CPU with AVX-VNNI support, use:
#   ARCH_FLAGS='-mavxvnni -mavx2' tools/build_v321_pgo.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_DIR="${PROFILE_DIR:-$ROOT/.pgo/v321}"
ARCH_FLAGS="${ARCH_FLAGS:--mavx2}"
STATIC_FLAG="${STATIC_FLAG:--static}"

rm -rf "$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"

ARCH_FLAGS="$ARCH_FLAGS -flto -fprofile-generate=$PROFILE_DIR" \
STATIC_FLAG="$STATIC_FLAG" \
make -C "$ROOT/engine/build" ENGINE=v321 clean native

"$ROOT/engine/c/zchezz_v321/zchezz.exe" bench 8  >/dev/null
"$ROOT/engine/c/zchezz_v321/zchezz.exe" bench 10 >/dev/null
"$ROOT/engine/c/zchezz_v321/zchezz.exe" bench 12 >/dev/null

ARCH_FLAGS="$ARCH_FLAGS -flto -fprofile-use=$PROFILE_DIR -fprofile-correction" \
STATIC_FLAG="$STATIC_FLAG" \
make -C "$ROOT/engine/build" ENGINE=v321 clean native

"$ROOT/engine/c/zchezz_v321/zchezz.exe" bench 11
