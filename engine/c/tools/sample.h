/* sample.h — the packed .bin training-sample record, shared by every
 * tool that produces training data (tools/selfplay.c, tools/arena.c).
 *
 * THIS STRUCT IS A CROSS-LANGUAGE CONTRACT with train/dataset.py's
 * SAMPLE_DTYPE. It lives in its own header precisely so that two
 * producers cannot drift apart: a second copy-pasted definition would
 * compile fine and silently write a differently-laid-out file that the
 * Python reader would reinterpret as garbage. The _Static_assert below
 * is the only thing standing between a field reorder and a training run
 * on noise — keep it, and update dataset.py in the same commit as any
 * change here.
 *
 * Field semantics (see selfplay.c's header for the full rationale):
 *   board[64]    mailbox, Zchezz encoding (0=empty, WP=9..BK=22), sq 0 = a8
 *   stm          0 = white to move, 1 = black
 *   rule50       halfmove clock
 *   castling     bitmask
 *   ep_file      0..7, 8 = none
 *   eval_cp      STM-relative score from the search that CHOSE the move
 *   game_result  +1/0/-1 from the point of view of the side to move in
 *                THIS position (requires a second pass once the game ends)
 *   move_played  packed move, low 16 bits of search.c's pack_move layout

 * ═══════════════════════════════════════════════════════════════════
 *  LABEL CONVENTION (CLAUDE.md rule 10) — result / cp / wdl
 * ═══════════════════════════════════════════════════════════════════
 *
 *  | Name    | Meaning                          | Range / frame        |
 *  |---------|----------------------------------|----------------------|
 *  | result  | the REAL game outcome            | parquet: 0.0/0.5/1.0 |
 *  |         |                                  | WHITE-relative       |
 *  | cp      | evaluation in centipawns         | int                  |
 *  | wdl     | sigmoid(cp/320), a FUNCTION of   | 0..1                 |
 *  |         | cp — NOT an outcome              |                      |
 *  | target  | lam*result + (1-lam)*wdl         | computed at TRAINING |
 *  |         |                                  | time, never stored   |
 *
 *  THIS FILE'S .bin RECORD USES ITS OWN INTERNALLY-CONSISTENT FRAME:
 *  `eval_cp` and `game_result` (+1/0/-1) are BOTH STM-relative here.
 *  train/dataset.py converts game_result to the 0..1 probability
 *  ((g+1)/2) at read time. Do not mix the two frames.
 *
 *  Never bake the lam blend into a stored record — lam is per-dataset and
 *  is annealed across bootstrap generations.
 * ═══════════════════════════════════════════════════════════════════
 */
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#pragma pack(push, 1)
typedef struct {
    uint8_t  board[64];
    uint8_t  stm;
    uint8_t  rule50;
    uint8_t  castling;
    uint8_t  ep_file;
    int16_t  eval_cp;
    int8_t   game_result;
    uint16_t move_played;
    uint16_t _pad;
} SelfplaySample;
#pragma pack(pop)

#define SELFPLAY_SAMPLE_SIZE 75
/* Compile-time contract check (C11 _Static_assert) — if this ever
 * fires, the struct drifted from dataset.py's SAMPLE_DTYPE and
 * every downstream .bin read would silently reinterpret garbage. */
_Static_assert(sizeof(SelfplaySample) == SELFPLAY_SAMPLE_SIZE,
    "SelfplaySample size drifted from train/dataset.py's "
    "SAMPLE_DTYPE (75 bytes) — update both in lockstep.");

/* ═══════════════════════════════════════════════════════════════════
 *  PROVENANCE HEADER — which engine build and which NNUE weights
 *  produced this .bin file's eval_cp column.
 * ═══════════════════════════════════════════════════════════════════
 *
 * A bare SelfplaySample carries no provenance: nothing in the 75-byte
 * record says which engine version searched the position or which
 * weight file was loaded when eval_cp was computed. That is exactly
 * why arena.c's --bin path HARD-REFUSES a mixed net:-vs-net: match
 * with different weight files (see the gate in arena_run()) — without
 * per-file identity, a mixed-evaluator run would silently blend two
 * engines' opinions into one eval_cp column.
 *
 * This header is FILE-LEVEL, not per-row: sample_open_bin_append()
 * writes ONE header at the front of a fresh .bin file, and every
 * SelfplaySample that follows (from this run and any later run that
 * appends to the same path, as long as it matches) shares that single
 * provenance. A per-row field was considered and rejected — this
 * struct's records already number in the hundreds of millions, and a
 * per-row source-id would add 2+ bytes x every row for information
 * that is constant across an entire arena/selfplay run (arena.c's
 * --bin gate already enforces one evaluator per file; selfplay.c
 * always has exactly one). If a single .bin file ever needs to mix
 * evaluators mid-file, switch to a per-row uint16 source-id indexing
 * a table of headers instead of widening every record.
 *
 * Format is versioned by MAGIC, not by guessing from file size: a
 * file starting with SAMPLE_FILE_MAGIC is format v2 (this header);
 * anything else — including every .bin written before this change —
 * is format v1 (headerless, legacy), and its provenance reads back as
 * "unknown" rather than being misinterpreted as v2 fields. See
 * train/dataset.py's read_bin_header() for the mirrored reader.
 *
 * `reserved` gives room to grow the header later without bumping the
 * magic, PROVIDED `header_size` (recorded in the header itself) is
 * used to compute the record-start offset instead of a hardcoded
 * constant — dataset.py does this.
 */
#define SAMPLE_FILE_MAGIC       "ZCHZSMP2"   /* 8 bytes, format v2 */
#define SAMPLE_FILE_MAGIC_LEN   8

#ifndef SAMPLE_ENGINE_VERSION
/* major*100 + minor, e.g. 400 == v4.00. Must track ENGINE_VERSION in
 * engine/c/zchezz_vXXX/main.c — bump this alongside a version release,
 * or pass -DSAMPLE_ENGINE_VERSION=NNN on the compile line to override
 * without editing this header. */
#define SAMPLE_ENGINE_VERSION 400
#endif

#pragma pack(push, 1)
typedef struct {
    char     magic[8];             /* SAMPLE_FILE_MAGIC, unterminated */
    uint32_t header_size;          /* sizeof(SampleFileHeader) at write time —
                                     * readers must skip THIS many bytes before
                                     * the first SelfplaySample, not a constant,
                                     * so the header can grow via `reserved`
                                     * without breaking older readers. */
    uint32_t engine_version;       /* SAMPLE_ENGINE_VERSION at write time */
    uint64_t weight_fingerprint;   /* FNV-1a 64 over the raw NNUE weight file
                                     * bytes that produced every eval_cp that
                                     * follows (see sample_hash_weight_file) */
    char     weight_path[128];     /* informational only: NUL-terminated,
                                     * truncated, NOT used for identity —
                                     * weight_fingerprint is the identity */
    uint8_t  reserved[32];         /* zero-filled; future fields grow here */
} SampleFileHeader;
#pragma pack(pop)

#define SAMPLE_FILE_HEADER_SIZE 184
_Static_assert(sizeof(SampleFileHeader) == SAMPLE_FILE_HEADER_SIZE,
    "SampleFileHeader size drifted from train/dataset.py's "
    "HEADER_DTYPE (184 bytes) — update both in lockstep.");

/* FNV-1a 64-bit, used both to fingerprint a whole weight file (below)
 * and available standalone if a caller needs it. Not cryptographic —
 * this is provenance bookkeeping (detect "wrong file got loaded"), not
 * a security boundary. */
static inline uint64_t sample_fnv1a64(const uint8_t *data, size_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* Hashes an NNUE weight file's raw bytes for the .bin header's
 * weight_fingerprint field. Returns 0 and fills *out on success, -1 on
 * I/O failure (caller decides whether that's fatal — both current
 * callers treat it as fatal, since a --bin run without a fingerprint
 * would defeat the point of this header). */
static inline int sample_hash_weight_file(const char *path, uint64_t *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint8_t buf[65536];
    uint64_t h = 14695981039346656037ULL;
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        for (size_t i = 0; i < n; i++) {
            h ^= buf[i];
            h *= 1099511628211ULL;
        }
    }
    fclose(f);
    *out = h;
    return 0;
}

/* Opens `path` in append mode for packed .bin training-sample output,
 * writing or verifying the SampleFileHeader described above.
 *
 *   - New or empty file: writes a fresh header, then returns the FILE*
 *     positioned right after it, ready for SelfplaySample records.
 *   - Existing file starting with SAMPLE_FILE_MAGIC: the header is read
 *     back and compared against (engine_version, weight_fingerprint).
 *     A mismatch is refused — appending records from a different engine
 *     build or a different NNUE weight file into one .bin would blend
 *     two evaluators under one eval_cp column, exactly like arena.c's
 *     --bin same-net gate refuses at the player level. (This header is
 *     what would eventually let that gate be relaxed from "same weight
 *     PATH string" to "same weight CONTENT" — not done here, arena.c's
 *     gate is left exactly as strict as before.)
 *   - Existing file with data but NOT starting with the magic: a
 *     pre-header (format v1) legacy .bin. Appending v2 records after
 *     headerless ones would make the file ambiguous to read (no marker
 *     for where the boundary is), so this is refused too — start a new
 *     --bin path for new output; the legacy file stays fully readable,
 *     just not extendable in place.
 *
 * Returns NULL on any failure, having already printed a diagnostic to
 * stderr — callers exit(1) on NULL, matching selfplay.c/arena.c's
 * existing fopen-failure style for --bin/--out. */
static inline FILE *sample_open_bin_append(const char *path, uint32_t engine_version,
                                            const char *weight_path) {
    uint64_t fingerprint = 0;
    if (sample_hash_weight_file(weight_path, &fingerprint) != 0) {
        fprintf(stderr, "[sample] cannot read NNUE weight file '%s' to fingerprint it\n", weight_path);
        return NULL;
    }

    /* Peek at any existing content BEFORE opening in append mode —
     * append-mode streams can't be read back portably on all libc's. */
    FILE *peek = fopen(path, "rb");
    int write_header = 1;
    if (peek) {
        SampleFileHeader existing;
        size_t got = fread(&existing, 1, sizeof(existing), peek);
        fclose(peek);
        if (got == 0) {
            write_header = 1;   /* file exists but is empty */
        } else if (got == sizeof(existing) && memcmp(existing.magic, SAMPLE_FILE_MAGIC, SAMPLE_FILE_MAGIC_LEN) == 0) {
            if (existing.engine_version != engine_version || existing.weight_fingerprint != fingerprint) {
                fprintf(stderr,
                    "[sample] --bin/--out file '%s' already carries a provenance header\n"
                    "         (engine_version=%u, weight_fingerprint=%016llx) that does not match\n"
                    "         this run (engine_version=%u, weight_fingerprint=%016llx). Appending\n"
                    "         would blend two evaluators under one eval_cp column — use a different\n"
                    "         output path, or match the engine build/weight file.\n",
                    path, existing.engine_version, (unsigned long long)existing.weight_fingerprint,
                    engine_version, (unsigned long long)fingerprint);
                return NULL;
            }
            write_header = 0;   /* header matches this run — just append records */
        } else {
            fprintf(stderr,
                "[sample] --bin/--out file '%s' has data but no recognizable provenance\n"
                "         header — it is a pre-header (format v1) legacy .bin. Appending v2\n"
                "         records after headerless ones would make the file ambiguous to read.\n"
                "         Use a fresh output path (the legacy file remains fully readable by\n"
                "         train/dataset.py, just not extendable in place).\n", path);
            return NULL;
        }
    }

    FILE *f = fopen(path, "ab");
    if (!f) return NULL;

    if (write_header) {
        SampleFileHeader hdr;
        memset(&hdr, 0, sizeof(hdr));
        memcpy(hdr.magic, SAMPLE_FILE_MAGIC, SAMPLE_FILE_MAGIC_LEN);
        hdr.header_size = (uint32_t)sizeof(hdr);
        hdr.engine_version = engine_version;
        hdr.weight_fingerprint = fingerprint;
        strncpy(hdr.weight_path, weight_path, sizeof(hdr.weight_path) - 1);
        if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
            fclose(f);
            return NULL;
        }
        fflush(f);
    }
    return f;
}


/* Pack a move into SAMPLE_DTYPE's 16-bit move_played field.
 *
 * Takes plain ints rather than a `Move` so this header stays free of
 * board.h — sample.h is included by tools that already include board.h
 * and (potentially) by tests that do not.
 *
 * This is the LOW 16 BITS of search.c's 20-bit pack_move layout:
 * from[0:5] to[6:11] prom[12:14] epc[15]. The 4 castle bits that
 * search.c's TT packing carries are dropped — not a loss for consumers,
 * since a castling move is always the king moving two files on its home
 * rank and from/to identify it unambiguously. If exact parity with the
 * 20-bit layout is ever needed, widen move_played to uint32 in BOTH this
 * header and dataset.py rather than silently reusing these bits. */
static inline uint16_t sample_pack_move(int from, int to, int prom, int epc) {
    return (uint16_t)((from & 63) | ((to & 63) << 6) |
                      ((prom & 7) << 12) | (epc ? (1 << 15) : 0));
}
