/* tools/relabel_static.c — relabel packed self-play samples with raw NNUE eval.
 *
 * This is deliberately engine-version agnostic: compile it with the include
 * path and nnue.c of the teacher you want to use.  The input board/stm/result
 * fields are copied byte-for-byte; only eval_cp is replaced by nnue_eval().
 *
 * Example (Linux, v3.14 teacher):
 *   gcc -O3 -std=c11 -mavx2 -DSAMPLE_ENGINE_VERSION=314 \
 *     -I../c/zchezz_v314 -I../c/tools \
 *     ../c/tools/relabel_static.c ../c/zchezz_v314/nnue.c -lm -o relabel_static
 *   ./relabel_static --input selfplay.bin --output static314.bin \
 *     --nnue ../c/zchezz_v314/nnue_weights.bin
 *
 * Why this exists: search scores are expensive and are not the same object as
 * the static evaluation consumed at every leaf.  For NNUE distillation we can
 * cheaply label millions of already-generated positions with the teacher's raw
 * static value, while preserving the real game_result for optional later use.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nnue.h"
#include "sample.h"

/* nnue.c's legacy/global API references this symbol.  The UCI engine normally
 * gets it from board.c; this standalone tool intentionally does not link the
 * board/search layer, so it owns the one accumulator itself. */
NnueAccum g_nnue_accum;

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s --input IN.bin --output OUT.bin --nnue weights.bin\n",
        argv0);
}

static long long record_offset(FILE *f, const char *path) {
    SampleFileHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    rewind(f);
    size_t got = fread(&hdr, 1, sizeof(hdr), f);
    if (got >= SAMPLE_FILE_MAGIC_LEN &&
        memcmp(hdr.magic, SAMPLE_FILE_MAGIC, SAMPLE_FILE_MAGIC_LEN) == 0) {
        if (got < sizeof(hdr) || hdr.header_size < sizeof(hdr)) {
            fprintf(stderr, "[relabel] invalid provenance header in %s\n", path);
            return -1;
        }
        return (long long)hdr.header_size;
    }
    return 0; /* legacy/headerless v1 */
}

static long long count_records(FILE *f, long long off, const char *path) {
    if (fseek(f, 0, SEEK_END) != 0) return -1;
    long end = ftell(f);
    if (end < 0 || (long long)end < off) return -1;
    long long bytes = (long long)end - off;
    if (bytes % (long long)sizeof(SelfplaySample) != 0) {
        fprintf(stderr,
            "[relabel] %s: record region %lld bytes is not divisible by %zu\n",
            path, bytes, sizeof(SelfplaySample));
        return -1;
    }
    return bytes / (long long)sizeof(SelfplaySample);
}

int main(int argc, char **argv) {
    const char *in_path = NULL, *out_path = NULL, *nnue_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--input") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(argv[i], "--output") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--nnue") && i + 1 < argc) nnue_path = argv[++i];
        else { usage(argv[0]); return 2; }
    }
    if (!in_path || !out_path || !nnue_path) { usage(argv[0]); return 2; }

    FILE *in = fopen(in_path, "rb");
    if (!in) {
        fprintf(stderr, "[relabel] cannot open input %s: %s\n", in_path, strerror(errno));
        return 1;
    }
    long long off = record_offset(in, in_path);
    long long total = off >= 0 ? count_records(in, off, in_path) : -1;
    if (off < 0 || total < 0) { fclose(in); return 1; }
    if (fseek(in, (long)off, SEEK_SET) != 0) { fclose(in); return 1; }

    memset(&g_nnue_accum, 0, sizeof(g_nnue_accum));
    if (nnue_load(nnue_path) != 0) {
        fprintf(stderr, "[relabel] NNUE load failed: %s\n", nnue_path);
        fclose(in);
        return 1;
    }

    FILE *out = sample_open_bin_append(out_path, SAMPLE_ENGINE_VERSION, nnue_path);
    if (!out) { fclose(in); return 1; }

    long long n = 0;
    SelfplaySample rec;
    while (fread(&rec, sizeof(rec), 1, in) == 1) {
        nnue_reset(&g_nnue_accum);
        nnue_rebuild(&g_nnue_accum, rec.board);
        int cp = nnue_eval(&g_nnue_accum, rec.stm ? 1 : 0, rec.board);
        if (cp < -32000) cp = -32000;
        if (cp >  32000) cp =  32000;
        rec.eval_cp = (int16_t)cp; /* STM-relative, same contract as SAMPLE_DTYPE */
        if (fwrite(&rec, sizeof(rec), 1, out) != 1) {
            fprintf(stderr, "[relabel] write failed at row %lld\n", n);
            fclose(out); fclose(in); return 1;
        }
        n++;
        if ((n % 100000) == 0)
            fprintf(stderr, "[relabel] %lld/%lld\n", n, total);
    }

    if (ferror(in)) {
        fprintf(stderr, "[relabel] read error after %lld rows\n", n);
        fclose(out); fclose(in); return 1;
    }
    fflush(out);
    fclose(out);
    fclose(in);
    fprintf(stderr, "[relabel] done: %lld rows -> %s\n", n, out_path);
    return n == total ? 0 : 1;
}
