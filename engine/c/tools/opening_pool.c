/* tools/opening_pool.c — see opening_pool.h for the "why this file
 * exists" comment (factored out of arena.c's originally-static
 * san_resolve()/move_to_san(), plus a new memory-efficient byte-offset
 * OpeningIndex for selfplay.c's opening-book support).
 */
#include "opening_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#include <dirent.h>

/* ═══════════════════════════════════════════════════════════════════
 * SAN <-> Move — moved verbatim from arena.c (see that file's original
 * comments, preserved here; arena.c now #includes this header instead
 * of defining these itself).
 * ═══════════════════════════════════════════════════════════════════ */
int san_resolve(const Board *b, const char *san_in, Move *out) {
    char san[16]; strncpy(san, san_in, sizeof(san) - 1); san[sizeof(san) - 1] = 0;
    size_t len = strlen(san);
    while (len > 0 && (san[len-1] == '+' || san[len-1] == '#' ||
                        san[len-1] == '!' || san[len-1] == '?')) san[--len] = 0;
    if (len == 0) return 0;

    Move moves[MAX_MOVES];
    int n = board_gen_moves(b, moves);

    if (!strcmp(san, "O-O") || !strcmp(san, "0-0")) {
        for (int i = 0; i < n; i++)
            if (moves[i].castle == 1 || moves[i].castle == 3) { *out = moves[i]; return 1; }
        return 0;
    }
    if (!strcmp(san, "O-O-O") || !strcmp(san, "0-0-0")) {
        for (int i = 0; i < n; i++)
            if (moves[i].castle == 2 || moves[i].castle == 4) { *out = moves[i]; return 1; }
        return 0;
    }

    int piece_type = 1;   /* pawn */
    size_t i = 0;
    if (strchr("NBRQK", san[0])) {
        switch (san[0]) { case 'N': piece_type=2; break; case 'B': piece_type=3; break;
                           case 'R': piece_type=4; break; case 'Q': piece_type=5; break;
                           case 'K': piece_type=6; break; }
        i = 1;
    }

    int promo = 0;
    char *eq = strchr(san, '=');
    if (eq) {
        char pc = toupper((unsigned char)eq[1]);
        switch (pc) { case 'N': promo=2; break; case 'B': promo=3; break;
                      case 'R': promo=4; break; case 'Q': promo=5; break; }
        *eq = 0;
        len = strlen(san);
    }

    if (len < i + 2) return 0;
    /* Destination square is always the last 2 chars. */
    char df = san[len - 2], dr = san[len - 1];
    if (df < 'a' || df > 'h' || dr < '1' || dr > '8') return 0;
    int to_file = df - 'a', to_rank = '8' - dr;   /* zsq rank 0 = rank 8 */
    int to_sq = to_rank * 8 + to_file;

    int disambig_file = -1, disambig_rank = -1;
    for (size_t k = i; k + 2 < len; k++) {
        char c = san[k];
        if (c >= 'a' && c <= 'h') disambig_file = c - 'a';
    }
    for (size_t k = i; k + 2 < len; k++) {
        char c = san[k];
        if (c >= '1' && c <= '8') disambig_rank = '8' - c;
    }

    int match = -1, matches = 0;
    for (int mi = 0; mi < n; mi++) {
        Move *m = &moves[mi];
        if (m->to != to_sq) continue;
        int pt = PC_TYPE(b->b[m->from]);
        if (pt != piece_type) continue;
        if (promo && m->prom != promo) continue;
        if (!promo && m->prom != 0 && piece_type == 1) continue;   /* pawn move that IS a promo but none requested */
        int ff = m->from % 8, fr = m->from / 8;
        if (disambig_file >= 0 && ff != disambig_file) continue;
        if (disambig_rank >= 0 && fr != disambig_rank) continue;
        match = mi; matches++;
    }
    if (matches != 1) {
        if (match < 0) return 0;
    }
    *out = moves[match];
    return 1;
}

void move_to_san(const Board *b, const Move *m, char *out, size_t cap) {
    if (m->castle == 1 || m->castle == 3) { strncpy(out, "O-O", cap - 1); out[cap-1]=0; return; }
    if (m->castle == 2 || m->castle == 4) { strncpy(out, "O-O-O", cap - 1); out[cap-1]=0; return; }

    int ptype = PC_TYPE(b->b[m->from]);
    int is_capture = (b->b[m->to] != 0) || m->epc;
    char to_file = (char)('a' + (m->to % 8));
    char to_rank = (char)('8' - (m->to / 8));
    char buf[16]; size_t n = 0;

    if (ptype == 1) {   /* pawn */
        if (is_capture) { buf[n++] = (char)('a' + (m->from % 8)); buf[n++] = 'x'; }
        buf[n++] = to_file; buf[n++] = to_rank;
        if (m->prom) {
            static const char promo_letters[6] = " ?NBRQ";   /* index by m->prom (2=N..5=Q) */
            buf[n++] = '='; buf[n++] = promo_letters[m->prom];
        }
    } else {
        static const char piece_letters[7] = " PNBRQK";   /* index by PC_TYPE */
        buf[n++] = piece_letters[ptype];

        Move legal[MAX_MOVES];
        int cnt = board_gen_moves(b, legal);
        int ambiguous = 0, same_file = 0, same_rank = 0;
        for (int i = 0; i < cnt; i++) {
            if (legal[i].to == m->to && legal[i].from != m->from &&
                PC_TYPE(b->b[legal[i].from]) == ptype) {
                ambiguous = 1;
                if (legal[i].from % 8 == m->from % 8) same_file = 1;
                if (legal[i].from / 8 == m->from / 8) same_rank = 1;
            }
        }
        if (ambiguous) {
            if (!same_file) buf[n++] = (char)('a' + (m->from % 8));
            else if (!same_rank) buf[n++] = (char)('8' - (m->from / 8));
            else { buf[n++] = (char)('a' + (m->from % 8)); buf[n++] = (char)('8' - (m->from / 8)); }
        }
        if (is_capture) buf[n++] = 'x';
        buf[n++] = to_file; buf[n++] = to_rank;
    }
    buf[n] = 0;
    strncpy(out, buf, cap - 1); out[cap - 1] = 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Recursive directory walk (dirent.h — available on mingw-w64 as well
 * as Linux, so this compiles unchanged on both this project's Windows
 * dev machines and any Linux CI).
 * ═══════════════════════════════════════════════════════════════════ */
static int has_suffix_ci(const char *s, const char *suf) {
    size_t ls = strlen(s), lsuf = strlen(suf);
    if (ls < lsuf) return 0;
#if defined(_WIN32)
    return _stricmp(s + ls - lsuf, suf) == 0;
#else
    return strcasecmp(s + ls - lsuf, suf) == 0;
#endif
}

static int is_directory(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return (st.st_mode & S_IFDIR) != 0;
}

static int walk_recursive(const char *dir, OpeningFileCB cb, void *ud) {
    DIR *d = opendir(dir);
    if (!d) return 0;
    int count = 0;
    /* Collect entry names first so we can sort them — matches Python's
     * sorted(files) per directory in tests/run_selfplay.py's
     * load_all_openings(), which keeps indexing order (and therefore
     * .idx cache validity / entry ordering) stable across runs. */
    char **names = NULL; int n = 0, cap = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..")) continue;
        if (n >= cap) { cap = cap ? cap * 2 : 32; names = (char **)realloc(names, cap * sizeof(char *)); }
        names[n] = strdup(ent->d_name);
        n++;
    }
    closedir(d);
    /* Simple insertion sort — directory listings here are at most a
     * few dozen entries (openings/lines, openings/positions), no need
     * for qsort ceremony. */
    for (int i = 1; i < n; i++) {
        char *key = names[i]; int j = i - 1;
        while (j >= 0 && strcmp(names[j], key) > 0) { names[j+1] = names[j]; j--; }
        names[j+1] = key;
    }

    for (int i = 0; i < n; i++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", dir, names[i]);
        if (is_directory(path)) {
            count += walk_recursive(path, cb, ud);
        } else if (has_suffix_ci(path, ".pgn") || has_suffix_ci(path, ".epd")) {
            cb(path, ud);
            count++;
        }
        free(names[i]);
    }
    free(names);
    return count;
}

int opening_walk_files(const char *root, OpeningFileCB cb, void *ud) {
    struct stat st;
    if (stat(root, &st) != 0) return -1;
    if (!(st.st_mode & S_IFDIR)) {
        cb(root, ud);
        return 1;
    }
    return walk_recursive(root, cb, ud);
}

/* ═══════════════════════════════════════════════════════════════════
 * OpeningIndex — byte-offset corpus index (see header comment).
 * ═══════════════════════════════════════════════════════════════════ */
static int oi_add_file(OpeningIndex *idx, const char *path) {
    if (idx->n_files >= idx->cap_files) {
        idx->cap_files = idx->cap_files ? idx->cap_files * 2 : 16;
        idx->files = (char **)realloc(idx->files, idx->cap_files * sizeof(char *));
    }
    idx->files[idx->n_files] = strdup(path);
    return idx->n_files++;
}

static void oi_push(OpeningIndex *idx, int file_id, uint64_t offset, OpeningKind kind) {
    if (idx->n_entries >= idx->cap_entries) {
        idx->cap_entries = idx->cap_entries ? idx->cap_entries * 2 : 4096;
        idx->entries = (OpeningEntry *)realloc(idx->entries, idx->cap_entries * sizeof(OpeningEntry));
    }
    OpeningEntry *e = &idx->entries[idx->n_entries++];
    e->file_id = file_id; e->offset = offset; e->kind = (uint8_t)kind;
}

/* Count whitespace-separated tokens in a line (EPD lines need >= 4:
 * board stm castling ep — matches tests/run_selfplay.py's
 * OpeningIndex._build_epd()). */
static int count_tokens(const char *s) {
    int n = 0; int in_tok = 0;
    for (; *s; s++) {
        if (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') { in_tok = 0; }
        else if (!in_tok) { in_tok = 1; n++; }
    }
    return n;
}

static void oi_index_epd(OpeningIndex *idx, int file_id, const char *path) {
    char idx_path[1200];
    snprintf(idx_path, sizeof(idx_path), "%s.idx", path);

    struct stat st_epd, st_idx;
    if (stat(path, &st_epd) != 0) return;
    if (stat(idx_path, &st_idx) == 0 && st_idx.st_mtime >= st_epd.st_mtime) {
        /* Cache format: raw array of little-endian/native uint64_t byte
         * offsets — matches Python's struct.pack(f"<{N}Q", *offsets).
         * On any x86/x64 machine (this project's only target) native
         * order IS little-endian, so a plain fread() round-trips with
         * Python's cache exactly. */
        FILE *f = fopen(idx_path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            size_t count = (size_t)sz / sizeof(uint64_t);
            if (count > 0) {
                uint64_t *offs = (uint64_t *)malloc(count * sizeof(uint64_t));
                size_t got = fread(offs, sizeof(uint64_t), count, f);
                for (size_t i = 0; i < got; i++) oi_push(idx, file_id, offs[i], OPEN_KIND_EPD);
                free(offs);
                fclose(f);
                fprintf(stderr, "[selfplay] [EPD] %s: %zu pos (cached)\n", path, got);
                return;
            }
            fclose(f);
        }
    }

    FILE *f = fopen(path, "rb");
    if (!f) return;
    uint64_t *offs = NULL; size_t n = 0, cap = 0;
    char line[2048];
    for (;;) {
        long off = ftell(f);
        if (!fgets(line, sizeof(line), f)) break;
        char *s = line;
        while (*s == ' ' || *s == '\t') s++;
        if (*s == '#' || *s == '\n' || *s == '\r' || *s == 0) continue;
        if (count_tokens(s) >= 4) {
            if (n >= cap) { cap = cap ? cap * 2 : 4096; offs = (uint64_t *)realloc(offs, cap * sizeof(uint64_t)); }
            offs[n++] = (uint64_t)off;
        }
    }
    fclose(f);

    FILE *cf = fopen(idx_path, "wb");
    if (cf) { fwrite(offs, sizeof(uint64_t), n, cf); fclose(cf); }

    for (size_t i = 0; i < n; i++) oi_push(idx, file_id, offs[i], OPEN_KIND_EPD);
    fprintf(stderr, "[selfplay] [EPD] %s: %zu pos indexed\n", path, n);
    free(offs);
}

/* Mirrors tests/run_selfplay.py's OpeningIndex._build_pgn(): the entry
 * offset is the START of the "[Event " tag line; the index only ever
 * needs to know where a game BEGINS (opening_index_fetch() reads
 * forward from there until it sees two consecutive blank lines, same
 * as the Python fetch()), not where it ends. */
static void oi_index_pgn(OpeningIndex *idx, int file_id, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return;
    char line[4096];
    long game_start = -1;
    int games = 0;
    for (;;) {
        long off = ftell(f);
        if (!fgets(line, sizeof(line), f)) {
            if (game_start >= 0) { oi_push(idx, file_id, (uint64_t)game_start, OPEN_KIND_PGN); games++; }
            break;
        }
        if (!strncmp(line, "[Event ", 7)) {
            game_start = off;
        } else {
            /* blank line (ignoring trailing \r\n) */
            int blank = 1;
            for (char *c = line; *c; c++) if (!isspace((unsigned char)*c)) { blank = 0; break; }
            if (blank && game_start >= 0) {
                oi_push(idx, file_id, (uint64_t)game_start, OPEN_KIND_PGN);
                games++;
                game_start = -1;
            }
        }
    }
    fclose(f);
    fprintf(stderr, "[selfplay] [PGN] %s: %d games indexed\n", path, games);
}

typedef struct { OpeningIndex *idx; } WalkCtx;

static void oi_walk_cb(const char *path, void *ud) {
    WalkCtx *wc = (WalkCtx *)ud;
    int file_id = oi_add_file(wc->idx, path);
    if (has_suffix_ci(path, ".epd")) oi_index_epd(wc->idx, file_id, path);
    else if (has_suffix_ci(path, ".pgn")) oi_index_pgn(wc->idx, file_id, path);
}

int opening_index_build(OpeningIndex *idx, const char *root) {
    memset(idx, 0, sizeof(*idx));
    WalkCtx wc = { idx };
    int rc = opening_walk_files(root, oi_walk_cb, &wc);
    return rc < 0 ? -1 : 0;
}

void opening_index_free(OpeningIndex *idx) {
    for (int i = 0; i < idx->n_files; i++) free(idx->files[i]);
    free(idx->files);
    free(idx->entries);
    memset(idx, 0, sizeof(*idx));
}

static int fetch_epd(const OpeningIndex *idx, const OpeningEntry *e, OpeningPick *out) {
    FILE *f = fopen(idx->files[e->file_id], "rb");
    if (!f) return 0;
    fseek(f, (long)e->offset, SEEK_SET);
    char line[2048];
    int ok = fgets(line, sizeof(line), f) != NULL;
    fclose(f);
    if (!ok) return 0;

    char tok[8][256]; int ntok = 0;
    char tmp[2048]; strncpy(tmp, line, sizeof(tmp) - 1); tmp[sizeof(tmp)-1] = 0;
    char *save = NULL;
    char *t = strtok_r(tmp, " \t\r\n", &save);
    while (t && ntok < 8) { strncpy(tok[ntok], t, sizeof(tok[ntok]) - 1); tok[ntok][sizeof(tok[ntok])-1]=0; ntok++; t = strtok_r(NULL, " \t\r\n", &save); }
    if (ntok < 4) return 0;

    out->has_fen = 1;
    if (ntok >= 6)
        snprintf(out->fen, sizeof(out->fen), "%s %s %s %s %s %s", tok[0], tok[1], tok[2], tok[3], tok[4], tok[5]);
    else
        snprintf(out->fen, sizeof(out->fen), "%s %s %s %s 0 1", tok[0], tok[1], tok[2], tok[3]);
    out->n_moves = 0;
    return 1;
}

static int fetch_pgn(const OpeningIndex *idx, const OpeningEntry *e, Board *scratch, OpeningPick *out) {
    FILE *f = fopen(idx->files[e->file_id], "rb");
    if (!f) return 0;
    fseek(f, (long)e->offset, SEEK_SET);

    static const char *STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    /* scratch->undo/scratch->nnue are assumed already bound by the
     * caller (see opening_pool.h) — we only need board_gen_moves() and
     * board_make() to work, not a full evaluation-capable Board. */
    board_load_fen(scratch, STARTPOS);

    out->has_fen = 0;
    out->n_moves = 0;

    /* PGN structure from here: a block of "[Tag ...]" header lines, one
     * blank separator line, then movetext, terminated by the NEXT blank
     * line (or EOF). We don't need to build a block like the Python
     * fetch() does (which then hands it to chess.pgn.read_game()) — we
     * can tokenize movetext directly with the same SAN resolver
     * load_pgn_openings() (arena.c) already uses. */
    char line[4096];
    int header_done = 0;
    int stop = 0;
    while (!stop && out->n_moves < OPENING_MAX_FORCED_MOVES && fgets(line, sizeof(line), f)) {
        char *s = line;
        while (*s == ' ' || *s == '\t') s++;
        if (!header_done && s[0] == '[') continue;   /* still in the tag block */

        int blank = 1;
        for (char *c = s; *c; c++) if (!isspace((unsigned char)*c)) { blank = 0; break; }
        if (blank) {
            if (!header_done) { header_done = 1; continue; }   /* header/movetext separator */
            break;                                              /* blank after movetext: game over */
        }
        header_done = 1;

        char tmp[4096]; strncpy(tmp, s, sizeof(tmp) - 1); tmp[sizeof(tmp)-1] = 0;
        char *save = NULL;
        char *tokraw = strtok_r(tmp, " \t\r\n", &save);
        while (tokraw && out->n_moves < OPENING_MAX_FORCED_MOVES) {
            char *tok = tokraw;
            char *dot = strrchr(tok, '.');
            if (dot) {
                int all_digits = 1;
                for (char *c = tok; c < dot; c++) if (!isdigit((unsigned char)*c)) { all_digits = 0; break; }
                if (all_digits && dot != tok) {
                    tok = dot + 1;
                    if (*tok == 0) { tokraw = strtok_r(NULL, " \t\r\n", &save); continue; }
                }
            }
            if (!strcmp(tok, "1-0") || !strcmp(tok, "0-1") ||
                !strcmp(tok, "1/2-1/2") || !strcmp(tok, "*")) { stop = 1; break; }

            Move mv;
            if (!san_resolve(scratch, tok, &mv)) { stop = 1; break; }
            board_make(scratch, &mv);
            out->moves[out->n_moves++] = mv;
            tokraw = strtok_r(NULL, " \t\r\n", &save);
        }
    }
    fclose(f);
    return out->n_moves > 0 || out->has_fen;
}

int opening_index_fetch(const OpeningIndex *idx, size_t i, Board *scratch, OpeningPick *out) {
    if (i >= idx->n_entries) return 0;
    const OpeningEntry *e = &idx->entries[i];
    memset(out, 0, sizeof(*out));
    if (e->kind == OPEN_KIND_EPD) return fetch_epd(idx, e, out);
    return fetch_pgn(idx, e, scratch, out);
}
