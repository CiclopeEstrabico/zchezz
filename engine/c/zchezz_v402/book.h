/* book.h — Polyglot opening book reader
 *
 * Probes a standard Polyglot .bin opening book file.
 * Compile with -DNO_BOOK to stub everything out (WASM builds).
 */
#pragma once
#include "board.h"

#ifdef NO_BOOK
/* ── Stubs for WASM / no-book builds ──────────────────────────── */
static inline int  book_open(const char *path) { (void)path; return 0; }
static inline int  book_probe(const Board *b, Move *out) { (void)b; (void)out; return 0; }
static inline void book_close(void) {}
static inline int  book_is_loaded(void) { return 0; }

#else
/* ── Full implementation ──────────────────────────────────────── */

/*
 * Open a Polyglot .bin book file.  Reads entire file into memory.
 * Returns: number of entries loaded (0 on failure).
 */
int book_open(const char *path);

/*
 * Probe the book for the current position.
 * Uses weighted random selection among all matching entries.
 * out: receives the selected move in Zchezz notation.
 * Returns: 1 if a book move was found, 0 if position not in book.
 */
int book_probe(const Board *b, Move *out);

/*
 * Free book resources.
 */
void book_close(void);

/*
 * Returns 1 if a book is currently loaded.
 */
int book_is_loaded(void);

#endif /* NO_BOOK */
