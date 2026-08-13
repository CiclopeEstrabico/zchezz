/* Throwaway synthetic SPRT/Elo verification — NOT part of the build,
 * links arena.c compiled with -DARENA_NO_MAIN to call sprt_llr()/
 * elo_diff_ci() directly on hand-picked W/D/L, no games played. */
#include <stdio.h>
#include "arena.h"

int main(void) {
    double llr, lo, hi, elo, ci;

    /* Lopsided record: candidate way stronger than elo1 -> expect ACCEPT_H1 */
    long long w=90, d=5, l=5;
    SprtVerdict v = sprt_llr(w,d,l, 0.0, 5.0, 0.05, 0.05, &llr, &lo, &hi);
    elo_diff_ci(w,d,l,&elo,&ci);
    printf("lopsided +90=5-5: elo=%.1f+/-%.1f llr=%.3f bounds=[%.3f,%.3f] verdict=%d (want %d ACCEPT_H1)\n",
           elo, ci, llr, lo, hi, v, SPRT_ACCEPT_H1);

    /* Even record -> expect CONTINUE or ACCEPT_H0 (never ACCEPT_H1) */
    w=48; d=4; l=48;
    v = sprt_llr(w,d,l, 0.0, 5.0, 0.05, 0.05, &llr, &lo, &hi);
    elo_diff_ci(w,d,l,&elo,&ci);
    printf("even +48=4-48: elo=%.1f+/-%.1f llr=%.3f bounds=[%.3f,%.3f] verdict=%d (want %d or %d, NOT %d)\n",
           elo, ci, llr, lo, hi, v, SPRT_CONTINUE, SPRT_ACCEPT_H0, SPRT_ACCEPT_H1);

    /* Clearly worse than elo0 -> expect ACCEPT_H0 */
    w=5; d=5; l=90;
    v = sprt_llr(w,d,l, 0.0, 5.0, 0.05, 0.05, &llr, &lo, &hi);
    elo_diff_ci(w,d,l,&elo,&ci);
    printf("lopsided-loss +5=5-90: elo=%.1f+/-%.1f llr=%.3f bounds=[%.3f,%.3f] verdict=%d (want %d ACCEPT_H0)\n",
           elo, ci, llr, lo, hi, v, SPRT_ACCEPT_H0);

    /* Score-from-elo sanity: 0 elo -> 0.5, +400 elo -> ~0.909 */
    printf("score_from_elo(0)=%.4f (want 0.5000)\n", score_from_elo(0.0));
    printf("score_from_elo(400)=%.4f (want ~0.9091)\n", score_from_elo(400.0));
    return 0;
}
