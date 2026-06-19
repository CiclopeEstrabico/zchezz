#include <stdio.h>
#include <stdint.h>

#define COL_W  8
#define COL_B 16
#define PC_COLOR(p)  ((p) & 24)
#define PC_TYPE(p)   ((p) &  7)

int piece_type_idx(uint8_t p) {
    if (p < 9 || p > 22) return -1;
    int t = PC_TYPE(p);
    if (t < 1 || t > 6) return -1;
    return t - 1;
}

int main() {
    uint8_t start_pos[64] = {
        20,18,19,21,22,19,18,20,
        17,17,17,17,17,17,17,17,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         9, 9, 9, 9, 9, 9, 9, 9,
        12,10,11,13,14,11,10,12,
    };

    for (int sq = 0; sq < 64; sq++) {
        uint8_t p = start_pos[sq];
        if (!p) continue;
        int pt = piece_type_idx(p);
        if (pt < 0) continue;
        int isW  = (PC_COLOR(p) == COL_W);
        int pySq = sq ^ 56;
        int coW  = isW ? 0 : 6;
        int idx = coW*64 + pt*64 + pySq;
        printf("sq=%d p=%d pt=%d color=%d => idx=%d\n", sq, p, pt, PC_COLOR(p), idx);
    }
    return 0;
}
