#include <stdio.h>
#include <stdint.h>
#include "nnue.h"
extern int32_t *_ext_buf[256][2][256];
extern int16_t _acc_buf_w[256][256];
extern int32_t *_nnL1B;
int main() {
    nnue_load("nnue_weights.bin");
    uint8_t board[64]={
        20,18,19,21,22,19,18,20,
        17,17,17,17,17,17,17,17,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,
         9, 9, 9, 9, 9, 9, 9, 9,
        12,10,11,13,14,11,10,12,
    };
    nnue_rebuild(board);
    nnue_eval(0, board); // computes _ext_buf etc
    return 0;
}
