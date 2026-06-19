#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define NNUE_TEST 1
int nnue_load(const char *path);
int nnue_eval(int stm, const uint8_t *board);
void nnue_rebuild(const uint8_t *board);

extern int32_t _nnL1B[];

int main() {
    if (nnue_load("nnue_weights.bin")!=0) return 1;

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
    int score = nnue_eval(0, board);
    printf("EVAL: %d\n", score);
    return 0;
}
