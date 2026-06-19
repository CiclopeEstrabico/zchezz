#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "nnue.h"
#include <string.h>

static long now_ns2(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1000000000L+ts.tv_nsec;
}

int main(int argc, char **argv) {
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
    
    int N = 500000;
    long t0 = now_ns2();
    volatile int sum = 0;
    
    for (int i=0; i<N; i++) {
        // Pseudo-random move
        NNMove m = {.from_sq = (i % 8) + 8, .to_sq = (i % 8) + 16, .prom = 0, .is_epc = 0, .castle = 0};
        uint8_t moved_piece = board[m.from_sq];
        uint8_t captured = board[m.to_sq];
        board[m.from_sq] = 0;
        board[m.to_sq] = moved_piece;
        
        nnue_push(board, &m);
        sum += nnue_eval(i & 1, board);
        nnue_pop();
        
        // Restore
        board[m.from_sq] = moved_piece;
        board[m.to_sq] = captured;
    }
    
    long t1 = now_ns2();
    double ms = (t1-t0)/1e6;
    printf("Simulated search %d nodes: %.1f ms (%.2f MNodes/s)\n", N, ms, N/ms/1000.0);
    return 0;
}
