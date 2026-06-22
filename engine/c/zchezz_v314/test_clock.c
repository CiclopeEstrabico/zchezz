#include <stdio.h>
#include <time.h>
int main() {
    struct timespec ts;
    for (int i = 0; i < 5; i++) {
        clock_gettime(CLOCK_MONOTONIC, &ts);
        long ms = (long)(ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL);
        long long ms_ll = (long long)(ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL);
        printf("now_ms=%ld ms_ll=%lld sizeof(long)=%d\n", ms, ms_ll, (int)sizeof(long));
        struct timespec sleep_ts = {0, 100000000};
        nanosleep(&sleep_ts, NULL);
    }
    return 0;
}
