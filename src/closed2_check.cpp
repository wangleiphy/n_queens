// Verifies the closed-form two-row leaf count against plain DFS.
//
// For a node whose placed-queen count is n-2 (two rows remain), with
// v = valid cells of the next row and M = last & ~c & ~(l<<1) & ~(r>>1):
//   completions = popc(v)*popc(M) - popc(M&v) - popc(M&(v<<1)) - popc(M&(v>>1))
//
// Build: c++ -O2 -o closed2_check closed2_check.cpp
// Usage: ./closed2_check N   (compares full count against known-good DFS)

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int popc(uint32_t x) { return __builtin_popcount(x); }

static void solve_plain(int n, int cur, int left, int right, long long &sum) {
    int last = (1 << n) - 1;
    if (cur == last) {
        sum++;
        return;
    }
    int valid = last & (~(cur | left | right));
    while (valid) {
        int p = valid & (-valid);
        valid -= p;
        solve_plain(n, cur | p, (left | p) << 1, (right | p) >> 1, sum);
    }
}

static void solve_closed2(int n, int cur, int left, int right, long long &sum) {
    int last = (1 << n) - 1;
    uint32_t v = (uint32_t)(last & (~(cur | left | right)));
    if (popc((uint32_t)cur) == n - 2) {
        uint32_t M = (uint32_t)(last & ~cur & ~(left << 1) & ~(((uint32_t)right) >> 1));
        long long k = popc(v), m = popc(M);
        sum += k * m - popc(M & v) - popc(M & (v << 1)) - popc(M & (v >> 1));
        return;
    }
    while (v) {
        uint32_t p = v & (~v + 1);
        v -= p;
        solve_closed2(n, cur | (int)p, (left | (int)p) << 1, (int)((((uint32_t)right) | p) >> 1), sum);
    }
}

int main(int argc, char **argv) {
    int nmax = argc > 1 ? atoi(argv[1]) : 14;
    for (int n = 5; n <= nmax; n++) {
        long long a = 0, b = 0;
        solve_plain(n, 0, 0, 0, a);
        solve_closed2(n, 0, 0, 0, b);
        printf("N=%2d plain=%lld closed2=%lld %s\n", n, a, b, a == b ? "MATCH" : "MISMATCH!!");
    }
    return 0;
}
