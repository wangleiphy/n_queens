// Measures how many depth-d subproblem states are duplicates.
//
// The DFS subtree below a subproblem depends only on the masks
// (cur, left & board, right); prefixes that reach the same masks have equal
// completion counts, and a mirror-image state has the same count as well
// (reflection maps completions bijectively). This probe generates the exact
// production subproblem list, canonicalizes each state, and reports the
// collapse factor P/S. With --solve it also verifies that the weighted count
// over unique states equals the plain count, and times both.
//
// Build: g++ -O2 -fopenmp -o dedup_probe dedup_probe.cpp
// Usage: ./dedup_probe N rows [--solve]

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <vector>

using namespace std;
typedef unsigned __int128 u128;

static inline u128 pack(uint32_t c, uint32_t l, uint32_t r) {
    return ((u128)c << 64) | ((uint64_t)l << 32) | r;
}

static inline uint32_t rev_n(uint32_t x, int n) {
    uint32_t y = 0;
    for (int i = 0; i < n; i++) y |= ((x >> i) & 1u) << (n - 1 - i);
    return y;
}

// canonical form: min(state, mirrored state); mirror reverses columns,
// swapping the roles of the two diagonal masks
static inline u128 canon(uint32_t c, uint32_t l, uint32_t r, int n) {
    uint32_t last = (1u << n) - 1;
    l &= last;  // bits shifted past the board never come back
    u128 a = pack(c, l, r);
    u128 b = pack(rev_n(c, n), rev_n(r, n), rev_n(l, n));
    return a < b ? a : b;
}

// exact copies of the production generators (n_queens.cpp)
static void partial_gen(int n, int cur, int left, int right, vector<u128> &tot, int rows, bool raw) {
    int last = (1 << n) - 1;
    if (cur == 0) last = (1 << n / 2) - 1;
    int valid_pos = last & (~(cur | left | right));
    while (valid_pos) {
        int p = valid_pos & (-valid_pos);
        valid_pos -= p;
        if (rows == 1) {
            uint32_t c = cur | p, l = (uint32_t)((left | p) << 1), r = (uint32_t)((right | p) >> 1);
            tot.push_back(raw ? pack(c, l, r) : canon(c, l, r, n));
            continue;
        }
        partial_gen(n, cur | p, (left | p) << 1, (right | p) >> 1, tot, rows - 1, raw);
    }
}

static void partial_gen_odd(int n, int cur, int left, int right, vector<u128> &tot, int rows, bool raw) {
    int last = (1 << n) - 1;
    if (cur == 0) {
        last = (1 << n / 2);
    } else if ((cur & (cur - 1)) == 0) {
        last = (1 << (n - 2) / 2) - 1;
    }
    int valid_pos = last & (~(cur | left | right));
    while (valid_pos) {
        int p = valid_pos & (-valid_pos);
        valid_pos -= p;
        if (rows == 1) {
            uint32_t c = cur | p, l = (uint32_t)((left | p) << 1), r = (uint32_t)((right | p) >> 1);
            tot.push_back(raw ? pack(c, l, r) : canon(c, l, r, n));
            continue;
        }
        partial_gen_odd(n, cur | p, (left | p) << 1, (right | p) >> 1, tot, rows - 1, raw);
    }
}

static void solve(int n, int cur, int left, int right, long long &sum) {
    int last = (1 << n) - 1;
    if (cur == last) {
        sum++;
        return;
    }
    int valid_pos = last & (~(cur | left | right));
    while (valid_pos) {
        int p = valid_pos & (-valid_pos);
        valid_pos -= p;
        solve(n, cur | p, (left | p) << 1, (right | p) >> 1, sum);
    }
}

static double now_s() {
    return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("usage: %s N rows [--solve]\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]), rows = atoi(argv[2]);
    bool do_solve = argc > 3 && strcmp(argv[3], "--solve") == 0;

    // raw list (uncanonicalized) for the reference solve
    vector<u128> raw;
    double t0 = now_s();
    partial_gen(n, 0, 0, 0, raw, rows, true);
    if (n & 1) partial_gen_odd(n, 0, 0, 0, raw, rows, true);
    double t_gen = now_s() - t0;

    // canonical list
    vector<u128> can;
    can.reserve(raw.size());
    partial_gen(n, 0, 0, 0, can, rows, false);
    if (n & 1) partial_gen_odd(n, 0, 0, 0, can, rows, false);

    t0 = now_s();
    sort(can.begin(), can.end());
    long long P = (long long)can.size(), S = 0;
    vector<u128> uniq;
    vector<long long> mult;
    for (size_t i = 0; i < can.size();) {
        size_t j = i;
        while (j < can.size() && can[j] == can[i]) j++;
        uniq.push_back(can[i]);
        mult.push_back((long long)(j - i));
        i = j;
    }
    S = (long long)uniq.size();
    double t_dedup = now_s() - t0;

    printf("N=%d rows=%d  P=%lld  S=%lld  collapse=%.4f  (gen %.2fs, dedup %.2fs)\n",
           n, rows, P, S, (double)P / (double)S, t_gen, t_dedup);

    if (do_solve) {
        int lastmask = (1 << n) - 1;
        t0 = now_s();
        long long sum_raw = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : sum_raw)
        for (long long i = 0; i < P; i++) {
            uint32_t c = (uint32_t)(raw[i] >> 64), l = (uint32_t)(raw[i] >> 32), r = (uint32_t)raw[i];
            long long s = 0;
            solve(n, (int)c, (int)l, (int)r, s);
            sum_raw += s;
        }
        double t_raw = now_s() - t0;

        t0 = now_s();
        long long sum_dedup = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : sum_dedup)
        for (long long i = 0; i < S; i++) {
            uint32_t c = (uint32_t)(uniq[i] >> 64), l = (uint32_t)(uniq[i] >> 32), r = (uint32_t)uniq[i];
            long long s = 0;
            solve(n, (int)c, (int)l, (int)r, s);
            sum_dedup += s * mult[i];
        }
        double t_dedup_solve = now_s() - t0;

        printf("  total(raw)   = %lld  (%.2fs)\n", 2 * sum_raw, t_raw);
        printf("  total(dedup) = %lld  (%.2fs)  match=%s  speedup=%.3f\n",
               2 * sum_dedup, t_dedup_solve, sum_raw == sum_dedup ? "YES" : "NO!!",
               t_raw / t_dedup_solve);
    }
    return 0;
}
