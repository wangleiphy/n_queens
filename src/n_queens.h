#pragma once

#include <vector>

using namespace std;

void n_queens(int N, int cur, int left, int right, long long &sum);

void n_queens_iterative(int N, int cur, int left, int right, long long &sum);

void partial_n_queens(int N, int cur, int left, int right, vector<int> &tot, int rows);

void partial_n_queens_for_odd(int N, int cur, int left, int right, vector<int> &tot, int rows);

long long serial_n_queens(int N, int rows);

long long parallel_n_queens(int N, int rows);

// range_start/range_end select a subproblem range [range_start, range_end) for
// multi-node sharding; range_end = -1 means "to the end". Outputs of disjoint
// ranges covering [0, cnt) sum to the full count.
long long cuda_n_queens(int N, int rows, long long range_start = 0, long long range_end = -1);
