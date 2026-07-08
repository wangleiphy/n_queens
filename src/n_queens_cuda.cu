#include <cuda_runtime.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>

#include "macro.h"
#include "n_queens.h"
#include "utils.h"

__device__ unsigned long long global_counter = 0;

// v2: register-cached top-of-stack. The current DFS level lives in registers
// (cur,left,right,valid_pos); shared memory is only touched when descending
// past a level that still has remaining choices (push) or when backtracking
// (pop). Levels with no remaining choices are never pushed, so every pop
// yields a level with work. Visits the same tree as v1 in the same order.
__global__ void n_queens_v2(int N, int *tot, long long *partial_sum, long long cnt) {
    unsigned long long tid = atomicAdd(&global_counter, 1);
    const int last = (1 << N) - 1;

    while (tid < cnt) {
        long long sum = 0;
        int bottom = ((threadIdx.x / 32) * 32 * STACKSIZE + threadIdx.x % 32) << 4;
        int cur = tot[tid * 3];
        int left = tot[tid * 3 + 1];
        int right = tot[tid * 3 + 2];
        int valid_pos = last & ~cur & ~left & ~right;

        if(valid_pos == 0) {
            tid = atomicAdd(&global_counter, 1);
            continue;
        }

        asm(".reg .s32 top, p, cc, ll, rr, vv, tmp;\n\t"
            ".reg .s64 ltmp;\n\t"
            ".reg .pred pz, pq, pe, z;\n\t"
            ".shared .align 16 .b8 stack2[" NQ_STR(STACKBYTES) "];\n\t"

            " mov.u32 tmp, stack2;\n\t"
            " add.s32 %5, %5, tmp;\n\t"                                     // bottom += stack base
            " mov.u32 top, %5;\n\t"                                         // top = bottom (empty stack)

            " LOOP2:\n\t"
            " neg.s32 p, %4;\n\t"                                           // p = -v0
            " and.b32 p, p, %4;\n\t"                                        // p = v0 & -v0
            " sub.s32 %4, %4, p;\n\t"                                       // v0 -= p
            " or.b32  cc, %1, p;\n\t"                                       // cc = c0 | p
            " or.b32  ll, %2, p;\n\t"
            " shl.b32 ll, ll, 1;\n\t"                                       // ll = (l0 | p) << 1
            " or.b32  rr, %3, p;\n\t"
            " shr.b32 rr, rr, 1;\n\t"                                       // rr = (r0 | p) >> 1
            " lop3.b32 tmp, cc, ll, rr, 0x1;\n\t"                           // ~cc & ~ll & ~rr
            " and.b32 vv, tmp, %6;\n\t"                                     // vv = last & tmp
            " popc.b32 tmp, cc;\n\t"
            " setp.eq.s32 pz, tmp, %7;\n\t"                                 // popc(cc) == N - 1
            " setp.eq.s32 pq, vv, 0;\n\t"                                   // vv == 0
            " or.pred z, pz, pq;\n\t"                                       // leaf or dead end
            " @z bra ZPATH;\n\t"

            // descend: push current level if it still has choices, child -> regs
            " setp.ne.s32 pq, %4, 0;\n\t"                                   // v0 != 0
            " @pq st.shared.v4.u32 [top], {%1, %2, %3, %4};\n\t"            // push {c0,l0,r0,v0}
            " @pq add.s32 top, top, 512;\n\t"
            " mov.b32 %1, cc;\n\t"
            " mov.b32 %2, ll;\n\t"
            " mov.b32 %3, rr;\n\t"
            " mov.b32 %4, vv;\n\t"
            " bra.uni LOOP2;\n\t"

            " ZPATH:\n\t"                                                   // count solutions (0 if dead end)
            " popc.b32 tmp, vv;\n\t"
            " cvt.s64.s32 ltmp, tmp;\n\t"
            " add.s64 %0, %0, ltmp;\n\t"                                    // sum += popc(vv)
            " setp.ne.s32 pq, %4, 0;\n\t"
            " @pq bra.uni LOOP2;\n\t"                                       // same level still has choices
            " setp.eq.s32 pe, top, %5;\n\t"
            " @pe bra FINISH2;\n\t"                                         // stack empty -> done
            " sub.s32 top, top, 512;\n\t"
            " ld.shared.v4.u32 {%1, %2, %3, %4}, [top];\n\t"                // pop (always has choices)
            " bra.uni LOOP2;\n\t"

            " FINISH2:\n\t"
            :"+l"(sum), "+r"(cur), "+r"(left), "+r"(right), "+r"(valid_pos), "+r"(bottom)  // output
            :"r"(last), "r"(N - 1)                                          // input
        );

        partial_sum[tid] = sum;
        tid = atomicAdd(&global_counter, 1);
    }
}

// v4: v1 plus a two-row leaf unroll. A child at depth N-2 (two rows left) is
// never pushed; instead an inner register-only loop counts its completions:
// for each choice q in its valid mask, add popc of the resulting last-row
// mask. Depth-(N-1) nodes are the majority of the tree, and v1 spends a full
// iteration incl. a shared-memory round trip on each; here they cost ~14
// issue slots and no shared traffic. Stack depth requirement drops to
// N - rows - 2. Otherwise identical to v1 (same tree, same order).
__global__ void n_queens_v4(int N, int *tot, long long *partial_sum, long long cnt) {
    unsigned long long tid = atomicAdd(&global_counter, 1);
    const int last = (1 << N) - 1;

    while (tid < cnt) {
        long long sum = 0;
        int bottom = ((threadIdx.x / 32) * 32 * STACKSIZE + threadIdx.x % 32) << 4;
        int cur = tot[tid * 3];
        int left = tot[tid * 3 + 1];
        int right = tot[tid * 3 + 2];
        int valid_pos = last & ~cur & ~left & ~right;

        if(valid_pos == 0) {
            tid = atomicAdd(&global_counter, 1);
            continue;
        }

        asm(".reg .s32 top, tmp, tmp2, cq, rq;\n\t"
            ".reg .s64 ltmp;\n\t"
            ".reg .pred p, q, z, y, nz;\n\t"
            ".shared .align 16 .b8 stack4[" NQ_STR(STACKBYTES) "];\n\t"

            " mov.u32 top, %5;\n\t"
            " mov.u32 tmp, stack4;\n\t"
            " add.s32 %5, %5, tmp;\n\t"
            " add.s32 top, top, tmp;\n\t"

            " st.shared.v4.u32 [top], {%1, %2, %3, %4};\n\t"                // stack[top] = {cur, left, right, valid_pos}
            " add.s32 top, top, 512;\n\t"                                   // top += 512

            " LOOP4:\n\t"
            " setp.eq.s32 p, top, %5;\n\t"                                  // top == bottom
            " @p bra FINISH4;\n\t"                                          // done

            " ld.shared.v4.u32 {%1, %2, %3, %4}, [top + -512];\n\t"         // {cur, left, right, valid_pos} = stack[top - 512]
            " neg.s32 tmp, %4;\n\t"                                         // p = -valid_pos
            " and.b32 tmp, %4, tmp;\n\t"                                    // p = valid_pos & (-valid_pos)
            " sub.s32 %4, %4, tmp;\n\t"                                     // valid_pos -= p
            " st.shared.s32 [top + -500], %4;\n\t"                          // stack[top - 500] = valid_pos
            " setp.eq.s32 p, %4, 0;\n\t"                                    // p = (valid_pos == 0)
            " selp.b32 tmp2, 512, 0, p;\n\t"                                // tmp2 = (p == 1 ? 512 : 0)
            " sub.s32 top, top, tmp2;\n\t"                                  // top -= 512

            " or.b32 %1, %1, tmp;\n\t"                                      // cur = cur | p
            " or.b32 %2, %2, tmp;\n\t"                                      // left = left | p
            " shl.b32 %2, %2, 1;\n\t"                                       // left = left << 1
            " or.b32 %3, %3, tmp;\n\t"                                      // right = right | p
            " shr.b32 %3, %3, 1;\n\t"                                       // right = right >> 1
            " lop3.b32 tmp, %1, %2, %3, 0x1;\n\t"                           // tmp = ~cur & ~left & ~right;
            " and.b32 %4, %6, tmp;\n\t"                                     // valid_pos = last & tmp
            " popc.b32 tmp, %1;\n\t"                                        // tmp = popc(cur)
            " setp.eq.s32 p, tmp, %7;\n\t"                                  // popc(cur) == N - 1
            " setp.eq.s32 y, tmp, %8;\n\t"                                  // popc(cur) == N - 2
            " setp.eq.s32 q, %4, 0;\n\t"                                    // valid_pos == 0
            " or.pred z, p, q;\n\t"                                         // valid_pos == 0 || popc(cur) == N - 1

            " popc.b32 tmp, %4;\n\t"                                        // tmp = popc(valid_pos)
            " cvt.s64.s32 ltmp, tmp;\n\t"                                   // s32 -> s64
            " selp.b64 ltmp, ltmp, 0, z;\n\t"                               // ltmp = (z == 1 ? ltmp : 0)
            " add.s64 %0, %0, ltmp;\n\t"                                    // sum += popc(valid_pos)

            " not.pred nz, z;\n\t"
            " and.pred y, y, nz;\n\t"                                       // live node with two rows left
            " @y bra LEAF4;\n\t"

            " @!z st.shared.v4.u32 [top], {%1, %2, %3, %4};\n\t"            // stack[top] = {cur, left, right, valid_pos}
            " selp.b32 tmp, 0, 512, z;\n\t"                                 // tmp = (z == 1 ? 0 : 512)
            " add.s32 top, top, tmp;\n\t"                                   // top += 512
            " bra.uni LOOP4;\n\t"

            " LEAF4:\n\t"                                                   // {cur,left,right,valid_pos} = live depth-(N-2) node
            " LEAFLOOP4:\n\t"
            " neg.s32 tmp, %4;\n\t"
            " and.b32 tmp, %4, tmp;\n\t"                                    // q = lowest bit of valid_pos
            " sub.s32 %4, %4, tmp;\n\t"
            " or.b32  cq, %1, tmp;\n\t"                                     // cur | q
            " or.b32  tmp2, %2, tmp;\n\t"
            " shl.b32 tmp2, tmp2, 1;\n\t"                                   // (left | q) << 1
            " or.b32  rq, %3, tmp;\n\t"
            " shr.b32 rq, rq, 1;\n\t"                                       // (right | q) >> 1
            " lop3.b32 tmp, cq, tmp2, rq, 0x1;\n\t"                         // free cells of the last row
            " and.b32 tmp, tmp, %6;\n\t"
            " popc.b32 tmp, tmp;\n\t"
            " cvt.s64.s32 ltmp, tmp;\n\t"
            " add.s64 %0, %0, ltmp;\n\t"                                    // each free cell is a solution
            " setp.ne.s32 p, %4, 0;\n\t"
            " @p bra LEAFLOOP4;\n\t"
            " bra.uni LOOP4;\n\t"

            " FINISH4:\n\t"
            :"+l"(sum), "+r"(cur), "+r"(left), "+r"(right), "+r"(valid_pos), "+r"(bottom)  // output
            :"r"(last), "r"(N - 1), "r"(N - 2)                              // input
        );

        partial_sum[tid] = sum;
        tid = atomicAdd(&global_counter, 1);
    }
}

__global__ void n_queens(int N, int *tot, long long *partial_sum, long long cnt) {
    unsigned long long tid = atomicAdd(&global_counter, 1);
    const int last = (1 << N) - 1;

    while (tid < cnt) {
        long long sum = 0;
        int bottom = ((threadIdx.x / 32) * 32 * STACKSIZE + threadIdx.x % 32) << 4;
        int cur = tot[tid * 3];
        int left = tot[tid * 3 + 1];
        int right = tot[tid * 3 + 2];
        int valid_pos = last & ~cur & ~left & ~right;

        if(valid_pos == 0) {
            tid = atomicAdd(&global_counter, 1);
            continue;
        }

        asm(".reg .s32 top, tmp, tmp2;\n\t"
            ".reg .s64 ltmp;\n\t"
            ".reg .pred p, q, z;\n\t"
            ".shared .align 16 .b8 stack[" NQ_STR(STACKBYTES) "];\n\t"

            " mov.u32 top, %5;\n\t"
            " mov.u32 tmp, stack;\n\t"
            " add.s32 %5, %5, tmp;\n\t"
            " add.s32 top, top, tmp;\n\t"

            " st.shared.v4.u32 [top], {%1, %2, %3, %4};\n\t"                // stack[top] = {cur, left, right, valid_pos}
            " add.s32 top, top, 512;\n\t"                                   // top += 512

            " LOOP:\n\t"
            " setp.eq.s32 p, top, %5;\n\t"                                  // top == bottom
            " @p bra FINISH;\n\t"                                           // done

            " ld.shared.v4.u32 {%1, %2, %3, %4}, [top + -512];\n\t"         // {cur, left, right, valid_pos} = stack[top - 512]
            " neg.s32 tmp, %4;\n\t"                                         // p = -valid_pos
            " and.b32 tmp, %4, tmp;\n\t"                                    // p = valid_pos & (-valid_pos)
            " sub.s32 %4, %4, tmp;\n\t"                                     // valid_pos -= p
            " st.shared.s32 [top + -500], %4;\n\t"                          // stack[top - 500] = valid_pos
            " setp.eq.s32 p, %4, 0;\n\t"                                    // p = (valid_pos == 0)
            " selp.b32 tmp2, 512, 0, p;\n\t"                                // tmp = (p == 1 ? 512 : 0)
            " sub.s32 top, top, tmp2;\n\t"                                  // top -= 512

            " or.b32 %1, %1, tmp;\n\t"                                      // cur = cur | p
            " or.b32 %2, %2, tmp;\n\t"                                      // left = left | p
            " shl.b32 %2, %2, 1;\n\t"                                       // left = left << 1
            " or.b32 %3, %3, tmp;\n\t"                                      // right = right | p
            " shr.b32 %3, %3, 1;\n\t"                                       // right = right >> 1
            " lop3.b32 tmp, %1, %2, %3, 0x1;\n\t"                           // tmp = ~cur & ~left & ~right;
            " and.b32 %4, %6, tmp;\n\t"                                     // valid_pos = last & tmp
            " popc.b32 tmp, %1;\n\t"                                        // tmp = popc(cur)
            " setp.eq.s32 p, tmp, %7;\n\t"                                  // popc(cur) == N - 1
            " setp.eq.s32 q, %4, 0;\n\t"                                    // valid_pos == 0
            " or.pred z, p, q;\n\t"                                         // valid_pos == 0 || popc(cur) == N - 1

            " popc.b32 tmp, %4;\n\t"                                        // tmp = popc(valid_pos)
            " cvt.s64.s32 ltmp, tmp;\n\t"                                   // s32 -> s64
            " selp.b64 ltmp, ltmp, 0, z;\n\t"                               // ltmp = (z == 1 ? ltmp : 0)
            " add.s64 %0, %0, ltmp;\n\t"                                    // sum += popc(valid_pos)

            " @!z st.shared.v4.u32 [top], {%1, %2, %3, %4};\n\t"            // stack[top] = {cur, left, right, valid_pos}
            " selp.b32 tmp, 0, 512, z;\n\t"                                 // tmp = (z == 1 ? 0 : 512)
            " add.s32 top, top, tmp;\n\t"                                   // top += 512
            " bra.uni LOOP;\n\t"

            " FINISH:\n\t"
            :"+l"(sum), "+r"(cur), "+r"(left), "+r"(right), "+r"(valid_pos), "+r"(bottom)  // output
            :"r"(last), "r"(N - 1)                                          // input
        );

        partial_sum[tid] = sum;
        tid = atomicAdd(&global_counter, 1);
    }
}

long long cuda_n_queens(int N, int rows, long long range_start, long long range_end) {
    struct timeval start, end;
    long long sum = 0;
    vector<int> tot;

    // 1. get total subproblems.
    gettimeofday(&start, NULL);
    partial_n_queens(N, 0, 0, 0, tot, rows);

    if (N & 0x1) {
        partial_n_queens_for_odd(N, 0, 0, 0, tot, rows);
    }

    long long cnt = tot.size() / 3;
    vector<long long> partial_sum(cnt);
    gettimeofday(&end, NULL);

    print_with_time("Use %.2fms to generate %lld subproblems!\n", time_diff_ms(start, end), cnt);

    // optional subproblem range [range_start, range_end) for multi-node sharding;
    // outputs of disjoint ranges covering [0, cnt) sum to the full count.
    if (range_end < 0 || range_end > cnt) range_end = cnt;
    if (range_start < 0) range_start = 0;
    if (range_start > range_end) range_start = range_end;
    bool has_range = (range_start != 0 || range_end != cnt);
    if (has_range) {
        print_with_time("processing subproblem range [%lld, %lld) of %lld\n", range_start, range_end, cnt);
    }

    int gpu_num = 0;
    cudaGetDeviceCount(&gpu_num);
    if (gpu_num == 0) {
        printf("Failed to find any gpu!\n");
        return -1;
    }

    // kernel version, grid size and scheduler are runtime-selectable for A/B experiments
    const char *kenv = getenv("NQ_KERNEL");
    int kernel_version = kenv ? atoi(kenv) : 1;
    const char *genv = getenv("NQ_GRID");
    int grid_size = genv ? atoi(genv) : 1024;
    const char *senv = getenv("NQ_SCHED");
    bool dynamic_sched = has_range || (senv && strcmp(senv, "dynamic") == 0);

    {
        int maxb1 = 0, maxb2 = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxb1, static_cast<void (*)(int, int *, long long *, long long)>(n_queens), CU1DBLOCK, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb2, n_queens_v2, CU1DBLOCK, 0);
        print_with_time("kernel=v%d grid=%d block=%d sched=%s occupancy: v1 %d blocks/SM, v2 %d blocks/SM\n",
                        kernel_version, grid_size, CU1DBLOCK, dynamic_sched ? "dynamic" : "static",
                        maxb1, maxb2);
    }

    if (dynamic_sched) {
        // Dynamic scheduler: GPUs pull fixed-size chunks from a shared queue until
        // the range is exhausted. Removes the tail imbalance of the static split
        // (the Q(27) run lost ~18% wall clock to it) and adapts to heterogeneous GPUs.
        long long span = range_end - range_start;
        long long chunk;
        const char *cenv = getenv("NQ_CHUNK");
        if (cenv) {
            chunk = atoll(cenv);
        } else {
            chunk = span / (gpu_num * 24) + 1;
            if (chunk < 65536) chunk = 65536;
            if (chunk > 8388608) chunk = 8388608;
        }

        vector<long long> gpu_sum(gpu_num, 0);
        std::atomic<long long> next_start(range_start);

#pragma omp parallel num_threads(gpu_num)
        {
            int idx = omp_get_thread_num();
            CU_SAFE_CALL(cudaSetDevice(idx));

            struct timeval t0, t1;
            gettimeofday(&t0, NULL);

            int *cuda_tot;
            long long *cuda_partial_sum;
            CU_SAFE_CALL(cudaMalloc(&cuda_tot, sizeof(int) * chunk * 3));
            CU_SAFE_CALL(cudaMalloc(&cuda_partial_sum, sizeof(long long) * chunk));
            vector<long long> host_partial(chunk);

            long long done = 0;
            int nchunks = 0;
            const unsigned long long zero = 0;

            while (true) {
                long long s = next_start.fetch_add(chunk);
                if (s >= range_end) break;
                long long e = s + chunk < range_end ? s + chunk : range_end;
                long long c = e - s;

                CU_SAFE_CALL(cudaMemcpyToSymbol(global_counter, &zero, sizeof(zero)));
                CU_SAFE_CALL(cudaMemcpy(cuda_tot, tot.data() + s * 3, sizeof(int) * c * 3, cudaMemcpyHostToDevice));
                CU_SAFE_CALL(cudaMemset(cuda_partial_sum, 0, sizeof(long long) * c));

                dim3 dimBlock(CU1DBLOCK);
                dim3 dimGrid(grid_size);
                if (kernel_version == 4) {
                    n_queens_v4<<<dimGrid, dimBlock>>>(N, cuda_tot, cuda_partial_sum, c);
                } else if (kernel_version == 2) {
                    n_queens_v2<<<dimGrid, dimBlock>>>(N, cuda_tot, cuda_partial_sum, c);
                } else {
                    n_queens<<<dimGrid, dimBlock>>>(N, cuda_tot, cuda_partial_sum, c);
                }
                cudaError_t err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    printf("kernel error: %s\n", cudaGetErrorString(err));
                    exit(-1);
                }

                CU_SAFE_CALL(cudaMemcpy(host_partial.data(), cuda_partial_sum, sizeof(long long) * c, cudaMemcpyDeviceToHost));
                for (long long i = 0; i < c; i++) {
                    gpu_sum[idx] += host_partial[i];
                }
                done += c;
                nchunks++;
            }

            CU_SAFE_CALL(cudaFree(cuda_tot));
            CU_SAFE_CALL(cudaFree(cuda_partial_sum));
            gettimeofday(&t1, NULL);
            print_with_time("gpu [%d] finish job: %d chunks, %lld subproblems, %.2fms.\n",
                            idx, nchunks, done, time_diff_ms(t0, t1));
        }

        for (int i = 0; i < gpu_num; i++) {
            sum += gpu_sum[i] * 2;
        }
        return sum;
    }

    // 2. divide total subproblems to different trunks
    vector<long long> new_cnt(gpu_num), start_pos(gpu_num);
    long long total = 0;
    if (gpu_num == 8) {
        float ratio[8] = {0.20, 0.15, 0.12, 0.11, 0.11, 0.11, 0.10, 0.10};
        for (int i = 0; i < gpu_num - 1; i++) {
            new_cnt[i] = cnt * ratio[i];
            start_pos[i] = total;
            total += new_cnt[i];
        }
    } else if (gpu_num == 4) {
        float ratio[4] = {0.35, 0.23, 0.22, 0.2};
        for (int i = 0; i < gpu_num - 1; i++) {
            new_cnt[i] = cnt * ratio[i];
            start_pos[i] = total;
            total += new_cnt[i];
        }
    } else if (gpu_num == 2) {
        float ratio[2] = {0.58, 0.42};
        for (int i = 0; i < gpu_num - 1; i++) {
            new_cnt[i] = cnt * ratio[i];
            start_pos[i] = total;
            total += new_cnt[i];
        }
    } else {
        long long partial_cnt = cnt / gpu_num;
        for (int i = 0; i < gpu_num - 1; i++) {
            new_cnt[i] = partial_cnt;
            start_pos[i] = total;
            total += partial_cnt;
        }
    }
    new_cnt[gpu_num - 1] = cnt - total;
    start_pos[gpu_num - 1] = total;

    // 3. use different gpu to process each trunk
#pragma omp parallel num_threads(gpu_num)
    {
        int idx = omp_get_thread_num();
        CU_SAFE_CALL(cudaSetDevice(idx));

        long long total = cnt;
        long long cnt = new_cnt[idx];

        print_with_time("gpu [%d] start job, with %lld(%.2f) subproblems.\n", idx, cnt, cnt * 1.0 / total);

        int *cuda_tot;
        CU_SAFE_CALL(cudaMalloc(&cuda_tot, sizeof(int) * cnt * 3));
        CU_SAFE_CALL(cudaMemcpy(cuda_tot, tot.data() + start_pos[idx] * 3, sizeof(int) * cnt * 3, cudaMemcpyHostToDevice));

        long long *cuda_partial_sum;
        CU_SAFE_CALL(cudaMalloc(&cuda_partial_sum, sizeof(long long) * cnt));
        CU_SAFE_CALL(cudaMemset(cuda_partial_sum, 0, sizeof(long long) * cnt));

        dim3 dimBlock(CU1DBLOCK);
        dim3 dimGrid(grid_size);

        if (kernel_version == 4) {
            n_queens_v4<<<dimGrid, dimBlock>>>(N, cuda_tot, cuda_partial_sum, cnt);
        } else if (kernel_version == 2) {
            n_queens_v2<<<dimGrid, dimBlock>>>(N, cuda_tot, cuda_partial_sum, cnt);
        } else {
            n_queens<<<dimGrid, dimBlock>>>(N, cuda_tot, cuda_partial_sum, cnt);
        }

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("kernel error: %s\n", cudaGetErrorString(err));
        }

        CU_SAFE_CALL(cudaMemcpy(partial_sum.data() + start_pos[idx], cuda_partial_sum, sizeof(long long) * cnt, cudaMemcpyDeviceToHost));

        CU_SAFE_CALL(cudaFree(cuda_tot));
        CU_SAFE_CALL(cudaFree(cuda_partial_sum));
        print_with_time("gpu [%d] finish job.\n", idx);
    }

    for (long long i = 0; i < cnt; i++) {
        sum += partial_sum[i] * 2;
    }

    return sum;
}
