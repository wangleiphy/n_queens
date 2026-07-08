#include <cuda_runtime.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>

#include "macro.h"
#include "n_queens.h"
#include "utils.h"

// Task fetch: one atomic per subproblem on a per-GPU device counter, reset
// before each launch. Kept as an unconditional pointer-based atomic so the
// compiler cannot unswitch the DFS loop (cloning the inline asm would
// duplicate its .reg declarations). Note: a host-pinned counter shared across
// GPUs via atomicAdd_system was tried and silently duplicated task ids on
// PCIe nodes -- do not share this counter between devices.
__device__ __forceinline__ unsigned long long fetch_task(unsigned long long *sys_counter) {
    return atomicAdd_system(sys_counter, 1ULL);
}

// Sums partial_sum[0..n) into *out (device memory, pre-zeroed).
__global__ void reduce_partial(const long long *v, long long n, unsigned long long *out) {
    long long acc = 0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n;
         i += (long long)gridDim.x * blockDim.x) {
        acc += v[i];
    }
    atomicAdd(out, (unsigned long long)acc);
}


__global__ void n_queens(int N, int *tot, long long *partial_sum, long long cnt, unsigned long long *sys_counter) {
    unsigned long long tid = fetch_task(sys_counter);
    const int last = (1 << N) - 1;

    while (tid < cnt) {
        long long sum = 0;
        int bottom = ((threadIdx.x / 32) * 32 * STACKSIZE + threadIdx.x % 32) << 4;
        int cur = tot[tid * 3];
        int left = tot[tid * 3 + 1];
        int right = tot[tid * 3 + 2];
        int valid_pos = last & ~cur & ~left & ~right;

        if(valid_pos == 0) {
            tid = fetch_task(sys_counter);
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
        tid = fetch_task(sys_counter);
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

    // shard drivers use this to size ranges without running the solve
    if (getenv("NQ_COUNT_ONLY")) {
        printf("TOTAL_SUBPROBLEMS %lld\n", cnt);
        exit(0);
    }

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

    // grid size and scheduler are runtime-selectable
    const char *genv = getenv("NQ_GRID");
    int grid_size = genv ? atoi(genv) : 1024;
    // NQ_SCHED: "static" (default) = one launch per GPU over a fixed split;
    // "chunk" = guided self-scheduling, GPUs pull geometrically shrinking
    // chunks from a host-side atomic queue (used automatically for ranges).
    // A host-pinned counter shared across GPUs via atomicAdd_system was tried
    // and produced duplicated task ids on PCIe nodes (non-atomic across
    // devices) -- do not resurrect it without hardware validation.
    const char *senv = getenv("NQ_SCHED");
    const char *sched = senv ? senv : (has_range ? "chunk" : "static");
    bool chunk_sched = strcmp(sched, "chunk") == 0 || strcmp(sched, "dynamic") == 0;

    {
        int maxb = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxb, static_cast<void (*)(int, int *, long long *, long long, unsigned long long *)>(n_queens), CU1DBLOCK, 0);
        print_with_time("grid=%d block=%d sched=%s occupancy: %d blocks/SM (%d threads/SM)\n",
                        grid_size, CU1DBLOCK, chunk_sched ? "chunk" : "static", maxb, maxb * CU1DBLOCK);
    }

    if (chunk_sched) {
        // Guided self-scheduling: GPUs pull chunks sized max(remaining/(2*G),
        // min_chunk) from a host-side atomic queue. Early chunks are large
        // (few launches, deep in-kernel dynamic balancing), the tail is
        // bounded by one small chunk. Removes the tail imbalance of the
        // static split (the Q(27) run lost ~18% wall clock to it) and adapts
        // to heterogeneous GPUs.
        long long span = range_end - range_start;
        long long min_chunk;
        const char *cenv = getenv("NQ_CHUNK");
        if (cenv) {
            min_chunk = atoll(cenv);
        } else {
            min_chunk = 262144;
        }
        long long max_take = span / (2 * gpu_num) + 1;
        if (max_take < min_chunk) max_take = min_chunk;

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
            unsigned long long *cuda_counter;
            unsigned long long *cuda_out;
            CU_SAFE_CALL(cudaMalloc(&cuda_tot, sizeof(int) * max_take * 3));
            CU_SAFE_CALL(cudaMalloc(&cuda_partial_sum, sizeof(long long) * max_take));
            CU_SAFE_CALL(cudaMalloc(&cuda_counter, sizeof(unsigned long long)));
            CU_SAFE_CALL(cudaMalloc(&cuda_out, sizeof(unsigned long long)));
            CU_SAFE_CALL(cudaMemset(cuda_out, 0, sizeof(unsigned long long)));

            long long done = 0;
            int nchunks = 0;

            while (true) {
                // grab a guided-size chunk [s, s+take)
                long long s = next_start.load();
                long long take = 0;
                do {
                    if (s >= range_end) break;
                    long long remaining = range_end - s;
                    take = remaining / (2 * gpu_num);
                    if (take < min_chunk) take = min_chunk;
                    if (take > remaining) take = remaining;
                    if (take > max_take) take = max_take;
                } while (!next_start.compare_exchange_weak(s, s + take));
                if (s >= range_end) break;
                long long c = take;

                CU_SAFE_CALL(cudaMemset(cuda_counter, 0, sizeof(unsigned long long)));
                CU_SAFE_CALL(cudaMemcpy(cuda_tot, tot.data() + s * 3, sizeof(int) * c * 3, cudaMemcpyHostToDevice));
                CU_SAFE_CALL(cudaMemset(cuda_partial_sum, 0, sizeof(long long) * c));

                dim3 dimBlock(CU1DBLOCK);
                dim3 dimGrid(grid_size);
                n_queens<<<dimGrid, dimBlock>>>(N, cuda_tot, cuda_partial_sum, c, cuda_counter);
                cudaError_t err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    printf("kernel error: %s\n", cudaGetErrorString(err));
                    exit(-1);
                }

                reduce_partial<<<256, 256>>>(cuda_partial_sum, c, cuda_out);
                done += c;
                nchunks++;
            }

            unsigned long long dev_sum = 0;
            CU_SAFE_CALL(cudaMemcpy(&dev_sum, cuda_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            gpu_sum[idx] = (long long)dev_sum;

            CU_SAFE_CALL(cudaFree(cuda_tot));
            CU_SAFE_CALL(cudaFree(cuda_partial_sum));
            CU_SAFE_CALL(cudaFree(cuda_counter));
            CU_SAFE_CALL(cudaFree(cuda_out));
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

        unsigned long long *cuda_counter;
        CU_SAFE_CALL(cudaMalloc(&cuda_counter, sizeof(unsigned long long)));
        CU_SAFE_CALL(cudaMemset(cuda_counter, 0, sizeof(unsigned long long)));

        dim3 dimBlock(CU1DBLOCK);
        dim3 dimGrid(grid_size);

        n_queens<<<dimGrid, dimBlock>>>(N, cuda_tot, cuda_partial_sum, cnt, cuda_counter);

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
