# Q(28) Feasibility: N-Queens GPU Solver Optimization — Design

Date: 2026-07-08
Status: in progress (autonomous background session)

## Goal

Reduce the projected wall time for computing Q(28) — currently **340 days on 8×RTX 5090**
(README: `28.4d × 10.52 × 1.14`, rows=6, CONFIG1) — while guaranteeing exact correctness.
Every optimization must be valid for N up to 28, not just tuned for small N.

## Evidence from existing runs (logs/)

- Q(27), 8×5090, rows=7, CONFIG2: 28.4 days total, but GPU finish times span
  **Sep 10 → Sep 15**, i.e. ≈5.4 days (≈18% of wall clock) is pure tail imbalance
  from the static hand-tuned inter-GPU ratio split (0.20/0.15/0.12/…).
- Subproblem generation is not a bottleneck (9.3 s for 453M subproblems at N=27 rows=7).
- Kernel: 24 registers, 49152 B static shared per block, zero barriers, zero bank
  conflicts (by stack layout). One block per SM on most parts due to 48 KB shared.

## Correctness protocol (gate for every change)

1. `expected_counts.txt` holds Q(4)..Q(27) fetched from OEIS A000170 and cross-checked
   against the repo README table (N=13..22) and the archived run logs (Q25, Q27 match).
2. `bench.sh` wraps every run and compares the exact count; prints `OK`/`FAIL`.
3. A change is adopted only if **all** swept N (odd and even, exercising both
   pre-placement paths) match exactly on ≥2 GPU architectures (5090 + A800 minimum),
   with no count deviation of any kind.
4. Arithmetic is pure integer; results must be bit-exact, not approximately equal.
5. Generalizability: no assumption that breaks at N=28 (bitmask width ≤ 32 with
   `& last` masking; stack depth `N - rows - 1 ≤ STACKSIZE`; 64-bit accumulators —
   Q(28) ≈ 2.3×10^18 fits in signed 64-bit).

## Optimization candidates (priority order)

| # | Idea | Expected win | Risk |
|---|------|--------------|------|
| 1 | Host-side dynamic inter-GPU chunking (atomic chunk queue instead of static ratios) | up to ~18% wall clock on long runs; auto-handles heterogeneous GPUs | low |
| 2 | Occupancy: allow ≥2 blocks/SM by moving stack to dynamic shared memory and tuning (CU1DBLOCK × blocks/SM); 5090 has 100 KB/SM, A100 164 KB, H100/H200 228 KB | unknown until measured; kernel today runs ≤160 threads/SM on 5090 | medium |
| 3 | Register-cached top of stack: skip `ld.shared.v4` on descend (state already in registers), reload only after pop | cuts shared traffic ~40–50%/node | medium (PTX rewrite) |
| 4 | 8-byte stack entries `{p ǀ lost_bit≪31, valid_pos}` with algebraic undo of (cur,left,right) on pop | halves shared per thread → more depth or occupancy; enables rows=7 CONFIG-variants for N=28 | medium-high |
| 5 | Two-row leaf unroll (count at depth N−2 without stack traffic) | leaf levels dominate node count | medium |
| 6 | Subproblem-range CLI (`start end`) + checkpoint/resume + Slurm shard driver | makes Q(28) runnable across many nodes/GPU types and crash-tolerant; wall clock = work / Σ throughput | low |
| 7 | Grid/launch shape tuning (fixed 1024 blocks today) | minor | low |

Explicitly out of scope: 8-fold symmetry ("constellation") redesign — large correctness
risk, whole-pipeline rewrite; tensor-network / permanent approaches (already ruled out
in `research/`).

## Methodology

- Primary iteration platform: single NV5090 (fastest arch for this kernel), bench
  N=19/20/21 rows=6 CONFIG2; final confirmation on N=22 (and N=23 for the shipped build).
- Regression check on A800 (sm_80) and V100 (sm_70) for every adopted change.
- Profile with `ncu` if counters are permitted on the cluster; otherwise infer the
  limiter from controlled sweeps (block size, blocks/SM, stack layout).
- Each optimization = separate commit, A/B-testable.

## Adoption gates

Adopt a change iff:
(a) correctness gate passes (above);
(b) ≥3% speedup on N=20/21 on 5090;
(c) no >5% regression on other archs without an arch-specific dispatch.

## Experiment log (updated as measured; all runs verified against exact counts)

Platform: BCM cluster; primary NV5090 (sm_120, CUDA 12.8, driver 570.172.08).
Reference workload: N=21, rows=6, static scheduler, single GPU unless noted.

| Experiment | Result | Verdict |
|---|---|---|
| Baseline v1 CONFIG2 (160 thr, 2 blk/SM = 320 thr/SM) | 20.8 s | reference (matches README) |
| v1 CONFIG1 (128 thr, 2 blk/SM = 256 thr/SM) | 24.3 s | Q(28)-capable but slowest |
| v1 CONFIG3 (192 thr, 2 blk/SM = 384 thr/SM) | 18.1–18.4 s | **+13% over CONFIG2**; N ≤ 22 only |
| v1 CONFIG4 (224×13, 448 thr/SM), rows=7 | 18.9 s (c3\@rows7 = 19.7 s) | occupancy curve saturating (+4%) |
| v1 CONFIG5 (128×15, 3 blk = 384 thr/SM) | 18.3 s | threads/SM is what matters, not blocks |
| rows=7 vs rows=6 (c3) | 19.7 s vs 18.1 s | rows=6 preferred (~9%) |
| grid 340 vs 1024 (c3) | 18.5 vs 18.1 s | keep 1024 |
| **v2** register-cached stack top (branchy descend/backtrack) | 55.2 s (c2) | **2.6× slower — rejected.** Divergent branches serialize the warp |
| **v4** two-row leaf unroll (branch into inner leaf loop) | 35.7 s (c2), 31.0 s (c3) | **72% slower — rejected.** Same reason: v1's uniform predicated loop is the whole trick |
| ncu profiling | ERR_NVGPUCTRPERM | not permitted on cluster; inference via occupancy sweeps |
| **queue scheduler**: one host-pinned counter shared by all GPUs via `atomicAdd_system` | 2×V100: N=19 count **too high** (5.13e9 vs 4.97e9); 2×A800 same | **WRONG on PCIe nodes — removed.** System atomics are not atomic across devices here; duplicated task ids |
| chunk scheduler, fixed 1/24 chunks (1 GPU) | 25.7 s vs 20.8 s static | 24% overhead → replaced by guided sizes |
| range mode (two halves of N=18, N=20) | sums exactly to Q(N) | correct |

Key insight so far: the v1 kernel's inner loop is effectively optimal for SIMT —
its only branch is the loop backedge, everything else is predicated, so warps
never diverge. Both attempted restructurings (v2, v4) lost far more to branch
divergence than they saved in issue slots or shared-memory traffic. Remaining
wins are in occupancy configs, scheduling, and scale-out.

## Q(28) projection method

Measure final build vs baseline on N=21/22 (single 5090) and on a multi-GPU node run;
projected Q(28) = 340 d ÷ (kernel speedup) ÷ (imbalance recovery) for 8×5090, plus a
cluster-wide estimate summing per-GPU-type throughputs with the shard driver.
