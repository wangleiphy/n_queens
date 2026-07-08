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

## Q(28) projection method

Measure final build vs baseline on N=21/22 (single 5090) and on a multi-GPU node run;
projected Q(28) = 340 d ÷ (kernel speedup) ÷ (imbalance recovery) for 8×5090, plus a
cluster-wide estimate summing per-GPU-type throughputs with the shard driver.
