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

## Algorithmic advances round (user directive: aim for 10×)

**State deduplication / transposition folding — measured, dead.** The DFS
subtree depends only on `(cur, left & board, right)`, so identical or
mirror-image states could be folded with multiplicities (exact). Probe
(`src/dedup_probe.cpp`) on the production subproblem lists:

| N | rows | P (paths) | S (unique canonical) | collapse |
|---|---|---:|---:|---|
| 14 | 6 | 120,742 | 119,688 | 1.009 |
| 18 | 6 | 1,199,146 | 1,195,146 | 1.003 |
| 18 | 7 | 7,409,150 | 7,354,439 | 1.007 |
| 14 | 9 | 2,195,994 | 2,048,303 | 1.072 |
| 16 | 10 | 54,239,483 | 50,737,675 | 1.069 |

Collapse ≤ 1.07 even at 62% of board depth and shrinks with N — for N=28 at
reachable depths it is ~1.00. Both one-shot dedup and layered BFS-with-merging
are dead on this problem.

**Closed-form two-row leaf count — the win of this round.** For a live node
with `popc(cur) == N-2`, choice mask `v`, and `M = last & ~cur & ~(left<<1) &
~(right>>1)`:

```
completions = popc(v)·popc(M) − popc(M&v) − popc(M&(v<<1)) − popc(M&(v>>1))
```

Each placement `p` (single bit) removes exactly the cells `p, p<<1, p>>1`
from `M`, and the cross-terms decompose bit-by-bit. Verified exhaustively vs
plain DFS for N=5…15 (`src/closed2_check.cpp`). Implemented as kernel v6
(fully predicated) and v7 (short fixed-length branch).

**Measured (5090, N=21 rows=6): v1 18.0 s / 22.3 s (c3/c7); v6 25.2 / 30.7;
v7 32.9 / 41.8. Correct everywhere, 40–90% slower — rejected.** The failure
exposes the decisive fact about this search tree: it is **middle-heavy**.
Eliminating the deepest walked level removed only ~13% of iterations (implied
by the timings), not the ~60% a "leaves dominate" intuition predicts — most
branches die in the middle rows, so the deepest levels are nearly empty while
the +17 predicated ops tax every fat middle-level node. v7 also shows that
even a rarely-taken branch loses: one diverging lane drags the whole warp.
A three-row closed form was analyzed and rejected on top of this: cross-terms
become products of p-dependent popcounts (~70 ops for another ×2 shrink of an
already thin level).

**Final conclusion of the algorithmic round:** seven kernel/algorithm
restructurings were implemented or probed and all measured negative (v2, v4,
v5, v6, v7, two dedup schemes), and the symmetry analysis shows ≤1.3× net on
GPU. The upstream kernel is at a genuine local optimum for this formulation;
the realized, verified wins are configuration occupancy (CONFIG3/CONFIG7),
guided-chunk scheduling (10.3% measured at N=23 on 3 GPUs; ~12% on the Q(27)
log), and multi-node sharding with checkpointing (linear scale-out).

**Symmetry beyond mirror — analyzed, not implemented (net ~1.1–1.3×).** The
Klein group {id, column-mirror, row-flip, 180°} acts on solutions with
orbit-size 4 for generic solutions (T = 4·A + 2·D with a clean last-row-mask
characterization). But in a row-sequential DFS the row-flip/180° canonical
constraints only bind at the *last* row — they reweight leaves without pruning
the walked tree. Orderings that make them prefix-compatible (middle-out or
top-bottom-alternating rows) require maintaining twice the diagonal masks:
stack entries grow 16→24-32 B, occupancy drops ~25-35%, per-node cost +20-30%,
canceling the ×2 tree reduction (net ≈1.1–1.3×) — while adding substantial
correctness surface (stabilizer classes). The FPGA record computations got
symmetry nearly free because custom datapaths pay no occupancy cost; this GPU
kernel does. Full 8-fold (diagonal reflections/90° rotations) additionally
breaks the row-DFS entirely. Documented as not worth it here.

## Q(28) projection method

Measure final build vs baseline on N=21/22 (single 5090) and on a multi-GPU node run;
projected Q(28) = 340 d ÷ (kernel speedup) ÷ (imbalance recovery) for 8×5090, plus a
cluster-wide estimate summing per-GPU-type throughputs with the shard driver.
