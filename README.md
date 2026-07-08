# n_queens

N-Queens solution counter using OpenMP and CUDA (iterative DFS with zero
shared-memory bank conflicts). This is a fork of
[ygch/n_queens](https://github.com/ygch/n_queens) focused on making Q(28)
practical: occupancy-tuned configurations, better multi-GPU scheduling, and
multi-node sharding with checkpointing. The CUDA kernel itself is unchanged
from upstream v3.0 — see [Research notes](#research-notes) for why.

# Usage
* `cd src && sh compile.sh`
* `./n_queens <N> <pre_placed_rows> [range_start range_end]`

Example: `./n_queens 19 6` counts all solutions of the 19-queens problem,
pre-placing the first 6 rows on the CPU.

## Runtime options (this fork)

| Option | Meaning |
|---|---|
| `NQ_SCHED=static` (default) | one kernel launch per GPU over a fixed split (upstream behavior) |
| `NQ_SCHED=chunk` | guided self-scheduling: GPUs pull geometrically shrinking chunks from a host-side atomic queue; use for runs of hours/days or heterogeneous GPUs |
| `NQ_CHUNK=<n>` | minimum chunk size for `chunk` (default 262144) |
| `NQ_GRID=<n>` | grid size (default 1024) |
| `NQ_COUNT_ONLY=1` | print `TOTAL_SUBPROBLEMS <cnt>` and exit (used by the shard driver) |
| `range_start range_end` | solve only subproblems `[start, end)`; outputs of disjoint ranges covering `[0, cnt)` sum to the full count. Implies `chunk`. |

## Choosing a configuration

Compile-time configs set threads/block (`CU1DBLOCK`) and per-thread DFS stack
depth (`STACKSIZE`); the per-block shared-memory footprint determines resident
blocks/SM. The kernel is occupancy-bound, so more resident threads/SM wins.
Constraint: `STACKSIZE + 1 + pre_placed_rows >= N`.

| Config | Threads × depth | thr/SM (5090) | Max N (rows=6) | Use for |
|---|---|---|---|---|
| CONFIG1 | 128 × 24 | 256 | 28 (rows≥3) | superseded by CONFIG7 |
| CONFIG2 | 160 × 19 | 320 | 26 | default (upstream compatible) |
| CONFIG3 | 192 × 16 | 384 | 23 | **fastest for N ≤ 23** |
| CONFIG7 | 96 × 21 | 288 | 28 | **N = 24…28** |

(Configs 4–6 were benchmark probes; see `docs/superpowers/specs/`.)

## Measured runtimes (single RTX 5090, rows=6, this fork's cluster)

All runs verified against exact counts (OEIS A000170).

| Config | N=20 | N=21 | N=22 |
|---|---:|---:|---:|
| CONFIG1 | 3.14 s | 24.0 s | 205 s |
| CONFIG2 | 2.69 s | 20.8 s | (pending) |
| CONFIG3 | 2.39 s | 18.1 s | (pending) |
| CONFIG7 | (pending) | 21.9 s | 187 s |

CONFIG3 is ~13% faster than CONFIG2 where its stack depth suffices (N ≤ 23 at
rows=6). For N up to 28, CONFIG7 replaces CONFIG1 and is ~9% faster than it.
`rows=6` beat both `rows=5` (+2% at N=21) and `rows=7` (+9% at N=21) in
direct tests.

## Multi-node sharding for very large N

The subproblem list is deterministic, so disjoint index ranges can be solved
by independent jobs (different nodes, different GPU types, different days)
and summed:

```bash
# one shard per job; shard.sh skips shards that already have results (resumable)
scripts/shard.sh ./n_queens 28 6 512 $SHARD_ID results_q28
# when all shards are done:
scripts/aggregate.sh results_q28 512
```

Each shard writes its partial count to a file; a killed campaign resumes by
resubmitting all shards, losing at most one shard's progress per node.

# Differences between versions
* v1.0: Basic implementation, plus int4 optimization;
* v2.0: Inline PTX optimization;
* v2.1: Dynamic task fetching optimization;
* v3.0: Further inline PTX optimization;
* fork: per-config PTX stack sizing + CONFIG7, guided-chunk multi-GPU
  scheduler, subproblem-range/sharding mode, occupancy reporting. Kernel
  unchanged from v3.0.

# Run time(ms) — upstream reference hardware
| N  |  CPU |Openmp| 4090 | A100 | 5090 | H800 |   Count     |
|:--:|-----:|-----:|-----:|-----:|-----:|-----:|------------:|
| 13 |    12|     2|   107|   123|   185|   558|        73712|
| 14 |    66|     4|   108|   166|   193|   614|       365596|
| 15 |   410|    20|   110|   174|   197|   627|      2279184|
| 16 |  2285|   110|   116|   209|   202|   645|     14772512|
| 17 | 15188|   756|   126|   247|   209|   670|     95815104|
| 18 |114134|  5834|   183|   267|   261|   791|    666090624|
| 19 |859313| 45275|   609|  1074|   557|  1138|   4968057848|
| 20 |  x   |368940|  3636|  6415|  2778|  4140|  39029188884|
| 21 |  x   |  x   | 28551| 51741| 21242| 29248| 314666222712|
| 22 |  x   |  x   |245090|444272|180364|249886|2691008701644|
1. CPU is AMD 9950x3D;
2. Openmp uses 32 threads on AMD 9950x3D;
3. single 4090/A100/5090/H800 with pre-placing first 6 rows under configuration2.

# Cuda runtimes(s) for larger N — upstream reference hardware
|  N   | 20 | 21  | 22  | 23  | 24 | 25  |
|:----:|---:|----:|----:|----:|---:|----:|
|8 5090|2.15|4.91 |29.10|232.7|2205|21880|
|8 4090|2.58|6.16 |37.46|308.5|2840|28332|
|8 A100|2.89|9.24 |67.58|562.0|5307|53104|

* Pre-placing first 6 rows under configuration2.

# Cuda runtime for 26-queens
2.5 days
* 8 RTX 5090 with pre-placing first 6 rows under configuration2.

# Cuda runtime for 27-queens
28.4 days
* 8 RTX 5090 with pre-placing first 7 rows under configuration2.

# Projected cuda runtime for 28-queens (this fork)

~270 days on a single 8×5090 node, down from the upstream projection of
340 days (= 28.4 d × 10.52 work growth × 1.14 CONFIG1 penalty):

* CONFIG7 instead of CONFIG1 for the depth-21 stack that rows=6 requires:
  measured 187 s vs 205 s at N=22 on the same GPU, a 0.91× factor.
* Guided-chunk scheduling instead of the static split: in the upstream Q(27)
  log the 8 GPUs finished between day 23.2 and day 28.4 (mean busy time
  25.0 days), so ~12% of wall clock was idle tail. Chunking bounds the tail
  by a single minimum-size chunk (minutes), a ~0.88× factor.
* 340 d × 0.91 × 0.88 ≈ **272 days**.

With range sharding the wall clock further divides across nodes; shards are
checkpointed, so a Q(28) campaign can run opportunistically on whatever GPUs
are free and survive interruptions.

# Research notes

Optimization attempts on the kernel itself, all rejected by measurement
(details in `docs/superpowers/specs/2026-07-08-q28-optimization-design.md`):

* Register-cached stack top with branchy descend/backtrack: 2.6× slower —
  warp divergence dwarfs the saved shared-memory traffic.
* Two-row leaf unroll via a branch into an inner loop: 1.7× slower — same
  reason. The kernel's speed comes from its fully predicated, uniform loop.
* Predication-only micro-opts (skip dead write-back, predicated 64-bit add):
  ±1%, not worth the code.
* One host-pinned task counter shared across GPUs via `atomicAdd_system`:
  **silently wrong** on PCIe nodes (duplicated task ids ⇒ inflated counts).
  The guided-chunk queue replaced it.
* Tensor-network and permanent-based counting: see `research/` — infeasible
  for N=28.

Every performance claim above was gated on exact-count verification for
N ∈ {13…23} on RTX 5090 (sm_120), A800/A100 (sm_80), and V100 (sm_70).

# Citation
If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝 :)

```BibTeX
@article{GPU-N-queens,
  title={High-Performance N-Queens Solver on GPU: Iterative DFS with Zero Bank Conflicts},
  author={Guangchao Yao, Yali Li},
  journal={arXiv preprint arXiv:2511.12009},
  year={2025}
}
```
Link: https://arxiv.org/pdf/2511.12009
