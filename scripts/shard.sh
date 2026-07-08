#!/bin/bash
# Multi-node sharding driver: computes one shard of an N-queens count.
#
# Usage: shard.sh BIN N ROWS NSHARDS SHARD_ID [OUTDIR]
#
# The subproblem list is deterministic (pure integer DFS), so every job sees
# the same ordering; shard i solves subproblems [i*cnt/K, (i+1)*cnt/K).
# Each shard writes its (already doubled) partial count to OUTDIR/shard_<i>.result;
# the final answer is the sum of all K shard results (see aggregate.sh).
# Re-running a finished shard is a no-op, so a killed campaign can be resumed
# by resubmitting all shards.
set -e
BIN=$1; N=$2; ROWS=$3; K=$4; I=$5; OUT=${6:-results_N$2_K$4}
mkdir -p "$OUT"

if [ -s "$OUT/shard_$I.result" ]; then
    echo "shard $I already done: $(cat $OUT/shard_$I.result)"
    exit 0
fi

CNT=$(NQ_COUNT_ONLY=1 $BIN $N $ROWS | grep -oP "TOTAL_SUBPROBLEMS \K[0-9]+")
S=$((CNT * I / K)); E=$((CNT * (I + 1) / K))
echo "shard $I of $K: subproblems [$S, $E) of $CNT"

R=$($BIN $N $ROWS $S $E | tee "$OUT/shard_$I.log" | grep -oP "queens result \K[-0-9]+")
echo "$R" > "$OUT/shard_$I.result.tmp" && mv "$OUT/shard_$I.result.tmp" "$OUT/shard_$I.result"
echo "shard $I done: $R"
