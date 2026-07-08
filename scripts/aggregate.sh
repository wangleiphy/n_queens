#!/bin/bash
# Sums shard results produced by shard.sh once all K shards are present.
# Usage: aggregate.sh OUTDIR NSHARDS
set -e
OUT=$1; K=$2
MISSING=0
TOTAL=0
for ((i = 0; i < K; i++)); do
    if [ -s "$OUT/shard_$i.result" ]; then
        TOTAL=$((TOTAL + $(cat "$OUT/shard_$i.result")))
    else
        echo "missing shard $i"
        MISSING=$((MISSING + 1))
    fi
done
if [ $MISSING -eq 0 ]; then
    echo "COMPLETE: total = $TOTAL"
else
    echo "INCOMPLETE ($MISSING of $K shards missing); partial sum = $TOTAL"
    exit 1
fi
