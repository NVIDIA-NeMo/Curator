#!/bin/bash

IN_BASE=$DATA_BASE/processed/dolma-v1_7-cc-parquet
OUT_BASE=$DATA_BASE/deduped/dolma-v1_7-cc-parquet
DUPES_BASE=$DATA_BASE/fuzzy/cc/dupes/dupes_dolma_to_remove.jsonl
N_WORKERS=$CPU_WORKERS

if test -d $IN_BASE; then
    echo "Processing dolma"
    python remove_dupes.py \
        --dupes-path $DUPES_BASE/$NAME \
        --input $IN_BASE \
        --output $OUT_BASE \
        --n-workers $N_WORKERS
fi
