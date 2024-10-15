#!/bin/bash

IN_BASE=$DATA_BASE/deduped/dolma-v1_7-cc-parquet
OUT_BASE=$DATA_BASE/zyda2/dolma-v1_7-cc-crossdeduped-filtered
N_WORKERS=$CPU_WORKERS

if test -d $IN_BASE; then
    echo "Processing dolma"
    python filter_quality.py \
        --input $IN_BASE \
        --output $OUT_BASE \
        --quality_pred High \
        --n-workers $N_WORKERS
fi
