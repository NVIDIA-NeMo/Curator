#!/bin/bash

IN_BASE=$DATA_BASE/deduped/zyda-parquet
OUT_BASE=$DATA_BASE/zyda2/zyda-crossdeduped-filtered
N_WORKERS=$CPU_WORKERS

NAME=zyda_arxiv
INPUT=$IN_BASE/$NAME
OUTPUT=$OUT_BASE/$NAME
if test -d $INPUT; then
    echo "Processing $NAME"
    python filter_quality.py \
        --input $INPUT \
        --output $OUTPUT \
        --quality_pred High \
        --n-workers $N_WORKERS
fi

NAME=zyda_c4-en
INPUT=$IN_BASE/$NAME
OUTPUT=$OUT_BASE/$NAME
if test -d $INPUT; then
    echo "Processing $NAME"
    python filter_quality.py \
        --input $INPUT \
        --output $OUTPUT \
        --quality_pred High \
        --n-workers $N_WORKERS
fi

NAME=zyda_peS2o
INPUT=$IN_BASE/$NAME
OUTPUT=$OUT_BASE/$NAME
if test -d $INPUT; then
    echo "Processing $NAME"
    python filter_quality.py \
        --input $INPUT \
        --output $OUTPUT \
        --quality_pred High \
        --n-workers $N_WORKERS
fi

NAME=zyda_pile-uncopyrighted
INPUT=$IN_BASE/$NAME
OUTPUT=$OUT_BASE/$NAME
if test -d $INPUT; then
    echo "Processing $NAME"
    python filter_quality.py \
        --input $INPUT \
        --output $OUTPUT \
        --quality_pred High \
        --n-workers $N_WORKERS
fi

NAME=zyda_refinedweb
INPUT=$IN_BASE/$NAME
OUTPUT=$OUT_BASE/$NAME
if test -d $INPUT; then
    echo "Processing $NAME"
    python filter_quality.py \
        --input $INPUT \
        --output $OUTPUT \
        --quality_pred High \
        --n-workers $N_WORKERS
fi

NAME=zyda_slimpajama
INPUT=$IN_BASE/$NAME
OUTPUT=$OUT_BASE/$NAME
if test -d $INPUT; then
    echo "Processing $NAME"
    python filter_quality.py \
        --input $INPUT \
        --output $OUTPUT \
        --quality_pred High \
        --n-workers $N_WORKERS
fi
