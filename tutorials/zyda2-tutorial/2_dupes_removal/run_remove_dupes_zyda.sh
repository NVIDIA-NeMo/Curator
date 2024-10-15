#!/bin/bash

IN_BASE=$DATA_BASE/processed/zyda-parquet
OUT_BASE=$DATA_BASE/deduped/zyda-parquet
DUPES_BASE=$DATA_BASE/fuzzy/cc/dupes/dupes_zyda_to_remove.jsonl
N_WORKERS=$CPU_WORKERS

NAME=zyda_arxiv
INPUT=$IN_BASE/$NAME
OUTPUT=$OUT_BASE/$NAME
if test -d $INPUT; then
    echo "Processing $NAME"
    python remove_dupes.py \
        --dupes-path $DUPES_BASE/$NAME \
        --input $INPUT \
        --output $OUTPUT \
        --n-workers $N_WORKERS
fi

NAME=zyda_c4-en
INPUT=$IN_BASE/$NAME
OUTPUT=$OUT_BASE/$NAME
if test -d $INPUT; then
    echo "Processing $NAME"
    python remove_dupes.py \
        --dupes-path $DUPES_BASE/$NAME \
        --input $INPUT \
        --output $OUTPUT \
        --n-workers $N_WORKERS
fi

NAME=zyda_peS2o
INPUT=$IN_BASE/$NAME
OUTPUT=$OUT_BASE/$NAME
if test -d $INPUT; then
    echo "Processing $NAME"
    python remove_dupes.py \
        --dupes-path $DUPES_BASE/$NAME \
        --input $INPUT \
        --output $OUTPUT \
        --n-workers $N_WORKERS
fi

NAME=zyda_pile-uncopyrighted
INPUT=$IN_BASE/$NAME
OUTPUT=$OUT_BASE/$NAME
if test -d $INPUT; then
    echo "Processing $NAME"
    python remove_dupes.py \
        --dupes-path $DUPES_BASE/$NAME \
        --input $INPUT \
        --output $OUTPUT \
        --n-workers $N_WORKERS
fi

NAME=zyda_refinedweb
INPUT=$IN_BASE/$NAME
OUTPUT=$OUT_BASE/$NAME
if test -d $INPUT; then
    echo "Processing $NAME"
    python remove_dupes.py \
        --dupes-path $DUPES_BASE/$NAME \
        --input $INPUT \
        --output $OUTPUT \
        --n-workers $N_WORKERS
fi

NAME=zyda_slimpajama
INPUT=$IN_BASE/$NAME
OUTPUT=$OUT_BASE/$NAME
if test -d $INPUT; then
    echo "Processing $NAME"
    python remove_dupes.py \
        --dupes-path $DUPES_BASE/$NAME \
        --input $INPUT \
        --output $OUTPUT \
        --n-workers $N_WORKERS
fi
