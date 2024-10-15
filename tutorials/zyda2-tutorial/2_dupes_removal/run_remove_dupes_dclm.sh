#!/bin/bash

IN_BASE=$DATA_BASE/processed/dclm-baseline-1.0-parquet
OUT_BASE=$DATA_BASE/zyda2/dclm-crossdeduped
DUPES_BASE=$DATA_BASE/fuzzy/cc/dupes/dupes_dclm_to_remove.jsonl
N_WORKERS=$CPU_WORKERS


NAME=global-shard_01_of_10
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

NAME=global-shard_02_of_10
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

NAME=global-shard_03_of_10
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

NAME=global-shard_04_of_10
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

NAME=global-shard_05_of_10
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

NAME=global-shard_06_of_10
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

NAME=global-shard_07_of_10
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

NAME=global-shard_08_of_10
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

NAME=global-shard_09_of_10
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

NAME=global-shard_10_of_10
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
