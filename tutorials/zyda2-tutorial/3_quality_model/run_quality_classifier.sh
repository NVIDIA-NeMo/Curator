#!/bin/bash

# dolma
echo "Processing dolma"
python run_quality_classifier.py --input $DATA_BASE/deduped/dolma-v1_7-cc-parquet --output $DATA_BASE/deduped-with-quality/dolma-v1_7-cc-parquet --batch-size 64

# zyda
NAME=zyda_refinedweb
echo "Processing $NAME"
python run_quality_classifier.py --input $DATA_BASE/deduped/zyda-parquet/$NAME --output $DATA_BASE/deduped-with-quality/zyda-parquet/$NAME --batch-size 64

NAME=zyda_slimpajama
echo "Processing $NAME"
python run_quality_classifier.py --input $DATA_BASE/deduped/zyda-parquet/$NAME --output $DATA_BASE/deduped-with-quality/zyda-parquet/$NAME --batch-size 64

NAME=zyda_c4-en
echo "Processing $NAME"
python run_quality_classifier.py --input $DATA_BASE/deduped/zyda-parquet/$NAME --output $DATA_BASE/deduped-with-quality/zyda-parquet/$NAME --batch-size 64

NAME=zyda_pile-uncopyrighted
echo "Processing $NAME"
python run_quality_classifier.py --input $DATA_BASE/deduped/zyda-parquet/$NAME --output $DATA_BASE/deduped-with-quality/zyda-parquet/$NAME --batch-size 64

NAME=zyda_peS2o
echo "Processing $NAME"
python run_quality_classifier.py --input $DATA_BASE/deduped/zyda-parquet/$NAME --output $DATA_BASE/deduped-with-quality/zyda-parquet/$NAME --batch-size 64

NAME=zyda_arxiv
echo "Processing $NAME"
python run_quality_classifier.py --input $DATA_BASE/deduped/zyda-parquet/$NAME --output $DATA_BASE/deduped-with-quality/zyda-parquet/$NAME --batch-size 64
