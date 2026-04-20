#!/usr/bin/env bash
set -euo pipefail

# Machine: aiapps-06052021  (2x RTX 6000 Ada, 48 GB each)
# Conda:   spkid4asr

cd "$(dirname "$0")/.."
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate spkid4asr

# Yodas lang IDs:
# bg cs da de el en es et fi fr hr hu it lt nl pl pt ro ru sk sv uk
# YTC lang IDs:
# bg cs da de el en es et fi fr hr hu it lt lv nl pl pt ro ru sk sl sv uk

#################### Speaker-ID Pipeline Config ####################
# YODAS Languages: "bg cs da de el en es et fi fr hr hu it lt nl pl pt ro ru sk sv uk"
# YTC Languages: "bg cs da de el en es et fi fr hr hu it lt lv nl pl pt ro ru sk sl sv uk"

# Done languages: bg cs da el et fi hr hu lt ro sk sv
LANGUAGES="de en es fr it nl pl pt ru uk"
DATASET="yodas"

# Done languages: bg cs da el 
LANGUAGES="de en es et fi fr hr hu it lt lv nl pl pt ro ru sk sl sv uk"
DATASET="ytc"


python run_pipeline.py \
    --dataset ytc \
    --languages cs da de el \
    --base-dir /disk_f_nvd/datasets/Granary_spkIDs/${DATASET}/ \
    --result-dir /disk_f_nvd/datasets/Granary_spkIDs/${DATASET}/ \
    --model-checkpoint /disk_f_nvd/datasets/Yodas/wespeaker/models/voxceleb_resnet293_LM \
    --corpusview-path /home/taejinp/projects/corpus_view/corpusview \
    --num-gpus 2 \
    --batch-dur 600 \
    --workers 16 \
    --s3cfg ~/.s3cfg[default]
