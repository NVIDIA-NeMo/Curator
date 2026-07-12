#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -----------------------------------------------------------------------------
# Set environment variables
# -----------------------------------------------------------------------------

# Parameters for 1_embed.py, 2_cluster.py, 3_prune.py, 4_tokenize.py, 5_mixture.py
export INPUT_PATH=/path/to/input/data/dir
export INPUT_FILETYPE="jsonl"
export TEXT_FIELD="text"
export ID_FIELD="_curator_climb_id"
export CURATOR_PATH=/path/to/Curator
export OUTPUT_PATH=/path/to/output/data/dir

FASTTEXT_MODEL_PATHS=(
    /path/to/best_model_advertisement.bin
    /path/to/best_model_cultural_value.bin
    /path/to/best_model_educational_value.bin
    /path/to/best_model_informational_value.bin
    /path/to/best_model_quality.bin
)

# Hugging Face token for accessing the tokenizer model
export HF_TOKEN=""

# Paths for training and evaluation
export WORK_BASE_DIR=/path/to/megatron_exps
export MEGATRON_PATH=/path/to/Megatron-LM
export TOKENIZER_MODEL=/path/to/tokenizer.model
# Optional: path to a Megatron checkpoint to fine-tune from. Leave empty to train from scratch.
export PRETRAINED_MODEL_PATH=""
export LM_EVAL_PATH=/path/to/lm-evaluation-harness

# Build optional 5th positional arg for 6_train.sh; omitted entirely when PRETRAINED_MODEL_PATH is empty.
PRETRAIN_ARGS=()
[ -n "${PRETRAINED_MODEL_PATH}" ] && PRETRAIN_ARGS=("${PRETRAINED_MODEL_PATH}")

# -----------------------------------------------------------------------------
# Run Python scripts: 1_embed.py, 2_cluster.py, 3_prune.py, 4_tokenize.py, 5_mixture.py
# These scripts only need to be run once each.
# -----------------------------------------------------------------------------

if [ -e ${OUTPUT_PATH}/computed_embeddings ]; then
    echo "Skipping 1_embed.py: ${OUTPUT_PATH}/computed_embeddings already exists"
else
    python ${CURATOR_PATH}/tutorials/text/nemotron-climb-data-curation/1_embed.py \
        --input-path ${INPUT_PATH} \
        --input-filetype ${INPUT_FILETYPE} \
        --output-path ${OUTPUT_PATH}/computed_embeddings \
        --text-field ${TEXT_FIELD} \
        --id-field ${ID_FIELD} \
        --use-sentence-transformer
fi

if [ -e ${OUTPUT_PATH}/clusters ]; then
    echo "Skipping 2_cluster.py: ${OUTPUT_PATH}/clusters already exists"
else
    python ${CURATOR_PATH}/tutorials/text/nemotron-climb-data-curation/2_cluster.py \
        --input-path ${OUTPUT_PATH}/computed_embeddings \
        --output-path ${OUTPUT_PATH}/clusters \
        --text-field ${TEXT_FIELD} \
        --id-field ${ID_FIELD} \
        --embedding-dim 3072 \
        --centroids-path ${OUTPUT_PATH}/centroids
fi

FASTTEXT_SCORE_FIELDS=(
    advertisement_score
    cultural_value_score
    educational_value_score
    informational_value_score
    quality_score
)
FASTTEXT_PRUNING_THRESHOLDS=(2.0 1.0 1.0 1.0 1.0)
if [ -e ${OUTPUT_PATH}/pruned_clusters ]; then
    echo "Skipping 3_prune.py: ${OUTPUT_PATH}/pruned_clusters already exists"
else
    python ${CURATOR_PATH}/tutorials/text/nemotron-climb-data-curation/3_prune.py \
        --input-path ${OUTPUT_PATH}/clusters \
        --output-path ${OUTPUT_PATH}/pruned_clusters \
        --fasttext-model-paths ${FASTTEXT_MODEL_PATHS[@]} \
        --score-fields ${FASTTEXT_SCORE_FIELDS[@]} \
        --text-field ${TEXT_FIELD} \
        --pruning-thresholds ${FASTTEXT_PRUNING_THRESHOLDS[@]} \
        --centroids-path ${OUTPUT_PATH}/centroids \
        --merge-threshold 1.5
fi

if [ -e ${OUTPUT_PATH}/domains ]; then
    echo "Skipping 4_tokenize.py: ${OUTPUT_PATH}/domains already exists"
else
    python ${CURATOR_PATH}/tutorials/text/nemotron-climb-data-curation/4_tokenize.py \
        --input-path ${OUTPUT_PATH}/pruned_clusters \
        --output-path ${OUTPUT_PATH}/domains \
        --hf-token ${HF_TOKEN} \
        --text-field ${TEXT_FIELD} \
        --append-eod
fi

# Generate 64 mixtures to start the curation loop
if [ -e ${OUTPUT_PATH}/mixtures_1 ]; then
    echo "Skipping 5_mixture.py: ${OUTPUT_PATH}/mixtures_1 already exists"
else
    python ${CURATOR_PATH}/tutorials/text/nemotron-climb-data-curation/5_mixture.py \
        --input-path ${OUTPUT_PATH}/domains \
        --output-path ${OUTPUT_PATH}/mixtures_1 \
        --num-mixtures 64
fi

# -----------------------------------------------------------------------------
# Run train, eval, and predict scripts: 6_train.sh, 7_evaluate.sh, 8_predict.py
# These scripts can be run in a loop until an optimal mixture is found.
# -----------------------------------------------------------------------------

# Train 64 proxy models
mapfile -t MIXTURE_SCRIPTS < <(find "${OUTPUT_PATH}/mixtures_1" -maxdepth 1 -name 'n[0-9]*.sh' | sort -V)
for MIXTURE_SCRIPT in "${MIXTURE_SCRIPTS[@]}"; do
    MODEL_NAME=$(basename "${MIXTURE_SCRIPT}" .sh)   # e.g. n1.sh -> n1
    WORK_PATH="${WORK_BASE_DIR}/megatron_exp_1/${MODEL_NAME}"

    if [ -e ${WORK_PATH} ]; then
        echo "Skipping 6_train.sh: ${WORK_PATH} already exists"
    else
        bash ${CURATOR_PATH}/tutorials/text/nemotron-climb-data-curation/6_train.sh \
            "${MEGATRON_PATH}/pretrain_gpt.py" \
            "${MIXTURE_SCRIPT}" \
            "${WORK_PATH}" \
            "${TOKENIZER_MODEL}" \
            "${PRETRAIN_ARGS[@]}"
    fi
done

# Evaluate the 64 proxy models
if [ -e ${OUTPUT_PATH}/lm_eval_results_1 ]; then
    echo "Skipping 7_evaluate.sh: ${OUTPUT_PATH}/lm_eval_results_1 already exists"
else
    bash ${CURATOR_PATH}/tutorials/text/nemotron-climb-data-curation/7_evaluate.sh \
        "${LM_EVAL_PATH}" \
        "${MEGATRON_PATH}" \
        "${WORK_BASE_DIR}/megatron_exp_1" \
        "${OUTPUT_PATH}/lm_eval_results_1" \
        "${TOKENIZER_MODEL}"
fi

# Generate 32 mixtures for the next iteration of the curation loop
if [ -e ${OUTPUT_PATH}/mixtures_2 ]; then
    echo "Skipping 8_predict.py: ${OUTPUT_PATH}/mixtures_2 already exists"
else
    python ${CURATOR_PATH}/tutorials/text/nemotron-climb-data-curation/8_predict.py \
        --input-paths ${OUTPUT_PATH}/lm_eval_results_1 \
        --domains-path ${OUTPUT_PATH}/domains \
        --mixtures-paths ${OUTPUT_PATH}/mixtures_1 \
        --output-path ${OUTPUT_PATH}/mixtures_2 \
        --metric "valid_avg" \
        --num-mixtures 32
fi

# Train 32 proxy models
mapfile -t MIXTURE_SCRIPTS < <(find "${OUTPUT_PATH}/mixtures_2" -maxdepth 1 -name 'n[0-9]*.sh' | sort -V)
for MIXTURE_SCRIPT in "${MIXTURE_SCRIPTS[@]}"; do
    MODEL_NAME=$(basename "${MIXTURE_SCRIPT}" .sh)   # e.g. n1.sh -> n1
    WORK_PATH="${WORK_BASE_DIR}/megatron_exp_2/${MODEL_NAME}"

    if [ -e ${WORK_PATH} ]; then
        echo "Skipping 6_train.sh: ${WORK_PATH} already exists"
    else
        bash ${CURATOR_PATH}/tutorials/text/nemotron-climb-data-curation/6_train.sh \
            "${MEGATRON_PATH}/pretrain_gpt.py" \
            "${MIXTURE_SCRIPT}" \
            "${WORK_PATH}" \
            "${TOKENIZER_MODEL}" \
            "${PRETRAIN_ARGS[@]}"
    fi
done

# Evaluate the 32 proxy models
if [ -e ${OUTPUT_PATH}/lm_eval_results_2 ]; then
    echo "Skipping 7_evaluate.sh: ${OUTPUT_PATH}/lm_eval_results_2 already exists"
else
    bash ${CURATOR_PATH}/tutorials/text/nemotron-climb-data-curation/7_evaluate.sh \
        "${LM_EVAL_PATH}" \
        "${MEGATRON_PATH}" \
        "${WORK_BASE_DIR}/megatron_exp_2" \
        "${OUTPUT_PATH}/lm_eval_results_2" \
        "${TOKENIZER_MODEL}"
fi

# Generate 16 mixtures for the next iteration of the curation loop
if [ -e ${OUTPUT_PATH}/mixtures_3 ]; then
    echo "Skipping 8_predict.py: ${OUTPUT_PATH}/mixtures_3 already exists"
else
    python ${CURATOR_PATH}/tutorials/text/nemotron-climb-data-curation/8_predict.py \
        --input-paths ${OUTPUT_PATH}/lm_eval_results_1 ${OUTPUT_PATH}/lm_eval_results_2 \
        --domains-path ${OUTPUT_PATH}/domains \
        --mixtures-paths ${OUTPUT_PATH}/mixtures_1 ${OUTPUT_PATH}/mixtures_2 \
        --output-path ${OUTPUT_PATH}/mixtures_3 \
        --metric "valid_avg" \
        --num-mixtures 16
fi

# Train 16 proxy models
mapfile -t MIXTURE_SCRIPTS < <(find "${OUTPUT_PATH}/mixtures_3" -maxdepth 1 -name 'n[0-9]*.sh' | sort -V)
for MIXTURE_SCRIPT in "${MIXTURE_SCRIPTS[@]}"; do
    MODEL_NAME=$(basename "${MIXTURE_SCRIPT}" .sh)   # e.g. n1.sh -> n1
    WORK_PATH="${WORK_BASE_DIR}/megatron_exp_3/${MODEL_NAME}"

    if [ -e ${WORK_PATH} ]; then
        echo "Skipping 6_train.sh: ${WORK_PATH} already exists"
    else
        bash ${CURATOR_PATH}/tutorials/text/nemotron-climb-data-curation/6_train.sh \
            "${MEGATRON_PATH}/pretrain_gpt.py" \
            "${MIXTURE_SCRIPT}" \
            "${WORK_PATH}" \
            "${TOKENIZER_MODEL}" \
            "${PRETRAIN_ARGS[@]}"
    fi
done

# Evaluate the 16 proxy models
if [ -e ${OUTPUT_PATH}/lm_eval_results_3 ]; then
    echo "Skipping 7_evaluate.sh: ${OUTPUT_PATH}/lm_eval_results_3 already exists"
else
    bash ${CURATOR_PATH}/tutorials/text/nemotron-climb-data-curation/7_evaluate.sh \
        "${LM_EVAL_PATH}" \
        "${MEGATRON_PATH}" \
        "${WORK_BASE_DIR}/megatron_exp_3" \
        "${OUTPUT_PATH}/lm_eval_results_3" \
        "${TOKENIZER_MODEL}"
fi

# Generate the optimal mixture and stop the curation loop
if [ -e ${OUTPUT_PATH}/optimal_mixture ]; then
    echo "Skipping 8_predict.py: ${OUTPUT_PATH}/optimal_mixture already exists"
else
    python ${CURATOR_PATH}/tutorials/text/nemotron-climb-data-curation/8_predict.py \
        --input-paths ${OUTPUT_PATH}/lm_eval_results_1 ${OUTPUT_PATH}/lm_eval_results_2 ${OUTPUT_PATH}/lm_eval_results_3 \
        --domains-path ${OUTPUT_PATH}/domains \
        --mixtures-paths ${OUTPUT_PATH}/mixtures_1 ${OUTPUT_PATH}/mixtures_2 ${OUTPUT_PATH}/mixtures_3 \
        --output-path ${OUTPUT_PATH}/optimal_mixture \
        --metric "valid_avg" \
        --num-mixtures 1
fi
