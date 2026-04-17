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

set -xeuo pipefail

if [ $# -lt 4 ]; then
  echo "Usage: $0 <megatron_path> <base_ckpt_dir> <results_dir> <tokenizer_model>"
  exit 1
fi

# Hack: auto-detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

export MEGATRON_PATH=$1

BASE_CKPT_DIR=$2
RESULTS_DIR=$3
TOKENIZER_MODEL=$4
mkdir -p "$RESULTS_DIR"

TASKS="arc_easy,hellaswag,piqa"

# Auto-discover all n{i} model dirs that have a checkpoint subdir
mapfile -t MODEL_DIRS < <(find "$BASE_CKPT_DIR" -maxdepth 1 -type d -name 'n[0-9]*' | sort -V)

if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
    echo "No model directories matching n{i} found under $BASE_CKPT_DIR"
    exit 1
fi

echo "Found ${#MODEL_DIRS[@]} models: $(basename -a "${MODEL_DIRS[@]}" | tr '\n' ' ')"

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_DIR")
    CKPT_PATH="$MODEL_DIR/checkpoint"
    OUT_DIR="$RESULTS_DIR/$MODEL_NAME"

    if [[ ! -d "$CKPT_PATH" ]]; then
        echo "[SKIP] $MODEL_NAME: no checkpoint dir found at $CKPT_PATH"
        continue
    fi

    mkdir -p "$OUT_DIR"

    torchrun --nproc_per_node="$NUM_GPUS" -m lm_eval \
        --model megatron_lm \
        --model_args "load=${CKPT_PATH},tokenizer_type=Llama2Tokenizer,tokenizer_model=${TOKENIZER_MODEL},seq_length=1024,devices=${NUM_GPUS}" \
        --tasks "$TASKS" \
        --batch_size auto \
        --output_path "$OUT_DIR"
done
