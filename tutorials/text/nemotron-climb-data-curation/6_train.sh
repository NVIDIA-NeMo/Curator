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
  echo "Usage: $0 <pretrain_gpt_path> <mixture_script> <work_path> <tokenizer_model>"
  exit 1
fi

# Hack: auto-detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

PRETRAIN_GPT_PATH=$1

MIXTURE_SCRIPT=$2
readarray -t MIXTURE_LINES < <(bash "$MIXTURE_SCRIPT")
DATA_ARGS=()
for line in "${MIXTURE_LINES[@]}"; do
    DATA_ARGS+=($line)
done

WORK_PATH=$3
CHECKPOINT_PATH="${WORK_PATH}/checkpoint"
TENSORBOARD_LOGS_PATH="${WORK_PATH}/tensorboard"
DATA_CACHE_PATH="${WORK_PATH}/data_cache"
mkdir -p "$CHECKPOINT_PATH" "$TENSORBOARD_LOGS_PATH" "$DATA_CACHE_PATH"

TOKENIZER_MODEL=$4

export NCCL_DEBUG=INFO

# TIPS AND CONSIDERATIONS:
#
# Note small --train-iters 10000 for proxy model training
# Consider increasing --train-iters depending on your GPU setup and throughput, but
# don't forget that this will impact the learning rate schedule
#
# Evaluation is disabled by default, but can be enabled by setting --eval-iters > 0 and a reasonably small --eval-interval
# Note: --eval-interval 50000000 is used below to avoid a zero division error, it's set to a large value so eval never triggers
#
# Consider lowering --lr for less noise, e.g., --lr 3e-4
#
# Make sure NUM_GPUS, --global-batch-size, and --micro-batch-size align with each other
# That is, global-batch-size = micro-batch-size x data-parallel-size x gradient-accumulation-steps
# Example: if NUM_GPUS = 8
#          and if tensor-model-parallel-size = 1, pipeline-model-parallel-size = 1, micro-batch-size = 64, and global-batch-size = 4096
#          then data-parallel-size = 8
#          and 4096 / (64 x 8) = 8
#          and gradient-accumulation-steps = 8
#
# Decreasing the micro-batch-size can help reduce memory pressure, e.g., --micro-batch-size 32
#
# Use --no-mmap-bin-files if dataset is on slow/network storage
# Otherwise, mmap is usually faster and more memory-efficient
#
# For the best performing mixtures, consider re-running with a new seed to see if the results are reproducible

torchrun --nproc_per_node $NUM_GPUS "$PRETRAIN_GPT_PATH" \
    --num-layers 12 \
    --hidden-size 1344 \
    --num-attention-heads 12 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 64 \
    --global-batch-size 4096 \
    --train-iters 10000 \
    --optimizer adam \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --clip-grad 1.0 \
    --bf16 \
    --distributed-timeout-minutes 120 \
    --auto-detect-ckpt-format \
    --exit-duration-in-mins 110 \
    --override-opt_param-scheduler \
    --lr-decay-style cosine \
    --lr 5e-4 \
    --min-lr 1.0e-5 \
    --lr-warmup-fraction .01 \
    --lr-decay-iters 10000 \
    --seed 1234 \
    --train-data-path "${DATA_ARGS[@]}" \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --data-cache-path "$DATA_CACHE_PATH" \
    --dataloader-type cyclic \
    --log-interval 100 \
    --save-interval 1000 \
    --eval-interval 50000000 \
    --save "$CHECKPOINT_PATH" \
    --load "$CHECKPOINT_PATH" \
    --eval-iters 0 \
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
