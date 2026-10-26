#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

: "${MANIFEST:?Set MANIFEST to a one-session JSONL manifest}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a destination directory}"
: "${SORTFORMER_MODEL:?Set SORTFORMER_MODEL to a checkpoint path or model name}"

PYTHON="${PYTHON:-python}"
BUILD_SEGLST="${BUILD_SEGLST:-1}"

args=(
  -m
  "tutorials.audio.vad_refinement.rectify_vad_boundaries"
  --manifest "$MANIFEST"
  --output-root "$OUTPUT_ROOT"
  --sortformer-model "$SORTFORMER_MODEL"
  --sortformer-device "${SORTFORMER_DEVICE:-auto}"
  --sortformer-onset "${SORTFORMER_ONSET:-0.5}"
  --sortformer-offset "${SORTFORMER_OFFSET:-0.5}"
)

if [[ -n "${SOUND_EXPERT_MODEL:-}" ]]; then
  args+=(
    --sound-expert-model "$SOUND_EXPERT_MODEL"
    --sound-expert-backend "${SOUND_EXPERT_BACKEND:-auto}"
  )
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi

"$PYTHON" "${args[@]}"

if [[ "$BUILD_SEGLST" == "1" ]]; then
  session_id="$(
    "$PYTHON" -c \
      'import json, sys; print(json.loads(next(line for line in open(sys.argv[1], encoding="utf-8") if line.strip()))["session_id"])' \
      "$MANIFEST"
  )"
  session_dir="$OUTPUT_ROOT/$session_id"
  seglst_args=(
    -m
    "tutorials.audio.vad_refinement.build_seglst_variants"
    --source-manifest "$MANIFEST"
    --refined-manifest "$session_dir/manifest_vad_refined.jsonl"
    --refined-rttm "$session_dir/rttm/${session_id}_vad_refined.rttm"
    --output-dir "$session_dir/seglst"
  )
  if [[ -n "${REFERENCE_SEGLST:-}" ]]; then
    seglst_args+=(--reference-seglst "$REFERENCE_SEGLST")
  fi
  if [[ -n "${SPEAKER_MAP:-}" ]]; then
    seglst_args+=(--speaker-map "$SPEAKER_MAP")
  fi
  "$PYTHON" "${seglst_args[@]}"
fi
