#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

compose=(docker compose -p "full-english-$$" -f "$(dirname "${BASH_SOURCE[0]}")/compose.yaml")
"${compose[@]}" config --quiet
if [[ "$ENGLISH_OUTPUT" != /* || -e "$ENGLISH_OUTPUT" ]]; then
  printf 'ENGLISH_OUTPUT must be a fresh absolute path: %s\n' "$ENGLISH_OUTPUT" >&2
  exit 1
fi
mkdir -p "$ENGLISH_OUTPUT"
trap '"${compose[@]}" down' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${compose[@]}" up --no-build --abort-on-container-exit --exit-code-from pipeline 2>&1 | tee "$ENGLISH_OUTPUT/compose.log"
