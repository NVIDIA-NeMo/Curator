#!/bin/bash
set -euo pipefail

LOG_DIR="mls_workdir/mls/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_german_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to $LOG_FILE"

conda run --live-stream -n curator_pain_1 python prepare_mls_with_mfa_for_fastmss_nfa_filter.py \
  --config-path . --config-name input_german \
  mls_src_dir=~/Curator/tutorials/audio/mls_mfa_2/multilingual_librispeech \
  data_dir=mls_workdir \
  manifests_dir=mls_manifests \
  2>&1 | tee "$LOG_FILE"
