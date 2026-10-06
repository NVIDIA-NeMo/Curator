#!/bin/bash
# Merge successful cluster shards into dataset-level WER and Lhotse outputs.

set -euo pipefail

: "${OUTPUT_DIR:?Missing OUTPUT_DIR}"
: "${PARAKEET_ROOT:?Missing PARAKEET_ROOT}"
: "${ASR_ENV:?Missing ASR_ENV}"

WER_THRESHOLD_PCT="${WER_THRESHOLD_PCT:-100}"
PYTHON="$ASR_ENV/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: merge node cannot access ASR_ENV=$ASR_ENV" >&2
    exit 1
fi

"$PYTHON" "$PARAKEET_ROOT/analyze_wer_distribution.py" \
    --output-dir "$OUTPUT_DIR" \
    --applied-threshold-pct "$WER_THRESHOLD_PCT"

"$PYTHON" "$PARAKEET_ROOT/merge_lhotse_variants.py" \
    --output-dir "$OUTPUT_DIR"

echo "[$(date -Is)] Dataset-level WER and Lhotse outputs completed"
