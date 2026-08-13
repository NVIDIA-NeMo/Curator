#!/bin/bash
# =============================================================================
# NeMo Curator — Hugging Face Jobs array launch script
#
# The HF Jobs analog of tutorials/slurm/submit_array.sh: fires one independent
# Job per shard. Curator's env-driven array sharding
# (NEMO_CURATOR_SLURM_ARRAY_*) splits the source files across shards, so
# array_pipeline.py and retry_array.py run unmodified. See the README for
# full details.
#
# ── Configure before launching ───────────────────────────────────────────────
#   Required — the script exits immediately with a clear error if missing:
#
#     BUCKET          — HF bucket used as the shared filesystem, e.g.
#                       hf://buckets/<user>/<bucket>
#                       (FUSE-mounted at /mnt in every Job)
#
#   Optional (sensible defaults are used when not set):
#
#     FLAVOR          — Jobs hardware flavor (default: cpu-basic)
#     TIMEOUT         — per-Job timeout (default: 30m)
#     TUTORIAL_SHARE  — shared-tree path inside the Job (default: /mnt/tutorial;
#                       input is read from ${TUTORIAL_SHARE}/input)
#
# ── Minimal usage ────────────────────────────────────────────────────────────
#   export BUCKET=hf://buckets/<user>/<bucket>
#   bash tutorials/hf_jobs/launch_array.sh 3        # launch shards 0..2
#   bash tutorials/hf_jobs/launch_array.sh 3 0 2    # relaunch only shards 0 and 2
#                                                   # (retry wave, per retry_array.py)
# =============================================================================
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "ERROR: total shard count is required." >&2
    echo "  Usage: launch_array.sh <total_shards> [shard_index ...]" >&2
    exit 2
fi
if [[ -z "${BUCKET:-}" ]]; then
    echo "ERROR: BUCKET is not set." >&2
    echo "  Set it before launching:" >&2
    echo "    export BUCKET=hf://buckets/<user>/<bucket>" >&2
    echo "    bash tutorials/hf_jobs/launch_array.sh 3" >&2
    exit 2
fi

K="$1"; shift
SHARDS=("$@")
[[ ${#SHARDS[@]} -eq 0 ]] && SHARDS=($(seq 0 $((K - 1))))

FLAVOR="${FLAVOR:-cpu-basic}"
TIMEOUT="${TIMEOUT:-30m}"
SCRIPT="$(dirname "$0")/run_shard.py"

for k in "${SHARDS[@]}"; do
    hf jobs uv run --detach --flavor "$FLAVOR" --timeout "$TIMEOUT" \
        -s HF_TOKEN \
        -e NEMO_CURATOR_SLURM_ARRAY_ENABLED=1 \
        -e NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX="$k" \
        -e NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS="$K" \
        -e TUTORIAL_SHARE="${TUTORIAL_SHARE:-/mnt/tutorial}" \
        -v "$BUCKET:/mnt" \
        --name "curator-tutorial-shard-$k" \
        "$SCRIPT" 2>&1 | grep -o "id=\S*"
    sleep 10   # space out firings: the mount driver can fail on freshly-allocated nodes
done
