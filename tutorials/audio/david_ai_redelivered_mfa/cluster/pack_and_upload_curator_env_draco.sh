#!/bin/bash
# Pack local curator_pain_1 env and upload to Draco fs12 lustre.
#
# Usage (laptop):
#   bash pack_and_upload_curator_env_draco.sh
#
# Then on draco login:
#   bash install_curator_pain_1_draco.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_CONDA="${LOCAL_CONDA:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-curator_pain_1}"
CLUSTER_BASE="${CLUSTER_BASE:-/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva}"
DEST_PREFIX="${DEST_PREFIX:-/tmp/miniconda3_tt/envs/$CONDA_ENV}"
OUT="${OUT:-$SCRIPT_DIR/../workdir/curator_pain_1_draco.tar.gz}"
DRACO_HOST="${DRACO_HOST:-draco-oci-login-01.draco-oci-iad.nvidia.com}"
DRACO_USER="${DRACO_USER:-ttimofeeva}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [[ ! -x "$LOCAL_CONDA/bin/conda" ]]; then
    echo "ERROR: conda not found at $LOCAL_CONDA" >&2
    exit 1
fi

log "Packing $CONDA_ENV (dest-prefix=$DEST_PREFIX)"
mkdir -p "$(dirname "$OUT")"
"$LOCAL_CONDA/bin/conda" pack -n "$CONDA_ENV" \
    --dest-prefix "$DEST_PREFIX" \
    --ignore-editable-packages \
    --ignore-missing-files \
    -o "$OUT"
log "Packed: $OUT ($(du -h "$OUT" | awk '{print $1}'))"

log "Uploading to ${DRACO_USER}@${DRACO_HOST}:$CLUSTER_BASE/"
rsync -avP "$OUT" "${DRACO_USER}@${DRACO_HOST}:${CLUSTER_BASE}/"
rsync -avP "$SCRIPT_DIR/install_curator_pain_1_draco.sh" \
    "${DRACO_USER}@${DRACO_HOST}:${CLUSTER_BASE}/Curator/tutorials/audio/david_ai_redelivered_mfa/cluster/"

log "On draco run:"
log "  ssh ${DRACO_USER}@${DRACO_HOST}"
log "  CLUSTER_BASE=$CLUSTER_BASE bash $CLUSTER_BASE/Curator/tutorials/audio/david_ai_redelivered_mfa/cluster/install_curator_pain_1_draco.sh"
