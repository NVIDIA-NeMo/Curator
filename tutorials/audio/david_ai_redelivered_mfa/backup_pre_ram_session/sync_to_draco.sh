#!/bin/bash
# Copy local david_ai_redelivered_mfa scripts to draco lustre (from laptop/workstation).
#
# Usage:
#   bash sync_to_draco.sh
#   DRACO_HOST=draco-oci-login-01 bash sync_to_draco.sh
#
# Then on draco:
#   bash /lustre/fsw/portfolios/nemotron/users/ttimofeeva/Curator/tutorials/audio/david_ai_redelivered_mfa/setup_draco_cluster.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRACO_USER="${DRACO_USER:-ttimofeeva}"
DRACO_HOST="${DRACO_HOST:-draco-oci-login-01}"
REMOTE_BASE="${REMOTE_BASE:-/lustre/fsw/portfolios/nemotron/users/ttimofeeva}"
REMOTE_DIR="$REMOTE_BASE/Curator/tutorials/audio/david_ai_redelivered_mfa"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if ! command -v rsync >/dev/null 2>&1; then
    echo "ERROR: rsync required" >&2
    exit 1
fi

log "Remote: ${DRACO_USER}@${DRACO_HOST}:$REMOTE_DIR"

ssh "${DRACO_USER}@${DRACO_HOST}" "mkdir -p '$REMOTE_DIR'"

rsync -av --progress \
    --exclude '__pycache__/' \
    --exclude 'workdir/' \
    --exclude '*.pyc' \
    --exclude '.done' \
    "$SCRIPT_DIR/" \
    "${DRACO_USER}@${DRACO_HOST}:${REMOTE_DIR}/"

log "Sync done. On draco run:"
log "  ssh ${DRACO_USER}@${DRACO_HOST}"
log "  bash $REMOTE_DIR/setup_draco_cluster.sh"
