#!/bin/bash
# Copy local david_ai_redelivered_mfa scripts to draco lustre (from laptop/workstation).
#
# Usage:
#   bash sync_to_draco.sh
#   DRACO_HOST=draco-oci-login-01 bash sync_to_draco.sh
#
# Default target (fs12, same as RAM cluster WORK_DIR base):
#   /lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva/
#     Curator/tutorials/audio/david_ai_redelivered_mfa/
#
# Legacy fsw path:
#   REMOTE_BASE=/lustre/fsw/portfolios/nemotron/users/ttimofeeva bash sync_to_draco.sh
#
# On draco (after sync):
#   cd $REMOTE_BASE/Curator/tutorials/audio/david_ai_redelivered_mfa
#   bash lexicon/run_preprocess_lexicon_cluster.sh
#   SKIP_LEXICON=1 bash run_david_ai_mfa_ram_session_cluster.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRACO_USER="${DRACO_USER:-ttimofeeva}"
DRACO_HOST="${DRACO_HOST:-draco-oci-login-01}"
REMOTE_BASE="${REMOTE_BASE:-/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva}"
REMOTE_DIR="$REMOTE_BASE/Curator/tutorials/audio/david_ai_redelivered_mfa"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if ! command -v rsync >/dev/null 2>&1; then
    echo "ERROR: rsync required" >&2
    exit 1
fi

# If fs12 is mounted locally (e.g. already on draco), copy directly.
if [[ -d "$(dirname "$REMOTE_BASE")" ]]; then
    log "Local lustre mount — copying to $REMOTE_DIR"
    mkdir -p "$REMOTE_DIR"
    rsync -av --progress \
        --exclude '__pycache__/' \
        --exclude 'workdir/' \
        --exclude 'workdir_ram_session/' \
        --exclude '*.pyc' \
        --exclude '.done' \
        "$SCRIPT_DIR/" \
        "$REMOTE_DIR/"
    log "Sync done (local). Scripts at: $REMOTE_DIR"
    exit 0
fi

log "Remote: ${DRACO_USER}@${DRACO_HOST}:$REMOTE_DIR"

ssh "${DRACO_USER}@${DRACO_HOST}" "mkdir -p '$REMOTE_DIR'"

rsync -av --progress \
    --exclude '__pycache__/' \
    --exclude 'workdir/' \
    --exclude 'workdir_ram_session/' \
    --exclude '*.pyc' \
    --exclude '.done' \
    "$SCRIPT_DIR/" \
    "${DRACO_USER}@${DRACO_HOST}:${REMOTE_DIR}/"

log "Sync done. On draco:"
log "  ssh ${DRACO_USER}@${DRACO_HOST}"
log "  cd $REMOTE_DIR"
log "  bash lexicon/run_preprocess_lexicon_cluster.sh"
log "  SKIP_LEXICON=1 bash run_david_ai_mfa_ram_session_cluster.sh"
