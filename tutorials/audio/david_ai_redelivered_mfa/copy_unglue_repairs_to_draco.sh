#!/bin/bash
# Install heuristic unglue repairs on Draco lustre for MFA + RAM pipelines.
#
# Run from laptop (with SSH to draco):
#   bash copy_unglue_repairs_to_draco.sh
#
# Or on draco login node (after copying the heuristic file to lustre):
#   SRC=/path/to/unglue_repairs_heuristic.tsv bash copy_unglue_repairs_to_draco.sh
#
# Installs to:
#   $CLUSTER_BASE/david_ai_mfa_workdir/lexicon/unglue_repairs.tsv
#   $CLUSTER_BASE/david_ai_mfa_ram_workdir/lexicon/unglue_repairs.tsv  (same content)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-$SCRIPT_DIR/workdir/lexicon/unglue_repairs_heuristic.tsv}"
DRACO_USER="${DRACO_USER:-ttimofeeva}"
DRACO_HOST="${DRACO_HOST:-draco-oci-login-01}"
# fs12 is used by run_david_ai_mfa_ram_session_cluster.sh; fsw by setup_draco_cluster.sh
CLUSTER_BASE_FS12="${CLUSTER_BASE_FS12:-/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva}"
CLUSTER_BASE_FSW="${CLUSTER_BASE_FSW:-/lustre/fsw/portfolios/nemotron/users/ttimofeeva}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: source not found: $SRC" >&2
    exit 1
fi

install_to_lexicon_dir() {
    local lex_dir="$1"
    mkdir -p "$lex_dir"
    cp -f "$SRC" "$lex_dir/unglue_repairs_heuristic.tsv"
    cp -f "$SRC" "$lex_dir/unglue_repairs.tsv"
    log "Installed $(wc -l < "$lex_dir/unglue_repairs.tsv") repairs -> $lex_dir/unglue_repairs.tsv"
}

if [[ -d "$CLUSTER_BASE_FS12" || -d "$CLUSTER_BASE_FSW" ]]; then
    log "Local lustre mount detected — installing directly"
    for base in "$CLUSTER_BASE_FS12" "$CLUSTER_BASE_FSW"; do
        [[ -d "$(dirname "$base")" ]] || continue
        install_to_lexicon_dir "$base/david_ai_mfa_workdir/lexicon"
        install_to_lexicon_dir "$base/david_ai_mfa_ram_workdir/lexicon"
    done
    exit 0
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "ERROR: rsync required for remote copy" >&2
    exit 1
fi

log "Remote install via ${DRACO_USER}@${DRACO_HOST}"
for base in "$CLUSTER_BASE_FS12" "$CLUSTER_BASE_FSW"; do
    for work in david_ai_mfa_workdir david_ai_mfa_ram_workdir; do
        remote_lex="${base}/${work}/lexicon"
        log "==> ${DRACO_USER}@${DRACO_HOST}:${remote_lex}"
        ssh "${DRACO_USER}@${DRACO_HOST}" "mkdir -p '$remote_lex'"
        rsync -av "$SRC" "${DRACO_USER}@${DRACO_HOST}:${remote_lex}/unglue_repairs_heuristic.tsv"
        ssh "${DRACO_USER}@${DRACO_HOST}" \
            "cp -f '${remote_lex}/unglue_repairs_heuristic.tsv' '${remote_lex}/unglue_repairs.tsv' && wc -l '${remote_lex}/unglue_repairs.tsv'"
    done
done

log "Done. RAM + manifest pipelines will load unglue_repairs.tsv from lexicon/."
