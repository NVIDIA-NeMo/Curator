#!/bin/bash
# Unpack curator_pain_1 conda env on Draco lustre (login node only).
#
# From laptop: pack + upload, then on draco:
#   bash install_curator_pain_1_draco.sh
#
# Or after rsync of the tarball to CLUSTER_BASE:
#   CONDA_ENV_TARBALL=/lustre/fs12/.../curator_pain_1_draco.tar.gz \
#     bash install_curator_pain_1_draco.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_BASE="${CLUSTER_BASE:-/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva}"
MINICONDA_DIR="${MINICONDA_DIR:-$CLUSTER_BASE/miniconda3}"
CONDA_ENV="${CONDA_ENV:-curator_pain_1}"
CONDA_ENV_TARBALL="${CONDA_ENV_TARBALL:-$CLUSTER_BASE/curator_pain_1_draco.tar.gz}"
ENV_DIR="$MINICONDA_DIR/envs/$CONDA_ENV"
MARKER="$ENV_DIR/.conda_unpack.done"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [[ ! -f "$CONDA_ENV_TARBALL" ]]; then
    echo "ERROR: tarball not found: $CONDA_ENV_TARBALL" >&2
    echo "Upload from laptop:" >&2
    echo "  rsync -avP workdir/curator_pain_1_draco.tar.gz draco:$CLUSTER_BASE/" >&2
    exit 1
fi

if [[ -f "$MARKER" && -x "$ENV_DIR/bin/python" && -x "$ENV_DIR/bin/mfa" ]]; then
    log "Env already installed: $ENV_DIR"
    "$ENV_DIR/bin/python" -c "import lhotse; print('lhotse ok')"
    "$ENV_DIR/bin/mfa" version
    exit 0
fi

log "Installing $CONDA_ENV -> $ENV_DIR"
log "Tarball: $CONDA_ENV_TARBALL ($(du -h "$CONDA_ENV_TARBALL" | awk '{print $1}'))"
mkdir -p "$ENV_DIR"
rm -rf "${ENV_DIR:?}/"*
tar -xzf "$CONDA_ENV_TARBALL" -C "$ENV_DIR"

if [[ -x "$ENV_DIR/bin/conda-unpack" ]]; then
    log "Running conda-unpack"
  # shellcheck source=/dev/null
    source "$ENV_DIR/bin/activate"
    conda-unpack
elif [[ -f "$ENV_DIR/bin/activate" ]]; then
  # shellcheck source=/dev/null
    source "$ENV_DIR/bin/activate"
    if command -v conda-unpack >/dev/null 2>&1; then
        conda-unpack
    fi
fi

touch "$MARKER"
log "Verify:"
"$ENV_DIR/bin/python" -c "import lhotse, num2words; print('python deps ok')"
"$ENV_DIR/bin/mfa" version
log "Done: $ENV_DIR"
