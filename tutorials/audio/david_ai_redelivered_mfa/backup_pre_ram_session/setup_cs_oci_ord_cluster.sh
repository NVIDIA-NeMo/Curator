#!/bin/bash
# One-time setup on cs-oci-ord-login-01 for David AI MFA pipeline.
#
# Run ON cs-oci-ord:
#   bash ~/ttimofeeva/Curator/tutorials/audio/david_ai_redelivered_mfa/setup_cs_oci_ord_cluster.sh
#
# Uses existing lustre miniconda + curator_pain_1 if present; otherwise installs them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# cs-oci-ord layout
CLUSTER_BASE="${CLUSTER_BASE:-/lustre/fsw/portfolios/nemotron/users/ttimofeeva}"
CURATOR_ROOT="${CURATOR_ROOT:-$HOME/ttimofeeva/Curator}"
MFA_TUTORIAL="${MFA_TUTORIAL:-$CURATOR_ROOT/tutorials/audio/david_ai_redelivered_mfa}"
MINICONDA_DIR="${MINICONDA_DIR:-$CLUSTER_BASE/miniconda3}"
CONDA_ENV="${CONDA_ENV:-curator_pain_1}"
MFA_ROOT_DIR="${MFA_ROOT_DIR:-$CLUSTER_BASE/MFA_models}"
GIT_REPO="${GIT_REPO:-https://github.com/Ssofja/Curator.git}"
GIT_BRANCH="${GIT_BRANCH:-main}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
SRC_DATA_ROOT="${SRC_DATA_ROOT:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/DavidAI_2026-05-29_redeliver}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# cs-oci-ord home has a tiny (10G) quota; keep all caches/temp on lustre so pip,
# conda, HF and MFA downloads never blow the home quota (Errno 122).
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CLUSTER_BASE/.cache/pip}"
export TMPDIR="${TMPDIR:-$CLUSTER_BASE/tmp}"
export HF_HOME="${HF_HOME:-$CLUSTER_BASE/.cache/huggingface}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$MINICONDA_DIR/pkgs}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME"
if [[ ! -e "$HOME/.cache/pip" || -L "$HOME/.cache/pip" ]]; then
    mkdir -p "$HOME/.cache"
    ln -sfn "$PIP_CACHE_DIR" "$HOME/.cache/pip"
fi

log "HOST=$(hostname)"
log "PIP_CACHE_DIR=$PIP_CACHE_DIR"
log "TMPDIR=$TMPDIR"
log "CLUSTER_BASE=$CLUSTER_BASE"
log "CURATOR_ROOT=$CURATOR_ROOT"
log "MINICONDA_DIR=$MINICONDA_DIR"
log "CONDA_ENV=$CONDA_ENV"
log "MFA_ROOT_DIR=$MFA_ROOT_DIR"
log "SRC_DATA_ROOT=$SRC_DATA_ROOT"

mkdir -p "$CLUSTER_BASE" "$MFA_TUTORIAL"

# --- Curator repo (merge if partial tree from rsync) -------------------------
if [[ ! -d "$CURATOR_ROOT/.git" ]]; then
    if [[ -e "$CURATOR_ROOT" ]]; then
        log "Partial Curator tree — cloning and merging"
        TMP_CLONE="$(mktemp -d)"
        git clone --depth 1 --branch "$GIT_BRANCH" "$GIT_REPO" "$TMP_CLONE/Curator"
        rsync -a "$TMP_CLONE/Curator/" "$CURATOR_ROOT/"
        rm -rf "$TMP_CLONE"
    else
        log "Cloning $GIT_REPO -> $CURATOR_ROOT"
        git clone --depth 1 --branch "$GIT_BRANCH" "$GIT_REPO" "$CURATOR_ROOT"
    fi
else
    log "Curator repo exists: $CURATOR_ROOT"
fi

# Re-sync MFA scripts from this script's directory (rsync may have newer files)
if [[ "$SCRIPT_DIR" != "$MFA_TUTORIAL" ]]; then
    rsync -a "$SCRIPT_DIR/" "$MFA_TUTORIAL/"
fi

# --- Miniconda3 --------------------------------------------------------------
if [[ ! -x "$MINICONDA_DIR/bin/conda" ]]; then
    INSTALLER="$CLUSTER_BASE/Miniconda3-latest-Linux-x86_64.sh"
    if [[ ! -f "$INSTALLER" ]]; then
        log "Downloading Miniconda installer..."
        curl -fsSL -o "$INSTALLER" \
            "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    fi
    log "Installing Miniconda -> $MINICONDA_DIR"
    bash "$INSTALLER" -b -p "$MINICONDA_DIR"
else
    log "Miniconda already installed: $MINICONDA_DIR"
fi

# shellcheck source=/dev/null
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    log "Creating conda env $CONDA_ENV (python=$PYTHON_VERSION)"
    conda create -y -n "$CONDA_ENV" "python=${PYTHON_VERSION}"
else
    log "Conda env exists: $CONDA_ENV"
fi

conda activate "$CONDA_ENV"

log "Installing MFA + Python deps into $CONDA_ENV"
conda install -y -c conda-forge montreal-forced-aligner=3.3.9 ffmpeg

pip install --upgrade pip wheel setuptools
pip install -e "$CURATOR_ROOT"
if [[ -f "$MFA_TUTORIAL/requirements.txt" ]]; then
    pip install -r "$MFA_TUTORIAL/requirements.txt"
fi

# --- MFA pretrained models ---------------------------------------------------
export MFA_ROOT_DIR
mkdir -p "$MFA_ROOT_DIR"
export PATH="$MINICONDA_DIR/envs/$CONDA_ENV/bin:$PATH"

for model in dictionary:english_us_arpa acoustic:english_us_arpa g2p:english_us_arpa; do
    kind="${model%%:*}"
    name="${model##*:}"
    if [[ "$kind" == "dictionary" && -f "$MFA_ROOT_DIR/pretrained_models/dictionary/${name}.dict" ]]; then
        log "MFA $kind $name already present"
        continue
    fi
    if [[ "$kind" != "dictionary" && -e "$MFA_ROOT_DIR/pretrained_models/${kind}/${name}.zip" ]]; then
        log "MFA $kind $name already present"
        continue
    fi
    log "Downloading MFA $kind $name"
    mfa model download "$kind" "$name" || log "WARN: mfa model download $kind $name failed"
done

# --- Env file ----------------------------------------------------------------
ENV_FILE="$CLUSTER_BASE/david_ai_mfa_env.sh"
cat >"$ENV_FILE" <<EOF
# Source on cs-oci-ord before running David AI MFA:
#   source $ENV_FILE
export CLUSTER_BASE="$CLUSTER_BASE"
export CURATOR_ROOT="$CURATOR_ROOT"
export MFA_TUTORIAL="$MFA_TUTORIAL"
export MINICONDA_DIR="$MINICONDA_DIR"
export MFA_ROOT_DIR="$MFA_ROOT_DIR"
export MFA_ENV="$MINICONDA_DIR/envs/$CONDA_ENV"
export PATH="\$MFA_ENV/bin:\$PATH"
# keep caches/temp off the 10G home quota
export PIP_CACHE_DIR="$CLUSTER_BASE/.cache/pip"
export TMPDIR="$CLUSTER_BASE/tmp"
export HF_HOME="$CLUSTER_BASE/.cache/huggingface"
export WORK_DIR="\${WORK_DIR:-$CLUSTER_BASE/david_ai_mfa_workdir}"
export DATA_ROOT="\${DATA_ROOT:-\$WORK_DIR/data_links}"
export SRC_DATA_ROOT="\${SRC_DATA_ROOT:-$SRC_DATA_ROOT}"
EOF

chmod +x "$MFA_TUTORIAL"/run_david_ai_mfa*.sh "$MFA_TUTORIAL"/setup_*.sh 2>/dev/null || true

log "Verifying install..."
"$MINICONDA_DIR/envs/$CONDA_ENV/bin/python" -c "import nemo_curator, lhotse, textgrid, num2words; print('python deps OK')"
mfa version || true
ffmpeg -version | head -1 || log "WARN: ffmpeg not on PATH"

log "SETUP COMPLETE (cs-oci-ord)"
log "  source $ENV_FILE"
log "  cd \$MFA_TUTORIAL"
log "  LOCAL_RUN=1 bash run_david_ai_mfa_cluster.sh"
