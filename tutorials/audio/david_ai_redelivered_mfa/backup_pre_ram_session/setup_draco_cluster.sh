#!/bin/bash
# One-time setup on draco (or any Linux cluster login node).
#
# Run ON the cluster:
#   bash setup_draco_cluster.sh
#
# From your laptop (after SSH works), copy this file first:
#   rsync -av tutorials/audio/david_ai_redelivered_mfa/ \
#     ttimofeeva@draco-oci-login-01:/lustre/fsw/portfolios/nemotron/users/ttimofeeva/Curator/tutorials/audio/david_ai_redelivered_mfa/
#
# Or use sync_to_draco.sh from your local Curator checkout.

set -euo pipefail

CLUSTER_BASE="${CLUSTER_BASE:-/lustre/fsw/portfolios/nemotron/users/ttimofeeva}"
CURATOR_ROOT="${CURATOR_ROOT:-$CLUSTER_BASE/Curator}"
MFA_TUTORIAL="${MFA_TUTORIAL:-$CURATOR_ROOT/tutorials/audio/david_ai_redelivered_mfa}"
MINICONDA_DIR="${MINICONDA_DIR:-$CLUSTER_BASE/miniconda3}"
CONDA_ENV="${CONDA_ENV:-curator_pain_1}"
MFA_ROOT_DIR="${MFA_ROOT_DIR:-$CLUSTER_BASE/MFA_models}"
GIT_REPO="${GIT_REPO:-https://github.com/Ssofja/Curator.git}"
GIT_BRANCH="${GIT_BRANCH:-main}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "CLUSTER_BASE=$CLUSTER_BASE"
log "CURATOR_ROOT=$CURATOR_ROOT"
log "MINICONDA_DIR=$MINICONDA_DIR"
log "CONDA_ENV=$CONDA_ENV"
log "MFA_ROOT_DIR=$MFA_ROOT_DIR"

mkdir -p "$CLUSTER_BASE"

# --- Clone Curator repo ------------------------------------------------------
if [[ ! -d "$CURATOR_ROOT/.git" ]]; then
    if [[ -e "$CURATOR_ROOT" ]]; then
        log "Partial Curator tree at $CURATOR_ROOT — cloning repo and merging"
        TMP_CLONE="$(mktemp -d)"
        git clone --depth 1 --branch "$GIT_BRANCH" "$GIT_REPO" "$TMP_CLONE/Curator"
        rsync -a "$TMP_CLONE/Curator/" "$CURATOR_ROOT/"
        rm -rf "$TMP_CLONE"
    else
        log "Cloning $GIT_REPO -> $CURATOR_ROOT"
        git clone --depth 1 --branch "$GIT_BRANCH" "$GIT_REPO" "$CURATOR_ROOT"
    fi
else
    log "Curator repo exists: $CURATOR_ROOT (skip clone)"
fi

mkdir -p "$MFA_TUTORIAL"

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

# Editable Curator + tutorial extras
pip install -e "$CURATOR_ROOT"
if [[ -f "$MFA_TUTORIAL/requirements.txt" ]]; then
    pip install -r "$MFA_TUTORIAL/requirements.txt"
else
    log "WARN: $MFA_TUTORIAL/requirements.txt missing — copy MFA scripts first, then re-run"
    pip install lhotse textgrid num2words hydra-core omegaconf soundfile tqdm pyloudnorm praatio
fi

# --- MFA pretrained models ---------------------------------------------------
export MFA_ROOT_DIR
mkdir -p "$MFA_ROOT_DIR"
if [[ -x "$MINICONDA_DIR/envs/$CONDA_ENV/bin/mfa" ]]; then
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
      mfa model download "$kind" "$name" || log "WARN: mfa model download $kind $name failed (retry manually)"
  done
else
    log "WARN: mfa not found after install"
fi

# --- Env snippet for interactive use -------------------------------------------
ENV_FILE="$CLUSTER_BASE/david_ai_mfa_env.sh"
cat >"$ENV_FILE" <<EOF
# Source before running the David AI MFA pipeline on the cluster:
#   source $ENV_FILE
export CLUSTER_BASE="$CLUSTER_BASE"
export CURATOR_ROOT="$CURATOR_ROOT"
export MFA_TUTORIAL="$MFA_TUTORIAL"
export MINICONDA_DIR="$MINICONDA_DIR"
export MFA_ROOT_DIR="$MFA_ROOT_DIR"
export MFA_ENV="$MINICONDA_DIR/envs/$CONDA_ENV"
export PATH="\$MFA_ENV/bin:\$PATH"
export WORK_DIR="\${WORK_DIR:-$CLUSTER_BASE/david_ai_mfa_workdir}"
export DATA_ROOT="\${DATA_ROOT:-\$WORK_DIR/data_links}"
export SRC_DATA_ROOT="\${SRC_DATA_ROOT:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/DavidAI_2026-05-29_redeliver}"
EOF

chmod +x "$MFA_TUTORIAL/run_david_ai_mfa.sh" \
    "$MFA_TUTORIAL/run_david_ai_mfa_ram.sh" \
    "$MFA_TUTORIAL/run_david_ai_mfa_cluster.sh" 2>/dev/null || true

log "SETUP COMPLETE"
log "  Curator:     $CURATOR_ROOT"
log "  MFA scripts: $MFA_TUTORIAL"
log "  Conda env:   $MINICONDA_DIR/envs/$CONDA_ENV"
log "  MFA models:  $MFA_ROOT_DIR"
log ""
log "Next:"
log "  1) Copy local MFA scripts if not done yet:"
log "       bash sync_to_draco.sh   # from your laptop"
log "  2) source $ENV_FILE"
log "  3) cd \$MFA_TUTORIAL && LOCAL_RUN=1 bash run_david_ai_mfa_cluster.sh"
log "     # or: sbatch-friendly: bash run_david_ai_mfa_cluster.sh"
