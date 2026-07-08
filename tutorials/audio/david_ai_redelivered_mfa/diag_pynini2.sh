#!/bin/bash
# Diagnose which pynini MFA actually loads: the container's (/opt/venv) vs the
# packed conda env's own bundled OpenFST. Reuses an already-unpacked env if
# present on the node.
set -euo pipefail

SCRIPT_DIR="${DIAG_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export DIAG_SCRIPT_DIR="$SCRIPT_DIR"
CLUSTER_BASE="${CLUSTER_BASE:-/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/portfolios/llmservice/users/pzelasko/containers/nemo-25.11-pytorch2.9-automodel-23apr26.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre/fsw:/lustre/fsw,/lustre/fs12:/lustre/fs12}"
MINICONDA_DIR="${MINICONDA_DIR:-/tmp/miniconda3_tt}"
CONDA_ENV="${CONDA_ENV:-curator_pain_1}"
CONDA_ENV_TARBALL="${CONDA_ENV_TARBALL:-$CLUSTER_BASE/curator_pain_1_draco.tar.gz}"
MFA_MODELS_TARBALL="${MFA_MODELS_TARBALL:-$CLUSTER_BASE/MFA_models_draco.tar.gz}"
JOB_NAME="${JOB_NAME:-diag_pynini2}"
LOGDIR="$CLUSTER_BASE/david_ai_mfa_ram_workdir/logs"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p "$LOGDIR"
    exec sbatch --job-name "$JOB_NAME" --nodes 1 --ntasks 1 --cpus-per-task 4 \
        --time 00:30:00 --output "$LOGDIR/${JOB_NAME}_%j.out" --export ALL \
        ${SLURM_ACCOUNT:+--account "$SLURM_ACCOUNT"} \
        ${SLURM_PARTITION:+--partition "$SLURM_PARTITION"} \
        "$0"
fi

if [[ "${IN_CONTAINER:-0}" != "1" ]]; then
    echo "[diag2] entering container"
    exec srun --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
        bash -lc "IN_CONTAINER=1 DIAG_SCRIPT_DIR='$SCRIPT_DIR' MINICONDA_DIR='$MINICONDA_DIR' \
        CLUSTER_BASE='$CLUSTER_BASE' CONDA_ENV='$CONDA_ENV' \
        CONDA_ENV_TARBALL='$CONDA_ENV_TARBALL' MFA_MODELS_TARBALL='$MFA_MODELS_TARBALL' \
        bash '$SCRIPT_DIR/diag_pynini2.sh'"
fi

ENV_DIR="$MINICONDA_DIR/envs/$CONDA_ENV"
PY="$ENV_DIR/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "[diag2] unpacking env"
    rm -rf "$ENV_DIR"; mkdir -p "$ENV_DIR"
    tar -xzf "$CONDA_ENV_TARBALL" -C "$ENV_DIR"
    [[ -x "$ENV_DIR/bin/conda-unpack" ]] && "$ENV_DIR/bin/conda-unpack" || true
fi
[[ -d /tmp/MFA_models ]] || tar -xzf "$MFA_MODELS_TARBALL" -C /tmp

echo "[diag2] container PYTHONPATH=${PYTHONPATH:-<unset>}"
echo "[diag2] ls conda-env pynini libs:"
ls "$ENV_DIR"/lib/python3.12/site-packages/pynini.libs/ 2>&1 | sed 's/^/[diag2]   /' | head
echo "[diag2] ls container pynini libs:"
ls /opt/venv/lib/python3.12/site-packages/pynini.libs/ 2>&1 | sed 's/^/[diag2]   /' | head

echo "[diag2] ===== force CONDA env pynini (clear PYTHONPATH) ====="
env -u PYTHONPATH "$PY" - <<PYEOF
import sys, os, tempfile, zipfile, traceback
# ensure /opt/venv is not on path
sys.path = [p for p in sys.path if "/opt/venv" not in p]
import pywrapfst, pynini
print("[diag2] _pywrapfst:", pywrapfst.__file__)
print("[diag2] pynini:", pynini.__file__)
z="/tmp/MFA_models/pretrained_models/g2p/english_us_arpa.zip"
d=tempfile.mkdtemp()
with zipfile.ZipFile(z) as f: f.extractall(d)
fst=[os.path.join(r,x) for r,_,fs in os.walk(d) for x in fs if x.endswith(".fst")][0]
try:
    m=pywrapfst.Fst.read(fst); print("[diag2] CONDA READ OK start=", m.start())
except Exception as e:
    print("[diag2] CONDA READ FAILED:", repr(e)); traceback.print_exc()
PYEOF

echo "[diag2] ===== which pynini does 'mfa' use? ====="
env -u PYTHONPATH "$ENV_DIR/bin/mfa" version 2>&1 | sed 's/^/[diag2] mfa: /' | head
echo "[diag2] ----- ldd conda-env _pywrapfst -----"
CSO=$(ls "$ENV_DIR"/lib/python3.12/site-packages/_pywrapfst*.so 2>/dev/null | head -1)
echo "[diag2] conda _pywrapfst: $CSO"
[[ -n "$CSO" ]] && ldd "$CSO" 2>&1 | sed 's/^/[diag2] ldd: /'
echo "[diag2] done"
