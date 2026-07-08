#!/bin/bash
# Diagnose the MFA G2P `FstIOError: Read failed` inside the pyxis container.
# Submits a tiny batch job, enters the same container, unpacks the packed conda
# env, extracts the g2p zip, and tries pynini.Fst.read — printing lib info so we
# can tell env corruption from a container ABI mismatch.
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
JOB_NAME="${JOB_NAME:-diag_pynini}"
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
    echo "[diag] entering container: $CONTAINER_IMAGE"
    exec srun --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
        bash -lc "IN_CONTAINER=1 SCRIPT_DIR='$SCRIPT_DIR' MINICONDA_DIR='$MINICONDA_DIR' \
        CLUSTER_BASE='$CLUSTER_BASE' CONDA_ENV='$CONDA_ENV' \
        CONDA_ENV_TARBALL='$CONDA_ENV_TARBALL' MFA_MODELS_TARBALL='$MFA_MODELS_TARBALL' \
        bash '$SCRIPT_DIR/diag_pynini_container.sh'"
fi

echo "[diag] ===== inside container ====="
cat /etc/os-release 2>/dev/null | head -2 || true
echo "[diag] glibc: $(ldd --version 2>/dev/null | head -1)"

ENV_DIR="$MINICONDA_DIR/envs/$CONDA_ENV"
PY="$ENV_DIR/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "[diag] unpacking env -> $ENV_DIR"
    rm -rf "$ENV_DIR"; mkdir -p "$ENV_DIR"
    tar -xzf "$CONDA_ENV_TARBALL" -C "$ENV_DIR"
    [[ -x "$ENV_DIR/bin/conda-unpack" ]] && "$ENV_DIR/bin/conda-unpack" || true
fi
echo "[diag] python: $PY"

if [[ ! -d /tmp/MFA_models ]]; then
    echo "[diag] extracting MFA models"
    tar -xzf "$MFA_MODELS_TARBALL" -C /tmp
fi
G2P_ZIP=/tmp/MFA_models/pretrained_models/g2p/english_us_arpa.zip
echo "[diag] g2p zip: $G2P_ZIP ($(stat -c%s "$G2P_ZIP" 2>/dev/null) bytes)"

echo "[diag] ----- pywrapfst .so ldd -----"
SO=$("$PY" -c "import _pywrapfst,os;print(os.path.dirname(_pywrapfst.__file__))" 2>/dev/null || true)
"$PY" -c "import _pywrapfst; print(_pywrapfst.__file__)" 2>/dev/null || echo "[diag] import _pywrapfst failed"
PYWRAP_SO=$("$PY" -c "import _pywrapfst; print(_pywrapfst.__file__)" 2>/dev/null || true)
[[ -n "$PYWRAP_SO" ]] && ldd "$PYWRAP_SO" 2>&1 | sed 's/^/[diag] ldd: /' || true

echo "[diag] ----- FST read test (default LD path) -----"
"$PY" - <<PYEOF
import os, tempfile, zipfile, traceback
z="/tmp/MFA_models/pretrained_models/g2p/english_us_arpa.zip"
d=tempfile.mkdtemp()
with zipfile.ZipFile(z) as f: f.extractall(d)
fst=None
for r,_,fs in os.walk(d):
    for x in fs:
        if x.endswith(".fst"): fst=os.path.join(r,x)
print("[diag] fst:", fst, os.path.getsize(fst))
try:
    import pywrapfst
    m=pywrapfst.Fst.read(fst); print("[diag] READ OK start=", m.start())
except Exception as e:
    print("[diag] READ FAILED:", repr(e))
    traceback.print_exc()
PYEOF

echo "[diag] ----- FST read test (LD_LIBRARY_PATH=env/lib first) -----"
LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}" "$PY" - <<PYEOF
import os, tempfile, zipfile, traceback
z="/tmp/MFA_models/pretrained_models/g2p/english_us_arpa.zip"
d=tempfile.mkdtemp()
with zipfile.ZipFile(z) as f: f.extractall(d)
fst=None
for r,_,fs in os.walk(d):
    for x in fs:
        if x.endswith(".fst"): fst=os.path.join(r,x)
try:
    import pywrapfst
    m=pywrapfst.Fst.read(fst); print("[diag] READ OK (env lib) start=", m.start())
except Exception as e:
    print("[diag] READ FAILED (env lib):", repr(e))
PYEOF

echo "[diag] done"
