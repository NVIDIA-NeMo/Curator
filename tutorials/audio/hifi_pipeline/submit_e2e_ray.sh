#!/bin/bash
# Single-allocation legacy HIFI e2e pipeline submission, Ray-distributed.
#
# This is the multi-node-Ray twin of submit_e2e.sh.  Whereas
# submit_e2e.sh chains one sbatch per stage with --array=0-N%128 manual
# chunking and --dependency=afterok, this script submits ONE sbatch that
# brings up a Ray cluster via SlurmRayClient and then runs each stage's
# Pipeline.run() across that cluster.  Per-stage Resources specs drive
# Xenna's actor scheduler — no manual --array per stage.
#
# Output safety:
#   * submit_e2e.sh         → ${WORK}/e2e_output/<sub>/      (legacy, untouched)
#   * submit_e2e_ray.sh     → ${WORK}/e2e_output_ray/<sub>/  (this script)
#   * submit_beta.sh        → ${WORK}/hifi_beta_output/<sub>/ (beta)
#
# Hard guard refuses any output path under the legacy e2e_output tree.
#
# Usage (from cluster login node):
#   sbatch tutorials/audio/hifi_pipeline/submit_e2e_ray.sh --corpus ytc_ru
#   bash   tutorials/audio/hifi_pipeline/submit_e2e_ray.sh --corpus ytc_ru --dry-run
#
# Required env: WORK, GRANARY, CURATOR, SQSH_DIR, ACCOUNT.
# AIS token is sourced from ~/.ais_token if present.

#SBATCH --job-name=curator-hifi-e2e-ray
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=logs/e2e_ray_%j.log
#SBATCH --error=logs/e2e_ray_%j.log

set -euo pipefail

# ---- args ----
CORPUS=""
START_FROM=""
STAGES_OVERRIDE=""
CLUSTERING="${CLUSTERING:-per_video}"   # per_video or scotch (mirrors submit_e2e.sh)
DRY_RUN=false
DATA_CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --corpus) CORPUS="$2"; shift 2;;
        --start-from) START_FROM="$2"; shift 2;;
        --stages) STAGES_OVERRIDE="$2"; shift 2;;
        --clustering) CLUSTERING="$2"; shift 2;;
        --data-config) DATA_CONFIG="$2"; shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done
[[ -z "$CORPUS" ]] && { echo "Usage: sbatch $0 --corpus <name> [--clustering per_video|scotch] [--start-from <stage>] [--stages <csv>]"; exit 1; }

# ---- env ----
WORK="${WORK:?Set WORK}"
GRANARY="${GRANARY:?Set GRANARY}"
CURATOR="${CURATOR:?Set CURATOR}"
SQSH_DIR="${SQSH_DIR:?Set SQSH_DIR}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT}"

if [[ -f "${HOME}/.ais_token" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.ais_token"
fi
: "${AIS_ENDPOINT:=http://asr.iad.oci.aistore.nvidia.com:51080}"
: "${AIS_AUTHN_TOKEN:=}"

# ---- container ----
# The legacy run_pipeline.py covers transcribe (cascade vLLM) + nemo
# stages; the curator-hifi-pipeline image has both.
CONT_E2E="${CONT_E2E:-${SQSH_DIR}/curator-hifi-pipeline.sqsh}"
if ! $DRY_RUN && [[ ! -f "$CONT_E2E" ]]; then
    echo "ERROR: container not found: $CONT_E2E"
    exit 1
fi

# ---- corpus mapping ----
declare -A CORPUS_SUB CORPUS_SHARDS
CORPUS_SUB[ytc_ru]="ytc/ru";              CORPUS_SHARDS[ytc_ru]=64
CORPUS_SUB[yodas_0fc_ru]="yodas/0_from_captions/ru"; CORPUS_SHARDS[yodas_0fc_ru]=256
CORPUS_SUB[yodas_0bw_ru]="yodas/0_by_whisper/ru";    CORPUS_SHARDS[yodas_0bw_ru]=1024
CORPUS_SUB[yodas_1bw_ru]="yodas/1_by_whisper/ru";    CORPUS_SHARDS[yodas_1bw_ru]=8192

SUB="${CORPUS_SUB[$CORPUS]:-}"
NUM_SHARDS="${CORPUS_SHARDS[$CORPUS]:-}"
[[ -z "$SUB" ]] && { echo "Unknown corpus: $CORPUS"; exit 1; }

INPUT_MANIFEST="${INPUT_MANIFEST_OVERRIDE:-${GRANARY}/${SUB}}"
OUTPUT_DIR="${WORK}/e2e_output_ray/${SUB}"
LOG_DIR="${WORK}/e2e_ray_logs/${CORPUS}"

# ---- hard guard: never overwrite legacy e2e_output ----
LEGACY_E2E="${WORK}/e2e_output/${SUB}"
if [[ "${OUTPUT_DIR}" == "${LEGACY_E2E}" ]] || [[ "${OUTPUT_DIR}" == "${LEGACY_E2E}/"* ]]; then
    echo "REFUSING: OUTPUT_DIR (${OUTPUT_DIR}) collides with legacy e2e_output."
    exit 1
fi

export RAY_PORT_BROADCAST_DIR="${LOG_DIR}/ray_ports"
if ! $DRY_RUN; then
    mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${RAY_PORT_BROADCAST_DIR}"
fi

MOUNTS="${CONTAINER_MOUNTS:-/lustre/fs11:/lustre/fs11,/lustre/fsw:/lustre/fsw}"

# ---- args to legacy run_pipeline.py ----
START_FLAG=""
[[ -n "$START_FROM" ]] && START_FLAG="--start_from ${START_FROM}"   # NB: legacy uses --start-from? Check existing run_pipeline.py.

STAGES_FLAG=""
[[ -n "$STAGES_OVERRIDE" ]] && STAGES_FLAG="--stages ${STAGES_OVERRIDE}"

DATA_CFG_FLAG=""
INPUT_FLAG="--input_manifest ${INPUT_MANIFEST}"
if [[ -n "$DATA_CONFIG" ]]; then
    DATA_CFG_FLAG="--data_config ${DATA_CONFIG}"
    INPUT_FLAG=""
fi

SED_CKPT="${SED_CKPT:-/opt/checkpoints/Cnn14_DecisionLevelMax.pth}"
SED_FLAG=""
[[ -n "$SED_CKPT" ]] && SED_FLAG="--sed_checkpoint ${SED_CKPT}"

echo "=================================================="
echo "  HIFI E2E Pipeline (Ray, single allocation)"
echo "=================================================="
echo "  Job ID      : ${SLURM_JOB_ID:-(not in slurm)}"
echo "  Nodes       : ${SLURM_JOB_NODELIST:-N/A} (${SLURM_JOB_NUM_NODES:-1} node(s))"
echo "  Corpus      : ${CORPUS}  (${NUM_SHARDS} shards)"
if [[ -n "$DATA_CONFIG" ]]; then
    echo "  Source      : AIS-streamed via Granary YAML"
    echo "    yaml      : ${DATA_CONFIG}"
else
    echo "  Source      : file-based JSONL"
    echo "    manifest  : ${INPUT_MANIFEST}"
fi
echo "  Output      : ${OUTPUT_DIR}"
echo "  Legacy e2e  : ${LEGACY_E2E}  (NOT touched)"
echo "  Container   : ${CONT_E2E}"
echo "  AIS token   : $([[ -n "$AIS_AUTHN_TOKEN" ]] && echo "set" || echo "EMPTY")"
echo "  Stages      : ${STAGES_OVERRIDE:-(default)}"
echo "  Clustering  : ${CLUSTERING}"
echo "=================================================="

if $DRY_RUN; then
    cat <<DRY
[DRY RUN] Would submit:

srun --ntasks-per-node=1 \\
     --container-image=${CONT_E2E} \\
     --container-mounts=${MOUNTS} \\
     --container-writable \\
     bash -c "set -e
       cd ${CURATOR}
       export PYTHONUNBUFFERED=1
       export PYTHONPATH=${CURATOR}:\${PYTHONPATH:-}
       export AIS_ENDPOINT=${AIS_ENDPOINT}
       export AIS_AUTHN_TOKEN=<redacted>
       export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}
       export RAY_PORT_BROADCAST_DIR=${RAY_PORT_BROADCAST_DIR}
       /opt/conda/bin/python ${CURATOR}/tutorials/audio/hifi_pipeline/run_pipeline.py \\
           --slurm \\
           ${INPUT_FLAG} \\
           ${DATA_CFG_FLAG} \\
           --output_dir ${OUTPUT_DIR} \\
           ${SED_FLAG} \\
           ${STAGES_FLAG}"
DRY
    exit 0
fi

# Pass through GPU binding so Ray sees the allocated devices.
GPU_BIND=""
if [[ -n "${SLURM_GPUS_PER_NODE:-}" && "${SLURM_GPUS_PER_NODE:-0}" != "0" ]]; then
    GPU_BIND="--gpus-per-task=${SLURM_GPUS_PER_NODE}"
fi

srun \
    --ntasks-per-node=1 \
    ${GPU_BIND} \
    --container-image="${CONT_E2E}" \
    --container-mounts="${MOUNTS}" \
    --container-writable \
    bash -c "
set -e
cd '${CURATOR}'
export PYTHONUNBUFFERED=1
export PYTHONPATH='${CURATOR}:\${PYTHONPATH:-}'
export HF_HOME='${WORK}/hf_cache'
export UTMOSV2_CHACHE='${WORK}/utmosv2_cache'
export AIS_ENDPOINT='${AIS_ENDPOINT}'
export AIS_AUTHN_TOKEN='${AIS_AUTHN_TOKEN}'
export RAY_TMPDIR=\"/tmp/ray_\${SLURM_JOB_ID:-\$\$}\"
export RAY_PORT_BROADCAST_DIR='${RAY_PORT_BROADCAST_DIR}'

echo \"[\$(hostname)] SLURM_NODEID=\${SLURM_NODEID:-0}\"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null \
    | sed \"s/^/  [\$(hostname)] GPU /\" || true

# cosmos_xenna's resource discovery needs pynvml; ensure it's present.
/opt/conda/bin/python -c \"import pynvml\" 2>/dev/null || /opt/conda/bin/python -m pip install --quiet --no-input pynvml || true
# Refresh /opt/Curator (baked, inside actor sys.path) with our latest source.
if [[ -d '${CURATOR}/nemo_curator' && -d /opt/Curator ]]; then
    rm -rf /opt/Curator/nemo_curator
    ln -s '${CURATOR}/nemo_curator' /opt/Curator/nemo_curator
fi

/opt/conda/bin/python '${CURATOR}/tutorials/audio/hifi_pipeline/run_pipeline.py' \
    --slurm \
    ${INPUT_FLAG} \
    ${DATA_CFG_FLAG} \
    --output_dir '${OUTPUT_DIR}' \
    ${SED_FLAG} \
    ${STAGES_FLAG}
"

echo "DONE"
