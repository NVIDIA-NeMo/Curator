#!/bin/bash
# Single-allocation HIFI beta pipeline submission.
#
# Spins up ONE multi-node Ray cluster via SlurmRayClient and runs all
# stages (ASR -> SED -> SED post -> segment -> diarize -> embed ->
# utmos2 -> cluster_scotch) inside it.  Replaces submit_e2e.sh's
# stage-per-sbatch chaining with one allocation; per-stage Resources
# specs drive Xenna's actor scheduler.
#
# Output goes to ${WORK}/hifi_beta_output/<sub>/ -- a brand-new path
# so existing ${WORK}/e2e_output/<sub>/ results are never touched.
#
# Usage:
#   sbatch tutorials/audio/hifi_pipeline/submit_beta.sh --corpus ytc_ru
#   sbatch tutorials/audio/hifi_pipeline/submit_beta.sh --corpus ytc_ru --start-from embed
#   bash   tutorials/audio/hifi_pipeline/submit_beta.sh --corpus ytc_ru --dry-run

#SBATCH --job-name=curator-hifi-beta
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=logs/hifi_beta_%j.log
#SBATCH --error=logs/hifi_beta_%j.log

set -euo pipefail

# ---- Parse args (after sbatch) ----
CORPUS=""
START_FROM=""
DRY_RUN=false
SCOTCH_PRESET="${SCOTCH_PRESET:-librispeech-2026-04}"
STAGES_OVERRIDE=""
DATA_CONFIG=""
SEGMENT_EVENTS_KEY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --corpus) CORPUS="$2"; shift 2;;
        --start-from) START_FROM="$2"; shift 2;;
        --scotch-preset) SCOTCH_PRESET="$2"; shift 2;;
        --stages) STAGES_OVERRIDE="$2"; shift 2;;
        --data-config) DATA_CONFIG="$2"; shift 2;;
        --segment-events-key) SEGMENT_EVENTS_KEY="$2"; shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done
[[ -z "$CORPUS" ]] && { echo "Usage: sbatch $0 --corpus <name> [--data-config <yaml>] [--start-from <stage>] [--stages <csv>] [--segment-events-key <key>] [--scotch-preset <name>]"; exit 1; }

# ---- Required env (same as submit_e2e.sh) ----
WORK="${WORK:?Set WORK to your working directory}"
GRANARY="${GRANARY:?Set GRANARY to your granary_filtered directory}"
CURATOR="${CURATOR:?Set CURATOR to your NeMo Curator checkout}"
SQSH_DIR="${SQSH_DIR:?Set SQSH_DIR to your container directory}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT to your Slurm account}"

# AIS token: per SUBMISSION_GUIDE.md gotcha, must be set in the submitter's
# shell so --export=ALL propagates it.  We source ~/.ais_token if present.
if [[ -f "${HOME}/.ais_token" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.ais_token"
fi
: "${AIS_ENDPOINT:=http://asr.iad.oci.aistore.nvidia.com:51080}"
: "${AIS_AUTHN_TOKEN:=}"
export AIS_ENDPOINT AIS_AUTHN_TOKEN

# ---- Container ----
# Beta needs ASR + SED + Sortformer + TitaNet + utmosv2 + scotch in one image.
# First-pass: try the existing nemo-stages image; document the rebuild path.
CONT_BETA="${CONT_BETA:-${SQSH_DIR}/curator-hifi-beta.sqsh}"
if [[ ! -f "$CONT_BETA" ]]; then
    CONT_BETA="${SQSH_DIR}/curator-hifi-nemo-stages.sqsh"
    echo "WARN: curator-hifi-beta.sqsh not found, falling back to ${CONT_BETA}"
    echo "      utmos2 stage will fail unless utmosv2 is installed at runtime."
fi
# In dry-run we may be off-cluster; only enforce the file check at submit time.
if ! $DRY_RUN && [[ ! -f "$CONT_BETA" ]]; then
    echo "ERROR: container not found: $CONT_BETA"
    exit 1
fi

# ---- Corpus mapping (mirrors submit_e2e.sh) ----
declare -A CORPUS_SUB CORPUS_SHARDS
CORPUS_SUB[ytc_ru]="ytc/ru";              CORPUS_SHARDS[ytc_ru]=64
CORPUS_SUB[yodas_0fc_ru]="yodas/0_from_captions/ru"; CORPUS_SHARDS[yodas_0fc_ru]=256
CORPUS_SUB[yodas_0bw_ru]="yodas/0_by_whisper/ru";    CORPUS_SHARDS[yodas_0bw_ru]=1024
CORPUS_SUB[yodas_1bw_ru]="yodas/1_by_whisper/ru";    CORPUS_SHARDS[yodas_1bw_ru]=8192

SUB="${CORPUS_SUB[$CORPUS]:-}"
NUM_SHARDS="${CORPUS_SHARDS[$CORPUS]:-}"
[[ -z "$SUB" ]] && { echo "Unknown corpus: $CORPUS"; exit 1; }

# Default input is the raw Granary manifest dir. For smoke tests,
# point INPUT_MANIFEST_OVERRIDE at an existing per-stage output dir
# (e.g., ${WORK}/e2e_output/ytc/ru/sed) — that path stays read-only;
# we only ever WRITE under ${WORK}/hifi_beta_output/.
INPUT_MANIFEST="${INPUT_MANIFEST_OVERRIDE:-${GRANARY}/${SUB}}"
OUTPUT_DIR="${WORK}/hifi_beta_output/${SUB}"
LOG_DIR="${WORK}/hifi_beta_logs/${CORPUS}"

# ---- Hard guard: never overwrite existing e2e_output paths ----
LEGACY_E2E="${WORK}/e2e_output/${SUB}"
if [[ "${OUTPUT_DIR}" == "${LEGACY_E2E}" ]] || [[ "${OUTPUT_DIR}" == "${LEGACY_E2E}/"* ]]; then
    echo "REFUSING: OUTPUT_DIR (${OUTPUT_DIR}) collides with legacy e2e_output."
    echo "          The beta pipeline must write to a different path to preserve prior runs."
    exit 1
fi

# Shared dir for SlurmRayClient port broadcast (must be visible to ALL nodes).
export RAY_PORT_BROADCAST_DIR="${LOG_DIR}/ray_ports"

# mkdir only when actually submitting (skip in dry-run so it works off-cluster).
if ! $DRY_RUN; then
    mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${RAY_PORT_BROADCAST_DIR}"
fi

MOUNTS="${CONTAINER_MOUNTS:-/lustre/fs11:/lustre/fs11,/lustre/fsw:/lustre/fsw}"

START_FLAG=""
[[ -n "$START_FROM" ]] && START_FLAG="--start_from ${START_FROM}"

STAGES_FLAG=""
[[ -n "$STAGES_OVERRIDE" ]] && STAGES_FLAG="--stages ${STAGES_OVERRIDE}"

SEGEVENTS_FLAG=""
[[ -n "$SEGMENT_EVENTS_KEY" ]] && SEGEVENTS_FLAG="--segment_events_key ${SEGMENT_EVENTS_KEY}"

# Source selector: AIS streaming (granary YAML) takes precedence over file-based
# input.  --input_manifest still defaults to ${GRANARY}/${SUB} for the
# legacy file-based mode, but if --data-config is given we omit that flag.
DATA_CFG_FLAG=""
INPUT_FLAG="--input_manifest ${INPUT_MANIFEST}"
if [[ -n "$DATA_CONFIG" ]]; then
    DATA_CFG_FLAG="--data_config ${DATA_CONFIG}"
    # Map cluster --corpus to the YAML's corpus name (no --corpus_filter for now,
    # users can edit the YAML or pass via env if they need).
    INPUT_FLAG=""
fi

# Optional SED checkpoint (only needed when SED stage is in --stages).
SED_CKPT="${SED_CKPT:-/opt/checkpoints/Cnn14_DecisionLevelMax.pth}"
SED_FLAG=""
[[ -n "$SED_CKPT" ]] && SED_FLAG="--sed_checkpoint ${SED_CKPT}"

echo "=================================================="
echo "  HIFI Beta Pipeline (single allocation)"
echo "=================================================="
echo "  Job ID      : ${SLURM_JOB_ID:-(not in slurm)}"
echo "  Nodes       : ${SLURM_JOB_NODELIST:-N/A} (${SLURM_JOB_NUM_NODES:-1} node(s))"
echo "  GPUs/node   : ${SLURM_GPUS_ON_NODE:-${SBATCH_GPUS_PER_NODE:-?}}"
echo "  Corpus      : ${CORPUS}  (${NUM_SHARDS} shards)"
if [[ -n "$DATA_CONFIG" ]]; then
    echo "  Source      : AIS-streamed via Granary YAML"
    echo "    yaml      : ${DATA_CONFIG}"
else
    echo "  Source      : file-based JSONL"
    echo "    manifest  : ${INPUT_MANIFEST}$([[ -n "${INPUT_MANIFEST_OVERRIDE:-}" ]] && echo "  (override)")"
fi
echo "  Output      : ${OUTPUT_DIR}"
echo "  Legacy e2e  : ${LEGACY_E2E}  (NOT touched)"
echo "  Container   : ${CONT_BETA}"
echo "  AIS endpoint: ${AIS_ENDPOINT}"
echo "  AIS token   : $([[ -n "$AIS_AUTHN_TOKEN" ]] && echo "set" || echo "EMPTY (will fail on s3:// reads)")"
echo "  Stages      : ${STAGES_OVERRIDE:-(default)}"
echo "  Start from  : ${START_FROM:-(none)}"
echo "=================================================="

if $DRY_RUN; then
    echo "[DRY RUN] Would submit the following srun command:"
    echo
    cat <<DRY
srun --ntasks-per-node=1 \\
     --container-image=${CONT_BETA} \\
     --container-mounts=${MOUNTS} \\
     --container-writable \\
     bash -c "set -e
       cd ${CURATOR}
       export PYTHONUNBUFFERED=1
       export PYTHONPATH=${CURATOR}:\${PYTHONPATH:-}
       export HF_HOME=${WORK}/hf_cache
       export UTMOSV2_CHACHE=${WORK}/utmosv2_cache
       export AIS_ENDPOINT=${AIS_ENDPOINT}
       export AIS_AUTHN_TOKEN=<redacted>
       export RAY_TMPDIR=/tmp/ray_\${SLURM_JOB_ID}
       export RAY_PORT_BROADCAST_DIR=${RAY_PORT_BROADCAST_DIR}
       /opt/conda/bin/python ${CURATOR}/tutorials/audio/hifi_pipeline/run_pipeline_beta.py \\
           --slurm \\
           ${INPUT_FLAG} \\
           ${DATA_CFG_FLAG} \\
           --output_dir ${OUTPUT_DIR} \\
           --num_shards ${NUM_SHARDS} \\
           --scotch_preset ${SCOTCH_PRESET} \\
           ${SED_FLAG} \\
           ${SEGEVENTS_FLAG} \\
           ${START_FLAG} \\
           ${STAGES_FLAG}"
DRY
    exit 0
fi

# GPU binding for the srun step.  ${SLURM_GPUS_PER_NODE:-0} > 0 means the
# allocation has GPUs; pass them explicitly so Ray inside the container
# detects nvidia devices (without this srun runs with no GPU binding even
# though sbatch allocated GPUs to the job).
GPU_BIND=""
if [[ -n "${SLURM_GPUS_PER_NODE:-}" && "${SLURM_GPUS_PER_NODE:-0}" != "0" ]]; then
    GPU_BIND="--gpus-per-task=${SLURM_GPUS_PER_NODE}"
fi

srun \
    --ntasks-per-node=1 \
    ${GPU_BIND} \
    --container-image="${CONT_BETA}" \
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
    | sed \"s/^/  [\$(hostname)] GPU /\" || echo \"  [\$(hostname)] no GPUs\"

# cosmos_xenna's resource discovery needs pynvml; ensure it's present.
/opt/conda/bin/python -c \"import pynvml\" 2>/dev/null || /opt/conda/bin/python -m pip install --quiet --no-input pynvml || true
# Refresh /opt/Curator (baked into actor sys.path) with the Lustre tree.
# Symlink rather than rsync because the vLLM container lacks rsync.
if [[ -d '${CURATOR}/nemo_curator' && -d /opt/Curator ]]; then
    rm -rf /opt/Curator/nemo_curator
    ln -s '${CURATOR}/nemo_curator' /opt/Curator/nemo_curator
fi

/opt/conda/bin/python '${CURATOR}/tutorials/audio/hifi_pipeline/run_pipeline_beta.py' \
    --slurm \
    ${INPUT_FLAG} \
    ${DATA_CFG_FLAG} \
    --output_dir '${OUTPUT_DIR}' \
    --num_shards ${NUM_SHARDS} \
    --scotch_preset ${SCOTCH_PRESET} \
    ${SED_FLAG} \
    ${SEGEVENTS_FLAG} \
    ${START_FLAG} \
    ${STAGES_FLAG}
"

echo "=================================================="
echo "  HIFI Beta Pipeline DONE"
echo "=================================================="
