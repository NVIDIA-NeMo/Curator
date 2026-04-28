#!/bin/bash
# Smoke chain: ASR pipeline (qwen_omni_inprocess) -> beta pipeline.
#
# Runs two sbatch jobs:
#   1. ASR job  — examples/audio/qwen_omni_inprocess/run_pipeline.py with
#                 vLLM Qwen3-Omni (TP=2), writes sharded manifests under
#                 ${WORK}/asr_smoke/<corpus>/sharded/.
#   2. Beta job — submit_beta.sh --start-from sed, INPUT_MANIFEST_OVERRIDE
#                 pointing at the ASR output dir.  Beta runs sed -> sed_post
#                 -> segment -> diarize -> embed -> utmos2 -> cluster_scotch.
#
# The two jobs are chained via Slurm --dependency=afterok, so beta only
# runs if ASR finishes cleanly.
#
# Usage (from cluster login node):
#   sbatch tutorials/audio/hifi_pipeline/submit_asr_then_beta.sh \
#       --corpus ytc_ru                  # smallest corpus, 64 shards
#   bash tutorials/audio/hifi_pipeline/submit_asr_then_beta.sh \
#       --corpus ytc_ru --dry-run        # preview only
#
# Prereqs (on cluster):
#   - curator-hifi-pipeline.sqsh in $SQSH_DIR (vLLM image, for ASR step)
#   - curator-hifi-beta.sqsh     in $SQSH_DIR (NeMo + UTMOSv2, for beta step)
#   - ${MANIFESTS_DIR}, $WORK, $CURATOR, $ACCOUNT, $SQSH_DIR env vars set
#   - ~/.ais_token sourced (sets AIS_AUTHN_TOKEN)
#
# Output collision-safe: writes to ${WORK}/asr_smoke/<sub>/ and
# ${WORK}/hifi_beta_output/<sub>/ — never touches ${WORK}/e2e_output/.

set -euo pipefail

# ---- args ----
CORPUS=""
DRY_RUN=false
SMOKE_YAML=""
HALL_PHRASES=""
REGEX_YAML=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --corpus) CORPUS="$2"; shift 2;;
        --smoke-yaml) SMOKE_YAML="$2"; shift 2;;
        --hall-phrases) HALL_PHRASES="$2"; shift 2;;
        --regex-yaml) REGEX_YAML="$2"; shift 2;;
        --dry-run) DRY_RUN=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done
[[ -z "$CORPUS" ]] && { echo "Usage: $0 --corpus <name> [--dry-run]"; exit 1; }

# ---- env ----
WORK="${WORK:?Set WORK}"
CURATOR="${CURATOR:?Set CURATOR}"
SQSH_DIR="${SQSH_DIR:?Set SQSH_DIR}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT}"
MANIFESTS_DIR="${MANIFESTS_DIR:?Set MANIFESTS_DIR (Granary manifests root)}"

if [[ -f "${HOME}/.ais_token" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/.ais_token"
fi
: "${AIS_ENDPOINT:=http://asr.iad.oci.aistore.nvidia.com:51080}"
: "${AIS_AUTHN_TOKEN:=}"

CONFIG_DIR="${CURATOR}/tutorials/audio/hifi_pipeline/configs"
SMOKE_YAML="${SMOKE_YAML:-${CONFIG_DIR}/granary_ru_smoke.resolved.yaml}"
HALL_PHRASES="${HALL_PHRASES:-${CONFIG_DIR}/hall_phrases.txt}"
REGEX_YAML="${REGEX_YAML:-${CONFIG_DIR}/regex_min.yaml}"

# Resolve granary YAML (substitute MANIFESTS_DIR placeholder) on the fly.
if [[ ! -f "$SMOKE_YAML" ]]; then
    SRC_YAML="${CONFIG_DIR}/granary_ru_smoke.yaml"
    [[ ! -f "$SRC_YAML" ]] && { echo "ERROR: $SRC_YAML missing"; exit 1; }
    echo "[setup] resolving $SRC_YAML -> $SMOKE_YAML"
    if ! $DRY_RUN; then
        MANIFESTS_DIR="$MANIFESTS_DIR" envsubst < "$SRC_YAML" > "$SMOKE_YAML"
    fi
fi

# ---- corpus -> sub mapping (mirrors submit_beta.sh) ----
declare -A CORPUS_SUB
CORPUS_SUB[ytc_ru]="ytc/ru"
CORPUS_SUB[yodas_0fc_ru]="yodas/0_from_captions/ru"
CORPUS_SUB[yodas_0bw_ru]="yodas/0_by_whisper/ru"
CORPUS_SUB[yodas_1bw_ru]="yodas/1_by_whisper/ru"
SUB="${CORPUS_SUB[$CORPUS]:-}"
[[ -z "$SUB" ]] && { echo "Unknown corpus: $CORPUS"; exit 1; }

ASR_OUTPUT_DIR="${WORK}/asr_smoke/${SUB}"
ASR_SHARDED_DIR="${ASR_OUTPUT_DIR}/sharded"
LOG_DIR="${WORK}/hifi_beta_logs/asr_smoke_${CORPUS}"

# ---- containers ----
CONT_ASR="${CONT_ASR:-${SQSH_DIR}/curator-hifi-pipeline.sqsh}"
CONT_BETA="${CONT_BETA:-${SQSH_DIR}/curator-hifi-beta.sqsh}"
if ! $DRY_RUN; then
    [[ ! -f "$CONT_ASR" ]]  && { echo "ERROR: missing $CONT_ASR";  exit 1; }
    [[ ! -f "$CONT_BETA" ]] && { echo "ERROR: missing $CONT_BETA"; exit 1; }
fi

MOUNTS="${CONTAINER_MOUNTS:-/lustre/fs11:/lustre/fs11,/lustre/fsw:/lustre/fsw}"
if ! $DRY_RUN; then
    mkdir -p "$LOG_DIR" "$ASR_OUTPUT_DIR"
fi

# ---- Step 1: ASR sbatch ----
# In dry-run, write to a temp file so we don't touch real cluster paths
# AND replace the AIS token with <redacted> so it doesn't leak.
if $DRY_RUN; then
    ASR_SBATCH="$(mktemp -t asr.XXXXXX.sbatch)"
    EFFECTIVE_AIS_TOKEN='<redacted>'
else
    ASR_SBATCH="${LOG_DIR}/asr.sbatch"
    EFFECTIVE_AIS_TOKEN="${AIS_AUTHN_TOKEN}"
fi
# Sbatch file may contain the AIS token; restrict to user only.
umask 077

# curator-hifi-pipeline image uses the venv python, not conda — see SUBMISSION_GUIDE.md.
if [[ "$CONT_ASR" == *curator-hifi-pipeline* ]]; then
    PY_INTERP="/usr/bin/python3"
    EXTRA_PYPATH=":/opt/curator_venv/lib/python3.10/site-packages"
else
    PY_INTERP="/opt/conda/bin/python"
    EXTRA_PYPATH=""
fi

cat > "$ASR_SBATCH" <<EOFASR
#!/bin/bash
#SBATCH --job-name=asr-smoke-${CORPUS}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=batch_singlenode
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/asr_%j.log
#SBATCH --error=${LOG_DIR}/asr_%j.log

set -euo pipefail
# Per SUBMISSION_GUIDE.md §8: source ~/.ais_token at the top of the
# sbatch so AIS_AUTHN_TOKEN is correct in the job env even when the
# submitting shell had a stale value (otherwise we get 401 missing
# 'sub' claim).
[[ -f \$HOME/.ais_token ]] && source \$HOME/.ais_token

srun --gpus-per-task=2 \\
     --container-image=${CONT_ASR} \\
     --container-mounts=${MOUNTS} \\
     --container-writable \\
     bash -c "
set -e
export PYTHONUNBUFFERED=1
export PYTHONPATH=${CURATOR}${EXTRA_PYPATH}:\${PYTHONPATH:-}
export HF_HOME=${WORK}/hf_cache
export AIS_ENDPOINT=${AIS_ENDPOINT}
# Re-source AIS token inside the container too.  The outer sbatch
# already sourced it (so AIS_AUTHN_TOKEN is in the env), this is a
# belt-and-suspenders override for any baked stale value.
[[ -f \$HOME/.ais_token ]] && source \$HOME/.ais_token
export AIS_AUTHN_TOKEN

# pynvml is required by cosmos_xenna's GPU resource discovery — without
# it the worker pool is built with gpus=0 even when CUDA_VISIBLE_DEVICES
# is set.  curator-hifi-pipeline.sqsh ships without it; install at
# runtime (cheap, ~3s, --container-writable allows site-packages writes).
${PY_INTERP} -c \"import pynvml\" 2>/dev/null || \\
    ${PY_INTERP} -m pip install --quiet --no-input pynvml || true

# Refresh /opt/Curator inside the container with the latest source from
# Lustre.  Ray actors put /opt/Curator on sys.path (it's the container's
# baked code) and runtime_env env_vars can't override sys.path (Python
# reads PYTHONPATH at interpreter startup, before runtime_env applies).
# We replace /opt/Curator/nemo_curator with a symlink to the Lustre tree
# so actors always see up-to-date code.  rsync isn't installed in the
# vLLM container, so we use rm + ln; both are universally available.
if [[ -d \"${CURATOR}/nemo_curator\" && -d /opt/Curator ]]; then
    rm -rf /opt/Curator/nemo_curator
    ln -s \"${CURATOR}/nemo_curator\" /opt/Curator/nemo_curator
fi

# Use load_ais_and_run.py to load AIS_AUTHN_TOKEN from .ais_token file
# directly into os.environ before the pipeline imports anything.  Bash's
# 'export' silently fails to carry the freshly-sourced token across to
# python's os.environ in this container — sanity 9451754 confirmed
# the file-read approach gives driver=228, actor=228, valid tar bytes.
${PY_INTERP} ${CURATOR}/tutorials/audio/hifi_pipeline/scripts/load_ais_and_run.py \\
    ${CURATOR}/examples/audio/qwen_omni_inprocess/run_pipeline.py \\
    --data_config ${SMOKE_YAML} \\
    --output_dir ${ASR_SHARDED_DIR} \\
    --tensor_parallel_size 2 \\
    --batch_size 16 \\
    --hall_phrases ${HALL_PHRASES} \\
    --regex_yaml ${REGEX_YAML} \\
    --target_lang ru \\
    --skip_pnc
"
EOFASR

# ---- Step 2: beta sbatch (depends on ASR) ----
BETA_DRIVER="${CURATOR}/tutorials/audio/hifi_pipeline/submit_beta.sh"

# We invoke submit_beta.sh as a separate sbatch with --dependency=afterok.
# INPUT_MANIFEST_OVERRIDE points at the ASR pipeline's sharded output dir.

echo "=================================================="
echo "  ASR -> Beta smoke chain"
echo "=================================================="
echo "  Corpus       : ${CORPUS}"
echo "  Granary YAML : ${SMOKE_YAML}"
echo "  ASR output   : ${ASR_SHARDED_DIR}"
echo "  Beta output  : ${WORK}/hifi_beta_output/${SUB}"
echo "  ASR sbatch   : ${ASR_SBATCH}"
echo "  Logs         : ${LOG_DIR}"
echo "=================================================="

if $DRY_RUN; then
    echo "[DRY RUN] ASR sbatch:"
    sed 's/^/    /' "$ASR_SBATCH"
    rm -f "$ASR_SBATCH"
    echo
    echo "[DRY RUN] beta would be invoked as:"
    echo "    sbatch --dependency=afterok:<asr_jobid> --account=${ACCOUNT} \\"
    echo "           --nodes=1 --gpus-per-node=2 --time=04:00:00 \\"
    echo "           ${BETA_DRIVER} --corpus ${CORPUS} --start-from sed"
    echo "    (with INPUT_MANIFEST_OVERRIDE=${ASR_SHARDED_DIR})"
    exit 0
fi

ASR_JOBID=$(sbatch --parsable "$ASR_SBATCH")
echo "ASR job submitted: ${ASR_JOBID}"

INPUT_MANIFEST_OVERRIDE="${ASR_SHARDED_DIR}" \
sbatch --dependency=afterok:${ASR_JOBID} \
       --account=${ACCOUNT} \
       --nodes=1 --gpus-per-node=2 --time=04:00:00 \
       "${BETA_DRIVER}" \
       --corpus "${CORPUS}" --start-from sed
echo "Beta job submitted (chained on ASR=${ASR_JOBID})"
