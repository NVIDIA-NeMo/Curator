#!/usr/bin/env bash
# =============================================================================
# submit_array.sh — SLURM-array driver for the WARC pipeline.
#
# Modes:
#   submit          fresh sbatch for indices 0..ARRAY_SIZE-1
#   status          list completed / missing shards (from _SUCCESS markers)
#   retry-missing   resubmit only the missing shard indices
#   worker          per-shard srun entrypoint (called by sbatch)
#
# Required env (submit/status/retry):
#   WARC_DIR        directory containing the WARC files
#   WARC_PATTERN    sprintf pattern with one %s slot, indexed by shard
#                   e.g. "CC-MAIN-20250612112840-20250612142840-%05d.warc.gz"
#   OUTPUT_PATH     where to write per-shard output + _SUCCESS markers
#   PRESET          omnicorpus | omnicorpus_cpu | omnicorpus_text_only
#   ARRAY_SIZE      number of shards (one WARC each)
#
# Common knobs (all optional, sensible defaults):
#   PARTITION         cpu_short | batch (default: cpu_short)
#   ACCOUNT           default: nemotron_n4_pre
#   TIME_LIMIT        default: 01:00:00
#   CPUS_PER_TASK     default: 16
#   GPUS_PER_NODE     default: 0 on cpu_short, 1 on batch
#   MEM               default: 200G on cpu_short, 190G on batch
#   MAX_CONCURRENT    array %N throttle; default: ARRAY_SIZE
#   LID_PATH          path to lid.176.bin (default: $USER_DIR/models/lid.176.bin)
#   MODEL_DIR         path to NSFW + aesthetic models (default: $USER_DIR/models/curator)
#   CONTAINER_IMAGE   sqsh path (default: /home/aot/scratch/sqsh/curator_2604.sqsh)
#
# Example — 5-WARC CPU validation:
#   WARC_DIR=$USER_DIR/CC-MAIN-2025-26/segments/1749709481111.44/warc \
#   WARC_PATTERN="CC-MAIN-20250612112840-20250612142840-%05d.warc.gz" \
#   OUTPUT_PATH=$USER_DIR/out/5warcs_cpu \
#   PRESET=omnicorpus_cpu ARRAY_SIZE=5 \
#   ./submit_array.sh submit
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_DIR=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/aot

# Required ----------------------------------------------------------------
require_env() {
    local missing=()
    for v in WARC_DIR WARC_PATTERN OUTPUT_PATH PRESET ARRAY_SIZE; do
        [[ -z "${!v:-}" ]] && missing+=("$v")
    done
    if (( ${#missing[@]} > 0 )); then
        echo "Missing required env: ${missing[*]}" >&2
        exit 2
    fi
}

# Defaults ----------------------------------------------------------------
: "${PARTITION:=cpu_short}"
: "${ACCOUNT:=nemotron_n4_pre}"
: "${TIME_LIMIT:=01:00:00}"
: "${CPUS_PER_TASK:=16}"
: "${LID_PATH:=${USER_DIR}/models/lid.176.bin}"
: "${MODEL_DIR:=${USER_DIR}/models/curator}"
: "${CONTAINER_IMAGE:=/home/aot/scratch/sqsh/curator_2604.sqsh}"

# Partition-specific defaults
if [[ "$PARTITION" == batch* ]]; then
    : "${GPUS_PER_NODE:=1}"
    : "${MEM:=190G}"
else
    : "${GPUS_PER_NODE:=0}"
    : "${MEM:=200G}"
fi
: "${MAX_CONCURRENT:=${ARRAY_SIZE:-1}}"

# ---- Status / marker helpers -------------------------------------------

marker_path() {  # $1: shard index
    printf "%s/_SUCCESS/shard_%05d.json" "${OUTPUT_PATH%/}" "$1"
}

collect_status() {  # populates COMPLETED, MISSING arrays
    COMPLETED=(); MISSING=()
    for ((i = 0; i < ARRAY_SIZE; i++)); do
        if [[ -f "$(marker_path "$i")" ]]; then
            COMPLETED+=("$i")
        else
            MISSING+=("$i")
        fi
    done
}

print_status() {
    require_env
    collect_status
    echo "Output root: ${OUTPUT_PATH}"
    echo "Completed (${#COMPLETED[@]}/${ARRAY_SIZE}): ${COMPLETED[*]:-none}"
    echo "Missing   (${#MISSING[@]}/${ARRAY_SIZE}): ${MISSING[*]:-none}"
}

# ---- sbatch submission --------------------------------------------------

submit_array() {  # $1: array spec
    require_env
    mkdir -p "${OUTPUT_PATH%/}/_logs"

    # Propagate the original array size so sparse retries
    # (e.g. --array=3,7,12) still know N for sanity checks.
    export CURATOR_ORIGINAL_ARRAY_SIZE="${ARRAY_SIZE}"
    export WARC_DIR WARC_PATTERN OUTPUT_PATH PRESET LID_PATH MODEL_DIR \
        CONTAINER_IMAGE USER_DIR SCRIPT_DIR

    local sbatch_args=(
        "--account=${ACCOUNT}"
        "--partition=${PARTITION}"
        "--job-name=nemotron_n4_pre:ccmm-array"
        "--time=${TIME_LIMIT}"
        "--nodes=1" "--ntasks=1"
        "--cpus-per-task=${CPUS_PER_TASK}"
        "--mem=${MEM}"
        "--array=$1"
        "--output=${OUTPUT_PATH%/}/_logs/slurm-%A_%a.out"
        "--export=ALL,CURATOR_ORIGINAL_ARRAY_SIZE=${ARRAY_SIZE}"
    )
    [[ "${GPUS_PER_NODE}" -gt 0 ]] && sbatch_args+=("--gpus-per-node=${GPUS_PER_NODE}")
    [[ "${PARTITION}" == cpu* ]] && sbatch_args+=("--export=ALL,CURATOR_ORIGINAL_ARRAY_SIZE=${ARRAY_SIZE},NVIDIA_VISIBLE_DEVICES=void")

    echo "Submitting array $1"
    echo "  partition=${PARTITION} mem=${MEM} cpus=${CPUS_PER_TASK} gpus_per_node=${GPUS_PER_NODE}"
    echo "  output=${OUTPUT_PATH}"
    sbatch "${sbatch_args[@]}" "$0" worker
}

submit_new() {
    (( ARRAY_SIZE >= 1 )) || { echo "ARRAY_SIZE must be >= 1" >&2; exit 2; }
    submit_array "0-$((ARRAY_SIZE - 1))%${MAX_CONCURRENT}"
}

retry_missing() {
    require_env
    collect_status
    if (( ${#MISSING[@]} == 0 )); then
        echo "All ${ARRAY_SIZE} shards complete; nothing to retry."
        return 0
    fi
    local spec
    spec="$(IFS=, ; echo "${MISSING[*]}")%${MAX_CONCURRENT}"
    echo "Retrying missing shards: ${MISSING[*]}"
    submit_array "${spec}"
}

# ---- Worker (called by sbatch) ------------------------------------------

run_worker() {
    [[ -n "${OUTPUT_PATH:-}" ]] || { echo "OUTPUT_PATH must be exported into the worker" >&2; exit 2; }

    # Shard env for the python script
    export CURATOR_SHARD_INDEX="${CURATOR_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
    export CURATOR_NUM_SHARDS="${CURATOR_NUM_SHARDS:-${CURATOR_ORIGINAL_ARRAY_SIZE:-1}}"

    local pad
    pad="$(printf '%05d' "${CURATOR_SHARD_INDEX}")"
    local warc_file
    # shellcheck disable=SC2059
    warc_file="$(printf "${WARC_PATTERN}" "${CURATOR_SHARD_INDEX}")"
    local warc="${WARC_DIR%/}/${warc_file}"
    local task_out="${OUTPUT_PATH%/}/idx_${pad}"
    # Keep --log-path *outside* the output dir so the writer's --mode overwrite
    # rmtree doesn't delete it.  The _logs/ dir was created by submit.
    local task_log="${OUTPUT_PATH%/}/_logs/idx_${pad}.log"
    mkdir -p "$(dirname "$task_log")"

    echo "Worker shard ${CURATOR_SHARD_INDEX}/${CURATOR_NUM_SHARDS}"
    echo "  WARC: ${warc}"
    echo "  OUT:  ${task_out}"

    local extra_args=()
    if [[ "${PRESET}" == omnicorpus ]]; then  # full GPU pipeline
        extra_args+=("--image-nsfw-model-dir" "${MODEL_DIR}"
                     "--image-aesthetic-model-dir" "${MODEL_DIR}")
    fi

    local srun_export="ALL"
    [[ "${PARTITION:-}" == cpu* ]] && srun_export="ALL,NVIDIA_VISIBLE_DEVICES=void"

    srun --export="${srun_export}" \
         --container-image="${CONTAINER_IMAGE}" \
         --container-mounts=/scratch:/scratch,/home:/home \
         bash -c "
            set -euo pipefail
            cd ${USER_DIR}/codebase/Curator
            pip install --no-deps -e . >/dev/null 2>&1 || true
            cd ${USER_DIR}/codebase/Curator/tutorials/nemotron_cc_mm
            stdbuf -oL -eL python run_warc_pipeline.py \
                --preset ${PRESET} \
                --input-path ${warc} \
                --output-path ${task_out} \
                --mode overwrite \
                --lang-id-model ${LID_PATH} \
                --log-path ${task_log} \
                ${extra_args[*]} \
                2>&1 | tee -a ${task_log}
         "
}

# ---- Dispatch -----------------------------------------------------------

case "${1:-}" in
    submit)         submit_new ;;
    status)         print_status ;;
    retry-missing)  retry_missing ;;
    worker)         run_worker ;;
    -h|--help|"")
        sed -n '2,42p' "$0"
        ;;
    *)
        echo "Unknown mode: $1" >&2
        exit 2
        ;;
esac
