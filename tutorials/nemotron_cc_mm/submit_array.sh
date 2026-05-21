#!/usr/bin/env bash
# =============================================================================
# submit_array.sh — SLURM-array driver for the WARC/Parquet pipeline.
#
# Modes:
#   submit          fresh sbatch for indices 0..ARRAY_SIZE-1
#   status          list completed / missing shards (from _SUCCESS markers)
#   retry-missing   resubmit only the missing shard indices
#   worker          per-shard srun entrypoint (called by sbatch)
#
# Required env (all modes):
#   OUTPUT_PATH     where to write per-shard output + _SUCCESS markers
#   PRESET          extract | text_filter | image_acquire | image_quality |
#                   omnicorpus | omnicorpus_cpu | omnicorpus_text_only
#   ARRAY_SIZE      number of shards (one WARC or one prior-stage idx_NN each)
#
# Input contract — choose ONE based on the preset's input type:
#   When INPUT_TYPE=warc (default; for `extract`, `omnicorpus*`):
#     WARC_DIR      directory containing the WARC files
#     WARC_PATTERN  sprintf pattern with one %d slot, indexed by shard
#                   e.g. "CC-MAIN-20250612112840-20250612142840-%05d.warc.gz"
#   When INPUT_TYPE=parquet (for `text_filter`, `image_acquire`, `image_quality`):
#     INPUT_PATH    output root of a PRIOR stage group; each shard reads
#                   $INPUT_PATH/idx_<padded>/  (a Parquet directory)
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
# Example — 4-group chained pipeline on 50 WARCs:
#   ROOT=$USER_DIR/out/50warcs_chained
#   # group 1: WARC → extracted Parquet
#   PRESET=extract INPUT_TYPE=warc \
#   WARC_DIR=… WARC_PATTERN=… OUTPUT_PATH=$ROOT/01_extract \
#   ARRAY_SIZE=50 ./submit_array.sh submit
#
#   # group 2: extracted → text-filtered
#   PRESET=text_filter INPUT_TYPE=parquet INPUT_PATH=$ROOT/01_extract \
#   OUTPUT_PATH=$ROOT/02_text ARRAY_SIZE=50 ./submit_array.sh submit
#
#   # group 3: text-filtered → with downloaded images
#   PRESET=image_acquire INPUT_TYPE=parquet INPUT_PATH=$ROOT/02_text \
#   OUTPUT_PATH=$ROOT/03_images ARRAY_SIZE=50 ./submit_array.sh submit
#
#   # group 4: with images → final (NSFW + aesthetic on GPU)
#   PRESET=image_quality INPUT_TYPE=parquet INPUT_PATH=$ROOT/03_images \
#   OUTPUT_PATH=$ROOT/04_quality ARRAY_SIZE=50 \
#   PARTITION=batch ./submit_array.sh submit
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_DIR=/scratch/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/aot

# Required ----------------------------------------------------------------
: "${INPUT_TYPE:=warc}"

require_env() {
    local missing=()
    for v in OUTPUT_PATH PRESET ARRAY_SIZE; do
        [[ -z "${!v:-}" ]] && missing+=("$v")
    done
    if [[ "${INPUT_TYPE}" == "warc" ]]; then
        for v in WARC_DIR WARC_PATTERN; do
            [[ -z "${!v:-}" ]] && missing+=("$v (required for INPUT_TYPE=warc)")
        done
    elif [[ "${INPUT_TYPE}" == "parquet" ]]; then
        [[ -z "${INPUT_PATH:-}" ]] && missing+=("INPUT_PATH (required for INPUT_TYPE=parquet)")
    else
        missing+=("INPUT_TYPE must be 'warc' or 'parquet' (got: ${INPUT_TYPE})")
    fi
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
    export INPUT_TYPE OUTPUT_PATH PRESET LID_PATH MODEL_DIR \
        CONTAINER_IMAGE USER_DIR SCRIPT_DIR PARTITION
    [[ "${INPUT_TYPE}" == "warc" ]] && export WARC_DIR WARC_PATTERN
    [[ "${INPUT_TYPE}" == "parquet" ]] && export INPUT_PATH

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

    local pad input_path
    pad="$(printf '%05d' "${CURATOR_SHARD_INDEX}")"

    # Resolve this shard's input based on INPUT_TYPE.
    case "${INPUT_TYPE:-warc}" in
        warc)
            # shellcheck disable=SC2059
            input_path="${WARC_DIR%/}/$(printf "${WARC_PATTERN}" "${CURATOR_SHARD_INDEX}")"
            ;;
        parquet)
            # Prior stage wrote to $INPUT_PATH/idx_<pad>/  → use it as the
            # Parquet directory for this shard.
            input_path="${INPUT_PATH%/}/idx_${pad}"
            ;;
        *)
            echo "Unknown INPUT_TYPE: ${INPUT_TYPE}" >&2
            exit 2
            ;;
    esac

    local task_out="${OUTPUT_PATH%/}/idx_${pad}"
    # Keep --log-path *outside* the output dir so the writer's --mode overwrite
    # rmtree doesn't delete it.  The _logs/ dir was created by submit.
    local task_log="${OUTPUT_PATH%/}/_logs/idx_${pad}.log"
    mkdir -p "$(dirname "$task_log")"

    echo "Worker shard ${CURATOR_SHARD_INDEX}/${CURATOR_NUM_SHARDS} (input_type=${INPUT_TYPE})"
    echo "  INPUT:  ${input_path}"
    echo "  OUTPUT: ${task_out}"

    # NSFW + aesthetic model dirs are only needed when those stages run.
    local extra_args=()
    if [[ "${PRESET}" == omnicorpus || "${PRESET}" == image_quality ]]; then
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
                --input-type ${INPUT_TYPE} \
                --input-path ${input_path} \
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
