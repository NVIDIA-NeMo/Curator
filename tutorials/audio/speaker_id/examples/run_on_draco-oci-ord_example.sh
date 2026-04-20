#!/bin/bash
#SBATCH -A llmservice_nemo_speechlm
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH --time-min 02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0
#SBATCH -J "llmservice_nemo_speechlm-speaker_id"

#################### User / Cluster Config ####################

USER_ID=taejinp
CLUSTER_TAG='DRACO-OCI-ORD'

DFS_PREFIX=/lustre/fs12/portfolios/llmservice
PROJECT_DIR=${DFS_PREFIX}/projects/llmservice_nemo_speechlm
USER_DIR=${DFS_PREFIX}/users/${USER_ID}

#################### Speaker-ID Pipeline Config ####################
# YODAS Languages: "bg cs da de el en es et fi fr hr hu it lt nl pl pt ro ru sk sv uk"
# YTC Languages: "bg cs da de el en es et fi fr hr hu it lt lv nl pl pt ro ru sk sl sv uk"

# Done languages: bg cs da el et fi hr hu lt ro sk sv
LANGUAGES="de en es fr it nl pl pt ru uk"
DATASET="yodas"

# Done languages: bg cs da el 
LANGUAGES="de en es et fi fr hr hu it lt lv nl pl pt ro ru sk sl sv uk"
DATASET="ytc"
NUM_GPUS=8
BATCH_DUR=800
WORKERS=16
PREFETCH_TARS=4
MODEL_CHECKPOINT_DIR=/lustre/fs12/portfolios/llmservice/users/taejinp/projects/Granary_spkIDs/models/voxceleb_resnet293_LM
CORPUSVIEW_PATH=/lustre/fs12/portfolios/llmservice/users/taejinp/projects/corpusview
S3CFG_PATH=/lustre/fs12/portfolios/llmservice/users/taejinp/.s3cfg
S3CFG_PROFILE=default
S3CFG_ARG="${S3CFG_PATH}[${S3CFG_PROFILE}]"

WAV_DATA_SAVE_DIR=/lustre/fs12/portfolios/llmservice/users/taejinp/projects/Granary_spkIDs/wav_files/${DATASET}
# CODE_DIR points at a Draco-OCI clone of this codebase.  As of Apr 2026 the
# pipeline lives inside Curator at nemo_curator/stages/audio/speaker_id/, so
# CODE_DIR should point at a Curator checkout instead (the YODAS driver is at
# ${CODE_DIR}/nemo_curator/stages/audio/speaker_id/run_pipeline.py).
CODE_DIR=/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/taejinp/projects/speaker_id_for_asr_data
RESULTS_DIR=/lustre/fs12/portfolios/llmservice/users/taejinp/projects/Granary_spkIDs/${DATASET}

mkdir -p ${WAV_DATA_SAVE_DIR}

mkdir -p ${RESULTS_DIR}

#################### Container & Mounts ####################

CONTAINER=/lustre/fs12/portfolios/llmservice/users/weiqingw/local_containers/nvidia+nemo+25.04.rc2_fixed.sqsh
MOUNTS="--container-mounts=/lustre:/lustre"

OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

#################### Build Command ####################

read -r -d '' cmd <<EOF
set -euxo pipefail \
&& echo "******** Speaker-ID Pipeline ********" \
&& echo "CODE_DIR=${CODE_DIR}" \
&& echo "WAV_DATA_SAVE_DIR=${WAV_DATA_SAVE_DIR}" \
&& echo "MODEL_CHECKPOINT_DIR=${MODEL_CHECKPOINT_DIR}" \
&& echo "S3CFG_ARG=${S3CFG_ARG}" \
&& nvidia-smi \
&& [ -f "${CODE_DIR}/run_pipeline.py" ] || { echo "ERROR: Missing ${CODE_DIR}/run_pipeline.py"; exit 1; } \
&& [ -f "${CODE_DIR}/requirements.txt" ] || { echo "ERROR: Missing ${CODE_DIR}/requirements.txt"; exit 1; } \
&& [ -d "${WAV_DATA_SAVE_DIR}" ] || { echo "ERROR: Missing WAV_DATA_SAVE_DIR ${WAV_DATA_SAVE_DIR}"; exit 1; } \
&& [ -d "${MODEL_CHECKPOINT_DIR}" ] || { echo "ERROR: Missing MODEL_CHECKPOINT_DIR ${MODEL_CHECKPOINT_DIR}"; exit 1; } \
&& [ -f "${MODEL_CHECKPOINT_DIR}/avg_model.pt" ] || { echo "ERROR: Missing ${MODEL_CHECKPOINT_DIR}/avg_model.pt"; exit 1; } \
&& [ -f "${MODEL_CHECKPOINT_DIR}/config.yaml" ] || { echo "ERROR: Missing ${MODEL_CHECKPOINT_DIR}/config.yaml"; exit 1; } \
&& [ -f "${S3CFG_PATH}" ] || { echo "ERROR: Missing S3 config ${S3CFG_PATH}"; exit 1; } \
&& cd ${CODE_DIR} \
&& pip install --no-deps --force-reinstall wespeaker@git+https://github.com/tango4j/wespeaker.git \
&& pip install -r requirements.txt \
&& python -u run_pipeline.py \
    --dataset ${DATASET} \
    --languages ${LANGUAGES} \
    --base-dir ${WAV_DATA_SAVE_DIR}/ \
    --result-dir ${RESULTS_DIR} \
    --model-checkpoint ${MODEL_CHECKPOINT_DIR} \
    --corpusview-path ${CORPUSVIEW_PATH} \
    --s3cfg "${S3CFG_ARG}" \
    --num-gpus ${NUM_GPUS} \
    --batch-dur ${BATCH_DUR} \
    --workers ${WORKERS} \
    --streaming \
    --no-cluster \
    --no-confidence \
    --prefetch-tars ${PREFETCH_TARS}
EOF

#################### Launch ####################

echo "OUTFILE: $OUTFILE"
echo "ERRFILE: $ERRFILE"
echo "CONTAINER: $CONTAINER"
echo "MOUNTS: $MOUNTS"
echo "Command: ${cmd}"
srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash --noprofile --norc -c "${cmd}"
