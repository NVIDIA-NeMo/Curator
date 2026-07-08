#!/bin/bash
# Submit the David AI RAM-by-session pipeline as a 200-shard SLURM array.
#
# Run from a healthy Draco login node (login-02 / login-03; avoid login-01
# during fs12 maintenance):
#   ssh draco-oci-login-02.draco-oci-iad.nvidia.com
#   bash /lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva/Curator/tutorials/audio/david_ai_redelivered_mfa/submit_ram_200nodes.sh
#
# Each array task owns a disjoint session shard (session_index % 200 == task_id)
# and skips sessions already flagged in .done/sessions, so resubmitting is safe
# and only reprocesses unfinished sessions.
set -euo pipefail

SCRIPT=/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva/Curator/tutorials/audio/david_ai_redelivered_mfa
cd "$SCRIPT"
export RAM_SCRIPT_DIR="$SCRIPT"
export CLUSTER_BASE=/lustre/fs12/portfolios/nemotron/projects/nemotron_speechprod_asr/users/ttimofeeva
export WORK_DIR="$CLUSTER_BASE/david_ai_mfa_workdir"
export LINK_WORK_DIR="$WORK_DIR"
export RAM_ARRAY_COUNT="${RAM_ARRAY_COUNT:-15}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-nemotron_speechprod_asr}"
export SLURM_PARTITION="${SLURM_PARTITION:-cpu_short}"
export CPUS="${CPUS:-96}"
export WORKERS=96
export MFA_NUM_JOBS=1     # 96 slots, oversubscribes hyperthreads to hide Lustre I/O waits
export TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
export JOB_NAME="${JOB_NAME:-david_ai_ram_session}"

bash run_david_ai_mfa_ram_session_cluster.sh
