#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source before starting Ray in either container.

while IFS= read -r full_english_name; do
  case "$full_english_name" in
    SLURM_*|OMPI_*|PMI_*|PMIX_*|HYDRA_*|I_MPI_*|MPI_*|MPICH_*|MV2_*|WORLD_SIZE|RANK|LOCAL_RANK|LOCAL_WORLD_SIZE|GROUP_RANK|ROLE_RANK|MASTER_ADDR|MASTER_PORT)
      unset "$full_english_name" ;;
  esac
done < <(compgen -e)
unset RAY_ADDRESS RAY_AUTH_TOKEN UV_PYTHON UV_VENV_NAME UV_VENV_PATH UV_PYTHON_INSTALL_DIR VIRTUAL_ENV
source /opt/venv/env.sh
export RAY_MAX_LIMIT_FROM_API_SERVER=40000 RAY_MAX_LIMIT_FROM_DATA_SOURCE=40000
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONFAULTHANDLER=1 RAY_DEDUP_LOGS=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAY_AUTH_MODE=token
export RAY_AUTH_TOKEN_PATH="$(mktemp /tmp/full-english-ray-auth.XXXXXXXX)"
chmod 600 "$RAY_AUTH_TOKEN_PATH"
python - <<'PY'
import os
import secrets
from pathlib import Path

Path(os.environ['RAY_AUTH_TOKEN_PATH']).write_text(secrets.token_hex(32))
PY
trap 'rm -f -- "$RAY_AUTH_TOKEN_PATH"' EXIT
unset full_english_name
