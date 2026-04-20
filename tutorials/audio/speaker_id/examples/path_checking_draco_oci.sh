#!/usr/bin/env bash
set -euo pipefail

# Preflight path checker for DRACO-OCI speaker-id run.
# Usage:
#   bash examples/path_checking_draco_oci.sh

DATASET="${DATASET:-yodas}"
LANGUAGES="${LANGUAGES:-bg cs da de el en es et fi fr hr hu it lt nl pl pt ro ru sk sv uk}"

# See run_on_draco-oci-ord_example.sh for the up-to-date Curator layout note.
CODE_DIR="/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/users/taejinp/projects/speaker_id_for_asr_data"
BASE_DIR="/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/Yodas"
MODEL_DIR="/lustre/fs12/portfolios/llmservice/users/taejinp/projects/Granary_spkIDs/models/voxceleb_resnet293_LM"
RESULTS_DIR="/lustre/fs12/portfolios/llmservice/users/taejinp/projects/Granary_spkIDs/${DATASET}"

MISSING=0

check_dir() {
  local p="$1"
  if [ -d "$p" ]; then
    echo "OK dir:     $p"
  else
    echo "MISSING dir: $p"
    MISSING=$((MISSING + 1))
  fi
}

check_file() {
  local p="$1"
  if [ -f "$p" ]; then
    echo "OK file:    $p"
  else
    echo "MISSING file: $p"
    MISSING=$((MISSING + 1))
  fi
}

echo "=== DRACO-OCI Path Preflight ==="
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Dataset: ${DATASET}"
echo "Languages: ${LANGUAGES}"
echo

echo "[1/4] Core directories"
check_dir "$CODE_DIR"
check_dir "$BASE_DIR"
check_dir "$MODEL_DIR"
check_dir "$RESULTS_DIR"
echo

echo "[2/4] Required files"
check_file "$CODE_DIR/run_pipeline.py"
check_file "$CODE_DIR/requirements.txt"
check_file "$MODEL_DIR/avg_model.pt"
check_file "$MODEL_DIR/config.yaml"
echo

echo "[3/4] Language directories under BASE_DIR"
for lang in ${LANGUAGES}; do
  check_dir "$BASE_DIR/$lang"
done
echo

echo "[4/4] Optional runtime commands"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "OK cmd:     nvidia-smi"
else
  echo "MISSING cmd: nvidia-smi"
  MISSING=$((MISSING + 1))
fi

if command -v python >/dev/null 2>&1; then
  echo "OK cmd:     python ($(python --version 2>&1))"
else
  echo "MISSING cmd: python"
  MISSING=$((MISSING + 1))
fi
echo

if [ "$MISSING" -eq 0 ]; then
  echo "PASS: all checks succeeded."
  exit 0
fi

echo "FAIL: ${MISSING} check(s) failed."
exit 1
