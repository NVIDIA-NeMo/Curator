#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Build file list if it doesn't exist
if [ ! -f file_list.json ]; then
    echo ">>> Building file_list.json ..."
    conda run --no-capture-output -n aec_py311 python build_file_list.py
fi

echo ">>> Running QA checks ..."
PYTHONUNBUFFERED=1 conda run --no-capture-output -n aec_py311 python run_qa_checks.py \
    --file_list file_list.json \
    --output qa_results.json

echo ">>> Done. Results in qa_results.json"
