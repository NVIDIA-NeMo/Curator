#!/bin/bash
# Entrypoint for HIFI data curation pipeline container.
#
# Starts vLLM server (system Python) in background, waits for health,
# then runs the Curator pipeline (venv Python).
#
# Environment variables:
#   VLLM_MODEL         - Omni model (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)
#   VLLM_PORT          - Server port (default: 8200)
#   VLLM_DTYPE         - Data type (default: bfloat16)
#   NUM_GPU            - Tensor parallel size (default: 1)
#   HEALTH_TIMEOUT     - Health check timeout (default: 1200)
#   SKIP_VLLM          - Set to "1" to skip vLLM server (use external)

set -euo pipefail

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8200}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
NUM_GPU="${NUM_GPU:-1}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"
SKIP_VLLM="${SKIP_VLLM:-0}"

cleanup() {
    if [ -n "${VLLM_PID:-}" ]; then
        echo "[entrypoint] Shutting down vLLM (PID $VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# --- Start vLLM (unless skipped) ---
if [ "$SKIP_VLLM" != "1" ]; then
    echo "[entrypoint] Starting vLLM: model=$VLLM_MODEL port=$VLLM_PORT tp=$NUM_GPU"
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL" \
        --port "$VLLM_PORT" \
        --host 0.0.0.0 \
        --dtype "$VLLM_DTYPE" \
        --max-model-len "$VLLM_MAX_MODEL_LEN" \
        --allowed-local-media-path / \
        -tp "$NUM_GPU" &
    VLLM_PID=$!

    echo "[entrypoint] Waiting for vLLM (timeout: ${HEALTH_TIMEOUT}s)..."
    ELAPSED=0
    while [ "$ELAPSED" -lt "$HEALTH_TIMEOUT" ]; do
        if curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
            echo "[entrypoint] vLLM ready after ${ELAPSED}s"
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[entrypoint] ERROR: vLLM exited"
            wait "$VLLM_PID" || true
            exit 1
        fi
        sleep 2
        ELAPSED=$((ELAPSED + 2))
    done
    if [ "$ELAPSED" -ge "$HEALTH_TIMEOUT" ]; then
        echo "[entrypoint] ERROR: vLLM timeout"
        exit 1
    fi
else
    echo "[entrypoint] Skipping vLLM (SKIP_VLLM=1)"
fi

# --- Run Curator pipeline ---
# Use PYTHONPATH to prefer /opt/Curator (repo copy with prompt YAMLs)
# over the pip-installed package (which lacks data files).
export PYTHONPATH=/opt/Curator:$PYTHONPATH
echo "[entrypoint] Running HIFI pipeline..."
/opt/curator_venv/bin/python /opt/Curator/tutorials/audio/hifi_pipeline/run_pipeline.py \
    --vllm_host localhost \
    --vllm_port "$VLLM_PORT" \
    --omni_model "$VLLM_MODEL" \
    "$@"

echo "[entrypoint] Pipeline complete."
