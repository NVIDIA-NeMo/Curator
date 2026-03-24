#!/bin/bash
# Entrypoint for the single-container Qwen3-Omni + Curator setup.
#
# Starts vLLM server (using system Python) in the background, waits for
# it to become healthy, then runs the Curator pipeline (using the venv
# Python).  All additional arguments are forwarded to run_qwen3.py.
#
# Environment variables (with defaults):
#   VLLM_MODEL           - Model ID (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)
#   VLLM_PORT            - Server port (default: 8200)
#   VLLM_DTYPE           - Data type (default: bfloat16)
#   VLLM_MAX_MODEL_LEN   - Max model length (default: 65536)
#   NUM_GPU              - Tensor parallel size (default: 1)
#   HEALTH_TIMEOUT       - Health check timeout in seconds (default: 600)

set -euo pipefail

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8200}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
NUM_GPU="${NUM_GPU:-1}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"

# ------------------------------------------------------------------
# Cleanup: kill vLLM on any exit (normal, error, or signal)
# ------------------------------------------------------------------
cleanup() {
    if [ -n "${VLLM_PID:-}" ]; then
        echo "[entrypoint] Shutting down vLLM server (PID $VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------
# 1. Start vLLM using system Python (NOT the Curator venv)
# ------------------------------------------------------------------
echo "[entrypoint] Starting vLLM server: model=$VLLM_MODEL port=$VLLM_PORT tp=$NUM_GPU"
python3 -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL" \
    --port "$VLLM_PORT" \
    --host 0.0.0.0 \
    --dtype "$VLLM_DTYPE" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --allowed-local-media-path / \
    -tp "$NUM_GPU" &
VLLM_PID=$!

# ------------------------------------------------------------------
# 2. Health check: poll /v1/models until ready
# ------------------------------------------------------------------
echo "[entrypoint] Waiting for vLLM to become healthy (timeout: ${HEALTH_TIMEOUT}s)..."
ELAPSED=0
while [ "$ELAPSED" -lt "$HEALTH_TIMEOUT" ]; do
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "[entrypoint] vLLM is ready after ${ELAPSED}s"
        break
    fi
    # Abort early if vLLM process died
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[entrypoint] ERROR: vLLM process exited prematurely"
        wait "$VLLM_PID" || true
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done
if [ "$ELAPSED" -ge "$HEALTH_TIMEOUT" ]; then
    echo "[entrypoint] ERROR: vLLM did not become healthy within ${HEALTH_TIMEOUT}s"
    exit 1
fi

# ------------------------------------------------------------------
# 3. Run Curator pipeline using the venv Python
# ------------------------------------------------------------------
echo "[entrypoint] Running Curator pipeline..."
/opt/curator_venv/bin/python /opt/Curator/tutorials/audio/qwen_omni/run_qwen3.py \
    --host localhost \
    --port "$VLLM_PORT" \
    --model-name "$VLLM_MODEL" \
    "$@"

echo "[entrypoint] Pipeline complete."
