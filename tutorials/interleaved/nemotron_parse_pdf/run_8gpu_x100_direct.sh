#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CURATOR_ROOT=${CURATOR_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
PYTHON=${PYTHON:-${CURATOR_ROOT}/.venv-parse35-cuda13/bin/python}
export PATH="${PYTHON%/*}:$PATH"

ROOT=${ROOT:?set ROOT to the benchmark output root}
RUN_DIR=${ROOT}/direct-r16-m45-s96-t8192-p16-w36-c64-nemotron-parse-pdf-x100
SERVER_LOG_DIR=${RUN_DIR}/server-logs
MODEL=${MODEL:?set MODEL to the Nemotron Parse 3.5 model directory}
PROXY=${PROXY:-${SCRIPT_DIR}/round_robin_proxy.py}
mkdir -p "${SERVER_LOG_DIR}"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

launch_replica() {
  local gpu=$1
  local port=$2
  CUDA_VISIBLE_DEVICES="${gpu}" VLLM_WORKER_MULTIPROC_METHOD=spawn \
    "${PYTHON}" -m vllm.entrypoints.openai.api_server \
      --model "${MODEL}" \
      --served-model-name nemotron-parse \
      --host 127.0.0.1 \
      --port "${port}" \
      --trust-remote-code \
      --gpu-memory-utilization 0.45 \
      --max-model-len 16384 \
      --max-num-seqs 96 \
      --max-num-batched-tokens 8192 \
      --attention-config '{"flash_attn_version":3}' \
      --no-enable-log-requests \
      >"${SERVER_LOG_DIR}/gpu${gpu}-port${port}.log" 2>&1 &
  pids+=("$!")
}

wait_for_replica() {
  local port=$1
  local deadline=$((SECONDS + 300))
  until curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null; do
    if (( SECONDS >= deadline )); then
      echo "Replica on port ${port} did not become healthy" >&2
      return 1
    fi
    sleep 2
  done
  echo "Replica ${port} healthy"
}

# Launch one replica per GPU first, then the second wave after memory profiling.
for gpu in {0..7}; do
  launch_replica "${gpu}" "$((8101 + gpu * 2))"
done
for gpu in {0..7}; do
  wait_for_replica "$((8101 + gpu * 2))"
done

for gpu in {0..7}; do
  launch_replica "${gpu}" "$((8102 + gpu * 2))"
done
for gpu in {0..7}; do
  wait_for_replica "$((8102 + gpu * 2))"
done

proxy_args=()
for port in {8101..8116}; do
  proxy_args+=(--upstream "http://127.0.0.1:${port}")
done
"${PYTHON}" "${PROXY}" --listen-port 8000 "${proxy_args[@]}" \
  >"${SERVER_LOG_DIR}/proxy.log" 2>&1 &
pids+=("$!")
until curl --silent "http://127.0.0.1:8000/health" >/dev/null; do sleep 1; done

cd "${CURATOR_ROOT}"
"${PYTHON}" tutorials/interleaved/nemotron_parse_pdf/benchmark_run.py \
  --run-dir "${RUN_DIR}" \
  --gpu-index 0 --gpu-index 1 --gpu-index 2 --gpu-index 3 \
  --gpu-index 4 --gpu-index 5 --gpu-index 6 --gpu-index 7 \
  -- \
  "${PYTHON}" tutorials/interleaved/nemotron_parse_pdf/main.py \
    --manifest "${MANIFEST:?set MANIFEST}" \
    --pdf-dir "${PDF_DIR:?set PDF_DIR}" \
    --output-dir "${RUN_DIR}/output" \
    --dataset-name nemotron-parse-pdf-x100-direct-r16-w36 \
    --model-path "${MODEL}" \
    --inference-server-endpoint http://127.0.0.1:8000/v1 \
    --inference-server-model-name nemotron-parse \
    --parse-mode single \
    --inference-batch-size 64 \
    --client-num-workers 36 \
    --pdfs-per-task 16 \
    --dpi 300 \
    --max-pages 50 \
    --executor ray-data
