#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CURATOR_ROOT=${CURATOR_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
PYTHON=${PYTHON:-${CURATOR_ROOT}/.venv-parse35-cuda13/bin/python}
export PATH="${PYTHON%/*}:$PATH"

ROOT=${ROOT:?set ROOT to the DC10K experiment directory}
RUN_ROOT=${ROOT}/extraction
SERVER_LOG_DIR=${RUN_ROOT}/server-logs
MODEL=${MODEL:?set MODEL to the Nemotron Parse 3.5 model directory}
PROXY=${PROXY:-${SCRIPT_DIR}/round_robin_proxy.py}
mkdir -p "${SERVER_LOG_DIR}"

for command in curl setsid nvidia-smi; do command -v "${command}" >/dev/null; done
[[ -x "${PYTHON}" && -d "${MODEL}" && -f "${PROXY}" ]]
: "${PDF_DIR:?set PDF_DIR}"
[[ -d "${PDF_DIR}" ]]
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "${gpu_count}" -ne 8 ]]; then
  echo "This B200 reproduction profile requires eight assigned GPUs" >&2
  exit 1
fi
"${PYTHON}" -c 'import aiohttp; import vllm'
"${PYTHON}" - <<'CHECK_PORTS'
import socket
for port in (8000, *range(8101, 8117)):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))
CHECK_PORTS
condition=${1:-both}
case "${condition}" in
  both|capped50|uncapped) ;;
  *) echo "Usage: $0 [both|capped50|uncapped]" >&2; exit 2 ;;
esac
for label in capped50 uncapped; do
  if [[ "${condition}" == both || "${condition}" == "${label}" ]]; then
    manifest="${ROOT}/manifest-${label}.jsonl"
    if [[ "${label}" == uncapped ]]; then manifest="${ROOT}/manifest-uncapped-window50.jsonl"; fi
    if [[ ! -f "${manifest}" ]]; then
      echo "Missing manifest: ${manifest}" >&2
      exit 1
    fi
    [[ ! -e "${RUN_ROOT}/${label}" ]]
  fi
done

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill -- "-${pid}" 2>/dev/null || true
  done
  local deadline=$((SECONDS + 10))
  while (( SECONDS < deadline )); do
    local running=0
    for pid in "${pids[@]}"; do
      if kill -0 -- "-${pid}" 2>/dev/null; then running=1; fi
    done
    if (( running == 0 )); then break; fi
    sleep 0.1
  done
  for pid in "${pids[@]}"; do kill -KILL -- "-${pid}" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
declare -A replica_pids

launch_replica() {
  local gpu=$1
  local port=$2
  setsid env CUDA_VISIBLE_DEVICES="${gpu}" VLLM_WORKER_MULTIPROC_METHOD=spawn \
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
  replica_pids["${port}"]=$!
}

wait_for_replica() {
  local port=$1
  local deadline=$((SECONDS + 300))
  until curl --fail --silent --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null; do
    if ! kill -0 "${replica_pids[${port}]}" 2>/dev/null; then
      echo "Replica on port ${port} exited before readiness" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Replica on port ${port} did not become healthy" >&2
      return 1
    fi
    sleep 2
  done
  kill -0 "${replica_pids[${port}]}" 2>/dev/null
  echo "Replica ${port} healthy"
}

run_condition() {
  local label=$1
  local manifest=$2
  local max_pages=$3
  local run_dir=${RUN_ROOT}/${label}

  if [[ -e "${run_dir}/run-summary.json" || -d "${run_dir}/output" ]]; then
    echo "Refusing to overwrite existing run: ${run_dir}" >&2
    return 1
  fi

  cd "${CURATOR_ROOT}"
  "${PYTHON}" tutorials/interleaved/nemotron_parse_pdf/benchmark_run.py \
    --run-dir "${run_dir}" \
    --gpu-index 0 --gpu-index 1 --gpu-index 2 --gpu-index 3 \
    --gpu-index 4 --gpu-index 5 --gpu-index 6 --gpu-index 7 \
    -- \
    "${PYTHON}" tutorials/interleaved/nemotron_parse_pdf/main.py \
      --manifest "${manifest}" \
      --pdf-dir "${PDF_DIR:?set PDF_DIR}" \
      --output-dir "${run_dir}/output" \
      --dataset-name "dc10k-${label}" \
      --model-path "${MODEL}" \
      --inference-server-endpoint http://127.0.0.1:8000/v1 \
      --inference-server-model-name nemotron-parse \
      --parse-mode single \
      --inference-batch-size 64 \
      --client-num-workers 36 \
      --pdfs-per-task 16 \
      --dpi 300 \
      --max-pages "${max_pages}" \
      --executor ray-data
}

# Start one replica per physical GPU, then the second wave after memory
# profiling completes. This is the established 16-replica B200 topology.
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
setsid "${PYTHON}" "${PROXY}" --listen-port 8000 "${proxy_args[@]}" \
  >"${SERVER_LOG_DIR}/proxy.log" 2>&1 &
pids+=("$!")
proxy_pid=$!
proxy_deadline=$((SECONDS + 30))
until curl --fail --silent --max-time 2 "http://127.0.0.1:8000/health" >/dev/null; do
  if ! kill -0 "${proxy_pid}" 2>/dev/null || (( SECONDS >= proxy_deadline )); then
    echo "Proxy failed to become healthy; see ${SERVER_LOG_DIR}/proxy.log" >&2
    exit 1
  fi
  sleep 1
done

condition=${1:-both}
case "${condition}" in
  both)
    run_condition capped50 "${ROOT}/manifest-capped50.jsonl" 50
    run_condition uncapped "${ROOT}/manifest-uncapped-window50.jsonl" 1000000
    ;;
  capped50)
    run_condition capped50 "${ROOT}/manifest-capped50.jsonl" 50
    ;;
  uncapped)
    run_condition uncapped "${ROOT}/manifest-uncapped-window50.jsonl" 1000000
    ;;
  *)
    echo "Usage: $0 [both|capped50|uncapped]" >&2
    exit 2
    ;;
esac
