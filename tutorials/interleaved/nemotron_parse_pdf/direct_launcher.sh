#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CURATOR_ROOT=${CURATOR_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
PYTHON=${PYTHON:-${CURATOR_ROOT}/.venv-parse35-cuda13/bin/python}
export PATH="${PYTHON%/*}:$PATH"

# Select the devices for the fixed eight-GPU reproduction profile.
select_direct_gpus() {
  local inventory index uuid selector
  local ordinal='(0|[1-9][0-9]*)'
  local gpu_uuid='GPU-[[:xdigit:]]{8}(-[[:xdigit:]]{4}){3}-[[:xdigit:]]{12}'
  local -A available=() seen=()

  if ! inventory=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader); then
    echo "Cannot enumerate the devices for the B200 reproduction profile" >&2
    return 1
  fi
  while IFS=', ' read -r index uuid; do
    if [[ ! "${index}" =~ ^${ordinal}$ || ! "${uuid}" =~ ^${gpu_uuid}$ ]]; then
      echo "Unexpected GPU inventory for the B200 reproduction profile" >&2
      return 1
    fi
    available["${index}"]=${index}
    available["${uuid}"]=${index}
  done <<< "${inventory}"

  if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^${ordinal}(,${ordinal}){7}$ &&
          ! "${CUDA_VISIBLE_DEVICES}" =~ ^${gpu_uuid}(,${gpu_uuid}){7}$ ]]; then
      echo "CUDA_VISIBLE_DEVICES must name eight assigned GPU indexes or eight full GPU UUIDs" >&2
      return 1
    fi
    IFS=, read -r -a direct_gpu_selectors <<< "${CUDA_VISIBLE_DEVICES}"
  else
    if (( ${#available[@]} != 16 )); then
      echo "This B200 reproduction profile requires eight assigned GPUs" >&2
      return 1
    fi
    direct_gpu_selectors=({0..7})
  fi

  direct_gpu_indexes=()
  for selector in "${direct_gpu_selectors[@]}"; do
    if [[ ! -v 'available[$selector]' || -v 'seen[$selector]' ]]; then
      echo "Expected distinct, available GPU selectors for the B200 reproduction profile: ${selector}" >&2
      return 1
    fi
    seen["${selector}"]=1
    direct_gpu_indexes+=("${available[${selector}]}")
  done
}

prepare_direct_launcher() {
  local tool
  MODEL=${MODEL:?set MODEL to the Nemotron Parse 3.5 model directory}
  PROXY=${PROXY:-${SCRIPT_DIR}/round_robin_proxy.py}
  for tool in curl setsid nvidia-smi; do command -v "${tool}" >/dev/null; done
  [[ -x "${PYTHON}" && -d "${MODEL}" && -f "${PROXY}" ]]
  : "${PDF_DIR:?set PDF_DIR}"
  [[ -d "${PDF_DIR}" ]]
  select_direct_gpus
  mkdir -p "${SERVER_LOG_DIR}"
  "${PYTHON}" -c 'import aiohttp; import vllm'
  "${PYTHON}" - <<'CHECK_PORTS'
import socket
for port in (8000, *range(8101, 8117)):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))
CHECK_PORTS
}

pids=()
declare -A replica_pids=()
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
launch_replica() {
  local gpu=$1
  local port=$2
  setsid env CUDA_VISIBLE_DEVICES="${direct_gpu_selectors[gpu]}" VLLM_WORKER_MULTIPROC_METHOD=spawn \
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

start_direct_servers() {
  local base_port gpu port
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  # Start the second replica wave only after the first is healthy.
  for base_port in 8101 8102; do
    for gpu in "${!direct_gpu_selectors[@]}"; do
      launch_replica "${gpu}" "$((base_port + gpu * 2))"
    done
    for gpu in "${!direct_gpu_selectors[@]}"; do
      wait_for_replica "$((base_port + gpu * 2))"
    done
  done

  local -a proxy_args=()
  for port in {8101..8116}; do
    proxy_args+=(--upstream "http://127.0.0.1:${port}")
  done
  setsid "${PYTHON}" "${PROXY}" --listen-port 8000 "${proxy_args[@]}" \
    >"${SERVER_LOG_DIR}/proxy.log" 2>&1 &
  pids+=("$!")
  proxy_pid=$!
  local proxy_deadline=$((SECONDS + 30))
  until curl --fail --silent --max-time 2 "http://127.0.0.1:8000/health" >/dev/null; do
    if ! kill -0 "${proxy_pid}" 2>/dev/null || (( SECONDS >= proxy_deadline )); then
      echo "Proxy failed to become healthy; see ${SERVER_LOG_DIR}/proxy.log" >&2
      exit 1
    fi
    sleep 1
  done
}

record_direct_benchmark() {
  local run_dir=$1 manifest=$2 dataset_name=$3 max_pages=$4
  local gpu
  local -a gpu_args=()
  for gpu in "${direct_gpu_indexes[@]}"; do
    gpu_args+=(--gpu-index "${gpu}")
  done

  cd "${CURATOR_ROOT}"
  "${PYTHON}" tutorials/interleaved/nemotron_parse_pdf/benchmark_run.py \
    --run-dir "${run_dir}" \
    "${gpu_args[@]}" \
    -- \
    "${PYTHON}" tutorials/interleaved/nemotron_parse_pdf/main.py \
      --manifest "${manifest}" \
      --pdf-dir "${PDF_DIR:?set PDF_DIR}" \
      --output-dir "${run_dir}/output" \
      --dataset-name "${dataset_name}" \
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

run_direct_profile() {
  local suffix=$1 dataset_name=$2
  ROOT=${ROOT:?set ROOT to the benchmark output root}
  local run_dir=${ROOT}/direct-r16-m45-s96-t8192-p16-w36-c64-nemotron-parse-pdf${suffix}
  SERVER_LOG_DIR=${ROOT}/server-logs/$(basename -- "${run_dir}")
  prepare_direct_launcher
  : "${MANIFEST:?set MANIFEST}"
  [[ -f "${MANIFEST}" && ! -e "${run_dir}" ]]

  start_direct_servers
  kill -0 "${proxy_pid}" 2>/dev/null
  record_direct_benchmark "${run_dir}" "${MANIFEST}" "${dataset_name}" 50
}
