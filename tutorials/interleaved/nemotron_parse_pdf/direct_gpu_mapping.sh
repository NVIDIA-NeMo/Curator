#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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
    available["${index}"]=1
    available["${uuid}"]=1
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

  for selector in "${direct_gpu_selectors[@]}"; do
    if [[ ! -v 'available[$selector]' || -v 'seen[$selector]' ]]; then
      echo "Expected distinct, available GPU selectors for the B200 reproduction profile: ${selector}" >&2
      return 1
    fi
    seen["${selector}"]=1
  done
}
