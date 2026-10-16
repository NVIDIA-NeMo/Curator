# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Check the dedicated Indic image, including native Canary imports on a GPU."""

import argparse
import importlib
import platform
import sys
from importlib.metadata import version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true", help="Check packages without a host NVIDIA driver")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or platform.system() != "Linux" or platform.machine() != "x86_64":
        msg = "Indic TensorRT-LLM requires Python 3.12 on Linux x86_64"
        raise RuntimeError(msg)

    for name, expected in {
        "torch": "2.9.1+cu128",
        "torchaudio": "2.9.1+cu128",
        "torchvision": "0.24.1+cu128",
        "torchcodec": "0.9.1",
        "tensorrt": "10.14.1.48.post1",
        "tensorrt-llm": "1.2.1",
    }.items():
        installed = version(name)
        if installed != expected:
            msg = f"{name}: expected {expected}, found {installed}"
            raise RuntimeError(msg)
        print(f"{name}=={installed}", flush=True)

    import torch

    importlib.import_module("tensorrt")
    if args.build_only:
        print("Build checks passed; native TensorRT-LLM still requires the GPU smoke check.")
        return

    from nemo_curator.stages.audio.inference.indic_canary_trtllm_runtime import _require_tensorrt_llm

    _require_tensorrt_llm()
    from cuda.bindings import runtime as cuda_runtime

    # Torch uses CUDA 12; check the CUDA 13 runtime used by TensorRT separately.
    status, device_count = cuda_runtime.cudaGetDeviceCount()
    if status != cuda_runtime.cudaError_t.cudaSuccess or device_count < 1:
        msg = f"TensorRT's CUDA runtime cannot use this host GPU: {status!s}, devices={device_count}"
        raise RuntimeError(msg)
    for name in (
        "nemo.collections.asr.models",
        "nemo_text_processing.text_normalization",
        "nemo_curator.models.asr.indic_parakeet_rnnt_tensorrt",
        "nemo_curator.stages.audio.metrics.wer",
        "nemo_curator.stages.audio.preprocessing.indic_asr",
        "nemo_curator.stages.audio.inference.indic_canary",
        "nemo_curator.stages.audio.inference.lid.stage",
        "nemo_curator.stages.audio.inference.parakeet",
        "nemo_curator.models.audio.lid.speechbrain",
        "nemo_curator.models.audio.lid.indic_canary",
        "nemo_curator.models.audio.lid.whisper",
        "nemo_curator.stages.audio.text_filtering",
        "nemo_curator.config.run",
        "benchmarking.scripts.audio_indic_asr_benchmark",
        "benchmarking.scripts.utils",
        "torchcodec.decoders",
        "onnxruntime",
        "cuda.bindings.runtime",
    ):
        importlib.import_module(name)
    if not torch.cuda.is_available():
        msg = "No usable CUDA GPU; the native smoke check requires --gpus all and a compatible driver"
        raise RuntimeError(msg)
    values = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    torch.testing.assert_close(values @ torch.eye(4, device="cuda"), values)
    torch.cuda.synchronize()
    print("Native Canary/Parakeet/LID imports and CUDA execution passed; model-engine parity is a separate check.")


if __name__ == "__main__":
    main()
