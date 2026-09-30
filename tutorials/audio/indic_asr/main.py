# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the canonical Hindi Indic ASR benchmark from a Hydra tutorial config.

This runner deliberately delegates to
``benchmarking/scripts/audio_indic_asr_benchmark.py``. The tutorial therefore
uses the same input normalization, processor graph, executor setup, timer, and
output validation as the benchmark entries instead of maintaining a second
copy of that contract.

Usage (from the Curator repository root)::

    python tutorials/audio/indic_asr/main.py \
        --config-path . \
        --config-name pipeline \
        input_manifest=/data/audio_indic_asr/manifest.jsonl \
        indic_canary_engine_dir=/models/indic_canary/engine_bfloat16_64_new \
        parakeet_tensorrt_engine_dir=/models/parakeet_indic/encoder_fp16 \
        backend=xenna
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.core.client import RayClient

if TYPE_CHECKING:
    from types import ModuleType

_BACKENDS = {"xenna", "ray_data"}


def _load_benchmark_module() -> ModuleType:
    """Import the repository's canonical benchmark implementation."""
    repository_root = Path(__file__).resolve().parents[3]
    scripts_dir = repository_root / "benchmarking" / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        return importlib.import_module("audio_indic_asr_benchmark")
    finally:
        sys.path.remove(str(scripts_dir))


def _benchmark_args(cfg: DictConfig) -> dict[str, Any]:
    return {
        "benchmark_results_path": to_absolute_path(cfg.benchmark_results_path),
        "input_manifest": to_absolute_path(cfg.input_manifest),
        "indic_canary_engine_dir": to_absolute_path(cfg.indic_canary_engine_dir),
        "parakeet_tensorrt_engine_dir": to_absolute_path(cfg.parakeet_tensorrt_engine_dir),
        "regex_yaml": to_absolute_path(cfg.regex_yaml),
        "hall_phrases": to_absolute_path(cfg.hall_phrases),
        "executor": cfg.backend,
        "expected_num_rows": cfg.expected_num_rows,
        "read_concurrency": cfg.read_concurrency,
        "prep_workers": cfg.prep_workers,
        "primary_workers": cfg.primary_workers,
        "fallback_workers": cfg.fallback_workers,
    }


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Run one exact benchmark entry using the selected backend."""
    if cfg.backend not in _BACKENDS:
        msg = f"Unknown backend '{cfg.backend}'. Choose from: {sorted(_BACKENDS)}"
        raise ValueError(msg)

    for key, value in cfg.environment.items():
        os.environ[str(key)] = str(value)

    benchmark = _load_benchmark_module()
    run_args = _benchmark_args(cfg)
    result: dict[str, Any] = {
        "params": run_args,
        "metrics": {"is_success": False},
        "tasks": [],
    }
    ray_client = RayClient(
        num_cpus=cfg.ray.num_cpus,
        num_gpus=cfg.ray.num_gpus,
        object_store_memory=cfg.ray.object_store_memory,
    )

    logger.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Using canonical benchmark: {Path(benchmark.__file__).resolve()}")
    try:
        ray_client.start()
        result.update(benchmark.run_audio_indic_asr_benchmark(**run_args))
    finally:
        benchmark.write_benchmark_results(result, run_args["benchmark_results_path"])
        ray_client.stop()

    metrics = result["metrics"]
    logger.success(
        "Indic ASR benchmark complete: "
        f"backend={cfg.backend}, rows={metrics['num_output_rows']}, "
        f"wall={metrics['time_taken_s']:.2f}s, "
        f"audio={metrics['total_audio_duration_hours']:.4f}h"
    )


if __name__ == "__main__":
    main()
