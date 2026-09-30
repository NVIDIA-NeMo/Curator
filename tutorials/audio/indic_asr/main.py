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

"""Run one canonical Hindi Indic ASR benchmark entry from Hydra configuration.

This runner delegates to ``benchmarking/run.py`` with the checked-in
``benchmarking/benchmarks.yaml`` and an exact entry name. The tutorial therefore
uses the benchmark's resource allocation, timeout, input normalization,
processor graph, GPU recorder, timing boundary, output validation, and
requirements instead of maintaining a second copy of that contract.

Usage (from the Curator repository root)::

    python tutorials/audio/indic_asr/main.py \
        --config-path . \
        --config-name pipeline \
        datasets_path=/data/curator_datasets \
        model_weights_path=/models/curator \
        backend=xenna
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf

_ENTRY_BY_BACKEND = {
    "xenna": "audio_indic_asr_xenna",
    "ray_data": "audio_indic_asr_raydata",
}


def _path_overlay(cfg: DictConfig) -> dict[str, list[dict[str, str]]]:
    """Map the tutorial's host roots onto the canonical config path names."""

    def path_entry(name: str, value: str) -> dict[str, str]:
        resolved = to_absolute_path(value)
        return {"name": name, "host_path": resolved, "container_path": resolved}

    return {
        "paths": [
            path_entry("datasets_path", cfg.datasets_path),
            path_entry("model_weights_path", cfg.model_weights_path),
            path_entry("results_path", cfg.results_path),
        ]
    }


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Run one exact benchmark entry and require every configured gate to pass."""
    if cfg.backend not in _ENTRY_BY_BACKEND:
        msg = f"Unknown backend '{cfg.backend}'. Choose from: {sorted(_ENTRY_BY_BACKEND)}"
        raise ValueError(msg)

    repository_root = Path(__file__).resolve().parents[3]
    benchmark_config = Path(to_absolute_path(cfg.benchmark_config))
    results_path = Path(to_absolute_path(cfg.results_path))
    session_path = results_path / cfg.session_name
    entry_name = _ENTRY_BY_BACKEND[cfg.backend]
    result_path = session_path / entry_name / "results.json"
    if session_path.exists():
        msg = f"Use a fresh session name or results path: {session_path}"
        raise FileExistsError(msg)

    results_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Running canonical entry '{entry_name}' from {benchmark_config}")
    with tempfile.TemporaryDirectory(prefix="indic-asr-tutorial-") as temp_dir:
        overlay_path = Path(temp_dir) / "paths.json"
        overlay_path.write_text(json.dumps(_path_overlay(cfg)))
        command = [
            sys.executable,
            str(repository_root / "benchmarking" / "run.py"),
            "--config",
            str(benchmark_config),
            "--config",
            str(overlay_path),
            "--session-name",
            cfg.session_name,
            "--entries-exact",
            entry_name,
            "--reason",
            cfg.reason,
        ]
        subprocess.run(command, cwd=repository_root, check=True)  # noqa: S603

    result = json.loads(result_path.read_text())
    if not result["success"] or result.get("requirements_not_met"):
        msg = f"Benchmark entry did not pass every requirement: {result_path}"
        raise RuntimeError(msg)
    metrics = result["metrics"]
    logger.success(
        "Indic ASR benchmark complete: "
        f"backend={cfg.backend}, rows={metrics['num_output_rows']}, "
        f"exec_time={result['exec_time_s']:.2f}s, "
        f"audio={metrics['total_audio_duration_hours']:.4f}h"
    )


if __name__ == "__main__":
    main()
