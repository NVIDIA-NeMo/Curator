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

"""Benchmark an audio pipeline YAML (tutorial-style) through nemo_curator.config.run."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from utils import write_benchmark_results

from nemo_curator.config.run import create_executor_from_yaml, create_pipeline_from_yaml


def count_output_rows(output_manifest: Path) -> int:
    if not output_manifest.is_file():
        return 0
    with output_manifest.open(encoding="utf-8") as manifest:
        return sum(1 for line in manifest if line.strip())


def apply_overrides(cfg: DictConfig, overrides: list[str]) -> None:
    """Merge ``key=value`` dotlist overrides into ``cfg``.

    OmegaConf silently adds keys it does not know, so an override aimed at a key the
    YAML has renamed would be ignored. Reject overrides whose top-level key is missing.
    """
    unknown = sorted({override.split("=", 1)[0].split(".", 1)[0] for override in overrides} - set(cfg.keys()))
    if unknown:
        msg = f"Overrides set top-level keys the pipeline YAML does not define: {unknown}"
        raise ValueError(msg)
    cfg.merge_with_dotlist(overrides)


def run_audio_yaml_pipeline_benchmark(
    benchmark_results_path: str,
    pipeline_yaml: str,
    output_manifest: str,
    overrides: list[str],
) -> dict[str, Any]:
    """Run the YAML with dotlist overrides and fail when it writes no rows."""
    cfg = OmegaConf.load(pipeline_yaml)
    apply_overrides(cfg, overrides)
    pipeline = create_pipeline_from_yaml(cfg)
    executor_obj = create_executor_from_yaml(cfg)

    start = time.perf_counter()
    pipeline.run(executor=executor_obj)
    run_time_s = time.perf_counter() - start
    num_output_rows = count_output_rows(Path(output_manifest))
    if num_output_rows == 0:
        logger.error(f"Pipeline wrote no rows to {output_manifest}")
    return {
        "params": {
            "executor": cfg.backend,
            "pipeline_yaml": pipeline_yaml,
            "output_manifest": output_manifest,
            "overrides": overrides,
            "benchmark_results_path": benchmark_results_path,
            "pipeline_config": OmegaConf.to_container(cfg, resolve=True),
        },
        "metrics": {
            "is_success": num_output_rows > 0,
            "time_taken_s": run_time_s,
            "num_output_rows": num_output_rows,
            "throughput_rows_per_sec": num_output_rows / run_time_s if run_time_s else 0.0,
        },
        "tasks": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--pipeline-yaml", required=True, help="Pipeline YAML, e.g. a tutorial pipeline.yaml")
    parser.add_argument("--output-manifest", required=True, help="JSONL the pipeline writes; its rows are counted")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], help="OmegaConf dotlist override, key=value"
    )
    parser.add_argument("--save-tasks", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    save_tasks = vars(args).pop("save_tasks")

    results: dict[str, Any] = {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}
    try:
        results.update(run_audio_yaml_pipeline_benchmark(**vars(args)))
    except Exception as error:
        logger.exception("Audio YAML pipeline benchmark failed")
        results["metrics"]["error_message"] = str(error)
    finally:
        write_benchmark_results(results, args.benchmark_results_path, save_tasks=save_tasks)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
