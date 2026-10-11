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

"""Benchmark AudioTextLLMStage on preloaded audio transcripts."""

import argparse
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import hydra
from loguru import logger
from omegaconf import OmegaConf
from utils import write_benchmark_results

from nemo_curator.config.run import (
    create_executor_from_yaml,
    create_pipeline_from_yaml,
    create_ray_client_from_yaml,
)
from nemo_curator.stages.audio.common import ManifestReaderStage
from nemo_curator.stages.audio.text_filtering import AudioTextLLMStage
from nemo_curator.tasks import AudioTask, FileGroupTask

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "tutorials/audio/text_processing/config"


def _collect_output_metrics(tasks: list[AudioTask], stage: AudioTextLLMStage, num_input_rows: int) -> dict[str, int]:
    if len(tasks) != num_input_rows:
        msg = f"Audio text benchmark returned {len(tasks)} rows for {num_input_rows} input rows"
        raise RuntimeError(msg)

    applied = fallbacks = skipped = 0
    for task in tasks:
        if not isinstance(task, AudioTask) or not isinstance(task.data.get(stage.output_text_key), str):
            msg = "Audio text benchmark requires an AudioTask with a string output for every input"
            raise TypeError(msg)
        note = task.data.get(stage.notes_key, {}).get(stage.name, "")
        if note in {"applied (modified)", "applied (unchanged)"}:
            applied += 1
        elif note.startswith("fallback ("):
            fallbacks += 1
        elif note.startswith("skipped ("):
            skipped += 1
        else:
            msg = f"Missing or invalid processing note for {stage.name}: {note!r}"
            raise RuntimeError(msg)

    if applied + fallbacks == 0:
        msg = "Audio text benchmark made no inference requests"
        raise RuntimeError(msg)
    return {
        "num_input_rows": num_input_rows,
        "num_output_rows": len(tasks),
        "num_requests": applied + fallbacks,
        "num_applied": applied,
        "num_fallbacks": fallbacks,
        "num_skipped": skipped,
    }


def run_audio_text_llm_benchmark(input_manifest: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """Time one text stage, excluding manifest reads, Ray/server startup, and result writes."""
    with hydra.initialize_config_dir(version_base=None, config_dir=str(_CONFIG_DIR)):
        cfg = hydra.compose(config_name="text", overrides=overrides or [])

    tasks = ManifestReaderStage().process(FileGroupTask(dataset_name="audio_text_benchmark", data=[input_manifest]))
    if not tasks:
        msg = "Audio text benchmark requires a nonempty input manifest"
        raise ValueError(msg)

    # Use the tutorial's operation and serving configuration without its reader/writer.
    cfg.stages = [cfg.stages[1]]
    with ExitStack() as stack:
        stack.enter_context(create_ray_client_from_yaml(cfg))
        server_start = time.perf_counter()
        startup_time = 0.0
        if not cfg.endpoint:
            server = stack.enter_context(hydra.utils.instantiate(cfg.inference_server))
            cfg.endpoint = server.endpoint
            startup_time = time.perf_counter() - server_start

        pipeline = create_pipeline_from_yaml(cfg, log_config=False)
        stage = pipeline.stages[0]
        if not isinstance(stage, AudioTextLLMStage):
            msg = "Audio text benchmark supports only AudioTextLLMStage"
            raise TypeError(msg)
        executor = create_executor_from_yaml(cfg)
        logger.info(pipeline.describe())
        start = time.perf_counter()
        results = pipeline.run(executor=executor, initial_tasks=tasks)
        elapsed = time.perf_counter() - start
        metrics = _collect_output_metrics(results or [], stage, len(tasks))

    return {
        "params": {
            "input_manifest": input_manifest,
            "overrides": overrides or [],
            "operation": OmegaConf.to_container(cfg.operation, resolve=True),
            "backend": cfg.backend,
            "model_name": cfg.model_name,
            "batch_size": stage.batch_size,
            "max_concurrent_requests": stage.max_concurrent_requests,
        },
        "metrics": {
            "is_success": True,
            "time_taken_s": elapsed,
            "inference_server_startup_s": startup_time,
            **metrics,
            "throughput_requests_per_sec": metrics["num_requests"] / elapsed if elapsed > 0 else 0,
            "fallback_ratio": metrics["num_fallbacks"] / metrics["num_requests"],
        },
        "tasks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--input-manifest", required=True, help="JSONL audio manifest containing transcripts")
    parser.add_argument(
        "--override",
        dest="overrides",
        action="append",
        default=[],
        help="Repeatable tutorial Hydra override, e.g. operation=itn or backend=xenna",
    )
    args = parser.parse_args()
    result: dict[str, Any] = {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}
    success_code = 1
    try:
        result.update(run_audio_text_llm_benchmark(args.input_manifest, args.overrides))
        success_code = 0
    except Exception as error:
        logger.exception("Audio text benchmark failed")
        result["metrics"]["error_message"] = str(error)
    finally:
        write_benchmark_results(result, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
