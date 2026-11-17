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

"""Benchmark the Granary v2 full English pipeline (recordings to enriched transcripts) on Ray Data."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import hydra
from huggingface_hub import snapshot_download
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from utils import write_benchmark_results

from nemo_curator.config.run import (
    create_executor_from_yaml,
    create_pipeline_from_yaml,
    create_ray_client_from_yaml,
)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "tutorials/audio/granary_v2/full_pipeline"
_CONFIG_NAME = "en_granary_pipeline"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pipeline_languages(cfg: DictConfig) -> set[str]:
    """The languages the YAML's gate keeps (``source_lang`` equal to one of them), read from the YAML itself."""
    languages: set[str] = set()
    for stage in OmegaConf.to_container(cfg.stages, resolve=True):
        if stage["_target_"].endswith(".PreserveByValueStage") and stage.get("input_value_key") == "source_lang":
            languages.add(stage["target_value"])
        elif stage["_target_"].endswith(".PreserveByValueConditionsStage"):
            languages.update(
                c["target_value"] for c in stage["conditions"] if c.get("input_value_key") == "source_lang"
            )
    if not languages:
        msg = "The pipeline YAML has no source_lang gate to take the language set from"
        raise ValueError(msg)
    return languages


def _collect_output_metrics(
    output_dir: Path, num_recordings: int, output_file: str, languages: set[str]
) -> dict[str, float | int]:
    metadata = _read_jsonl(output_dir / "metadata.jsonl")
    rows = _read_jsonl(output_dir / output_file)
    if not rows:
        msg = f"No clips reached {output_file}; inspect metadata.jsonl"
        raise ValueError(msg)
    metadata_ids = {row["audio_item_id"] for row in metadata}
    flagged = 0
    duration_s = 0.0
    for row in rows:
        if row["audio_item_id"] not in metadata_ids or row.get("source_lang") not in languages:
            msg = f"Output row does not match the metadata audit: {row['audio_item_id']}"
            raise ValueError(msg)
        if row.get("_skipme"):
            flagged += 1
        elif not isinstance(row.get("best_prediction"), str) or not row["best_prediction"].strip():
            msg = f"Empty best_prediction for {row['audio_item_id']}"
            raise ValueError(msg)
        duration_s += float(row["duration"])
    return {
        "num_input_recordings": num_recordings,
        "num_metadata_rows": len(metadata),
        "num_output_rows": len(rows),
        "num_flagged_rows": flagged,
        "total_audio_duration_hours": duration_s / 3600,
    }


def run_granary_full_benchmark(  # noqa: PLR0913
    config_name: str,
    benchmark_results_path: str,
    input_manifest: str,
    model_dir: str,
    cache_root: str,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Time the pipeline run, excluding Ray start-up and the Dynamo server's start-up."""
    output_dir = Path(benchmark_results_path).resolve() / "results"
    if output_dir.exists():
        msg = f"Benchmark output already exists: {output_dir}; choose a fresh results directory"
        raise FileExistsError(msg)
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name,
            overrides=[
                f"manifest_path={Path(input_manifest).resolve()}",
                f"model_dir={model_dir}",
                f"output_dir={output_dir}",
                f"cache_root={cache_root}",
                *(overrides or []),
            ],
        )
    output_file = Path(cfg.output_path).name
    num_recordings = len(_read_jsonl(Path(input_manifest)))

    with ExitStack() as stack:
        stack.enter_context(create_ray_client_from_yaml(cfg))
        server_start = time.perf_counter()
        if not Path(cfg.model_identifier).is_dir():
            cfg.model_identifier = snapshot_download(
                repo_id=cfg.model_identifier, revision=cfg.inference_server.models[0].engine_kwargs.revision
            )
        cfg.model_name = str(cfg.model_name)
        server = stack.enter_context(hydra.utils.instantiate(cfg.inference_server))
        cfg.endpoint = server.endpoint
        languages = _pipeline_languages(cfg)
        startup_s = time.perf_counter() - server_start

        pipeline = create_pipeline_from_yaml(cfg, log_config=False)
        logger.info(pipeline.describe())
        start = time.perf_counter()
        pipeline.run(executor=create_executor_from_yaml(cfg))
        elapsed = time.perf_counter() - start

    output_metrics = _collect_output_metrics(output_dir, num_recordings, output_file, languages)
    return {
        "params": {
            "input_manifest": str(Path(input_manifest).resolve()),
            "model_dir": model_dir,
            "config_name": config_name,
            "overrides": overrides or [],
            "backend": cfg.backend,
        },
        "metrics": {
            "is_success": True,
            "time_taken_s": elapsed,
            "inference_server_startup_s": startup_s,
            **output_metrics,
            "throughput_audio_hours_per_hour": output_metrics["total_audio_duration_hours"] * 3600 / elapsed,
        },
        "tasks": [],
    }


def main(config_name: str = _CONFIG_NAME, description: str | None = __doc__) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--input-manifest", required=True, help="JSONL of recordings: audio_item_id, audio_filepath")
    parser.add_argument("--model-dir", required=True, help="Prepared metadata models directory")
    parser.add_argument("--cache-root", required=True, help="Persistent compilation cache directory")
    parser.add_argument(
        "--override", action="append", default=[], help="Hydra override, e.g. ray_client.num_gpus=8 (repeatable)"
    )
    parser.add_argument("--save-tasks", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    save_tasks = args.save_tasks

    results: dict[str, Any] = {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}
    try:
        results.update(
            run_granary_full_benchmark(
                config_name,
                args.benchmark_results_path,
                args.input_manifest,
                args.model_dir,
                args.cache_root,
                args.override,
            )
        )
    except Exception as error:
        logger.exception("Granary full English benchmark failed")
        results["metrics"]["error_message"] = str(error)
    finally:
        write_benchmark_results(results, args.benchmark_results_path, save_tasks=save_tasks)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
