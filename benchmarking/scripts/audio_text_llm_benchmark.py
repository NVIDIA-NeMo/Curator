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

"""Benchmark the complete audio text tutorial on a fixed transcript cohort."""

import argparse
import copy
import hashlib
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import hydra
from fsspec import open as fsspec_open
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

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "tutorials/audio/granary_v2/text_pipeline"


def _collect_output_metrics(  # noqa: C901
    tasks: list[AudioTask], stages: list[AudioTextLLMStage], num_input_rows: int
) -> dict[str, Any]:
    if len(tasks) != num_input_rows:
        msg = f"Audio text benchmark returned {len(tasks)} rows for {num_input_rows} input rows"
        raise RuntimeError(msg)

    operations = {}
    for stage in stages:
        counts = {"num_applied": 0, "num_fallbacks": 0, "num_skipped": 0}
        for task in tasks:
            if not isinstance(task, AudioTask) or not isinstance(task.data.get(stage.output_text_key), str):
                msg = f"Audio text benchmark requires an AudioTask with string output {stage.output_text_key!r}"
                raise TypeError(msg)
            notes = task.data.get(stage.notes_key)
            note = notes.get(stage.name) if isinstance(notes, dict) else None
            if not isinstance(note, str):
                msg = f"Missing or invalid processing note for {stage.name}: {note!r}"
                raise RuntimeError(msg)  # noqa: TRY004
            if note in {"applied (modified)", "applied (unchanged)", "recovered", "unchanged"}:
                counts["num_applied"] += 1
            elif note.startswith("fallback ("):
                counts["num_fallbacks"] += 1
            elif note.startswith("skipped ("):
                counts["num_skipped"] += 1
            else:
                msg = f"Missing or invalid processing note for {stage.name}: {note!r}"
                raise RuntimeError(msg)
            if (
                stage.mode == "language_id"
                and not note.startswith("skipped (")
                and not isinstance(notes.get("LLMLanguageVerification"), str)
            ):
                msg = "Missing language verification result"
                raise RuntimeError(msg)
        counts["num_requests"] = counts["num_applied"] + counts["num_fallbacks"]
        operations[stage.name] = counts

    totals = {
        key: sum(counts[key] for counts in operations.values())
        for key in ("num_applied", "num_fallbacks", "num_skipped", "num_requests")
    }
    if totals["num_requests"] == 0:
        msg = "Audio text benchmark made no inference requests"
        raise RuntimeError(msg)
    return {
        "num_input_rows": num_input_rows,
        "num_output_rows": len(tasks),
        "input_output_coverage_ratio": 1.0,
        "num_operations": len(stages),
        **totals,
        "operations": operations,
    }


def _input_inventory(tasks: list[AudioTask], id_key: str) -> dict[str, dict]:
    inventory = {}
    for task in tasks:
        if not isinstance(task.data, dict):
            msg = "Each input manifest row must be a JSON object"
            raise TypeError(msg)
        identity = task.data.get(id_key)
        if not isinstance(identity, str) or not identity.strip() or identity in inventory:
            msg = f"Each input must have a unique nonempty string {id_key!r}: {identity!r}"
            raise ValueError(msg)
        inventory[identity] = copy.deepcopy(task.data)
    return inventory


def _validate_preserved_inputs(
    tasks: list[AudioTask], inventory: dict[str, dict], stages: list[AudioTextLLMStage], id_key: str
) -> None:
    output_inventory = _input_inventory(tasks, id_key)
    if output_inventory.keys() != inventory.keys():
        msg = "Audio text benchmark output identities do not match the input cohort"
        raise RuntimeError(msg)
    mutable = {stage.output_text_key for stage in stages} | {stage.skip_me_key for stage in stages}
    note_keys = {stage.notes_key for stage in stages}
    changed_notes = {stage.name for stage in stages} | {"LLMLanguageVerification"}
    for identity, original in inventory.items():
        output = output_inventory[identity]
        for key, value in original.items():
            if key in note_keys and isinstance(value, dict):
                if not isinstance(output.get(key), dict) or any(
                    name not in output[key] or output[key][name] != note
                    for name, note in value.items()
                    if name not in changed_notes
                ):
                    msg = f"Audio text benchmark changed existing notes for {identity!r}"
                    raise RuntimeError(msg)
            elif key not in mutable | note_keys and (key not in output or output[key] != value):
                msg = f"Audio text benchmark changed input field {key!r} for {identity!r}"
                raise RuntimeError(msg)


def _manifest_identity(input_manifest: str) -> dict[str, str | int]:
    digest = hashlib.sha256()
    size = 0
    with fsspec_open(input_manifest, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"input_manifest_bytes": size, "input_manifest_sha256": digest.hexdigest()}


def _validate_workload_inputs(tasks: list[AudioTask], operations: list[AudioTextLLMStage], id_key: str) -> None:
    if not operations or len({stage.name for stage in operations}) != len(operations):
        msg = "Audio text benchmark requires operations with unique names"
        raise ValueError(msg)
    produced = set()
    for stage in operations:
        if id_key in {stage.output_text_key, stage.notes_key, stage.skip_me_key}:
            msg = "The benchmark identity key must not be overwritten by the configured stages"
            raise ValueError(msg)
        required = set(stage.inputs()[1])
        # Benchmark cohorts require references and source labels even when stages allow fallbacks.
        if stage.mode == "recover_entities":
            required.update([stage.text_key, stage.ground_truth_key])
        elif stage.mode == "language_id":
            required.add(stage.language_key)
        for task in tasks:
            if task.data.get(stage.skip_me_key):
                continue
            for key in required - produced:
                if not isinstance(task.data.get(key), str):
                    msg = f"Input {task.data[id_key]!r} requires string field {key!r} for {stage.name}"
                    raise TypeError(msg)
        produced.add(stage.output_text_key)


def _check_fresh_outputs(output_dir: Path, benchmark_results_path: Path) -> None:
    paths = [output_dir / "text_pipeline.jsonl"]
    paths.extend(benchmark_results_path / name for name in ("results", "scratch", "metrics.json", "tasks.pkl"))
    # The runner precreates params.json and an empty scratch directory.
    for path in paths:
        if (
            path == benchmark_results_path / "scratch"
            and path.is_dir()
            and not path.is_symlink()
            and not any(path.iterdir())
        ):
            continue
        if path.exists() or path.is_symlink():
            msg = f"Benchmark output already exists: {path}; choose fresh output and results directories"
            raise FileExistsError(msg)


def run_audio_text_llm_benchmark(  # noqa: C901, PLR0912, PLR0913, PLR0915
    input_manifest: str,
    output_dir: str,
    executor: str | None = None,
    endpoint: str | None = None,
    cache_root: str | None = None,
    model_path: str | None = None,
    model_name: str | None = None,
    id_key: str = "audio_item_id",
    expected_num_rows: int | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Time the complete reader/processing/writer graph, excluding preflight and server lifecycle."""
    if endpoint and model_path is not None:
        msg = "model_path cannot reconfigure an existing endpoint; omit endpoint to start that model"
        raise ValueError(msg)
    cfg = OmegaConf.load(_CONFIG_DIR / "text_pipeline.yaml")
    cfg.manifest_path = input_manifest
    output_manifest = Path(output_dir).resolve() / "text_pipeline.jsonl"
    if output_manifest.exists():
        msg = f"Audio text benchmark output already exists: {output_manifest}"
        raise ValueError(msg)
    cfg.output_path = str(output_manifest)
    cfg.endpoint = endpoint
    if executor is not None:
        cfg.backend = executor
    if model_name is not None:
        cfg.model_name = model_name
    if model_path is not None:
        cfg.model_name = str(cfg.model_name)
        cfg.model_identifier = model_path
    if not endpoint and not cache_root:
        msg = "cache_root is required when the benchmark starts the inference server"
        raise ValueError(msg)
    if cache_root is not None:
        cfg.cache_root = cache_root
    cfg.ray_client.num_gpus = 0 if endpoint else None

    manifest_identity = _manifest_identity(input_manifest)
    tasks = ManifestReaderStage().process(FileGroupTask(dataset_name="audio_text_benchmark", data=[input_manifest]))
    if not tasks:
        msg = "Audio text benchmark requires a nonempty input manifest"
        raise ValueError(msg)
    if _manifest_identity(input_manifest) != manifest_identity:
        msg = "Input manifest changed while loading the benchmark cohort"
        raise RuntimeError(msg)
    inventory = _input_inventory(tasks, id_key)
    if expected_num_rows is not None and len(tasks) != expected_num_rows:
        msg = f"Expected {expected_num_rows} input rows, found {len(tasks)}"
        raise ValueError(msg)
    if (
        expected_manifest_sha256 is not None
        and manifest_identity["input_manifest_sha256"] != expected_manifest_sha256.lower()
    ):
        msg = "Input manifest SHA-256 does not match the frozen cohort"
        raise ValueError(msg)

    with ExitStack() as stack:
        stack.enter_context(create_ray_client_from_yaml(cfg))
        server_start = time.perf_counter()
        startup_time = 0.0
        if not cfg.endpoint:
            server = stack.enter_context(hydra.utils.instantiate(cfg.inference_server))
            cfg.endpoint = server.endpoint
            startup_time = time.perf_counter() - server_start

        pipeline = create_pipeline_from_yaml(cfg, log_config=False)
        processing_stages = pipeline.stages[1:-1]
        if not processing_stages or not all(isinstance(stage, AudioTextLLMStage) for stage in processing_stages):
            msg = "Audio text benchmark requires AudioTextLLMStages between the reader and writer"
            raise TypeError(msg)
        operations = [operation for stage in processing_stages for operation in stage.operation_stages()]
        _validate_workload_inputs(tasks, operations, id_key)
        executor = create_executor_from_yaml(cfg)
        logger.info(pipeline.describe())
        start = time.perf_counter()
        results = pipeline.run(executor=executor)
        elapsed = time.perf_counter() - start
        if _manifest_identity(input_manifest) != manifest_identity:
            msg = "Input manifest changed during the benchmark"
            raise RuntimeError(msg)
        metrics = _collect_output_metrics(results or [], operations, len(tasks))
        _validate_preserved_inputs(results, inventory, operations, id_key)
        written = ManifestReaderStage().process(
            FileGroupTask(dataset_name="audio_text_benchmark_output", data=[str(output_manifest)])
        )
        _collect_output_metrics(written, operations, len(tasks))
        _validate_preserved_inputs(written, inventory, operations, id_key)
        if _input_inventory(written, id_key) != _input_inventory(results, id_key):
            msg = "Written audio text output does not match the returned tasks"
            raise RuntimeError(msg)

    return {
        "params": {
            "input_manifest": input_manifest,
            "output_dir": str(output_manifest.parent),
            "output_manifest": str(output_manifest),
            "timing_scope": "reader_processing_writer",
            "config_path": str(_CONFIG_DIR / "text_pipeline.yaml"),
            "id_key": id_key,
            "expected_num_rows": expected_num_rows,
            "expected_manifest_sha256": expected_manifest_sha256,
            "stages": OmegaConf.to_container(cfg.stages, resolve=True),
            "backend": cfg.backend,
            "model_name": cfg.model_name,
            "inference_server": OmegaConf.to_container(cfg.inference_server, resolve=True) if not endpoint else None,
        },
        "metrics": {
            "is_success": True,
            "time_taken_s": elapsed,
            "inference_server_startup_s": startup_time,
            **manifest_identity,
            **metrics,
            "throughput_rows_per_sec": len(tasks) / elapsed if elapsed > 0 else 0,
            "throughput_requests_per_sec": metrics["num_requests"] / elapsed if elapsed > 0 else 0,
            "fallback_ratio": metrics["num_fallbacks"] / metrics["num_requests"],
        },
        "tasks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--input-manifest", required=True, help="JSONL audio manifest containing transcripts")
    parser.add_argument("--output-dir", required=True, help="Directory for a new text_pipeline.jsonl output")
    parser.add_argument("--executor", choices=["xenna", "ray_data"], help="Defaults to the tutorial's backend")
    parser.add_argument(
        "--endpoint", help="Existing OpenAI-compatible endpoint; otherwise start the configured server"
    )
    parser.add_argument("--cache-root", help="Persistent cache directory; required when starting the server")
    parser.add_argument("--model-path", help="Local model snapshot for the configured server")
    parser.add_argument("--model-name", help="Served model name; defaults to the tutorial's model name")
    parser.add_argument("--id-key", default="audio_item_id", help="Unique string identity preserved in each input row")
    parser.add_argument("--expected-num-rows", type=int)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument(
        "--save-tasks", action=argparse.BooleanOptionalAction, default=True, help="Save completed tasks in tasks.pkl"
    )
    args = parser.parse_args()
    try:
        _check_fresh_outputs(Path(args.output_dir), Path(args.benchmark_results_path))
    except FileExistsError as error:
        logger.error(str(error))
        return 1

    result: dict[str, Any] = {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}
    success_code = 1
    try:
        result.update(
            run_audio_text_llm_benchmark(
                input_manifest=args.input_manifest,
                output_dir=args.output_dir,
                executor=args.executor,
                endpoint=args.endpoint,
                cache_root=args.cache_root,
                model_path=args.model_path,
                model_name=args.model_name,
                id_key=args.id_key,
                expected_num_rows=args.expected_num_rows,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
        )
        success_code = 0
    except Exception as error:
        logger.exception("Audio text benchmark failed")
        result["metrics"]["error_message"] = str(error)
    finally:
        write_benchmark_results(result, args.benchmark_results_path, save_tasks=args.save_tasks)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
