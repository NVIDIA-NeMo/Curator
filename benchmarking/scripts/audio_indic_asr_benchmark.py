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

"""Benchmark the production Hindi Indic Canary + Parakeet recovery pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import ManifestReader, ManifestWriterStage, load_audio_file
from nemo_curator.stages.audio.inference.indic_canary import InferenceIndicCanaryStage
from nemo_curator.stages.audio.inference.indic_canary_trtllm_runtime import _require_tensorrt_llm
from nemo_curator.stages.audio.inference.parakeet import InferenceParakeetStage
from nemo_curator.stages.audio.metrics.wer import GetPairwiseWerStage
from nemo_curator.stages.audio.text_filtering import (
    AbbreviationConcatStage,
    RegexSubstitutionStage,
    SelectBestPredictionStage,
    WhisperHallucinationStage,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

EXPECTED_NUM_ROWS = 216_164
EXPECTED_TOTAL_DURATION_MS = 1_914_327_468
EXPECTED_MANIFEST_BYTES = 116_382_162
EXPECTED_MANIFEST_SHA256 = "0a8ccc0f3ff8d4ad35b3e7e104e5e093b0de14a92542d71fa6911c8045373727"
SAMPLE_RATE = 16_000
MAX_DURATION_ERROR_S = 0.001
GPU_ACTOR_RUNTIME_ENV = {"env_vars": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}}
CANARY_REQUIRED_FILES = (
    "encoder/encoder.plan",
    "encoder/config.json",
    "decoder/config.json",
    "decoder/rank0.engine",
    "decoder/vocab.json",
    "preprocessor/config.json",
    "preprocessor/mel_basis.pt",
)
PARAKEET_ENGINE_REQUIRED_FILES = ("encoder.plan", "metadata.json", "model.nemo")


def _load_benchmark_utils() -> tuple[Any, Any]:
    """Load harness-only helpers without burdening Hydra stage imports."""
    if __package__:
        from benchmarking.scripts.utils import setup_executor, write_benchmark_results
    else:  # Direct script execution adds only benchmarking/scripts to sys.path.
        from utils import setup_executor, write_benchmark_results

    return setup_executor, write_benchmark_results


@dataclass(frozen=True)
class InputInventory:
    """Identity and duration contract for one staged benchmark cohort."""

    manifests: tuple[Path, ...]
    audio_paths_by_id: dict[str, str]
    durations_ms_by_id: dict[str, int]
    reference_texts_by_id: dict[str, str]
    manifest_bytes: int
    manifest_sha256: str

    @property
    def num_rows(self) -> int:
        return len(self.audio_paths_by_id)

    @property
    def total_duration_s(self) -> float:
        return self.total_duration_ms / 1000

    @property
    def total_duration_ms(self) -> int:
        return sum(self.durations_ms_by_id.values())


@dataclass
class PrepareIndicASRInputStage(ProcessingStage[AudioTask, AudioTask]):
    """Load validated 16 kHz audio and initialize the Granary-v2 fields."""

    name: str = "prepare_indic_asr_input"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    audio_root: str | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["audio_filepath", "audio_item_id", "duration", "source_lang", "text"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            "additional_notes",
            "granary_v1_prediction",
            "reference_text",
            "sampling_rate",
            "waveform",
            "_skipme",
        ]

    def process(self, task: AudioTask) -> AudioTask:
        data = task.data
        audio_item_id = data["audio_item_id"]
        reference_text = data.pop("text")
        if not isinstance(audio_item_id, str) or not audio_item_id:
            msg = "audio_item_id must be a nonempty string"
            raise RuntimeError(msg)
        if not isinstance(reference_text, str) or not reference_text.strip():
            msg = f"Reference text is empty for {audio_item_id}"
            raise RuntimeError(msg)
        if data["source_lang"] != "hi":
            msg = f"Expected source_lang='hi' for {audio_item_id}, found {data['source_lang']!r}"
            raise RuntimeError(msg)

        audio_path = Path(data["audio_filepath"])
        if not audio_path.is_absolute():
            if self.audio_root is None:
                msg = f"Relative audio path requires audio_root for {audio_item_id}: {audio_path}"
                raise RuntimeError(msg)
            audio_path = Path(self.audio_root) / audio_path
        audio_path = audio_path.expanduser().resolve()
        if not audio_path.is_file():
            msg = f"Input audio is missing for {audio_item_id}: {audio_path}"
            raise FileNotFoundError(msg)
        data["audio_filepath"] = str(audio_path)

        waveform, sample_rate = load_audio_file(str(audio_path), mono=True)
        duration_s = float(data["duration"])
        measured_duration_s = waveform.shape[-1] / sample_rate
        if sample_rate != SAMPLE_RATE or waveform.shape[0] != 1:
            msg = f"Expected mono {SAMPLE_RATE} Hz audio for {audio_item_id}"
            raise RuntimeError(msg)
        if not math.isclose(duration_s, measured_duration_s, rel_tol=0, abs_tol=MAX_DURATION_ERROR_S):
            msg = (
                f"Manifest duration mismatch for {audio_item_id}: "
                f"manifest={duration_s}, measured={measured_duration_s}"
            )
            raise RuntimeError(msg)

        notes = data.get("additional_notes")
        notes = notes if isinstance(notes, dict) else {}
        notes.update({"primary_model": "indic_canary", "recovery_model": "parakeet_riva"})
        data["additional_notes"] = notes
        data["granary_v1_prediction"] = reference_text
        data["reference_text"] = reference_text
        data["sampling_rate"] = sample_rate
        data["waveform"] = waveform
        data["_skipme"] = ""
        return task


def _normalize_input_row(
    row: object,
    manifest_path: Path,
    line_number: int,
) -> tuple[dict[str, Any], int]:
    if not isinstance(row, dict):
        msg = f"Input manifest row must be an object: {manifest_path}:{line_number}"
        raise TypeError(msg)
    try:
        audio_item_id = row["audio_item_id"]
        audio_filepath = row["audio_filepath"]
        duration = Decimal(str(row["duration"]))
        source_lang = row["source_lang"]
        text = row["text"]
    except (InvalidOperation, KeyError, TypeError, ValueError) as e:
        msg = f"Invalid input manifest row: {manifest_path}:{line_number}"
        raise RuntimeError(msg) from e
    duration_ms_decimal = duration * 1000
    if (
        not isinstance(audio_item_id, str)
        or not audio_item_id
        or not isinstance(audio_filepath, str)
        or not audio_filepath
        or not duration.is_finite()
        or duration <= 0
        or duration_ms_decimal != duration_ms_decimal.to_integral_value()
        or source_lang != "hi"
        or not isinstance(text, str)
        or not text.strip()
    ):
        msg = f"Invalid input manifest values: {manifest_path}:{line_number}"
        raise RuntimeError(msg)

    resolved_audio_path = Path(audio_filepath)
    if not resolved_audio_path.is_absolute():
        resolved_audio_path = manifest_path.parent / resolved_audio_path
    resolved_audio_path = resolved_audio_path.resolve()
    if not resolved_audio_path.is_file():
        msg = f"Input audio is missing: {resolved_audio_path}"
        raise FileNotFoundError(msg)

    normalized = dict(row)
    normalized["audio_filepath"] = str(resolved_audio_path)
    normalized["duration"] = float(duration)
    return normalized, int(duration_ms_decimal)


def _materialize_input_manifests(  # noqa: C901
    source_manifest: Path,
    scratch_dir: Path,
    num_shards: int,
) -> InputInventory:
    if num_shards <= 0:
        msg = f"num_shards must be positive, got {num_shards}"
        raise ValueError(msg)
    if not source_manifest.is_file():
        msg = f"Input manifest does not exist: {source_manifest}"
        raise FileNotFoundError(msg)

    rows: list[dict[str, Any]] = []
    audio_paths_by_id: dict[str, str] = {}
    durations_ms_by_id: dict[str, int] = {}
    reference_texts_by_id: dict[str, str] = {}
    seen_audio_paths: set[str] = set()
    manifest_digest = hashlib.sha256()
    manifest_bytes = 0
    with source_manifest.open("rb") as source_file:
        opened_manifest_bytes = os.fstat(source_file.fileno()).st_size
        for line_number, raw_line in enumerate(source_file, start=1):
            manifest_digest.update(raw_line)
            manifest_bytes += len(raw_line)
            if not raw_line.strip():
                continue
            try:
                raw_row = json.loads(raw_line, parse_float=Decimal)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                msg = f"Invalid JSON input manifest row: {source_manifest}:{line_number}"
                raise RuntimeError(msg) from e
            row, duration_ms = _normalize_input_row(raw_row, source_manifest, line_number)
            audio_item_id = row["audio_item_id"]
            audio_path = row["audio_filepath"]
            if audio_item_id in audio_paths_by_id or audio_path in seen_audio_paths:
                msg = f"Duplicate input identity or audio path at {source_manifest}:{line_number}"
                raise RuntimeError(msg)
            audio_paths_by_id[audio_item_id] = audio_path
            durations_ms_by_id[audio_item_id] = duration_ms
            reference_texts_by_id[audio_item_id] = row["text"]
            seen_audio_paths.add(audio_path)
            rows.append(row)
    if manifest_bytes != opened_manifest_bytes:
        msg = f"Input manifest changed size while it was read: opened={opened_manifest_bytes}, read={manifest_bytes}"
        raise RuntimeError(msg)
    manifest_sha256 = manifest_digest.hexdigest()
    if not rows:
        msg = f"Input manifest contains no rows: {source_manifest}"
        raise RuntimeError(msg)

    scratch_dir.mkdir(parents=True, exist_ok=False)
    shard_paths = tuple(scratch_dir / f"manifest-{index:02d}.jsonl" for index in range(num_shards))
    shard_files = [path.open("x", encoding="utf-8") for path in shard_paths]
    try:
        for index, row in enumerate(rows):
            shard_files[index % num_shards].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    finally:
        for shard_file in shard_files:
            shard_file.close()

    return InputInventory(
        manifests=shard_paths,
        audio_paths_by_id=audio_paths_by_id,
        durations_ms_by_id=durations_ms_by_id,
        reference_texts_by_id=reference_texts_by_id,
        manifest_bytes=manifest_bytes,
        manifest_sha256=manifest_sha256,
    )


def _require_files(root: Path, relative_paths: tuple[str, ...], label: str) -> None:
    missing = [str(root / relative_path) for relative_path in relative_paths if not (root / relative_path).is_file()]
    if missing:
        msg = f"{label} is missing required file(s): {missing}"
        raise FileNotFoundError(msg)


def _preflight_runtime_and_models(
    indic_canary_engine_dir: Path,
    parakeet_tensorrt_engine_dir: Path,
) -> None:
    if sys.version_info[:2] != (3, 12):
        msg = (
            "Indic ASR requires the root trt_llm profile with Python 3.12: "
            "uv sync --locked --python 3.12 --extra trt_llm --no-default-groups"
        )
        raise RuntimeError(msg)
    missing_modules = [module for module in ("tensorrt", "tensorrt_llm") if importlib.util.find_spec(module) is None]
    if missing_modules:
        msg = (
            "Indic ASR benchmark image is missing required runtime module(s): "
            f"{missing_modules}. Install the root trt_llm profile."
        )
        raise RuntimeError(msg)
    # Loading the native bindings catches ABI failures that find_spec cannot.
    _require_tensorrt_llm()
    for module in (
        "nemo.collections.asr",
        "nemo_text_processing.text_normalization",
        "nemo_curator.models.asr.indic_parakeet_rnnt_tensorrt",
    ):
        importlib.import_module(module)
    _require_files(indic_canary_engine_dir, CANARY_REQUIRED_FILES, "Indic Canary engine")
    _require_files(parakeet_tensorrt_engine_dir, PARAKEET_ENGINE_REQUIRED_FILES, "Indic Parakeet TensorRT bundle")
    logger.info(f"Indic ASR Python: {sys.executable}")
    for package_name in ("torch", "torchaudio", "tensorrt", "tensorrt-llm", "nemo-toolkit", "transformers"):
        try:
            logger.info(f"{package_name} version: {importlib.metadata.version(package_name)}")
        except importlib.metadata.PackageNotFoundError:
            logger.info(f"{package_name} is importable but has no package metadata")


def _build_pipeline(  # noqa: PLR0913
    input_manifests: tuple[Path, ...],
    output_manifest: Path,
    indic_canary_engine_dir: Path,
    parakeet_tensorrt_engine_dir: Path,
    regex_yaml: Path,
    hall_phrases: Path,
    read_concurrency: int,
    prep_workers: int,
    primary_workers: int,
    fallback_workers: int,
) -> Pipeline:
    reader = ManifestReader(
        manifest_path=[str(path) for path in input_manifests],
        files_per_partition=1,
    ).with_({"manifest_reader_stage": {"num_workers": read_concurrency}})
    return Pipeline(
        name="audio_indic_asr",
        description="Hindi Indic Canary ASR with Parakeet TensorRT recovery and Granary-v2 text cleanup",
        stages=[
            reader,
            PrepareIndicASRInputStage().with_(batch_size=64, num_workers=prep_workers),
            InferenceIndicCanaryStage(
                name="IndicCanary_primary",
                engine_dir=str(indic_canary_engine_dir),
                num_beams=4,
                max_new_tokens=347,
                pnc=False,
                kv_cache_free_gpu_memory_fraction=0.1,
                cross_kv_cache_fraction=0.1,
                source_lang_key="source_lang",
                pred_text_key="primary_model_prediction",
                keep_waveform=True,
                batch_size=64,
                num_workers_override=primary_workers,
                resources=Resources(gpu_memory_gb=50),
            ).with_(runtime_env=GPU_ACTOR_RUNTIME_ENV),
            WhisperHallucinationStage(
                name="WhisperHallucination_primary",
                common_hall_file=str(hall_phrases),
                text_key="primary_model_prediction",
                language_key="source_lang",
            ).with_(batch_size=64),
            InferenceParakeetStage(
                name="ParakeetRiva_recovery",
                model_id=str(parakeet_tensorrt_engine_dir / "model.nemo"),
                supported_langs={"hi"},
                backend="tensorrt",
                tensorrt_engine_dir=str(parakeet_tensorrt_engine_dir),
                chunking_mode="engine",
                source_lang_key="source_lang",
                pred_text_key="fallback_model_prediction",
                keep_waveform=False,
                batch_size=64,
                num_workers_override=fallback_workers,
                resources=Resources(gpu_memory_gb=24),
            ).with_(runtime_env=GPU_ACTOR_RUNTIME_ENV),
            WhisperHallucinationStage(
                name="WhisperHallucination_asr",
                common_hall_file=str(hall_phrases),
                text_key="fallback_model_prediction",
                language_key="source_lang",
                overwrite=True,
                recovery_value="Recovered:ASR",
            ).with_(batch_size=64),
            SelectBestPredictionStage(
                primary_text_key="primary_model_prediction",
                fallback_text_key="fallback_model_prediction",
                output_key="best_prediction",
                source_key="best_prediction_source",
                primary_source_label="primary",
                fallback_source_label="fallback",
                reference_text_key="granary_v1_prediction",
                use_ground_truth_for_short_audio=False,
                primary_model_type="indic_canary",
                language_key="source_lang",
            ).with_(batch_size=64),
            RegexSubstitutionStage(
                regex_params_yaml=str(regex_yaml),
                text_key="best_prediction",
                output_text_key="cleaned_text",
            ).with_(batch_size=64),
            AbbreviationConcatStage(
                text_key="cleaned_text",
                output_text_key="abbreviated_text",
                source_lang_key="source_lang",
            ).with_(batch_size=64),
            GetPairwiseWerStage(
                text_key="reference_text",
                pred_text_key="abbreviated_text",
                wer_key="wer_pct",
            ).with_(batch_size=64),
            ManifestWriterStage(output_path=str(output_manifest)).with_(batch_size=64, num_workers=1),
        ],
    )


def _coverage(rows: list[dict[str, Any]], key: str) -> float:
    return sum(isinstance(row.get(key), str) and bool(row[key].strip()) for row in rows) / len(rows)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _validate_output_manifest(output_manifest: Path, inventory: InputInventory) -> dict[str, float | int | str]:
    if not output_manifest.is_file():
        msg = f"Indic ASR pipeline wrote no output manifest: {output_manifest}"
        raise RuntimeError(msg)
    rows = [json.loads(line) for line in output_manifest.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != inventory.num_rows:
        msg = f"Indic ASR returned {len(rows)} rows for {inventory.num_rows} input rows"
        raise RuntimeError(msg)

    output_ids: set[str] = set()
    wers: list[float] = []
    output_durations_ms: list[int] = []
    source_counts = {"primary": 0, "fallback": 0, "reference": 0, "ground_truth": 0, "other": 0}
    skipped_rows = 0
    for row_index, row in enumerate(rows):
        try:
            audio_item_id = row["audio_item_id"]
            audio_filepath = row["audio_filepath"]
            duration_s = float(row["duration"])
            duration_ms_decimal = Decimal(str(row["duration"])) * 1000
            source_lang = row["source_lang"]
            reference_text = row["reference_text"]
            granary_v1_prediction = row["granary_v1_prediction"]
            wer_pct = float(row["wer_pct"])
            source = row["best_prediction_source"]
            skip_reason = row["_skipme"]
            model_text_values = [row["primary_model_prediction"], row["fallback_model_prediction"]]
            selected_text_values = [
                row["best_prediction"],
                row["cleaned_text"],
                row["abbreviated_text"],
            ]
        except (KeyError, TypeError, ValueError) as e:
            msg = f"Invalid Indic ASR output row {row_index}"
            raise RuntimeError(msg) from e
        if (
            not isinstance(audio_item_id, str)
            or audio_item_id in output_ids
            or inventory.audio_paths_by_id.get(audio_item_id) != audio_filepath
            or not math.isfinite(duration_s)
            or duration_ms_decimal != duration_ms_decimal.to_integral_value()
            or int(duration_ms_decimal) != inventory.durations_ms_by_id.get(audio_item_id, -1)
            or source_lang != "hi"
            or reference_text != inventory.reference_texts_by_id.get(audio_item_id)
            or granary_v1_prediction != reference_text
            or not math.isfinite(wer_pct)
            or wer_pct < 0
            or source not in {"primary", "fallback"}
            or not isinstance(skip_reason, str)
            or not all(isinstance(value, str) for value in model_text_values)
            or not isinstance(selected_text_values[0], str)
            or not selected_text_values[0].strip()
            or not all(isinstance(value, str) for value in selected_text_values[1:])
            or (not skip_reason and not all(bool(value.strip()) for value in selected_text_values[1:]))
        ):
            msg = f"Invalid Indic ASR output values in row {row_index}"
            raise RuntimeError(msg)
        output_ids.add(audio_item_id)
        wers.append(wer_pct)
        output_durations_ms.append(int(duration_ms_decimal))
        skipped_rows += bool(skip_reason)
        source_counts[source if source in source_counts else "other"] += 1

    expected_ids = set(inventory.audio_paths_by_id)
    if output_ids != expected_ids:
        msg = f"Indic ASR output identity mismatch: missing={len(expected_ids - output_ids)}, extra={len(output_ids - expected_ids)}"
        raise RuntimeError(msg)
    total_duration_ms = sum(output_durations_ms)
    if total_duration_ms != inventory.total_duration_ms:
        msg = (
            f"Indic ASR output duration changed: input_ms={inventory.total_duration_ms}, output_ms={total_duration_ms}"
        )
        raise RuntimeError(msg)

    return {
        "num_input_rows": inventory.num_rows,
        "num_output_rows": len(rows),
        "input_output_coverage_ratio": len(rows) / inventory.num_rows,
        "total_audio_duration_ms": total_duration_ms,
        "total_audio_duration_hours": total_duration_ms / 3_600_000,
        "input_manifest_bytes": inventory.manifest_bytes,
        "input_manifest_sha256": inventory.manifest_sha256,
        "primary_prediction_coverage_ratio": _coverage(rows, "primary_model_prediction"),
        "fallback_prediction_coverage_ratio": _coverage(rows, "fallback_model_prediction"),
        "best_prediction_coverage_ratio": _coverage(rows, "best_prediction"),
        "cleaned_text_coverage_ratio": _coverage(rows, "cleaned_text"),
        "abbreviated_text_coverage_ratio": _coverage(rows, "abbreviated_text"),
        "wer_output_coverage_ratio": len(wers) / inventory.num_rows,
        "mean_wer_pct": statistics.fmean(wers),
        "p50_wer_pct": _percentile(wers, 0.50),
        "p90_wer_pct": _percentile(wers, 0.90),
        "num_skipped_rows": skipped_rows,
        "prediction_source_primary_rows": source_counts["primary"],
        "prediction_source_fallback_rows": source_counts["fallback"],
        "prediction_source_reference_rows": source_counts["reference"],
        "prediction_source_ground_truth_rows": source_counts["ground_truth"],
        "prediction_source_other_rows": source_counts["other"],
        "recognized_prediction_source_ratio": (source_counts["primary"] + source_counts["fallback"])
        / inventory.num_rows,
    }


def run_audio_indic_asr_benchmark(  # noqa: PLR0913
    benchmark_results_path: str,
    input_manifest: str,
    indic_canary_engine_dir: str,
    parakeet_tensorrt_engine_dir: str,
    regex_yaml: str,
    hall_phrases: str,
    executor: str = "ray_data",
    expected_num_rows: int = EXPECTED_NUM_ROWS,
    expected_total_duration_ms: int = EXPECTED_TOTAL_DURATION_MS,
    expected_manifest_bytes: int = EXPECTED_MANIFEST_BYTES,
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
    read_concurrency: int = 4,
    prep_workers: int = 24,
    primary_workers: int = 8,
    fallback_workers: int = 8,
) -> dict[str, Any]:
    """Run the pinned production processor chain over the staged Hindi cohort."""
    benchmark_path = Path(benchmark_results_path)
    scratch_dir = benchmark_path / "scratch" / "audio_indic_asr_input"
    output_manifest = benchmark_path / "results" / "audio_indic_asr_output.jsonl"
    canary_path = Path(indic_canary_engine_dir)
    parakeet_engine = Path(parakeet_tensorrt_engine_dir)
    regex_path = Path(regex_yaml)
    phrases_path = Path(hall_phrases)

    if output_manifest.exists() or scratch_dir.exists():
        msg = f"Indic ASR benchmark output already exists under {benchmark_path}"
        raise ValueError(msg)
    for config_path in (regex_path, phrases_path):
        if not config_path.is_file():
            msg = f"Indic ASR configuration file is missing: {config_path}"
            raise FileNotFoundError(msg)
    _preflight_runtime_and_models(canary_path, parakeet_engine)
    inventory = _materialize_input_manifests(Path(input_manifest), scratch_dir, read_concurrency)
    if inventory.num_rows != expected_num_rows:
        msg = f"Expected {expected_num_rows} Hindi rows, found {inventory.num_rows}"
        raise RuntimeError(msg)
    if inventory.total_duration_ms != expected_total_duration_ms:
        msg = f"Expected {expected_total_duration_ms} ms of Hindi audio, found {inventory.total_duration_ms}"
        raise RuntimeError(msg)
    if inventory.manifest_bytes != expected_manifest_bytes:
        msg = f"Expected {expected_manifest_bytes} manifest bytes, found {inventory.manifest_bytes}"
        raise RuntimeError(msg)
    if inventory.manifest_sha256 != expected_manifest_sha256.lower():
        msg = f"Expected manifest SHA-256 {expected_manifest_sha256.lower()}, found {inventory.manifest_sha256}"
        raise RuntimeError(msg)

    pipeline = _build_pipeline(
        input_manifests=inventory.manifests,
        output_manifest=output_manifest,
        indic_canary_engine_dir=canary_path,
        parakeet_tensorrt_engine_dir=parakeet_engine,
        regex_yaml=regex_path,
        hall_phrases=phrases_path,
        read_concurrency=read_concurrency,
        prep_workers=prep_workers,
        primary_workers=primary_workers,
        fallback_workers=fallback_workers,
    )
    logger.info(pipeline.describe())
    setup_executor, _ = _load_benchmark_utils()
    start_time = time.perf_counter()
    tasks = pipeline.run(setup_executor(executor))
    pipeline_elapsed_s = time.perf_counter() - start_time
    output_metrics = _validate_output_manifest(output_manifest, inventory)
    elapsed_s = time.perf_counter() - start_time
    del tasks
    total_audio_hours = float(output_metrics["total_audio_duration_hours"])
    logger.success(f"Processed {inventory.num_rows} unique Hindi clips in {elapsed_s:.2f}s")
    return {
        "metrics": {
            "is_success": True,
            "time_taken_s": elapsed_s,
            "pipeline_time_s": pipeline_elapsed_s,
            **output_metrics,
            "throughput_tasks_per_sec": inventory.num_rows / elapsed_s if elapsed_s > 0 else 0,
            "throughput_audio_hours_per_hour": total_audio_hours * 3600 / elapsed_s if elapsed_s > 0 else 0,
        },
        "tasks": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--indic-canary-engine-dir", required=True)
    parser.add_argument("--parakeet-tensorrt-engine-dir", required=True)
    parser.add_argument("--regex-yaml", required=True)
    parser.add_argument("--hall-phrases", required=True)
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"])
    parser.add_argument("--expected-num-rows", type=int, default=EXPECTED_NUM_ROWS)
    parser.add_argument("--expected-total-duration-ms", type=int, default=EXPECTED_TOTAL_DURATION_MS)
    parser.add_argument("--expected-manifest-bytes", type=int, default=EXPECTED_MANIFEST_BYTES)
    parser.add_argument("--expected-manifest-sha256", default=EXPECTED_MANIFEST_SHA256)
    parser.add_argument("--read-concurrency", type=int, default=4)
    parser.add_argument("--prep-workers", type=int, default=24)
    parser.add_argument("--primary-workers", type=int, default=8)
    parser.add_argument("--fallback-workers", type=int, default=8)
    args = parser.parse_args()

    params = vars(args)
    logger.info(f"Audio Indic ASR benchmark arguments: {params}")
    results: dict[str, Any] = {"params": params, "metrics": {"is_success": False}, "tasks": []}
    exit_code = 1
    try:
        results.update(run_audio_indic_asr_benchmark(**params))
        exit_code = 0
    except Exception as e:
        logger.error(f"Indic ASR benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        results["metrics"]["error_message"] = str(e)
    finally:
        _, write_benchmark_results = _load_benchmark_utils()
        write_benchmark_results(results, args.benchmark_results_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
