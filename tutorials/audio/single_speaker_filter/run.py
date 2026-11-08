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

"""Filter an ASR manifest to keep only single-speaker audio using Streaming Sortformer.

Four-stage pipeline via XennaExecutor with stage-boundary checkpointing:
  ManifestReader → InferenceSortformerStage → SingleSpeakerFilterStage → ManifestWriterStage

Input:
  NeMo-style JSONL manifest — one JSON object per line, at minimum:
    {"text": "the cat sat on a mat", "audio_filepath": "/path/to/file.wav"}

Output:
  Filtered JSONL manifest containing only entries where Sortformer detects
  exactly one speaker.

Usage:
    python tutorials/audio/single_speaker_filter/run.py \\
        --manifest /path/to/manifest.jsonl --output-dir /path/to/output

    # Resume from the last completed stage (same --output-dir)
    python tutorials/audio/single_speaker_filter/run.py \\
        --manifest /path/to/manifest.jsonl --output-dir /path/to/output

    # Clean run (remove all previous outputs and checkpoints)
    python tutorials/audio/single_speaker_filter/run.py \\
        --manifest /path/to/manifest.jsonl --output-dir /path/to/output --clean
"""

import argparse
import hashlib
import json
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import ManifestReader, ManifestWriterStage
from nemo_curator.stages.audio.inference.speaker_diarization.stage import InferenceSortformerStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, FileGroupTask
from nemo_curator.utils.atomic_io import write_json_atomically

CKPT_HASH_KEY = "_ckpt_hash"
CKPT_ORDER_KEY = "_ckpt_order"
CKPT_SUCCESS = "_SUCCESS"
CKPT_RUN = "_RUN.json"


# ---------------------------------------------------------------------------
# Stage-boundary checkpointing
# ---------------------------------------------------------------------------


def _task_hash(task: AudioTask | FileGroupTask) -> str:
    """Stable identity within one frozen manifest snapshot, including duplicate paths."""
    if not task.task_id:
        msg = "Cannot checkpoint a task before Curator assigns its task_id"
        raise ValueError(msg)
    return hashlib.sha256(task.task_id.encode()).hexdigest()[:16]


def _stage_ckpt_dir(checkpoint_dir: Path, stage_index: int, stage_name: str) -> Path:
    return checkpoint_dir / f"stage_{stage_index:02d}_{stage_name}"


def _save_task(directory: Path, h: str, task: AudioTask | FileGroupTask) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task.task_id,
        "dataset_name": task.dataset_name,
        "data": dict(task.data) if isinstance(task.data, Mapping) else task.data,
        "_metadata": task._metadata,
        "_task_type": type(task).__name__,
    }
    if isinstance(task, FileGroupTask):
        payload["reader_config"] = task.reader_config
    write_json_atomically(directory / f"{h}.json", payload, indent=2)


def _load_task(path: Path) -> AudioTask | FileGroupTask:
    payload = json.loads(path.read_text())
    if payload.get("_task_type") == "FileGroupTask":
        task = FileGroupTask(
            dataset_name=payload["dataset_name"],
            data=payload["data"],
            _metadata=payload.get("_metadata", {}),
            reader_config=payload.get("reader_config", {}),
        )
    else:
        task = AudioTask(
            dataset_name=payload["dataset_name"],
            data=payload["data"],
            _metadata=payload.get("_metadata", {}),
        )
    task.task_id = payload["task_id"]
    return task


def _load_all_tasks(directory: Path) -> list[AudioTask | FileGroupTask]:
    if not directory.exists():
        return []
    tasks = [_load_task(p) for p in directory.glob("*.json")]
    return sorted(tasks, key=lambda task: task._metadata[CKPT_ORDER_KEY])


def _stage_complete(directory: Path) -> bool:
    return (directory / CKPT_SUCCESS).is_file()


def _mark_stage_complete(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomically(directory / CKPT_SUCCESS, {})


def _bind_checkpoint_run(checkpoint_dir: Path, identity: dict[str, object]) -> None:
    """Refuse checkpoints created for different inputs or inference settings."""
    marker = checkpoint_dir / CKPT_RUN
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            msg = "Checkpoint input or configuration changed; rerun with --clean"
            raise RuntimeError(msg)
        return
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        msg = "Checkpoint provenance is missing; rerun with --clean"
        raise RuntimeError(msg)
    write_json_atomically(marker, identity, indent=2)


def _manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


@dataclass
class SingleSpeakerFilterStage(ProcessingStage[AudioTask, AudioTask]):
    """Keep only audio samples where Sortformer detected exactly one speaker."""

    diar_segments_key: str = "diar_segments"
    num_speakers_key: str = "num_speakers"
    name: str = "SingleSpeakerFilter"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.diar_segments_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.num_speakers_key]

    def validate_input(self, task: AudioTask) -> bool:
        if not hasattr(task, "data") or task.data is None:
            return False
        return self.diar_segments_key in task.data

    def process(self, task: AudioTask) -> AudioTask:
        msg = "SingleSpeakerFilterStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        results = []
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task!s} failed validation for stage {self}"
                raise ValueError(msg)
            segments = task.data.get(self.diar_segments_key, [])
            speakers = {seg["speaker"] for seg in segments}
            if len(speakers) == 1:
                output_data = {k: v for k, v in task.data.items() if k != self.diar_segments_key}
                output_data[self.num_speakers_key] = 1
                results.append(
                    AudioTask(
                        dataset_name=task.dataset_name,
                        data=output_data,
                        _metadata=task._metadata,
                        _stage_perf=task._stage_perf,
                    )
                )
        return results


# ---------------------------------------------------------------------------
# Checkpointed execution
# ---------------------------------------------------------------------------


def _run_stages_with_checkpoints(  # noqa: C901, PLR0912, PLR0915
    stages: list[ProcessingStage],
    checkpoint_dir: Path,
) -> list[AudioTask | FileGroupTask]:
    """Execute stages sequentially, reusing only complete stage boundaries.

    A stage interrupted inside ``Pipeline.run`` is rerun as a whole because
    the executor returns its results only after that stage succeeds.
    """
    executor = XennaExecutor()
    current_tasks: list[AudioTask | FileGroupTask] | None = None
    t0 = time.time()

    for idx, stage in enumerate(stages):
        sdir = _stage_ckpt_dir(checkpoint_dir, idx, stage._name)

        is_writer = isinstance(stage, ManifestWriterStage)
        if _stage_complete(sdir) and not is_writer:
            cached = _load_all_tasks(sdir)
            logger.info(f"Stage {idx} ({stage._name}): loaded {len(cached)} cached tasks — skipping")
            current_tasks = cached
            continue
        if is_writer and sdir.exists():
            shutil.rmtree(sdir)

        # --- reader stage (no input tasks) ---
        if current_tasks is None:
            if sdir.exists():
                shutil.rmtree(sdir)

            logger.info(f"Running stage {idx}/{len(stages) - 1}: {stage._name}")
            stage_t0 = time.time()
            pipeline = Pipeline(name=f"stage_{idx}_{stage._name}", stages=[stage])
            output = pipeline.run(executor=executor)
            current_tasks = output or []

            for order, task in enumerate(current_tasks):
                h = _task_hash(task)
                task._metadata[CKPT_HASH_KEY] = h
                task._metadata[CKPT_ORDER_KEY] = order
                _save_task(sdir, h, task)
            _mark_stage_complete(sdir)
            logger.info(f"Stage {stage._name} done in {time.time() - stage_t0:.1f}s — {len(current_tasks)} tasks")
            continue

        # --- subsequent stages: split cached vs todo ---
        cached_tasks: list[AudioTask | FileGroupTask] = []
        todo_tasks: list[AudioTask | FileGroupTask] = []

        if is_writer:
            todo_tasks = current_tasks
        else:
            for task in current_tasks:
                h = task._metadata.get(CKPT_HASH_KEY) or _task_hash(task)
                if (sdir / f"{h}.json").exists():
                    cached_task = _load_task(sdir / f"{h}.json")
                    cached_task._metadata[CKPT_ORDER_KEY] = task._metadata[CKPT_ORDER_KEY]
                    cached_tasks.append(cached_task)
                else:
                    task._metadata[CKPT_HASH_KEY] = h
                    todo_tasks.append(task)

        if cached_tasks:
            logger.info(f"Stage {idx} ({stage._name}): {len(cached_tasks)} cached, {len(todo_tasks)} remaining")

        if todo_tasks:
            logger.info(f"Running stage {idx}/{len(stages) - 1}: {stage._name} ({len(todo_tasks)} tasks)")
            stage_t0 = time.time()
            pipeline = Pipeline(name=f"stage_{idx}_{stage._name}", stages=[stage])
            output = pipeline.run(executor=executor, initial_tasks=todo_tasks)
            new_tasks = output or []

            for task in new_tasks:
                if not is_writer:
                    h = task._metadata.get(CKPT_HASH_KEY) or _task_hash(task)
                    task._metadata[CKPT_HASH_KEY] = h
                    _save_task(sdir, h, task)
                cached_tasks.append(task)

            logger.info(f"Stage {stage._name} done in {time.time() - stage_t0:.1f}s — {len(new_tasks)} tasks")
        elif is_writer:
            stage.setup()
            stage.finalize()
        else:
            logger.info(f"Stage {idx} ({stage._name}): all tasks processed — skipping")

        cached_tasks.sort(key=lambda task: task._metadata[CKPT_ORDER_KEY])
        if not is_writer:
            active_hashes = {task._metadata[CKPT_HASH_KEY] for task in cached_tasks}
            for path in sdir.glob("*.json"):
                if path.stem not in active_hashes:
                    path.unlink()
            _mark_stage_complete(sdir)
        current_tasks = cached_tasks

    total = time.time() - t0
    logger.info(f"All stages done in {total / 60:.1f} min")
    return current_tasks or []


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter ASR manifest to single-speaker audio via Streaming Sortformer.")
    p.add_argument("--manifest", type=Path, required=True, help="Input NeMo-style JSONL manifest.")
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="Root for all outputs.")
    p.add_argument("--model", default="nvidia/diar_streaming_sortformer_4spk-v2.1", help="HF Sortformer model id.")
    p.add_argument("--clean", action="store_true", help="Remove output directory before running.")
    p.add_argument("--chunk-len", type=int, default=340, help="Streaming chunk size in 80ms frames.")
    p.add_argument("--chunk-right-context", type=int, default=40, help="Right context frames.")
    p.add_argument("--fifo-len", type=int, default=40, help="FIFO queue size in frames.")
    p.add_argument("--spkcache-update-period", type=int, default=300, help="Speaker cache update period in frames.")
    p.add_argument("--spkcache-len", type=int, default=188, help="Speaker cache size in frames.")

    args = p.parse_args()
    args.checkpoint_dir = args.output_dir / "checkpoints"
    args.output_manifest = args.output_dir / "filtered_manifest.jsonl"
    return args


def _ensure_safe_clean(output_dir: Path, protected_path: Path) -> None:
    if protected_path.resolve().is_relative_to(output_dir.resolve()):
        msg = f"Refusing --clean because output directory contains input path: {protected_path}"
        raise ValueError(msg)


def _ensure_manifest_sources_safe(output_dir: Path, manifest: Path) -> None:
    """Prevent --clean from deleting the manifest or any local audio it names."""
    _ensure_safe_clean(output_dir, manifest)
    with manifest.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            audio_filepath = json.loads(line).get("audio_filepath")
            if isinstance(audio_filepath, str) and audio_filepath:
                _ensure_safe_clean(output_dir, Path(audio_filepath).expanduser())


def main() -> None:
    args = parse_args()

    if args.clean and args.output_dir.exists():
        _ensure_manifest_sources_safe(args.output_dir, args.manifest)
        shutil.rmtree(args.output_dir)
        logger.info(f"Cleaned output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # shortcut: referenced audio bytes are not hashed; use --clean after replacing them.
    _bind_checkpoint_run(
        args.checkpoint_dir,
        {
            "version": 1,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": _manifest_sha256(args.manifest),
            "model": args.model,
            "chunk_len": args.chunk_len,
            "chunk_right_context": args.chunk_right_context,
            "fifo_len": args.fifo_len,
            "spkcache_update_period": args.spkcache_update_period,
            "spkcache_len": args.spkcache_len,
        },
    )

    with open(args.manifest) as f:
        total_entries = sum(1 for line in f if line.strip())
    if total_entries == 0:
        print(f"Empty manifest: {args.manifest}")
        return

    has_checkpoint = any(args.checkpoint_dir.glob("stage_*"))
    resume_info = " (resuming from checkpoint)" if has_checkpoint else ""
    print(f"Manifest: {total_entries} entries{resume_info}", flush=True)

    ray_client = RayClient()
    ray_client.start()

    try:
        stages: list[ProcessingStage] = [
            ManifestReader(manifest_path=str(args.manifest)),
            InferenceSortformerStage(
                model_id=args.model,
                adapter_kwargs={
                    "chunk_len": args.chunk_len,
                    "chunk_right_context": args.chunk_right_context,
                    "fifo_len": args.fifo_len,
                    "spkcache_update_period": args.spkcache_update_period,
                    "spkcache_len": args.spkcache_len,
                    "inference_batch_size": 1,
                },
            ).with_(resources=Resources(gpus=1), batch_size=1),
            SingleSpeakerFilterStage(),
            ManifestWriterStage(output_path=str(args.output_manifest)),
        ]

        print("Starting pipeline with inter-stage checkpointing...", flush=True)
        _run_stages_with_checkpoints(stages, args.checkpoint_dir)
    finally:
        ray_client.stop()

    if args.output_manifest.exists():
        with open(args.output_manifest) as f:
            entries_out = sum(1 for line in f if line.strip())
    else:
        entries_out = 0
    print(f"\n{'=' * 60}")
    print(f"Filtered: {entries_out} / {total_entries} entries have exactly 1 speaker")
    if entries_out > 0:
        print(f"Output manifest: {args.output_manifest}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
