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

Four-stage pipeline via XennaExecutor with per-task hash-based checkpointing:
  ALMManifestReader → InferenceSortformerStage → SingleSpeakerFilterStage → ALMManifestWriterStage

Input:
  NeMo-style JSONL manifest — one JSON object per line, at minimum:
    {"text": "the cat sat on a mat", "audio_filepath": "/path/to/file.wav"}

Output:
  Filtered JSONL manifest containing only entries where Sortformer detects
  exactly one speaker.

Usage:
    python tutorials/audio/single_speaker_filter/run.py \\
        --manifest /path/to/manifest.jsonl --output-dir /path/to/output

    # Resume from checkpoint after partial failure (same --output-dir)
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
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm.alm_manifest_reader import ALMManifestReader
from nemo_curator.stages.audio.alm.alm_manifest_writer import ALMManifestWriterStage
from nemo_curator.stages.audio.inference.sortformer import InferenceSortformerStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioBatch

CKPT_HASH_KEY = "_ckpt_hash"


# ---------------------------------------------------------------------------
# Per-task hash-based checkpointing
# ---------------------------------------------------------------------------


def _task_hash(task: AudioBatch) -> str:
    """Stable content hash derived from audio_filepath."""
    identifiers: list[str] = []
    for item in task.data or []:
        if isinstance(item, dict) and "audio_filepath" in item:
            identifiers.append(item["audio_filepath"])
    if not identifiers:
        identifiers.append(task.task_id)
    raw = "|".join(sorted(identifiers))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _stage_ckpt_dir(checkpoint_dir: Path, stage_index: int, stage_name: str) -> Path:
    return checkpoint_dir / f"stage_{stage_index:02d}_{stage_name}"


def _save_task(directory: Path, h: str, task: AudioBatch) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task.task_id,
        "dataset_name": task.dataset_name,
        "data": task.data,
        "_metadata": task._metadata,
    }
    (directory / f"{h}.json").write_text(json.dumps(payload, indent=2))


def _load_task(path: Path) -> AudioBatch:
    payload = json.loads(path.read_text())
    return AudioBatch(
        task_id=payload["task_id"],
        dataset_name=payload["dataset_name"],
        data=payload["data"],
        _metadata=payload.get("_metadata", {}),
    )


def _load_all_tasks(directory: Path) -> list[AudioBatch]:
    if not directory.exists():
        return []
    return [_load_task(p) for p in sorted(directory.glob("*.json"))]


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


@dataclass
class SingleSpeakerFilterStage(ProcessingStage[AudioBatch, AudioBatch]):
    """Keep only audio samples where Sortformer detected exactly one speaker."""

    diar_segments_key: str = "diar_segments"
    name: str = "SingleSpeakerFilter"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.diar_segments_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.diar_segments_key]

    def validate_input(self, task: AudioBatch) -> bool:
        if not hasattr(task, "data") or task.data is None:
            return False
        if not isinstance(task.data, list):
            return super().validate_input(task)
        return all(isinstance(item, dict) and self.diar_segments_key in item for item in task.data)

    def process(self, task: AudioBatch) -> AudioBatch:
        surviving = []
        for item in task.data:
            segments = item.get(self.diar_segments_key, [])
            speakers = {seg["speaker"] for seg in segments}
            if len(speakers) == 1:
                entry = {k: v for k, v in item.items() if k != self.diar_segments_key}
                entry["num_speakers"] = 1
                surviving.append(entry)
        return AudioBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=surviving,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


# ---------------------------------------------------------------------------
# Checkpointed execution
# ---------------------------------------------------------------------------


def _run_stages_with_checkpoints(
    stages: list[ProcessingStage],
    checkpoint_dir: Path,
) -> list[AudioBatch]:
    """Execute stages sequentially with per-task hash-based checkpointing."""
    executor = XennaExecutor()
    current_tasks: list[AudioBatch] | None = None
    t0 = time.time()

    for idx, stage in enumerate(stages):
        sdir = _stage_ckpt_dir(checkpoint_dir, idx, stage._name)

        # --- reader stage (no input tasks) ---
        if current_tasks is None:
            cached = _load_all_tasks(sdir)
            if cached:
                logger.info(f"Stage {idx} ({stage._name}): loaded {len(cached)} cached tasks — skipping")
                current_tasks = cached
                continue

            logger.info(f"Running stage {idx}/{len(stages) - 1}: {stage._name}")
            stage_t0 = time.time()
            pipeline = Pipeline(name=f"stage_{idx}_{stage._name}", stages=[stage])
            output = pipeline.run(executor=executor)
            current_tasks = output or []

            for task in current_tasks:
                h = _task_hash(task)
                task._metadata[CKPT_HASH_KEY] = h
                _save_task(sdir, h, task)
            logger.info(f"Stage {stage._name} done in {time.time() - stage_t0:.1f}s — {len(current_tasks)} tasks")
            continue

        # --- subsequent stages: split cached vs todo ---
        cached_tasks: list[AudioBatch] = []
        todo_tasks: list[AudioBatch] = []

        for task in current_tasks:
            h = task._metadata.get(CKPT_HASH_KEY) or _task_hash(task)
            if (sdir / f"{h}.json").exists():
                cached_tasks.append(_load_task(sdir / f"{h}.json"))
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
                h = task._metadata.get(CKPT_HASH_KEY) or _task_hash(task)
                task._metadata[CKPT_HASH_KEY] = h
                _save_task(sdir, h, task)
                cached_tasks.append(task)

            logger.info(f"Stage {stage._name} done in {time.time() - stage_t0:.1f}s — {len(new_tasks)} tasks")
        else:
            logger.info(f"Stage {idx} ({stage._name}): all tasks processed — skipping")

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
    p.add_argument("--model", default="nvidia/diar_streaming_sortformer_4spk-v2", help="HF Sortformer model id.")
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


def main() -> None:
    args = parse_args()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
        logger.info(f"Cleaned output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        total_entries = sum(1 for line in f if line.strip())
    if total_entries == 0:
        print(f"Empty manifest: {args.manifest}")
        return

    has_checkpoint = args.checkpoint_dir.exists() and any(args.checkpoint_dir.iterdir())
    resume_info = " (resuming from checkpoint)" if has_checkpoint else ""
    print(f"Manifest: {total_entries} entries{resume_info}", flush=True)

    stages: list[ProcessingStage] = [
        ALMManifestReader(manifest_path=str(args.manifest)),
        InferenceSortformerStage(
            model_name=args.model,
            chunk_len=args.chunk_len,
            chunk_right_context=args.chunk_right_context,
            fifo_len=args.fifo_len,
            spkcache_update_period=args.spkcache_update_period,
            spkcache_len=args.spkcache_len,
            inference_batch_size=1,
        ),
        SingleSpeakerFilterStage(),
        ALMManifestWriterStage(output_path=str(args.output_manifest)),
    ]

    print("Starting pipeline with inter-stage checkpointing...", flush=True)
    _run_stages_with_checkpoints(stages, args.checkpoint_dir)

    with open(args.output_manifest) as f:
        entries_out = sum(1 for line in f if line.strip())
    print(f"\n{'=' * 60}")
    print(f"Filtered: {entries_out} / {total_entries} entries have exactly 1 speaker")
    print(f"Output manifest: {args.output_manifest}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
