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

"""Speaker diarization on CallHome English using Streaming Sortformer via NeMo Curator.

Four-stage pipeline via XennaExecutor:
  CallHomeReaderStage → EnsureMonoStage → InferenceSortformerStage → DERComputationStage

Usage:
    python tutorials/audio/callhome_diar/run.py --data-dir /path/to/callhome_eng0
    python tutorials/audio/callhome_diar/run.py --data-dir /path/to/callhome_eng0 --output-dir /path/to/output
    python tutorials/audio/callhome_diar/run.py --data-dir /path/to/callhome_eng0 --clean
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.inference.sortformer import InferenceSortformerStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioBatch, _EmptyTask

COLLAR = 0.25
CKPT_HASH_KEY = "_ckpt_hash"


# ---------------------------------------------------------------------------
# Per-task hash-based checkpointing
# ---------------------------------------------------------------------------


def _task_hash(task: AudioBatch) -> str:
    """Derive a stable content hash for an AudioBatch.

    Uses session_name / audio_filepath from the task data as the identity
    key so the hash stays the same across stages.
    """
    identifiers: list[str] = []
    for item in (task.data or []):
        if isinstance(item, dict):
            if "session_name" in item:
                identifiers.append(item["session_name"])
            elif "audio_filepath" in item:
                identifiers.append(item["audio_filepath"])
    if not identifiers:
        identifiers.append(task.task_id)
    raw = "|".join(sorted(identifiers))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _stage_ckpt_dir(checkpoint_dir: Path, stage_index: int, stage_name: str) -> Path:
    return checkpoint_dir / f"stage_{stage_index:02d}_{stage_name}"


def _save_task(directory: Path, h: str, task: AudioBatch) -> None:
    """Write a single task checkpoint, keyed by its hash."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task.task_id,
        "dataset_name": task.dataset_name,
        "data": task.data,
        "_metadata": task._metadata,
    }
    (directory / f"{h}.json").write_text(json.dumps(payload, indent=2))


def _load_task(path: Path) -> AudioBatch:
    """Reconstruct a single AudioBatch from a checkpoint file."""
    payload = json.loads(path.read_text())
    return AudioBatch(
        task_id=payload["task_id"],
        dataset_name=payload["dataset_name"],
        data=payload["data"],
        _metadata=payload.get("_metadata", {}),
    )


def _load_all_tasks(directory: Path) -> list[AudioBatch]:
    """Load every task checkpoint in a stage directory."""
    if not directory.exists():
        return []
    return [_load_task(p) for p in sorted(directory.glob("*.json"))]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sortformer diarization on CallHome English + DER evaluation.")
    p.add_argument("--data-dir", type=Path, required=True, help="CallHome-eng0 dataset root.")
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="Root directory for all outputs.")
    p.add_argument("--model", default="nvidia/diar_streaming_sortformer_4spk-v2", help="HF Sortformer model id.")
    p.add_argument("--collar", type=float, default=COLLAR, help="Collar tolerance (seconds).")
    p.add_argument("--clean", action="store_true", help="Remove entire output directory before running.")
    p.add_argument("--chunk-len", type=int, default=340, help="Streaming chunk size in 80ms frames.")
    p.add_argument("--chunk-right-context", type=int, default=40, help="Right context frames.")
    p.add_argument("--fifo-len", type=int, default=40, help="FIFO queue size in frames.")
    p.add_argument("--spkcache-update-period", type=int, default=300, help="Speaker cache update period in frames.")
    p.add_argument("--spkcache-len", type=int, default=188, help="Speaker cache size in frames.")

    args = p.parse_args()
    out = args.output_dir
    args.rttm_out_dir = out / "rttm"
    args.results_json = out / "results.json"
    args.checkpoint_dir = out / "checkpoints"
    return args


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


@dataclass
class CallHomeReaderStage(ProcessingStage[_EmptyTask, AudioBatch]):
    """Discover CallHome WAV files with matching .cha annotations, skipping already-processed."""

    data_dir: str = ""
    cha_dir: str = ""
    rttm_out_dir: str = ""
    filepath_key: str = "audio_filepath"
    name: str = "CallHomeReaderStage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.filepath_key]

    def process(self, task: _EmptyTask) -> list[AudioBatch]:  # noqa: ARG002
        cha_path = Path(self.cha_dir)
        done = {p.stem for p in Path(self.rttm_out_dir).glob("*.rttm")} if self.rttm_out_dir else set()
        tasks: list[AudioBatch] = []
        for wav in sorted(Path(self.data_dir).glob("*.wav")):
            fid = wav.stem
            if fid in done or not (cha_path / f"{fid}.cha").exists():
                continue
            tasks.append(
                AudioBatch(
                    data=[{self.filepath_key: str(wav), "session_name": fid}],
                    task_id=f"callhome_{fid}",
                    dataset_name="callhome_eng0",
                )
            )
        return tasks


@dataclass
class EnsureMonoStage(ProcessingStage[AudioBatch, AudioBatch]):
    """Downmix stereo WAVs to mono 16 kHz via sox."""

    mono_dir: str = "mono"
    filepath_key: str = "audio_filepath"
    name: str = "EnsureMonoStage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.filepath_key]

    def _ensure_mono(self, wav_path: str) -> str:
        mono_path = os.path.join(self.mono_dir, os.path.basename(wav_path))
        if os.path.exists(mono_path):
            return mono_path
        os.makedirs(self.mono_dir, exist_ok=True)
        subprocess.run(  # noqa: S603
            ["sox", wav_path, "-c", "1", "-r", "16000", mono_path],  # noqa: S607
            check=True,
            capture_output=True,
        )
        return mono_path

    def process(self, task: AudioBatch) -> AudioBatch:
        items = [{**item, self.filepath_key: self._ensure_mono(item[self.filepath_key])} for item in task.data]
        return AudioBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=items,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


@dataclass
class DERComputationStage(ProcessingStage[AudioBatch, AudioBatch]):
    """Compute Diarization Error Rate against CHA ground-truth annotations."""

    cha_dir: str = ""
    diar_segments_key: str = "diar_segments"
    der_metrics_key: str = "der_metrics"
    collar: float = 0.25
    name: str = "DERComputationStage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.diar_segments_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.der_metrics_key]

    def validate_input(self, task: AudioBatch) -> bool:
        if not hasattr(task, "data") or task.data is None:
            return False
        # task.data is a list of dicts; base class expects task.data to have the key as attr
        if not isinstance(task.data, list):
            return super().validate_input(task)
        return all(
            isinstance(item, dict) and self.diar_segments_key in item
            for item in task.data
        )

    def process(self, task: AudioBatch) -> AudioBatch:
        cha_path = Path(self.cha_dir)
        items: list[dict[str, Any]] = []
        for item in task.data:
            item = dict(item)  # noqa: PLW2901
            sess = item.get("session_name", "unknown")
            cha_file = cha_path / f"{sess}.cha"
            metrics = None
            if cha_file.exists():
                gt, uem_start, uem_end = self._parse_cha(cha_file)
                if gt and item.get(self.diar_segments_key):
                    metrics = self._compute_der(gt, item[self.diar_segments_key], uem_start, uem_end)
            item[self.der_metrics_key] = metrics
            items.append(item)
        return AudioBatch(
            task_id=task.task_id, dataset_name=task.dataset_name,
            data=items, _metadata=task._metadata, _stage_perf=task._stage_perf,
        )

    @staticmethod
    def _parse_cha(path: Path) -> tuple[list[dict], float, float]:
        """Parse a CHA file → (segments, uem_start, uem_end)."""
        segs: list[dict] = []
        with open(path) as f:
            for line in f:
                m = re.match(r"^\*([A-Z]):\t", line)
                ts = re.search(r"\x15(\d+)_(\d+)\x15", line)
                if m and ts:
                    segs.append(
                        {"speaker": m.group(1), "start": int(ts.group(1)) / 1000, "end": int(ts.group(2)) / 1000}
                    )
        if not segs:
            return segs, 0.0, 0.0
        return segs, min(s["start"] for s in segs), max(s["end"] for s in segs)

    def _compute_der(  # noqa: C901, PLR0912
        self, gt: list[dict], pred: list[dict], uem_start: float, uem_end: float,
    ) -> dict | None:
        """Frame-level DER restricted to UEM region with collar tolerance."""
        if not gt or not pred:
            return None

        pred = [
            {"speaker": s["speaker"], "start": max(s["start"], uem_start), "end": min(s["end"], uem_end)}
            for s in pred if s["end"] > uem_start and s["start"] < uem_end
        ]
        pred = [s for s in pred if s["end"] > s["start"]]
        if not pred:
            return None

        collar_zones: list[tuple[float, float]] = []
        if self.collar > 0:
            for s in gt:
                collar_zones.append((s["start"] - self.collar, s["start"] + self.collar))
                collar_zones.append((s["end"] - self.collar, s["end"] + self.collar))

        def in_collar(t: float) -> bool:
            return any(lo <= t <= hi for lo, hi in collar_zones)

        # Greedy speaker mapping by overlap
        mv: Counter = Counter()
        for g in gt:
            for p in pred:
                ov = min(g["end"], p["end"]) - max(g["start"], p["start"])
                if ov > 0:
                    mv[(g["speaker"], p["speaker"])] += ov
        used, mapping = set(), {}
        for _v, gs, ps in sorted([(v, g, p) for (g, p), v in mv.items()], reverse=True):
            if gs not in mapping and ps not in used:
                mapping[gs] = ps
                used.add(ps)
        inv = {v: k for k, v in mapping.items()}

        # Frame-level scoring
        step = 0.01
        nf = int((uem_end - uem_start) / step) + 1
        gf: dict[int, set] = {}
        pf: dict[int, set] = {}
        for s in gt:
            for i in range(max(0, int((s["start"] - uem_start) / step)), min(nf, int((s["end"] - uem_start) / step))):
                gf.setdefault(i, set()).add(s["speaker"])
        for s in pred:
            mm = inv.get(s["speaker"], f"x_{s['speaker']}")
            for i in range(max(0, int((s["start"] - uem_start) / step)), min(nf, int((s["end"] - uem_start) / step))):
                pf.setdefault(i, set()).add(mm)

        miss = fa = conf = correct = total = 0
        for i in range(nf):
            t = uem_start + i * step
            if in_collar(t):
                continue
            gs = gf.get(i, set())
            ps = pf.get(i, set())
            if gs:
                total += len(gs)
                for s in gs:
                    if s in ps:
                        correct += 1
                    elif ps:
                        conf += 1
                    else:
                        miss += 1
            fa += len(ps - gs if gs else ps)

        ts = total * step
        if ts == 0:
            return None
        return {
            "der": (miss + fa + conf) * step / ts * 100, "miss": miss * step / ts * 100,
            "fa": fa * step / ts * 100, "conf": conf * step / ts * 100, "correct": correct * step / ts * 100,
            "gt_speech_s": ts, "pred_speech_s": sum(s["end"] - s["start"] for s in pred),
            "gt_speakers": len({s["speaker"] for s in gt}), "pred_speakers": len({s["speaker"] for s in pred}),
        }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_summary(results: list[dict], collar: float) -> None:
    n = len(results)
    print(f"\n{'=' * 60}")
    print(f"COMPLETED: {n} files evaluated (collar={collar}s)")
    print(f"{'=' * 60}", flush=True)
    if not results:
        return

    def avg(key: str) -> float:
        return sum(r[key] for r in results) / n

    total_gt = sum(r["gt_speech_s"] for r in results)

    def wavg(key: str) -> float:
        return sum(r[key] * r["gt_speech_s"] for r in results) / total_gt

    print(
        f"\n  Macro-avg  DER={avg('der'):.1f}%  Miss={avg('miss'):.1f}%  FA={avg('fa'):.1f}%  Conf={avg('conf'):.1f}%"
    )
    print(
        f"  Weighted   DER={wavg('der'):.1f}%  Miss={wavg('miss'):.1f}%  FA={wavg('fa'):.1f}%  Conf={wavg('conf'):.1f}%"
    )

    spk_match = sum(1 for r in results if r["gt_speakers"] == r["pred_speakers"])
    print(f"  Speaker count match: {spk_match}/{n} ({spk_match / n * 100:.0f}%)")

    by_der = sorted(results, key=lambda r: r["der"])
    print("\n  Best 5:", ", ".join(f"{r['file_id']}={r['der']:.1f}%" for r in by_der[:5]))
    print("  Worst 5:", ", ".join(f"{r['file_id']}={r['der']:.1f}%" for r in by_der[-5:]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run_stages_with_checkpoints(
    stages: list[ProcessingStage],
    checkpoint_dir: Path,
) -> list[AudioBatch]:
    """Execute stages one at a time with per-task hash-based checkpointing.

    For each stage the flow is:
      1. Compute a content hash for every input task.
      2. Check the stage's output directory for existing ``<hash>.json`` files.
      3. Tasks whose hash already exists are loaded from disk (cached).
      4. Only the remaining tasks are sent through the executor.
      5. Newly produced outputs are saved to disk, keyed by the *input* hash.

    This means even a partially completed stage is resumable — only the
    un-processed tasks will be re-run.
    """
    executor = XennaExecutor()
    current_tasks: list[AudioBatch] | None = None
    t0 = time.time()

    for idx, stage in enumerate(stages):
        sdir = _stage_ckpt_dir(checkpoint_dir, idx, stage._name)

        # --- reader stage (no input tasks) ---
        if current_tasks is None:
            cached = _load_all_tasks(sdir)
            if cached:
                logger.info(f"Stage {idx} ({stage._name}): loaded {len(cached)} cached tasks — skipping execution")
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
        existing_hashes = {p.stem for p in sdir.glob("*.json")} if sdir.exists() else set()

        cached_tasks: list[AudioBatch] = []
        todo_tasks: list[AudioBatch] = []

        for task in current_tasks:
            h = task._metadata.get(CKPT_HASH_KEY) or _task_hash(task)
            if h in existing_hashes:
                cached_tasks.append(_load_task(sdir / f"{h}.json"))
            else:
                task._metadata[CKPT_HASH_KEY] = h
                todo_tasks.append(task)

        if cached_tasks:
            logger.info(
                f"Stage {idx} ({stage._name}): {len(cached_tasks)} tasks already checkpointed, "
                f"{len(todo_tasks)} remaining"
            )

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

            logger.info(f"Stage {stage._name} done in {time.time() - stage_t0:.1f}s — {len(new_tasks)} new tasks")
        else:
            logger.info(f"Stage {idx} ({stage._name}): all tasks cached — skipping execution")

        current_tasks = cached_tasks

    total = time.time() - t0
    logger.info(f"All stages done in {total / 60:.1f} min")
    return current_tasks or []


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    cha_dir = data_dir / "eng"
    rttm_out: Path = args.rttm_out_dir
    ckpt_dir: Path = args.checkpoint_dir

    ray_client = RayClient()
    ray_client.start()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
        logger.info(f"Cleaned output directory: {args.output_dir}")
    rttm_out.mkdir(parents=True, exist_ok=True)

    # Pre-check: how many files will the reader emit (same logic as CallHomeReaderStage)
    done = {p.stem for p in rttm_out.glob("*.rttm")} if rttm_out.exists() else set()
    wavs_with_cha = [
        w for w in sorted(data_dir.glob("*.wav"))
        if w.stem not in done and (cha_dir / f"{w.stem}.cha").exists()
    ]
    n_skip = len(done)

    has_checkpoint = ckpt_dir.exists() and any(ckpt_dir.iterdir()) if ckpt_dir.exists() else False
    if not wavs_with_cha and not has_checkpoint:
        print(
            f"No files to process: {len(list(data_dir.glob('*.wav')))} WAV(s) in {data_dir}, "
            f"{n_skip} already have RTTM (skipped). Need WAVs and matching {cha_dir}/<stem>.cha. Use --clean to re-run all.",
            flush=True,
        )
        return
    print(f"Files to process: {len(wavs_with_cha)} (skipping {n_skip} with existing RTTM)", flush=True)

    stages: list[ProcessingStage] = [
        CallHomeReaderStage(data_dir=str(data_dir), cha_dir=str(cha_dir), rttm_out_dir=str(rttm_out)),
        EnsureMonoStage(mono_dir=str(data_dir / "mono")),
        InferenceSortformerStage(
            model_name=args.model,
            rttm_out_dir=str(rttm_out),
            chunk_len=args.chunk_len,
            chunk_right_context=args.chunk_right_context,
            fifo_len=args.fifo_len,
            spkcache_update_period=args.spkcache_update_period,
            spkcache_len=args.spkcache_len,
            inference_batch_size=1,
        ),
        DERComputationStage(cha_dir=str(cha_dir), collar=args.collar),
    ]

    print("Starting pipeline with inter-stage checkpointing...", flush=True)
    output_tasks = _run_stages_with_checkpoints(stages, ckpt_dir)

    output_tasks = output_tasks or []
    results = [
        {**item["der_metrics"], "file_id": item.get("session_name", "unknown")}
        for task in output_tasks
        for item in (task.data or [])
        if item.get("der_metrics") is not None
    ]

    _print_summary(results, args.collar)

    with open(args.results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {args.results_json}")


if __name__ == "__main__":
    main()
