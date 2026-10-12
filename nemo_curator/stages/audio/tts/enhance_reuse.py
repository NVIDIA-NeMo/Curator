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

"""Resemble-Enhance ("ReUse") speech enhancement of Granary segments (alternative to Maxine StudioVoice).

Writes the same annotation as ``StudioVoiceEnhancementStage`` under ``enhanced_audio``::

    {audio_filepath, duration, sampling_rate, method: "resemble_enhance", error}

so every downstream stage can switch to the enhanced copy with ``audio_source="enhanced"`` (see
``audio_io.TarAudioConfig``). Enhancement settings default to the ones used for the GranaryHiFi RE build
(denoise, then CFM enhancer with nfe=32, midpoint solver, lambd=0.9, tau=0.5, native 44.1 kHz output).

``resemble-enhance`` pins ``torch==2.1``, so the model runs out of process in its own interpreter
(``enhancer_python``) through ``enhance_reuse_worker.py``; the GPU assigned to the stage actor is inherited via
``CUDA_VISIBLE_DEVICES``. One worker invocation per task loads the model once for all its segments.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.stages.audio.tts.audio_io import (
    SegmentAudioError,
    TarAudioConfig,
    iter_segments,
    load_segment_audio,
    sample_id,
)
from nemo_curator.stages.audio.tts.fields import get_dotted, set_dotted
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

METHOD = "resemble_enhance"
ENHANCE_FAILED = "enhance_failed"
WORKER_FAILED = "enhance_worker_failed"
_WORKER = Path(__file__).with_name("enhance_reuse_worker.py")


def empty_annotation(error: str, detail: str | None = None) -> dict[str, Any]:
    annotation: dict[str, Any] = {
        "audio_filepath": None,
        "duration": None,
        "sampling_rate": None,
        "method": METHOD,
        "error": error,
    }
    if detail:
        annotation["error_detail"] = detail
    return annotation


def success_annotation(path: Path, sampling_rate: int, num_samples: int) -> dict[str, Any]:
    return {
        "audio_filepath": str(path),
        "duration": float(num_samples) / float(sampling_rate),
        "sampling_rate": int(sampling_rate),
        "method": METHOD,
        "error": None,
    }


def parse_worker_output(stdout: str) -> dict[str, dict[str, Any]]:
    """Map worker input path -> its JSON result line (non-JSON log lines are ignored)."""
    results = {}
    for line in stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "in" in record:
            results[record["in"]] = record
    return results


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value) or "_"


@dataclass
class ResembleEnhanceStage(ProcessingStage[AudioTask, AudioTask]):
    """Enhance every segment with Resemble-Enhance and record the enhanced copy (rows/segments never dropped).

    Args:
        output_dir: Enhanced WAVs go to ``<output_dir>/<sample_id>/<tar member stem>.wav``.
        enhancer_python: Interpreter of the resemble-enhance venv (torch 2.1). Defaults to the current one.
        run_dir: Pre-staged ``enhancer_stage2`` weights (``RESEMBLE_RUN_DIR``); ``None`` lets resemble download.
        nfe, solver, lambd, tau, denoise: enhancer settings (GranaryHiFi RE build: 32 / midpoint / 0.9 / 0.5 / True).
        target_sample_rate: Output rate; ``None`` keeps the enhancer's native 44.1 kHz.
        skip_existing: Reuse an existing non-empty output (idempotent re-runs).
        output_key: Annotation key (shared with ``StudioVoiceEnhancementStage``).
        timeout_sec: Worker timeout per task.
        tar_root, strip_tar_key_prefixes: Resolution of relative ``tar_key`` values (input is always the original).
    """

    output_dir: str = "resemble_enhanced"
    enhancer_python: str | None = None
    run_dir: str | None = None
    nfe: int = 32
    solver: str = "midpoint"
    lambd: float = 0.9
    tau: float = 0.5
    denoise: bool = True
    target_sample_rate: int | None = None
    skip_existing: bool = True
    output_key: str = "enhanced_audio"
    timeout_sec: float = 7200.0
    tar_root: str | None = None
    strip_tar_key_prefixes: tuple[str, ...] = ()
    command_runner: Any = None  # tests: callable(cmd, env, timeout) -> stdout str
    name: str = "ResembleEnhance"
    resources: Resources = field(default_factory=lambda: Resources(cpus=2.0, gpus=1.0))
    batch_size: int = 1

    def __post_init__(self) -> None:
        super().__init__()
        if self.nfe < 1:
            msg = "nfe must be >= 1"
            raise ValueError(msg)
        self._audio_config = TarAudioConfig(self.tar_root, tuple(self.strip_tar_key_prefixes))  # always original

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key.split(".")[0]]

    def output_path(self, row: dict[str, Any], segment: dict[str, Any], index: int) -> Path:
        member = get_dotted(segment, "GranaryHifi.AudioMetadata.tar_member") or segment.get("audio_filepath")
        stem = Path(str(member)).stem if member else f"segment_{index:06d}"
        return Path(self.output_dir).absolute() / _safe_name(sample_id(row)) / f"{_safe_name(stem)}.wav"

    def _command(self, manifest: Path) -> list[str]:
        cmd = [self.enhancer_python or sys.executable, str(_WORKER), "--manifest", str(manifest)]
        cmd += ["--nfe", str(self.nfe), "--solver", self.solver, "--lambd", str(self.lambd), "--tau", str(self.tau)]
        cmd += ["--sr", str(self.target_sample_rate or 0)]
        if not self.denoise:
            cmd.append("--no-denoise")
        return cmd

    def _run_worker(self, manifest: Path) -> str:
        import os

        env = dict(os.environ)
        if self.run_dir:
            env["RESEMBLE_RUN_DIR"] = str(Path(self.run_dir).expanduser())
        cmd = self._command(manifest)
        if self.command_runner is not None:
            return self.command_runner(cmd, env, self.timeout_sec)
        done = subprocess.run(cmd, env=env, timeout=self.timeout_sec, capture_output=True, text=True, check=False)  # noqa: S603
        if done.returncode != 0:
            logger.warning("resemble-enhance worker exited {}: {}", done.returncode, done.stderr[-2000:])
        return done.stdout

    def _existing(self, path: Path) -> dict[str, Any] | None:
        if not (self.skip_existing and path.is_file() and path.stat().st_size > 0):
            return None
        try:
            import soundfile as sf

            info = sf.info(str(path))
            return success_annotation(path, info.samplerate, info.frames)
        except Exception as exc:  # noqa: BLE001  (truncated by preemption -> re-enhance)
            logger.warning("Re-enhancing unreadable existing output {}: {}", path, exc)
            return None

    def _enhance_task(self, row: dict[str, Any]) -> None:
        import soundfile as sf

        pending: dict[str, tuple[dict[str, Any], Path]] = {}
        with tempfile.TemporaryDirectory(prefix="resemble_in_") as tmp:
            items = []
            for index, segment in iter_segments(row):
                out = self.output_path(row, segment, index)
                existing = self._existing(out)
                if existing is not None:
                    set_dotted(segment, self.output_key, existing)
                    continue
                try:
                    audio, sr = load_segment_audio(segment, None, self._audio_config)
                except SegmentAudioError as exc:
                    set_dotted(segment, self.output_key, empty_annotation(exc.code))
                    continue
                src = Path(tmp) / f"{index:06d}.wav"
                sf.write(str(src), audio, sr, subtype="FLOAT")  # lossless hand-off to the worker
                pending[str(src)] = (segment, out)
                items.append({"in": str(src), "out": str(out)})
            if not items:
                return
            manifest = Path(tmp) / "items.jsonl"
            manifest.write_text("".join(json.dumps(i) + "\n" for i in items))
            try:
                stdout = self._run_worker(manifest)
            except subprocess.TimeoutExpired:
                stdout = ""
                logger.error("resemble-enhance worker timed out after {}s", self.timeout_sec)
            results = parse_worker_output(stdout)
            for src, (segment, out) in pending.items():
                record = results.get(src)
                if record is None:
                    annotation = empty_annotation(WORKER_FAILED)
                elif record.get("ok") and out.is_file():
                    annotation = success_annotation(out, record["sampling_rate"], record["num_samples"])
                else:
                    annotation = empty_annotation(ENHANCE_FAILED, record.get("error"))
                set_dotted(segment, self.output_key, annotation)

    def process(self, task: AudioTask) -> AudioTask:
        self._enhance_task(task.data)
        return task
