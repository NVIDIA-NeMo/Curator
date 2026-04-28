# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Bridge stage for AIS-streamed audio to file-based downstream stages.

Several audio stages (SED, Sortformer, NeMo ASR, TitaNet) load audio
via ``librosa.load(audio_filepath)`` — they expect a real local file
path.  But ``NemoTarShardReaderStage`` decodes audio in memory from
AIS tars, leaving ``task.data["audio_filepath"]`` as the tar member
name (no actual file on disk).  Plugging the streaming reader directly
into SED therefore fails with ``FileNotFoundError: ... .flac``.

This stage materialises the in-memory waveform into a temporary WAV
file under ``temp_dir`` (default ``/tmp``), updates ``audio_filepath``
to the temp path, and tracks the path in ``task.data["_temp_files"]``
so a teardown step can remove it.  The original ``waveform``/
``sample_rate`` keys are retained — downstream stages that prefer
in-memory access (e.g. UTMOSv2) can still use them.

Usage::

    pipeline.add_stage(NemoTarredAudioReader(yaml_path=...))
    pipeline.add_stage(AsrBridgeStage(temp_dir="/tmp"))
    pipeline.add_stage(SEDInferenceStage(...))   # now sees real .wav

Mirrors the per-task temp-WAV pattern used by ``run_stage.py:run_sed``
in the legacy slurm-array e2e pipeline, but inside a Curator stage
so it composes with Ray-distributed Pipeline.run().
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    import numpy as np


@dataclass
class AsrBridgeStage(ProcessingStage[AudioTask, AudioTask]):
    """Materialise in-memory waveforms to temp WAVs for file-based stages.

    Args:
        temp_dir: Root directory for temporary WAV files.  Default
            ``"/tmp"`` is node-local on Slurm and cleaned by the job
            scheduler when the allocation ends.  Override to e.g.
            ``"/lustre/.../scratch"`` if you need shared-FS visibility
            across nodes (rare; most pipelines keep audio on the
            originating node).
        waveform_key: Key in ``task.data`` for the numpy waveform.
        sample_rate_key: Key in ``task.data`` for the integer sample rate.
        filepath_key: Key in ``task.data`` to populate with the temp
            WAV path.
        keep_waveform: If True (default), leave ``waveform`` in
            ``task.data`` so downstream stages that consume it
            in-memory still work.  If False, drop it to save memory.
        subtype: ``soundfile`` write subtype.  Default ``"PCM_16"``
            matches what librosa-based stages expect.
    """

    name: str = "AsrBridge"
    temp_dir: str = "/tmp"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    filepath_key: str = "audio_filepath"
    keep_waveform: bool = True
    subtype: str = "PCM_16"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _job_temp_dir: str = field(default="", init=False, repr=False)

    def setup(self, _worker_metadata: Any = None) -> None:  # noqa: ANN401
        # One subdir per Slurm job (or pid fallback) so concurrent
        # actors don't collide.  Job-level cleanup happens with the
        # allocation; we don't try to remove the subdir ourselves.
        slurm_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        self._job_temp_dir = os.path.join(self.temp_dir, f"asr_bridge_{slurm_id}")
        os.makedirs(self._job_temp_dir, exist_ok=True)
        logger.info(f"AsrBridgeStage: temp dir = {self._job_temp_dir}")

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.waveform_key, self.sample_rate_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.filepath_key]

    def process(self, task: AudioTask) -> AudioTask:
        import soundfile as sf

        waveform = task.data.get(self.waveform_key)
        sample_rate = task.data.get(self.sample_rate_key)
        if waveform is None or sample_rate is None:
            msg = (
                f"AsrBridgeStage: missing {self.waveform_key} or "
                f"{self.sample_rate_key} in task {task.task_id}"
            )
            raise ValueError(msg)

        # Use a stable name based on task_id so retries don't proliferate
        # files; mkstemp would also work but adds entropy on each retry.
        safe_id = task.task_id.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(self._job_temp_dir, f"{safe_id}.wav")
        sf.write(out_path, waveform, int(sample_rate), subtype=self.subtype)

        task.data[self.filepath_key] = out_path
        task.data.setdefault("_temp_files", []).append(out_path)
        if not self.keep_waveform:
            task.data.pop(self.waveform_key, None)

        return task

    def teardown(self) -> None:
        # Best-effort: leave the per-job subdir in place; Slurm cleans
        # /tmp on job end.  If callers point temp_dir somewhere
        # persistent they can wire their own cleanup.
        pass
