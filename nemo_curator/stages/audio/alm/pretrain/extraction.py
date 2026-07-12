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

"""Audio extraction stage: slice, mono-resample, write into a tar shard."""

from __future__ import annotations

import copy
import io
import math
import os
import tarfile
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taf
from loguru import logger

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.alm.pretrain.planning import relativize_segments
from nemo_curator.stages.audio.alm.pretrain.utils import (
    _PLAN_DATA_KEY,
    _SOUNDFILE_SUBTYPES,
    _TAR_SHARD_EXT,
    _is_origin_stub,
    _make_shard_path,
    _segment_text,
    make_snippet_id,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata


@dataclass
class SnippetExtractionStage(ProcessingStage[AudioTask, AudioTask]):
    """Slice the source audio per snippet plan, mono-resample, and write into a tar.

    For each planned snippet:

    1. Read just the slice ``[start, end]`` from the source file.
    2. Channel-average to mono if the source has > 1 channel.
    3. Resample to ``target_sample_rate`` using torchaudio if the source
       rate differs.
    4. Encode the mono waveform in-memory (via ``soundfile`` to a
       ``BytesIO``) and append it as ``<snippet_id>.<output_format>`` to
       this replica's tar shard (``output_audio_tar_path.shard-...``);
       all replicas' shards are merged into ``output_audio_tar_path`` by
       :func:`finalize_audio_pretrain_outputs`.
    5. Emit one ``AudioTask`` per snippet with the source row's metadata
       carried over (minus ``alignment``), the new ``snippet_id``,
       ``audio_filepath`` set to the **tar-internal basename**
       (``<snippet_id>.<output_format>``), updated ``duration``, and
       segments relativized to the snippet start.

    The tar-internal basename matches webdataset / Energon convention:
    sample key is ``<snippet_id>`` (everything before the first ``.``),
    extension is ``<output_format>``.  ``make_snippet_id`` already
    avoids ``.`` characters so the snippet id never spuriously splits.

    If the input produced zero snippets, a single "stub" ``AudioTask``
    is emitted (``snippet_id=None``, no audio written) so that
    per-original metrics can still flow to the aggregator.

    Dry-run mode (``dry_run=True``): skips steps 1-4 entirely (no
    ``soundfile`` reads, no resampling, no tar writes -- not even a tar
    shard is opened), and step 5 uses the planned ``end - start`` as the
    snippet ``duration`` instead of the post-resample frame count.  The
    emitted ``audio_filepath`` still uses the basename form for parity
    with real runs.  Useful for previewing the manifest and metrics on
    real data before committing to a full run.
    """

    output_dir: str
    output_audio_tar_path: str
    target_sample_rate: int = 16000
    output_format: str = "flac"
    audio_filepath_key: str = "audio_filepath"
    dry_run: bool = False

    name: str = "SnippetExtraction"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        if self.output_format not in _SOUNDFILE_SUBTYPES:
            msg = f"output_format must be one of {sorted(_SOUNDFILE_SUBTYPES)}, got {self.output_format!r}"
            raise ValueError(msg)
        if self.target_sample_rate <= 0:
            msg = "target_sample_rate must be > 0"
            raise ValueError(msg)
        self._tar_shard_path: str | None = None
        self._tar: Any = None  # tarfile.TarFile, opened lazily in setup()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, _PLAN_DATA_KEY]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, "snippet_id", "duration", "segments"]

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        parent = os.path.dirname(self.output_audio_tar_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        parent = os.path.dirname(self.output_audio_tar_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if self.dry_run:
            return
        self._tar_shard_path = _make_shard_path(self.output_audio_tar_path, _TAR_SHARD_EXT)
        # The TarFile stays open for the worker's lifetime and is closed in
        # teardown(); a context manager isn't usable here without making every
        # snippet write re-open the archive (forces tarfile to re-scan to the
        # end-of-archive marker, making writes O(n^2)).
        self._tar = tarfile.open(self._tar_shard_path, "w")  # noqa: SIM115
        logger.info(f"[{self.name}] writing audio tar shard to {self._tar_shard_path}")

    def teardown(self) -> None:
        if self._tar is not None:
            try:
                self._tar.close()
            except OSError as e:
                logger.warning(f"[{self.name}] failed to close tar shard {self._tar_shard_path}: {e}")
            finally:
                self._tar = None

    def process(self, task: AudioTask) -> list[AudioTask]:
        t0 = time.perf_counter()
        # Read the plan without mutating the input task -- Xenna may preempt
        # and replay the same task through this stage; popping would leave the
        # retried task without a plan and fail validate_input on retry.
        plan: list[dict] = list(task.data.get(_PLAN_DATA_KEY) or [])
        if not plan:
            return [self._make_stub_task(task)]

        original_id = str(task.data.get("id") or task.task_id)

        if self.dry_run:
            outputs = self._dry_run_emit(task, plan, original_id)
        else:
            outputs = self._extract_emit(task, plan, original_id)

        total_dur = 0.0 if _is_origin_stub(outputs[0]) else sum(t.data["duration"] for t in outputs)
        self._log_metrics(
            {
                "extract_time": time.perf_counter() - t0,
                "snippets_written": float(len(outputs) if not _is_origin_stub(outputs[0]) else 0),
                "snippets_total_duration": float(total_dur),
            }
        )
        return outputs

    def _extract_emit(self, task: AudioTask, plan: list[dict], original_id: str) -> list[AudioTask]:
        source_path = task.data.get(self.audio_filepath_key)
        if not source_path or not os.path.exists(source_path):
            logger.error(
                f"[{self.name}] {task.task_id}: source audio missing at {source_path!r}; "
                f"emitting stub for {len(plan)} planned snippets"
            )
            return [self._make_stub_task(task)]
        try:
            info = sf.info(source_path)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.name}] {task.task_id}: cannot read header of {source_path}: {e}")
            return [self._make_stub_task(task)]

        outputs: list[AudioTask] = []
        for snippet in plan:
            emitted = self._extract_one_snippet(task, snippet, source_path, info, original_id)
            if emitted is not None:
                outputs.append(emitted)
        if not outputs:
            outputs.append(self._make_stub_task(task))
        return outputs

    def _dry_run_emit(self, task: AudioTask, plan: list[dict], original_id: str) -> list[AudioTask]:
        """Emit snippet metadata only, without reading or writing audio.

        ``audio_filepath`` is the tar-internal basename
        ``<snippet_id>.<output_format>`` for parity with real runs --
        the tar itself is not opened in dry-run.  Snippet ``duration``
        is the planned ``end - start`` (vs. the resampled-frame-count
        duration the real path would compute -- the difference is at
        most one frame at ``target_sample_rate``).
        """
        outputs: list[AudioTask] = []
        for snippet in plan:
            start_sec = float(snippet["start"])
            end_sec = float(snippet["end"])
            snippet_id = make_snippet_id(original_id, start_sec, end_sec)
            out_path = f"{snippet_id}.{self.output_format}"
            outputs.append(
                self._make_snippet_task(
                    task=task,
                    snippet=snippet,
                    snippet_id=snippet_id,
                    out_path=out_path,
                    duration=end_sec - start_sec,
                )
            )
        if not outputs:
            outputs.append(self._make_stub_task(task))
        return outputs

    def _extract_one_snippet(
        self,
        task: AudioTask,
        snippet: dict,
        source_path: str,
        info: Any,  # noqa: ANN401  (soundfile._SoundFileInfo)
        original_id: str,
    ) -> AudioTask | None:
        source_sr = info.samplerate
        start_sec = float(snippet["start"])
        end_sec = float(snippet["end"])
        start_frame = max(0, math.floor(start_sec * source_sr))
        end_frame = min(info.frames, math.ceil(end_sec * source_sr))
        if end_frame <= start_frame:
            logger.warning(
                f"[{self.name}] {task.task_id}: empty frame range [{start_frame}, {end_frame}); skipping snippet"
            )
            return None

        try:
            audio, _ = sf.read(source_path, start=start_frame, stop=end_frame, dtype="float32", always_2d=True)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.name}] {task.task_id}: failed to read slice [{start_sec:.2f}, {end_sec:.2f}]: {e}")
            return None

        wave = torch.from_numpy(np.ascontiguousarray(audio.T))
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)
        if source_sr != self.target_sample_rate:
            wave = taf.resample(wave, source_sr, self.target_sample_rate)
        mono = wave.squeeze(0).contiguous().numpy()
        actual_duration = mono.shape[0] / float(self.target_sample_rate)

        snippet_id = make_snippet_id(original_id, start_sec, end_sec)
        member_name = f"{snippet_id}.{self.output_format}"
        try:
            buf = io.BytesIO()
            sf.write(
                buf,
                mono,
                self.target_sample_rate,
                format=self.output_format.upper(),
                subtype=_SOUNDFILE_SUBTYPES[self.output_format],
            )
            payload = buf.getvalue()
            tarinfo = tarfile.TarInfo(name=member_name)
            tarinfo.size = len(payload)
            self._tar.addfile(tarinfo, io.BytesIO(payload))
            # Flush the tar's BufferedWriter so this member's bytes hit
            # the kernel page cache. Cosmos-Xenna shuts actors down with
            # `ray.kill()` (see lines 74, 1220, 1473 and
            # cosmos_xenna/ray_utils/actor_pool.py), which does a Quick
            # exit that bypasses Python cleanup. Anything still in the
            # user-space buffer at kill time is lost. Page cache survives
            # process death — the downstream merger reads back the same
            # file and gets every fully-completed member regardless of
            # whether teardown() ever ran. Without this flush, ~50%+ of
            # snippets per shard get dropped during _merge_tar_shards's
            # Pass 2 streaming because their data sections are truncated
            # on disk.
            self._tar.fileobj.flush()
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.name}] failed to add {member_name} to tar shard {self._tar_shard_path}: {e}")
            return None

        return self._make_snippet_task(
            task=task,
            snippet=snippet,
            snippet_id=snippet_id,
            out_path=member_name,
            duration=actual_duration,
        )

    def _make_snippet_task(
        self,
        task: AudioTask,
        snippet: dict,
        snippet_id: str,
        out_path: str,
        duration: float,
    ) -> AudioTask:
        new_data = dict(task.data)
        new_data.pop("alignment", None)
        new_data.pop(_PLAN_DATA_KEY, None)
        # Drop source-file-specific fields that don't apply to the snippet.
        new_data.pop("audio_size", None)
        new_data.pop("resampled_audio_filepath", None)
        new_data["snippet_id"] = snippet_id
        new_data[self.audio_filepath_key] = out_path
        new_data["duration"] = duration
        # Update audio-property fields only if the source row had them.
        if "actual_duration" in new_data:
            new_data["actual_duration"] = duration
        if "proposed_duration" in new_data:
            new_data["proposed_duration"] = duration
        if "audio_sample_rate" in new_data:
            new_data["audio_sample_rate"] = self.target_sample_rate
        if "audio_num_channels" in new_data:
            new_data["audio_num_channels"] = 1
        # Reset to "" — a downstream pipeline is expected to set this correctly.
        if "swift_audio_filepath" in new_data:
            new_data["swift_audio_filepath"] = ""
        new_data["segments"] = relativize_segments(snippet["segments"], snippet["start"], snippet["end"])
        if "text" in new_data:
            new_data["text"] = " ".join(_segment_text(s) for s in snippet["segments"]).strip()
        return AudioTask(
            dataset_name=task.dataset_name,
            data=new_data,
            filepath_key=self.audio_filepath_key,
            _metadata=copy.deepcopy(task._metadata),
            _stage_perf=list(task._stage_perf),
        )

    def _make_stub_task(self, task: AudioTask) -> AudioTask:
        original_id = task.data.get("id")
        stub_data: dict = {
            "id": original_id,
            "snippet_id": None,
            self.audio_filepath_key: None,
            "duration": 0.0,
            "segments": [],
        }
        return AudioTask(
            dataset_name=task.dataset_name,
            data=stub_data,
            _metadata=copy.deepcopy(task._metadata),
            _stage_perf=list(task._stage_perf),
        )
