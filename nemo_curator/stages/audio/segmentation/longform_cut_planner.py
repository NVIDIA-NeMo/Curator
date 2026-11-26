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

"""Plan model-safe ASR chunks for long-form recordings, one manifest row per recording."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from loguru import logger

from nemo_curator.stages.audio._agent._agent_ready import (
    AgentReady,
    ConditionalWrite,
    Gates,
    IOSpec,
    Role,
    StageContract,
)
from nemo_curator.stages.audio.segmentation.longform_cuts import (
    LongformCutConfig,
    check_tiling,
    chunks_to_cuts,
    plan_longform_cuts,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from nemo_curator.backends.base import WorkerMetadata

_VAD_SAMPLE_RATE = 16000
_VAD_FRAME_SAMPLES = 512  # Silero v5 window at 16 kHz -> 32 ms frames
_NO_DIAR_SPEECH_FRAC = 0.05


def _set_note(data: dict[str, Any], notes_key: str, stage: str, value: str) -> None:
    notes = data.get(notes_key)
    if not isinstance(notes, dict):
        notes = {}
        data[notes_key] = notes
    notes[stage] = value


@dataclass
class LongformCutPlannerStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Plan where to cut a long recording into ASR-sized chunks, without splitting the row.

    The stage reads the recording, computes Silero VAD speech probabilities per
    32 ms frame, combines them with the diarization intervals already in the
    row, and writes ONE compact field, ``cuts_key``: a list that tiles the whole
    recording, ``[start, end]`` for a chunk that needs ASR and
    ``[start, end, "silence"]`` for a stretch that does not. Times are seconds
    from the start of the original audio, rounded to milliseconds. No audio is
    written and the row is not fanned out; ``ASRStage(segment_bounds_key=...)``
    transcribes the speech chunks and joins their text back into this row.

    Meaning of ``cuts_key``: entries are contiguous (each ``end`` equals the next
    ``start``), the first starts at 0 and the last ends at the decoded audio
    duration, and every speech entry is at most ``max_sec`` long. A silence entry
    means the diarizer heard nobody for at least ``skip_min_sec`` and VAD heard
    speech in at most ``skip_max_vad_speech_frac`` of its frames; the
    ``skip_margin_sec`` on each side stays with the neighbouring speech chunks.
    Counterexample: a silence entry is NOT proof of no sound - it may hold music,
    applause or crowd noise; and a speech entry may contain silence, which ASR
    then simply transcribes as nothing.

    Diarization (``diar_segments_key``, e.g. from ``InferenceSortformerStage``) is
    required. If it is missing the row is marked ``skip_me_key``; if it is empty
    while VAD hears speech in more than 5% of frames, silence skipping is
    disabled for that row (everything goes to ASR) and a note is written.

    Defaults are the values validated on 30 granary-v2 long-form recordings with
    Nemotron-3-Diarization: chunks 120-290 s (target 240 s) to stay under the
    300 s Qwen3-Omni feature-extractor limit.

    Args:
        audio_filepath_key: Path of the original long-form audio.
        diar_segments_key: Diarization intervals of the recording.
        diar_start_key: Start-time field (seconds) inside a diarization interval.
        diar_end_key: End-time field (seconds) inside a diarization interval.
        diar_speaker_key: Speaker-label field inside a diarization interval.
        duration_key: Optional manifest duration, compared with the decoded audio.
        cuts_key: Output cut plan; see the class description.
        skip_me_key: Shared skip-reason field. Rows already carrying a reason are not planned.
        notes_key: Shared per-stage notes field; written only when something is unusual.
        drop_diar_segments: Remove the diarization intervals after planning to keep the manifest lean.
        vad_model_path: Optional local Silero TorchScript file; default is the model bundled
            with the ``silero-vad`` package.
        target_sec: Preferred speech-chunk length in seconds.
        min_sec: Minimum speech-chunk length where the audio allows it.
        max_sec: Hard maximum speech-chunk length; keep below the ASR model input limit.
        min_pause_sec: Shortest VAD pause considered as a cut candidate.
        skip_min_sec: Shortest silence turned into a ``silence`` entry.
        skip_max_vad_speech_frac: VAD veto - maximum fraction of speech frames in a skipped silence.
        merge_below_sec: Speech chunks shorter than this are merged into a neighbour; 0 disables.
        planner_overrides: Expert overrides of other ``LongformCutConfig`` fields (scoring weights).
    """

    name: str = "LongformCutPlanner"
    audio_filepath_key: str = "audio_filepath"
    diar_segments_key: str = "diar_segments"
    diar_start_key: str = "start"
    diar_end_key: str = "end"
    diar_speaker_key: str = "speaker"
    duration_key: str = "duration"
    cuts_key: str = "cuts"
    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"
    drop_diar_segments: bool = True
    vad_model_path: str | None = None

    target_sec: float = 240.0
    min_sec: float = 120.0
    max_sec: float = 290.0
    min_pause_sec: float = 0.15
    skip_min_sec: float = 10.0
    skip_max_vad_speech_frac: float = 0.10
    merge_below_sec: float = 30.0
    planner_overrides: dict[str, Any] = field(default_factory=dict)

    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 1

    _vad_model: Any = field(default=None, init=False, repr=False)
    _frame_prob_fn: Callable[[np.ndarray], np.ndarray] | None = field(default=None, init=False, repr=False)

    KEY_ROLE_OVERRIDES: ClassVar[Mapping[str, Role]] = {
        "diar_start_key": "start",
        "diar_end_key": "end",
        "diar_speaker_key": "speaker_id",
    }
    INTERNAL_KEY_FIELDS: ClassVar[frozenset[str]] = frozenset({"skip_me_key", "notes_key"})

    def __post_init__(self) -> None:
        self.config = self._build_config()
        keys = [self.audio_filepath_key, self.diar_segments_key, self.cuts_key, self.skip_me_key, self.notes_key]
        if any(not isinstance(k, str) or not k.strip() for k in keys) or len(set(keys)) != len(keys):
            msg = f"[{self.name}] task keys must be distinct non-empty strings: {keys}"
            raise ValueError(msg)

    def _build_config(self) -> LongformCutConfig:
        known = {f.name for f in dataclasses.fields(LongformCutConfig)}
        unknown = sorted(set(self.planner_overrides) - known)
        if unknown:
            msg = f"[{self.name}] unknown planner_overrides: {unknown}"
            raise ValueError(msg)
        explicit = {
            "target_sec": self.target_sec,
            "min_sec": self.min_sec,
            "max_sec": self.max_sec,
            "min_pause_sec": self.min_pause_sec,
            "skip_min_sec": self.skip_min_sec,
            "skip_max_vad_speech_frac": self.skip_max_vad_speech_frac,
            "merge_below_sec": self.merge_below_sec,
        }
        clash = sorted(set(explicit) & set(self.planner_overrides))
        if clash:
            msg = f"[{self.name}] set {clash} as stage fields, not in planner_overrides"
            raise ValueError(msg)
        cfg = LongformCutConfig(**explicit, **self.planner_overrides)
        cfg.validate()
        return cfg

    # ---- contract --------------------------------------------------------------------------------
    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.cuts_key]

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.audio_filepath_key, self.diar_segments_key], accepts=["file"]),
            optional_reads=IOSpec(data_keys=[self.duration_key, self.skip_me_key, self.notes_key]),
            writes=IOSpec(data_keys=[self.cuts_key]),
            conditional_writes=[
                ConditionalWrite(
                    writes=IOSpec(data_keys=[self.skip_me_key]),
                    condition="audio cannot be read, diarization is missing, or no valid plan exists",
                ),
                ConditionalWrite(
                    writes=IOSpec(data_keys=[self.notes_key]),
                    condition="diarization is empty while VAD hears speech, the manifest duration "
                    "disagrees with the decoded audio, or the row is skipped",
                ),
            ],
            removes_keys=[self.diar_segments_key] if self.drop_diar_segments else [],
            cardinality="1:1",
            gates=Gates(requires_gpu=False),
        )

    # ---- lifecycle ---------------------------------------------------------------------------------
    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self._frame_prob_fn is not None:
            return
        import torch

        torch.set_num_threads(1)  # Silero on 512-sample windows is call-overhead bound
        if self.vad_model_path:
            model = torch.jit.load(self.vad_model_path, map_location="cpu")
        else:
            from silero_vad import load_silero_vad

            model = load_silero_vad()
        model.eval()
        self._vad_model = model
        self._frame_prob_fn = self._silero_frame_probs

    def teardown(self) -> None:
        self._vad_model = None
        self._frame_prob_fn = None

    def _silero_frame_probs(self, wav16k: np.ndarray) -> np.ndarray:
        import torch

        n = len(wav16k) // _VAD_FRAME_SAMPLES
        if n == 0:
            return np.zeros(0, np.float32)
        x = torch.from_numpy(np.ascontiguousarray(wav16k[: n * _VAD_FRAME_SAMPLES], dtype=np.float32))
        model = self._vad_model
        with torch.inference_mode():
            model.reset_states()
            if hasattr(model, "audio_forward"):
                probs = model.audio_forward(x.unsqueeze(0), _VAD_SAMPLE_RATE).squeeze(0).cpu().numpy()[:n]
                if len(probs) == n:
                    return probs.astype(np.float32)
                model.reset_states()
            chunks = x.view(n, _VAD_FRAME_SAMPLES)
            return np.array([model(chunks[i], _VAD_SAMPLE_RATE).item() for i in range(n)], np.float32)

    @staticmethod
    def _load_mono_16k(path: str) -> np.ndarray:
        import soundfile

        data, sr = soundfile.read(path, dtype="float32", always_2d=True)
        wav = data.mean(axis=1)
        if sr != _VAD_SAMPLE_RATE:
            import torch
            import torchaudio

            wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, _VAD_SAMPLE_RATE).numpy()
        return np.ascontiguousarray(wav, dtype=np.float32)

    # ---- processing ----------------------------------------------------------------------------------
    def _skip(self, task: AudioTask, reason: str) -> AudioTask:
        task.data[self.cuts_key] = []
        task.data[self.skip_me_key] = f"{self.name}: {reason}"
        _set_note(task.data, self.notes_key, self.name, reason)
        if self.drop_diar_segments:
            task.data.pop(self.diar_segments_key, None)
        return task

    def _diar_intervals(self, raw: list[Any]) -> list[tuple[float, float, str]]:
        out = []
        for seg in raw:
            if isinstance(seg, dict):
                a, b, spk = seg[self.diar_start_key], seg[self.diar_end_key], seg.get(self.diar_speaker_key, "")
            else:
                a, b, spk = seg[0], seg[1], seg[2] if len(seg) > 2 else ""  # noqa: PLR2004
            out.append((float(a), float(b), str(spk)))
        return out

    def process(self, task: AudioTask) -> AudioTask:  # noqa: C901, PLR0911
        data = task.data
        if data.get(self.skip_me_key):
            data[self.cuts_key] = []
            if self.drop_diar_segments:
                data.pop(self.diar_segments_key, None)
            return task
        raw_diar = data.get(self.diar_segments_key)
        if raw_diar is None:
            return self._skip(task, "missing diarization")
        if self._frame_prob_fn is None:
            msg = f"[{self.name}] setup() was not called"
            raise RuntimeError(msg)
        try:
            wav = self._load_mono_16k(str(data[self.audio_filepath_key]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[{}] cannot read {}: {}", self.name, data.get(self.audio_filepath_key), exc)
            return self._skip(task, f"audio read error ({type(exc).__name__})")
        duration = len(wav) / _VAD_SAMPLE_RATE
        if duration <= 0:
            return self._skip(task, "empty audio")

        with self._time_metric("vad_time"):
            prob = self._frame_prob_fn(wav)
        diar = self._diar_intervals(raw_diar)
        cfg = self.config
        notes = []
        if not diar and float(np.mean(prob >= cfg.vad_onset)) > _NO_DIAR_SPEECH_FRAC:
            cfg = dataclasses.replace(cfg, skip_min_sec=math.inf)
            notes.append("diarization empty but VAD hears speech: silence skipping disabled")
        manifest_dur = data.get(self.duration_key)
        if isinstance(manifest_dur, (int, float)) and abs(float(manifest_dur) - duration) > 1.0:
            notes.append(f"decoded duration {duration:.2f}s != manifest {float(manifest_dur):.2f}s")

        try:
            with self._time_metric("plan_time"):
                chunks = plan_longform_cuts(prob, _VAD_FRAME_SAMPLES / _VAD_SAMPLE_RATE, diar, duration, cfg)
        except RuntimeError as exc:
            return self._skip(task, f"no valid cut plan ({exc})")
        cuts = chunks_to_cuts(chunks)
        err = check_tiling(cuts, round(duration, 3), cfg.max_sec)
        if err:
            return self._skip(task, f"invalid cut plan ({err})")

        data[self.cuts_key] = cuts
        if notes:
            _set_note(data, self.notes_key, self.name, "; ".join(notes))
        if self.drop_diar_segments:
            data.pop(self.diar_segments_key, None)
        n_speech = sum(1 for c in cuts if len(c) == 2)  # noqa: PLR2004
        self._log_metrics(
            {
                "audio_sec": duration,
                "speech_chunks": n_speech,
                "silence_sec": sum(c[1] - c[0] for c in cuts if len(c) == 3),  # noqa: PLR2004
            }
        )
        return task
