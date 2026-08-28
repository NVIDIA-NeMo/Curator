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

"""
UTMOS (UTokyo-SaruLab MOS Prediction) filter stage.

Filters audio segments based on UTMOS predicted Mean Opinion Score.
Uses the utmos22_strong model from tarepan/SpeechMOS via torch.hub.

Accepts in-memory (waveform + sample_rate) or audio_filepath input.
Audio is resampled to 16 kHz internally for UTMOS inference.

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.filtering import UTMOSFilterStage
    from nemo_curator.stages.resources import Resources

    pipeline = Pipeline(name="quality_pipeline")
    pipeline.add_stage(
        UTMOSFilterStage(mos_threshold=3.5)
        .with_(resources=Resources(cpus=1.0, gpus=0.5))
    )
"""

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import torch
import torchaudio
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, StageContract
from nemo_curator.stages.audio._agent._residency import (
    resolve_audio,
    scoped_audio_conditional_writes,
    scoped_audio_io_specs,
)
from nemo_curator.stages.audio.common import ensure_mono, ensure_waveform_2d
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_UTMOS_REPO = "tarepan/SpeechMOS:v1.2.0"
_UTMOS_ENTRYPOINT = "utmos22_strong"
_UTMOS_TARGET_SR = 16000
_VALID_MODES = {"task", "segments", "auto"}
_VALID_ACTIONS = {"filter", "annotate"}


def _load_waveform_tensor(  # noqa: PLR0913 (complexity accepted: keyword-only residency/key knobs mirror the stage fields)
    item: dict[str, Any],
    task_id: str,
    *,
    input_residency: Literal["file", "waveform", "auto"] = "auto",
    audio_filepath_key: str = "audio_filepath",
    waveform_key: str = "waveform",
    sample_rate_key: str = "sample_rate",
) -> tuple[torch.Tensor, int] | None:
    """
    Extract a mono waveform tensor (1, N) and sample_rate from an item.

    Supports waveform (Tensor/ndarray) + sample_rate or audio_filepath.
    Returns None if unavailable.
    """
    # Thin wrapper over the shared resolver (_residency.resolve_audio): delegate
    # file/waveform resolution, then force mono (1, N). Two behaviors are kept
    # to match the original exactly: (1) a waveform present without sample_rate
    # is unusable (no file fallback for it); (2) audio-file load errors are
    # swallowed and reported as None.
    if input_residency != "file" and item.get(waveform_key) is not None and item.get(sample_rate_key) is None:
        logger.warning(f"[{task_id}] Waveform present but {sample_rate_key!r} missing - item skipped")
        return None

    try:
        resolved = resolve_audio(
            item,
            residency=input_residency,
            audio_filepath_key=audio_filepath_key,
            waveform_key=waveform_key,
            sample_rate_key=sample_rate_key,
            mono=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"[{task_id}] Failed to load audio file: {e}")
        return None

    if resolved is None:
        if input_residency == "waveform":
            logger.warning(f"[{task_id}] No {waveform_key}+{sample_rate_key} found")
        else:
            logger.warning(f"[{task_id}] No {waveform_key}+{sample_rate_key} or valid {audio_filepath_key} found")
        return None

    waveform, sample_rate = resolved
    return ensure_mono(ensure_waveform_2d(waveform)), int(sample_rate)


@dataclass
class UTMOSFilterStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """
    UTMOS quality assessment filter stage.

    Filters audio segments based on the UTMOS predicted MOS score.
    The model (utmos22_strong) is loaded via torch.hub from tarepan/SpeechMOS.
    Audio is resampled to 16 kHz for inference.

    Args:
        mos_threshold: Minimum MOS score to pass (None to disable)
        sample_rate: Target sample rate for UTMOS inference (default 16000)
        input_residency: Which input to use — "waveform" (in-memory only), "file"
            (audio_filepath only), or "auto" (waveform first, file fallback; default).
        mode: Where to score — "task" (top-level audio), "segments" (nested segments list),
            or "auto" (segments when segments_key is present, else task; default). Setting
            "task" or "segments" overrides the auto-detection.
        action: "filter" drops items below mos_threshold; "annotate" keeps every item —
            including items that fail or cannot be scored — and only writes score_key.
        audio_filepath_key: Key in data dict for the input audio file path.
        waveform_key: Key in data dict for the in-memory waveform tensor.
        sample_rate_key: Key in data dict for the waveform sample rate.
        segments_key: Key in data dict holding the nested segments list (segments/auto mode).
        score_key: Key where the UTMOS MOS score is written.

    Note:
        GPU assignment is handled by the executor via _resources.
        Use .with_(resources=Resources(gpus=X)) to configure GPU allocation.
    """

    SEPARABLE_DECISION_CONSTRAINTS: ClassVar[dict[str, Any]] = {
        "action": "annotate",
        "mode": "task",
    }
    SEPARABLE_DECISION_CONSTRAINTS_BY_SCOPE: ClassVar[dict[str, dict[str, Any]]] = {
        "task": {"action": "annotate", "mode": "task"},
        "segments": {"action": "annotate", "mode": "segments"},
    }

    mos_threshold: float | None = 3.5
    sample_rate: int = _UTMOS_TARGET_SR
    input_residency: Literal["file", "waveform", "auto"] = "auto"
    mode: Literal["task", "segments", "auto"] = "auto"
    action: Literal["filter", "annotate"] = "filter"
    audio_filepath_key: str = "audio_filepath"
    waveform_key: str = "waveform"
    sample_rate_key: str = "sample_rate"
    segments_key: str = "segments"
    score_key: str = "utmos_mos"

    name: str = "UTMOSFilter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=0.5))

    def __post_init__(self):
        super().__init__()
        if self.mode not in _VALID_MODES:
            msg = f"mode must be one of {_VALID_MODES!r}, got {self.mode!r}"
            raise ValueError(msg)
        if self.action not in _VALID_ACTIONS:
            msg = f"action must be one of {_VALID_ACTIONS!r}, got {self.action!r}"
            raise ValueError(msg)
        self._model = None
        self._model_failed = False
        self._resamplers: dict[int, Any] = {}

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.score_key]

    def describe(self) -> StageContract:
        reads, reads_one_of, writes = scoped_audio_io_specs(
            self.input_residency,
            mode=self.mode,
            audio_filepath_key=self.audio_filepath_key,
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
            segments_key=self.segments_key,
            output_keys=[self.score_key],
        )
        return StageContract(
            reads=reads,
            reads_one_of=reads_one_of,
            writes=writes,
            conditional_writes=scoped_audio_conditional_writes(
                self.mode,
                segments_key=self.segments_key,
                output_keys=[self.score_key],
                assignment_condition=(
                    "audio and model inference succeed, a finite numeric MOS is produced, "
                    f"and '{self.score_key}' is assigned"
                    + (
                        " on an item that meets the configured threshold and is retained"
                        if self.action == "filter"
                        else ""
                    )
                ),
            ),
            cardinality="filter" if self.action == "filter" else "1:1",
            cardinality_options=["filter", "annotate"],
            gates=Gates(
                requires_gpu=self.resources.requires_gpu, requires_internet_first_run=True, per_row_independent=True
            ),
        )

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        try:
            torch.hub.load(
                _UTMOS_REPO,
                _UTMOS_ENTRYPOINT,
                trust_repo=True,
                force_reload=False,
                skip_validation=True,
            )
        except Exception:  # noqa: BLE001
            logger.warning("UTMOS repo pre-download in setup_on_node failed.")

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self._ensure_model()
        if self._model is None:
            msg = "UTMOS model failed to load. Check network connectivity and torch.hub cache."
            raise RuntimeError(msg)

    def teardown(self) -> None:
        self._model = None
        self._model_failed = False
        self._resamplers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self._model_failed:
            return

        device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

        try:
            predictor = torch.hub.load(
                _UTMOS_REPO,
                _UTMOS_ENTRYPOINT,
                trust_repo=True,
                force_reload=False,
                skip_validation=True,
            )
        except Exception:  # noqa: BLE001
            logger.warning("UTMOS download failed, loading from cache...")
            try:
                predictor = torch.hub.load(
                    _UTMOS_REPO,
                    _UTMOS_ENTRYPOINT,
                    trust_repo=True,
                    source="local",
                    skip_validation=True,
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"UTMOS model unavailable (download and cache both failed): {e}")
                self._model_failed = True
                return

        predictor = predictor.to(device)
        predictor.eval()

        dummy = torch.randn(1, self.sample_rate, device=device)
        with torch.no_grad():
            _ = predictor(dummy, self.sample_rate)

        self._model = predictor
        logger.info(f"UTMOS model loaded on {device}")

    def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
        """Process a single AudioTask and filter or annotate by UTMOS MOS score.

        Segment mode applies when ``mode="segments"``, or when ``mode="auto"``
        (default) and ``task.data`` contains the ``segments_key``; each segment is
        then evaluated individually. With ``action="filter"`` only survivors are
        kept; with ``action="annotate"`` every item is kept — including items that
        fail mos_threshold or cannot be scored — and only the score is written.
        """
        use_segments = self.mode == "segments" or (self.mode == "auto" and self.segments_key in task.data)
        if use_segments:
            survivors = []
            for seg in task.data.get(self.segments_key, []):
                temp = AudioTask(data=seg)
                result = self._process_single(temp)
                if result is not None or self.action == "annotate":
                    survivors.append(temp.data)
            task.data[self.segments_key] = survivors
            return task if survivors or self.action == "annotate" else []
        return self._process_single(task) or (task if self.action == "annotate" else [])

    def _process_single(self, task: AudioTask) -> AudioTask | None:
        """Run UTMOS scoring on a single (non-nested) task."""
        # This stage owns ``score_key``. Clear a prior annotation before
        # inference so an unscorable rerun cannot leave a stale finite value
        # that a downstream missing=drop selector would incorrectly retain.
        task.data.pop(self.score_key, None)
        audio_result = _load_waveform_tensor(
            task.data,
            task.task_id,
            input_residency=self.input_residency,
            audio_filepath_key=self.audio_filepath_key,
            waveform_key=self.waveform_key,
            sample_rate_key=self.sample_rate_key,
        )
        if audio_result is None:
            return None
        waveform, sr = audio_result

        if self._model is None:
            return None

        try:
            device = next(self._model.parameters()).device
            waveform = waveform.to(device)

            if sr != self.sample_rate:
                if sr not in self._resamplers:
                    self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate).to(device)
                waveform = self._resamplers[sr](waveform)

            with torch.no_grad():
                score = self._model(waveform, sr=self.sample_rate)

            mos = float(score.item() if torch.is_tensor(score) else score)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[{task.task_id}] UTMOS prediction error: {e}")
            return None
        if not math.isfinite(mos):
            logger.warning(f"[{task.task_id}] UTMOS returned non-finite MOS; treating item as unscorable")
            return None

        logger.debug(f"[{task.task_id}] UTMOS MOS={mos:.3f}")

        task.data[self.score_key] = mos

        if self.mos_threshold is not None and mos < self.mos_threshold:
            logger.info(f"[{task.task_id}] UTMOS FAILED: MOS {mos:.3f} < {self.mos_threshold}")
            return task if self.action == "annotate" else None

        return task
