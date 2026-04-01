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

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from loguru import logger

try:
    import torchaudio

    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    torchaudio = None
    _TORCHAUDIO_AVAILABLE = False

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio.common import load_audio_file
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_UTMOS_REPO = "tarepan/SpeechMOS:v1.2.0"
_UTMOS_ENTRYPOINT = "utmos22_strong"
_UTMOS_TARGET_SR = 16000


def _load_waveform_tensor(item: dict[str, Any], task_id: str) -> tuple[torch.Tensor, int] | None:
    """
    Extract a mono waveform tensor (1, N) and sample_rate from an item.

    Supports waveform (Tensor/ndarray) + sample_rate or audio_filepath.
    Returns None if unavailable.
    """
    waveform = item.get("waveform")
    sample_rate = item.get("sample_rate")

    if waveform is not None and sample_rate is not None:
        if not torch.is_tensor(waveform):
            waveform = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, int(sample_rate)

    if waveform is not None and sample_rate is None:
        logger.warning(f"[{task_id}] Waveform present but 'sample_rate' missing - item skipped")
        return None

    path = item.get("audio_filepath")
    if path and os.path.isfile(path):
        try:
            return load_audio_file(path, mono=True)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{task_id}] Failed to load audio file: {e}")
            return None

    logger.warning(f"[{task_id}] No waveform+sample_rate or valid audio_filepath found")
    return None


@dataclass
class UTMOSFilterStage(ProcessingStage[AudioTask, AudioTask]):
    """
    UTMOS quality assessment filter stage.

    Filters audio segments based on the UTMOS predicted MOS score.
    The model (utmos22_strong) is loaded via torch.hub from tarepan/SpeechMOS.
    Audio is resampled to 16 kHz for inference.

    Args:
        mos_threshold: Minimum MOS score to pass (None to disable)
        sample_rate: Target sample rate for UTMOS inference (default 16000)

    Note:
        GPU assignment is handled by the executor via _resources.
        Use .with_(resources=Resources(gpus=X)) to configure GPU allocation.
    """

    mos_threshold: float | None = 3.5
    sample_rate: int = _UTMOS_TARGET_SR

    name: str = "UTMOSFilter"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=0.5))

    def __post_init__(self):
        super().__init__()
        self._model = None
        self._model_failed = False
        self._resamplers: dict[int, Any] = {}

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["utmos_mos"]

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

    def teardown(self) -> None:
        self._model = None
        self._model_failed = False
        self._resamplers.clear()
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
        """Process a single AudioTask and filter by UTMOS MOS score.

        When ``task.data`` contains a ``"segments"`` key (nested mode from VAD),
        each segment is evaluated individually and only survivors are kept.
        """
        if "segments" in task.data:
            survivors = []
            for seg in task.data["segments"]:
                temp = AudioTask(data=seg, task_id=task.task_id)
                result = self._process_single(temp)
                if result is not None:
                    survivors.append(temp.data)
            task.data["segments"] = survivors
            return task if survivors else []
        return self._process_single(task) or []

    def _process_single(self, task: AudioTask) -> AudioTask | None:
        """Run UTMOS scoring on a single (non-nested) task."""
        audio_result = _load_waveform_tensor(task.data, task.task_id)
        if audio_result is None:
            return None
        waveform, sr = audio_result

        self._ensure_model()
        if self._model is None:
            return None

        try:
            device = next(self._model.parameters()).device
            waveform = waveform.to(device)

            if sr != self.sample_rate:
                if not _TORCHAUDIO_AVAILABLE:
                    msg = "torchaudio is required for resampling. Install it with: pip install torchaudio"
                    raise ImportError(msg)  # noqa: TRY301
                if sr not in self._resamplers:
                    self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate).to(device)
                waveform = self._resamplers[sr](waveform)

            with torch.no_grad():
                score = self._model(waveform, sr=self.sample_rate)

            mos = float(score.item() if torch.is_tensor(score) else score)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[{task.task_id}] UTMOS prediction error: {e}")
            return None

        logger.debug(f"[{task.task_id}] UTMOS MOS={mos:.3f}")

        if self.mos_threshold is not None and mos < self.mos_threshold:
            logger.info(f"[{task.task_id}] UTMOS FAILED: MOS {mos:.3f} < {self.mos_threshold}")
            return None

        task.data["utmos_mos"] = mos
        return task
