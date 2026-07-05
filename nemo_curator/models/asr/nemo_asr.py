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

"""NeMo Framework ASR models behind the shared :class:`ASRAdapter` contract."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.models.asr.waveform import resample_waveform, to_mono_numpy_1d

_DEFAULT_FASTCONFORMER_CTC_MODEL = "nvidia/stt_en_fastconformer_ctc_large"
_DEFAULT_SAMPLE_RATE = 16_000


def _nemo_asr_module() -> Any:  # noqa: ANN401
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        msg = "NeMoASRAdapter requires the audio_common extra: uv sync --extra audio_common"
        raise ImportError(msg) from exc
    return nemo_asr


@dataclass
class NeMoASRAdapter:
    """Run a NeMo ASR checkpoint as exact ``ASRStage`` adapter batches.

    The default checkpoint is NVIDIA's English FastConformer CTC model. The
    adapter accepts the in-memory waveform items produced by ``ASRStage`` and
    passes all valid items from one adapter call to one NeMo transcription
    DataLoader batch. This preserves global dispatch boundaries instead of
    letting NeMo silently fall back to its public ``batch_size=4`` default.

    Any NeMo checkpoint compatible with ``ASRModel.from_pretrained`` may be
    selected through ``model_id``. ``revision`` is accepted to satisfy the
    shared adapter constructor, but NeMo's loader has no revision argument and
    therefore rejects non-``None`` revisions explicitly.
    """

    model_id: str = _DEFAULT_FASTCONFORMER_CTC_MODEL
    revision: str | None = None
    target_sample_rate: int | None = None
    num_workers: int = 0
    verbose: bool = False
    device: str | None = None
    refresh_cache: bool = False
    strict: bool = True
    last_metrics: dict[str, float] = field(default_factory=dict)
    _model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_id:
            msg = "NeMoASRAdapter.model_id must be non-empty"
            raise ValueError(msg)
        if self.revision is not None:
            msg = "NeMo ASRModel.from_pretrained does not support revision pinning"
            raise ValueError(msg)
        if self.target_sample_rate is not None and self.target_sample_rate <= 0:
            msg = "NeMoASRAdapter.target_sample_rate must be positive when set"
            raise ValueError(msg)
        if self.num_workers < 0:
            msg = "NeMoASRAdapter.num_workers must be non-negative"
            raise ValueError(msg)

    @classmethod
    def prefetch_weights(cls, model_id: str, revision: str | None = None) -> None:
        """Download the NeMo checkpoint without constructing a GPU model."""
        if revision is not None:
            msg = "NeMo ASRModel.from_pretrained does not support revision pinning"
            raise ValueError(msg)
        _nemo_asr_module().models.ASRModel.from_pretrained(model_name=model_id, return_model_file=True)

    def setup(self) -> None:
        """Load one worker-local NeMo model on the selected device."""
        if self._model is not None:
            return

        import torch

        device = (
            torch.device(self.device) if self.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = _nemo_asr_module().models.ASRModel.from_pretrained(
            model_name=self.model_id,
            map_location=device,
            refresh_cache=self.refresh_cache,
            strict=self.strict,
        )

    def teardown(self) -> None:
        """Release worker-local model and CUDA cache state."""
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def estimate_item_cost(self, item: dict[str, Any]) -> float | None:
        """Prefer explicit encoder/VRAM estimates, then audio duration."""
        for key in ("estimated_vram_units", "estimated_encoder_tokens", "audio_seconds"):
            value = item.get(key)
            if isinstance(value, Real):
                return max(0.0, float(value))
        return None

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Transcribe one adapter call while preserving input order."""
        if not items:
            self.last_metrics = self._metrics(input_count=0, valid_count=0, elapsed_s=0.0)
            return []
        if self._model is None:
            msg = "NeMoASRAdapter is not initialized; call setup() first"
            raise RuntimeError(msg)

        sample_rate = self._model_sample_rate()
        valid_indices: list[int] = []
        waveforms: list[np.ndarray] = []
        for index, item in enumerate(items):
            waveform = to_mono_numpy_1d(item.get("waveform"))
            source_rate = int(item.get("sample_rate") or 0)
            if waveform.size == 0 or source_rate <= 0:
                continue
            waveforms.append(resample_waveform(waveform, source_rate, sample_rate))
            valid_indices.append(index)

        results = [ASRResult(text="", skipped=True, model_id=self.model_id) for _ in items]
        if not waveforms:
            self.last_metrics = self._metrics(input_count=len(items), valid_count=0, elapsed_s=0.0)
            return results

        started = time.perf_counter()
        outputs = self._model.transcribe(
            audio=waveforms,
            batch_size=len(waveforms),
            return_hypotheses=False,
            num_workers=self.num_workers,
            verbose=self.verbose,
        )
        elapsed_s = time.perf_counter() - started
        texts = self._normalize_transcriptions(outputs)
        if len(texts) != len(valid_indices):
            msg = f"NeMo returned {len(texts)} transcriptions for {len(valid_indices)} valid inputs"
            raise RuntimeError(msg)

        for index, text in zip(valid_indices, texts, strict=True):
            results[index] = ASRResult(text=text, skipped=False, model_id=self.model_id)
        self.last_metrics = self._metrics(
            input_count=len(items),
            valid_count=len(valid_indices),
            elapsed_s=elapsed_s,
        )
        return results

    def _model_sample_rate(self) -> int:
        if self.target_sample_rate is not None:
            return int(self.target_sample_rate)

        preprocessor = getattr(self._model, "preprocessor", None)
        value = getattr(preprocessor, "_sample_rate", None)
        if isinstance(value, Real) and value > 0:
            return int(value)

        config = getattr(self._model, "cfg", None)
        get_value = getattr(config, "get", None)
        if callable(get_value):
            value = get_value("sample_rate")
            if isinstance(value, Real) and value > 0:
                return int(value)
        return _DEFAULT_SAMPLE_RATE

    @staticmethod
    def _normalize_transcriptions(outputs: object) -> list[str]:
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if outputs is None:
            return []
        if not isinstance(outputs, list):
            msg = f"Unsupported NeMo transcription output type: {type(outputs).__name__}"
            raise TypeError(msg)

        texts: list[str] = []
        for output in outputs:
            primary = (output[0] if output else "") if isinstance(output, list) else output
            text = getattr(primary, "text", primary)
            if not isinstance(text, str):
                msg = f"Unsupported NeMo transcription item type: {type(primary).__name__}"
                raise TypeError(msg)
            texts.append(text)
        return texts

    @staticmethod
    def _metrics(*, input_count: int, valid_count: int, elapsed_s: float) -> dict[str, float]:
        return {
            "utterances_input": float(input_count),
            "utterances_valid": float(valid_count),
            "utterances_skipped_preprocess": float(input_count - valid_count),
            "transcribe_calls": float(valid_count > 0),
            "transcribe_items": float(valid_count),
            "requested_batch_size": float(valid_count),
            "transcribe_time_s": float(elapsed_s),
        }
