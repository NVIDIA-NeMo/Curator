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

"""NeMo Framework ASR behind the shared :class:`ASRAdapter` contract."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np

from nemo_curator.models.asr.base import ASRResult

_DEFAULT_FASTCONFORMER_CTC_MODEL = "nvidia/stt_en_fastconformer_ctc_large"
_DEFAULT_SAMPLE_RATE = 16_000
_ATTENTION_CONTEXT_DIRECTIONS = 2


def _nemo_asr_module() -> Any:  # noqa: ANN401
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        msg = "NeMoASRAdapter requires the audio_common extra: uv sync --extra audio_common"
        raise ImportError(msg) from exc
    return nemo_asr


def normalize_nemo_transcriptions(outputs: object) -> list[str]:
    """Normalize the output shapes returned by supported NeMo ASR models."""
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


@dataclass
class NeMoASRAdapter:
    """Run a pretrained NeMo checkpoint using waveforms prepared by ``ASRStage``."""

    model_id: str = _DEFAULT_FASTCONFORMER_CTC_MODEL
    revision: str | None = None
    num_workers: int = 0
    verbose: bool = False
    enable_local_attention: bool = False
    local_attention_context_size: tuple[int, int] = (128, 128)
    refresh_cache: bool = False
    strict: bool = True
    _model: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_id:
            msg = "NeMoASRAdapter.model_id must be non-empty"
            raise ValueError(msg)
        self._reject_revision(self.revision)
        if self.num_workers < 0:
            msg = "NeMoASRAdapter.num_workers must be non-negative"
            raise ValueError(msg)
        if not isinstance(self.enable_local_attention, bool):
            msg = "NeMoASRAdapter.enable_local_attention must be a boolean"
            raise TypeError(msg)
        try:
            context_size = tuple(self.local_attention_context_size)
        except TypeError as exc:
            msg = "NeMoASRAdapter.local_attention_context_size must contain two positive integers"
            raise ValueError(msg) from exc
        if len(context_size) != _ATTENTION_CONTEXT_DIRECTIONS or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in context_size
        ):
            msg = "NeMoASRAdapter.local_attention_context_size must contain two positive integers"
            raise ValueError(msg)
        self.local_attention_context_size = (int(context_size[0]), int(context_size[1]))

    @staticmethod
    def _reject_revision(revision: str | None) -> None:
        if revision is not None:
            msg = "NeMo ASRModel.from_pretrained does not support revision pinning"
            raise ValueError(msg)

    @classmethod
    def download_weights_on_node(cls, model_id: str, revision: str | None = None) -> None:
        """Download a pretrained checkpoint without allocating a GPU model."""
        cls._reject_revision(revision)
        _nemo_asr_module().models.ASRModel.from_pretrained(model_name=model_id, return_model_file=True)

    def _load_checkpoint(self, device: Any) -> Any:  # noqa: ANN401
        return _nemo_asr_module().models.ASRModel.from_pretrained(
            model_name=self.model_id,
            map_location=device,
            refresh_cache=self.refresh_cache,
            strict=self.strict,
        )

    def load_model(self, *, num_gpus: int) -> None:
        """Load one worker-local model on the device requested by ``ASRStage``."""
        if self._model is not None:
            return
        if num_gpus < 0:
            msg = "num_gpus must be non-negative"
            raise ValueError(msg)

        import torch

        device = torch.device("cuda" if num_gpus else "cpu")
        model = self._load_checkpoint(device)
        if self.enable_local_attention:
            self._configure_local_attention(model)
        self._model = model

    def _configure_local_attention(self, model: Any) -> None:  # noqa: ANN401
        change_attention_model = getattr(model, "change_attention_model", None)
        change_subsampling_chunking = getattr(model, "change_subsampling_conv_chunking_factor", None)
        encoder = getattr(model, "encoder", None)
        encoder_change_attention = getattr(encoder, "change_attention_model", None)
        encoder_change_subsampling = getattr(encoder, "change_subsampling_conv_chunking_factor", None)
        if (
            not callable(change_attention_model)
            or not callable(change_subsampling_chunking)
            or not callable(encoder_change_attention)
            or not callable(encoder_change_subsampling)
        ):
            msg = f"NeMo checkpoint {self.model_id!r} does not support FastConformer local-attention conversion"
            raise TypeError(msg)
        change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=list(self.local_attention_context_size),
        )
        change_subsampling_chunking(1)

    def unload_model(self) -> None:
        """Release worker-local model and CUDA cache state."""
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        """Transcribe one adapter call while preserving input order."""
        if not items:
            return []
        if self._model is None:
            msg = "NeMoASRAdapter is not initialized; call load_model() first"
            raise RuntimeError(msg)

        model_sample_rate = self._model_sample_rate()
        valid_indices: list[int] = []
        waveforms: list[np.ndarray] = []
        for index, item in enumerate(items):
            waveform = np.asarray(item.get("waveform"), dtype=np.float32)
            if waveform.size == 0:
                continue
            if waveform.ndim != 1:
                msg = f"ASRStage must provide a mono 1-D waveform, got shape {waveform.shape}"
                raise ValueError(msg)
            sample_rate = int(item.get("sample_rate") or 0)
            if sample_rate != model_sample_rate:
                msg = (
                    f"ASRStage must provide {model_sample_rate} Hz audio for {self.model_id!r}; "
                    f"received {sample_rate} Hz"
                )
                raise ValueError(msg)
            waveforms.append(np.ascontiguousarray(waveform))
            valid_indices.append(index)

        results = [ASRResult(text="", skipped=True, skip_reason="empty_audio") for _ in items]
        if not waveforms:
            return results

        outputs = self._model.transcribe(
            audio=waveforms,
            batch_size=len(waveforms),
            return_hypotheses=False,
            num_workers=self.num_workers,
            verbose=self.verbose,
        )
        texts = normalize_nemo_transcriptions(outputs)
        if len(texts) != len(valid_indices):
            msg = f"NeMo returned {len(texts)} transcriptions for {len(valid_indices)} valid inputs"
            raise RuntimeError(msg)

        for index, text in zip(valid_indices, texts, strict=True):
            results[index] = ASRResult(text=text)
        return results

    def _model_sample_rate(self) -> int:
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
