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

"""Indic Parakeet RNN-T adapter with a TensorRT encoder and NeMo decoder."""

from __future__ import annotations

import gc
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from loguru import logger

from nemo_curator.models.asr.nemo_asr import NeMoASRAdapter
from nemo_curator.stages.audio.inference.audio_chunking import (
    engine_chunk_duration,
    has_audio_longer_than,
    merge_chunk_texts,
    split_waveforms,
)
from nemo_curator.stages.audio.inference.tensorrt_encoder import (
    ENGINE_FILENAME,
    MODEL_FILENAME,
    TensorRTEncoder,
)
from nemo_curator.stages.audio.inference.tensorrt_encoder import (
    load_engine_metadata as _load_engine_metadata,
)

if TYPE_CHECKING:
    import numpy as np

_MAX_CHUNK_DURATION_SEC = 40.0


def load_engine_metadata(engine_dir: str | Path) -> dict[str, Any]:
    """Load and validate an Indic Parakeet RNN-T engine bundle manifest."""
    return _load_engine_metadata(
        engine_dir,
        model_type="indic_parakeet_rnnt",
        required_positive_ints=(
            "sample_rate",
            "feature_count",
            "subsampling_factor",
            "vocabulary_size",
            "max_symbols_per_step",
        ),
    )


class TensorRTParakeetRNNTAdapter(NeMoASRAdapter):
    """Run a bundled Indic Parakeet model with only its encoder replaced by TensorRT.

    ``ASRStage.batch_size`` owns the externally visible inference batch. The
    adapter therefore has no independent batch-size setting; the shared
    ``TensorRTEncoder`` subdivides only when the serialized engine profile
    requires it.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_id: str = "",
        *,
        engine_dir: str | Path,
        chunking_mode: Literal["engine", "none"] = "engine",
        num_workers: int = 0,
        empty_audio_marks_skip: bool = False,
        verbose: bool = False,
    ) -> None:
        if chunking_mode not in {"engine", "none"}:
            msg = f"Unsupported Indic Parakeet chunking mode: {chunking_mode!r}"
            raise ValueError(msg)
        if not str(engine_dir):
            msg = "TensorRTParakeetRNNTAdapter.engine_dir must be non-empty"
            raise ValueError(msg)

        self.configured_model_id = model_id
        self.engine_dir = Path(engine_dir)
        self.metadata: dict[str, Any] | None = None
        self.chunking_mode = chunking_mode
        self._engine_path = self.engine_dir / ENGINE_FILENAME
        self._model_path = self.engine_dir / MODEL_FILENAME
        self._trt_encoder: TensorRTEncoder | None = None
        self._chunk_duration_sec: float | None = None
        super().__init__(
            model_id=str(self._model_path),
            num_workers=num_workers,
            empty_audio_marks_skip=empty_audio_marks_skip,
            verbose=verbose,
        )

    def _validate_bundle(self) -> dict[str, Any]:
        metadata = load_engine_metadata(self.engine_dir)
        if not self._engine_path.is_file():
            msg = f"TensorRT encoder engine not found: {self._engine_path}"
            raise FileNotFoundError(msg)
        if not self._model_path.is_file():
            msg = f"Bundled NeMo model not found: {self._model_path}"
            raise FileNotFoundError(msg)
        return metadata

    def download_weights_on_node(self) -> None:
        """Validate the complete node-local bundle without loading either model."""
        self.metadata = self._validate_bundle()

    def load_model(self, *, num_gpus: int) -> None:
        """Load NeMo decoding components and replace only the encoder with TensorRT."""
        if self._model is not None:
            return
        if isinstance(num_gpus, bool) or not isinstance(num_gpus, Integral) or num_gpus != 1:
            msg = f"Indic Parakeet TensorRT inference requires exactly one GPU, got {num_gpus!r}"
            raise ValueError(msg)
        if not torch.cuda.is_available():
            msg = "Indic Parakeet TensorRT inference requires CUDA"
            raise RuntimeError(msg)

        self.metadata = self._validate_bundle()
        try:
            super().load_model(num_gpus=1)
            self._validate_model()
            self._model.to(dtype=torch.float16)
            self._enable_batched_greedy_decoder()

            original_encoder = self._model.encoder
            self._model.encoder = None
            del original_encoder
            gc.collect()
            torch.cuda.empty_cache()

            optimized_encoder = TensorRTEncoder(
                self._engine_path,
                subsampling_factor=int(self.metadata["subsampling_factor"]),
            )
            self._trt_encoder = optimized_encoder
            self._configure_engine_chunking()
            self._model.encoder = optimized_encoder
            logger.info("Indic Parakeet TensorRT encoder loaded: {}", self._engine_path)
        except Exception:
            self._close_tensorrt_encoder()
            super().unload_model()
            self._chunk_duration_sec = None
            raise

    def _configure_engine_chunking(self) -> None:
        if self.chunking_mode == "none":
            return
        if self._model is None or self._trt_encoder is None:
            msg = "Indic Parakeet model and TensorRT encoder must be loaded before configuring chunking"
            raise RuntimeError(msg)
        max_feature_frames = self._trt_encoder.max_input_shape("audio_signal")[2]
        engine_duration = engine_chunk_duration(self._model, max_feature_frames)
        if engine_duration < _MAX_CHUNK_DURATION_SEC:
            msg = (
                "Indic Parakeet TensorRT engine does not support 40-second audio: "
                f"max_feature_frames={max_feature_frames}; rebuild with --max-frames 4001"
            )
            raise ValueError(msg)
        self._chunk_duration_sec = _MAX_CHUNK_DURATION_SEC

    def _validate_model(self) -> None:
        model = self._model
        metadata = self.metadata
        if model is None or metadata is None:
            msg = "Indic Parakeet model and TensorRT metadata must be loaded before validation"
            raise RuntimeError(msg)
        if not hasattr(model, "decoder") or not hasattr(model, "joint"):
            msg = "Indic Parakeet TensorRT backend requires a NeMo RNN-T model"
            raise TypeError(msg)

        encoder = getattr(model, "encoder", None)
        expected_subsampling = int(metadata["subsampling_factor"])
        actual_subsampling = int(getattr(encoder, "subsampling_factor", -1))
        if actual_subsampling != expected_subsampling:
            msg = (
                "Bundled NeMo model does not match the TensorRT engine: "
                f"subsampling_factor={actual_subsampling}, expected={expected_subsampling}"
            )
            raise ValueError(msg)

        actual_feature_count = int(getattr(encoder, "_feat_in", getattr(model.cfg.encoder, "feat_in", -1)))
        expected_feature_count = int(metadata["feature_count"])
        if actual_feature_count != expected_feature_count:
            msg = (
                "Bundled NeMo model does not match the TensorRT engine: "
                f"feature_count={actual_feature_count}, expected={expected_feature_count}"
            )
            raise ValueError(msg)

        preprocessor_cfg = getattr(model.cfg, "preprocessor", None)
        actual_sample_rate = int(getattr(preprocessor_cfg, "sample_rate", -1))
        expected_sample_rate = int(metadata["sample_rate"])
        if actual_sample_rate != expected_sample_rate:
            msg = (
                "Bundled NeMo model does not match the TensorRT engine: "
                f"sample_rate={actual_sample_rate}, expected={expected_sample_rate}"
            )
            raise ValueError(msg)

        actual_vocabulary_size = int(getattr(model.joint, "_vocab_size", getattr(model.cfg.joint, "num_classes", -1)))
        expected_vocabulary_size = int(metadata["vocabulary_size"])
        if actual_vocabulary_size != expected_vocabulary_size:
            msg = (
                "Bundled NeMo model does not match the TensorRT engine: "
                f"vocabulary_size={actual_vocabulary_size}, expected={expected_vocabulary_size}"
            )
            raise ValueError(msg)

    def _enable_batched_greedy_decoder(self) -> None:
        model = self._model
        metadata = self.metadata
        if model is None or metadata is None:
            msg = "Indic Parakeet model and TensorRT metadata must be loaded before configuring decoding"
            raise RuntimeError(msg)

        from omegaconf import OmegaConf, open_dict

        with open_dict(model.cfg):
            model.cfg.decoding.strategy = "greedy_batch"
            greedy_cfg = OmegaConf.select(model.cfg, "decoding.greedy")
            if greedy_cfg is None:
                model.cfg.decoding.greedy = OmegaConf.create({})
            model.cfg.decoding.greedy.max_symbols_per_step = int(metadata["max_symbols_per_step"])
            model.cfg.decoding.greedy.use_cuda_graph_decoder = True
        model.change_decoding_strategy(model.cfg.decoding)

    def _transcribe_waveforms(self, waveforms: list[np.ndarray]) -> list[str]:
        if self.chunking_mode == "none":
            return super()._transcribe_waveforms(waveforms)
        if self._chunk_duration_sec is None:
            msg = "Indic Parakeet chunk duration was not initialized from the engine"
            raise RuntimeError(msg)

        sample_rates = [self._DEFAULT_SAMPLE_RATE] * len(waveforms)
        requires_merge = has_audio_longer_than(waveforms, sample_rates, self._chunk_duration_sec)
        chunks, chunk_sample_rates, owners = split_waveforms(
            waveforms,
            sample_rates,
            self._chunk_duration_sec,
        )
        if not chunks:
            return [""] * len(waveforms)

        duration_order = sorted(
            range(len(chunks)),
            key=lambda index: chunks[index].shape[0] / chunk_sample_rates[index],
        )
        ordered_chunks = [chunks[index] for index in duration_order]
        if self._trt_encoder is None:
            msg = "Indic Parakeet TensorRT encoder must be loaded before transcription"
            raise RuntimeError(msg)
        engine_max_batch = self._trt_encoder.max_input_shape("audio_signal")[0]
        ordered_texts = []
        for start in range(0, len(ordered_chunks), engine_max_batch):
            ordered_texts.extend(super()._transcribe_waveforms(ordered_chunks[start : start + engine_max_batch]))
        chunk_texts = [""] * len(chunks)
        for original_index, text in zip(duration_order, ordered_texts, strict=True):
            chunk_texts[original_index] = text
        if not requires_merge:
            texts = [""] * len(waveforms)
            for text, owner in zip(chunk_texts, owners, strict=True):
                texts[owner] = text
            return texts
        return merge_chunk_texts(chunk_texts, owners, len(waveforms))

    def _close_tensorrt_encoder(self) -> None:
        if self._trt_encoder is not None:
            self._trt_encoder.close()
            self._trt_encoder = None

    def unload_model(self) -> None:
        self._close_tensorrt_encoder()
        self._chunk_duration_sec = None
        self.metadata = None
        super().unload_model()


__all__ = ["TensorRTParakeetRNNTAdapter", "load_engine_metadata"]
