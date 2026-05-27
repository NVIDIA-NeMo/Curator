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

"""AI4Bharat Indic Conformer ASR (Hugging Face ``trust_remote_code``).

See model card: https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
"""

from __future__ import annotations

import gc
import os
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

if TYPE_CHECKING:
    import numpy as np

from nemo_curator.models.base import ModelInterface

# Silence ONNX Runtime informational chatter from the model_onnx.py wrapper.
# Must be set before any ``import onnxruntime`` so the global env is captured.
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

INDIC_CONFORMER_DEFAULT_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
_TARGET_SR = 16000
_ORT_INTRA_OP_THREADS = 2
_ORT_INTER_OP_THREADS = 1


class IndicConformerASR(ModelInterface):
    """Indic Conformer multilingual ASR via Hugging Face ``AutoModel`` + custom forward."""

    def __init__(
        self,
        model_id: str = INDIC_CONFORMER_DEFAULT_MODEL_ID,
        decode_mode: Literal["ctc", "rnnt"] = "ctc",
    ):
        self.model_id = model_id
        self.decode_mode = decode_mode
        self._model: Any = None
        self._device: Any = None

    @property
    def model_id_names(self) -> list[str]:
        return [self.model_id]

    def setup(self) -> None:
        try:
            import onnxruntime as ort
            import torch
            from transformers import AutoModel
        except ImportError as e:
            msg = (
                "transformers, torch, and onnxruntime are required for IndicConformerASR. "
                "Install per the model card, e.g. pip install transformers torchaudio onnxruntime-gpu"
            )
            raise ImportError(msg) from e

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Indic Conformer model={self.model_id} decode={self.decode_mode} device={self._device}")

        # Patch ort.InferenceSession.__init__ for the duration of from_pretrained
        # so the HF wrapper's ~28 session creations pick up sensible defaults
        # (small thread pools, sequential execution, full graph optimisation,
        # quiet logger). Restore the original immediately after to avoid global
        # side effects on any other ORT user (e.g. SigMOS) in the same process.
        _orig_init = ort.InferenceSession.__init__

        def _patched_init(self_: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            if not kwargs.get("sess_options"):
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = _ORT_INTRA_OP_THREADS
                opts.inter_op_num_threads = _ORT_INTER_OP_THREADS
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.log_severity_level = 3
                kwargs["sess_options"] = opts
            return _orig_init(self_, *args, **kwargs)

        ort.InferenceSession.__init__ = _patched_init
        try:
            self._model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        finally:
            ort.InferenceSession.__init__ = _orig_init

        self._model.to(self._device)
        self._model.eval()

        logger.info("Indic Conformer model loaded")

    def teardown(self) -> None:
        del self._model
        self._model = None
        self._device = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110
            pass

    def generate(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
        lang_codes: list[str],
    ) -> tuple[list[str], list[str]]:
        """Transcribe waveforms using per-sample ISO language codes (e.g. ``hi``, ``ta``)."""
        if self._model is None:
            msg = "Model not initialized. Call setup() first."
            raise RuntimeError(msg)

        import torch
        import torchaudio.functional as ta_F  # noqa: N812

        texts: list[str] = []
        langs_out: list[str] = []

        with torch.inference_mode():
            for w, sr, lang in zip(waveforms, sample_rates, lang_codes, strict=True):
                if w.size == 0:
                    texts.append("")
                    langs_out.append(lang)
                    continue

                wav = torch.from_numpy(w.copy()).to(dtype=torch.float32, device=self._device)
                if wav.ndim > 1:
                    wav = wav.mean(dim=-1)
                wav = wav.unsqueeze(0)
                if int(sr) != _TARGET_SR:
                    wav = ta_F.resample(wav, orig_freq=int(sr), new_freq=_TARGET_SR)

                raw = self._model(wav, lang, self.decode_mode)
                text = raw if isinstance(raw, str) else str(raw)
                texts.append(text)
                langs_out.append(lang)

        return texts, langs_out
