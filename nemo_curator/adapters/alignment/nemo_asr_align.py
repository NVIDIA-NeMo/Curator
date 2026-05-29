# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""NeMo ASR forced-alignment adapter.

Implements :class:`~nemo_curator.adapters.alignment.ForcedAlignmentAdapter`
on top of NeMo's ``ASRModel.transcribe(timestamps=True)`` path
(FastConformer + CTC / RNNT decoders).

Logic moved verbatim from the pre-split
``nemo_curator.stages.audio.tagging.inference.nemo_asr_align.NeMoASRAlignerStage``
body so per-word timestamps, confidences, RNNT 0.08 s offset, ``⁇``
strip and one-by-one retry fallback are byte-for-byte preserved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import nemo.collections.asr as nemo_asr
import torch
from loguru import logger
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

from nemo_curator.adapters.alignment.base import AlignmentResult, WordAlignment


@dataclass
class NeMoASRAlignAdapter:
    """NeMo-backed implementation of :class:`ForcedAlignmentAdapter`.

    Tier-2 knobs (set via ``adapter_kwargs`` in YAML):

    Attributes:
        model_id: Pretrained model identifier passed to
            ``ASRModel.from_pretrained`` (e.g.
            ``"nvidia/parakeet-tdt_ctc-1.1b"``). Ignored when
            ``model_path`` is set.
        revision: Accepted for protocol uniformity. NeMo's
            ``from_pretrained`` does not currently accept a revision
            argument; passed through to ``extras`` for diagnostics.
        model_path: Optional local ``.nemo`` checkpoint path. When set
            overrides ``model_id``.
        device: ``"cuda"`` or ``"cpu"``; passed by the stage.
        is_fastconformer: Whether the model encoder is FastConformer
            (triggers ``change_attention_model`` /
            ``change_subsampling_conv_chunking_factor`` calls and
            adjusts the per-token time stride).
        decoder_type: ``"ctc"`` or ``"rnnt"``.
        timestamp_type: ``"word"`` or ``"char"``.
        transcribe_batch_size: Batch size for the NeMo
            ``transcribe`` call.
        num_workers: Number of data-loading workers.
        compute_timestamps: When False, returns alignments=[] (text
            only). Pre-split behaviour was True.
        disable_word_confidence: When True, the adapter does NOT
            populate ``WordAlignment.confidence``.
    """

    # ---- Required protocol fields ----
    model_id: str = "nvidia/parakeet-tdt_ctc-1.1b"
    revision: str | None = None

    # ---- Adapter-specific knobs ----
    model_path: str | None = None
    device: str = "cuda"
    is_fastconformer: bool = True
    decoder_type: str = "rnnt"
    timestamp_type: str = "word"
    transcribe_batch_size: int = 32
    num_workers: int = 10
    compute_timestamps: bool = True
    disable_word_confidence: bool = False

    # ---- Internal state ----
    _asr_model: Any = field(default=None, repr=False)
    _override_cfg: Any = field(default=None, repr=False)
    last_metrics: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Adapter contract
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.decoder_type not in ("ctc", "rnnt"):
            msg = f"decoder_type must be 'ctc' or 'rnnt', got {self.decoder_type}"
            raise ValueError(msg)
        if self.timestamp_type not in ("word", "char"):
            msg = f"timestamp_type must be 'word' or 'char', got {self.timestamp_type}"
            raise ValueError(msg)

    @classmethod
    def prefetch_weights(cls, model_id: str, revision: str | None = None) -> None:
        """Download model weights without instantiating the GPU runtime.

        ``ASRModel.from_pretrained(return_model_file=True)`` is the
        public entry point that triggers the HF / NGC download.
        """
        del revision  # NeMo's from_pretrained doesn't take a revision arg today.
        if not model_id:
            return
        try:
            nemo_asr.models.ASRModel.from_pretrained(model_name=model_id, return_model_file=True)
        except Exception as exc:  # noqa: BLE001
            msg = f"NeMoASRAlignAdapter: failed to download model {model_id}"
            raise RuntimeError(msg) from exc

    def setup(self) -> None:
        if self._asr_model is None:
            if self.model_path:
                self._asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=self.model_path)
            else:
                self._asr_model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.model_id,
                    map_location=torch.device(self.device),
                )

        self._asr_model.to(self.device)
        self._asr_model.eval()

        if self.is_fastconformer:
            self._asr_model.change_attention_model(
                self_attention_model="rel_pos_local_attn", att_context_size=[128, 128]
            )
            self._asr_model.change_subsampling_conv_chunking_factor(1)

        decoding_cfg = CTCDecodingConfig() if self.decoder_type == "ctc" else RNNTDecodingConfig()
        if self.decoder_type == "ctc":
            decoding_cfg.strategy = "greedy_batch"
        else:
            decoding_cfg.rnnt_timestamp_type = self.timestamp_type

        decoding_cfg.preserve_alignments = self.compute_timestamps
        decoding_cfg.confidence_cfg.preserve_word_confidence = not self.disable_word_confidence
        decoding_cfg.compute_timestamps = self.compute_timestamps
        decoding_cfg.greedy.compute_timestamps = self.compute_timestamps

        self._asr_model.change_decoding_strategy(decoding_cfg=decoding_cfg)

        self._override_cfg = self._asr_model.get_transcribe_config()
        self._override_cfg.batch_size = self.transcribe_batch_size
        self._override_cfg.num_workers = self.num_workers
        self._override_cfg.return_hypotheses = True
        self._override_cfg.timestamps = self.compute_timestamps

        logger.info("NeMoASRAlignAdapter ready on {} (model={})", self.device, self.model_id)

    def teardown(self) -> None:
        self._asr_model = None
        self._override_cfg = None

    def align_batch(self, items: list[dict[str, Any]]) -> list[AlignmentResult]:
        if not items:
            return []
        if self._asr_model is None:
            msg = "NeMoASRAlignAdapter.setup() must be called before align_batch()"
            raise RuntimeError(msg)

        t0 = time.perf_counter()
        # Classify the batch as path-mode or segment-mode. The stage
        # guarantees homogeneity per call; we double-check defensively.
        first = items[0]
        if first.get("audio_segment") is not None:
            transcribe_inputs = [it["audio_segment"] for it in items]
            mode = "segment"
        else:
            transcribe_inputs = [it.get("audio_path") for it in items]
            mode = "path"

        hypotheses_list = self._transcribe(transcribe_inputs, mode=mode)

        results: list[AlignmentResult] = []
        for hyp in hypotheses_list:
            if hyp is None:
                results.append(AlignmentResult(alignments=[], text="", model_id=self.model_id))
                continue
            alignments, text = self._get_alignments_text(hyp)
            results.append(
                AlignmentResult(
                    alignments=[
                        WordAlignment(
                            word=w["word"],
                            start=w["start"],
                            end=w["end"],
                            confidence=w.get("confidence"),
                        )
                        for w in alignments
                    ],
                    text=text,
                    model_id=self.model_id,
                )
            )

        self.last_metrics = {
            "batch_size": float(len(items)),
            "align_time_s_total": float(time.perf_counter() - t0),
            "mode_is_segment": 1.0 if mode == "segment" else 0.0,
        }
        return results

    # ------------------------------------------------------------------
    # Internal helpers (moved verbatim from NeMoASRAlignerStage)
    # ------------------------------------------------------------------

    def _transcribe(self, inputs: list[Any], mode: str) -> list[Any]:
        """Run ``ASRModel.transcribe`` with a one-by-one retry fallback."""
        try:
            with torch.no_grad():
                hypotheses_list = self._asr_model.transcribe(
                    inputs, override_config=self._override_cfg
                )
            if isinstance(hypotheses_list, tuple) and len(hypotheses_list) == 2:  # noqa: PLR2004
                hypotheses_list = hypotheses_list[0]
            return list(hypotheses_list)
        except Exception as exc:  # noqa: BLE001
            if mode == "segment":
                msg = f"NeMoASRAlignAdapter: batch transcribe failed in segment mode: {exc}"
                raise ValueError(msg) from exc
            logger.error(
                "NeMoASRAlignAdapter: batch transcribe failed ({}), retrying one-by-one", exc
            )
            out: list[Any] = []
            for single_input in inputs:
                try:
                    with torch.no_grad():
                        hyp = self._asr_model.transcribe(
                            [single_input], override_config=self._override_cfg
                        )
                    if isinstance(hyp, tuple) and len(hyp) == 2:  # noqa: PLR2004
                        hyp = hyp[0]
                    out.append(hyp[0] if hyp else None)
                except Exception as exc2:  # noqa: BLE001, PERF203
                    logger.error("NeMoASRAlignAdapter: per-item transcribe failed for {}: {}", single_input, exc2)
                    out.append(None)
            return out

    def _get_alignments_text(self, hypothesis: Any) -> tuple[list[dict[str, Any]], str]:
        """Extract word alignments + text from a single NeMo Hypothesis."""
        if not self.compute_timestamps:
            return [], hypothesis.text

        timestamp_dict = hypothesis.timestamp

        if self.is_fastconformer:
            time_stride = 8 * self._asr_model.cfg.preprocessor.window_stride
        else:
            time_stride = 4 * self._asr_model.cfg.preprocessor.window_stride

        word_timestamps = timestamp_dict[self.timestamp_type]

        alignments: list[dict[str, Any]] = []
        for i, stamp in enumerate(word_timestamps):
            conf: float | None = None
            if hypothesis.word_confidence is not None and i < len(hypothesis.word_confidence):
                raw = hypothesis.word_confidence[i]
                if isinstance(raw, torch.Tensor):
                    raw = raw.item()
                conf = round(float(raw), 4)

            if self.decoder_type == "ctc":
                start = stamp["start_offset"] * time_stride
                end = stamp["end_offset"] * time_stride
            else:
                start = max(0, stamp["start_offset"] * time_stride - 0.08)
                end = max(0, stamp["end_offset"] * time_stride - 0.08)

            word = stamp.get("word", stamp.get("char", ""))
            alignments.append(
                {
                    "word": word,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "confidence": conf,
                }
            )

        text = " ".join(w["word"] for w in alignments).replace("⁇", "")
        return alignments, text
