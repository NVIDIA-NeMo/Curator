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

"""Punctuation and Capitalization (PNC) with BERT stage."""

from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger
from nemo.collections.nlp.models import PunctuationCapitalizationModel

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio.common import LegacySpeechStage
from nemo_curator.tasks import AudioBatch


@dataclass
class PNCwithBERTStage(LegacySpeechStage):
    """Punctuation and capitalisation using a BERT-based NeMo model.

    Supports two operating modes controlled by ``is_audio_entry``:

    * **audio entry** (``is_audio_entry=True``, default) -- iterates over the
      ``segments`` list inside each manifest entry (or falls back to the
      top-level ``text_key``) and applies PNC to each text.
    * **segmented entry** (``is_audio_entry=False``) -- treats each manifest
      row as an individual segment, reconstructs text from the ``alignment``
      word list, runs PNC, and writes the punctuated words back into
      ``alignment``.

    Args:
        model_name:       Pretrained PNC model name.
        model_path:       Path to a local ``.nemo`` checkpoint (overrides *model_name*).
        batch_size:       Batch size passed to the PNC model.
        text_key:         Manifest key holding the text.
        use_bert_pnc_key: Gate processing on the ``use_bert_pnc`` field.
        is_audio_entry:   ``True`` → audio-entry mode; ``False`` → segmented-entry mode.
    """

    model_name: str = "punctuation_en_bert"
    model_path: str = ""
    batch_size: int = 64
    text_key: str = "text"
    use_bert_pnc_key: bool = False
    segments_key: str = "segments"
    update_alignment: bool = False
    device: str = "cuda"

    # Stage metadata
    name: str = "PNCwithBERT"
    # Internal state
    _pnc_model: Any = field(default=None, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def load_model(self) -> None:
        if self.model_path:
            self._pnc_model = PunctuationCapitalizationModel.restore_from(self.model_path)
        else:
            self._pnc_model = PunctuationCapitalizationModel.from_pretrained(self.model_name)

    def setup_on_node(self, node_info: NodeInfo, worker_metadata: WorkerMetadata) -> None:  # noqa: ARG002
        """Setup stage on node."""
        if self._pnc_model is None:
            self.load_model()

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002, ANN401
        """Setup stage."""
        if self._pnc_model is None:
            self.load_model()

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU for PNC")
            self.device = "cpu"
        self._pnc_model.to(self.device)

        logger.info(f"[{self.name}] Initialized PNC model")

    def _should_skip(self, entry: dict) -> bool:
        """Return True when use_bert_pnc_key is set and the entry opts out."""
        return self.use_bert_pnc_key and not entry.get("use_bert_pnc", False)

    def update_segment_alignment(self, segment: dict[str, Any], pnc_text: str) -> None:
        if self.update_alignment:
            alignment = segment.get("alignment", [])
            pnc_words = pnc_text.split()
            pnc_idx = 0
            for word in alignment:
                if word.get("word", "") != "":
                    if pnc_idx >= len(pnc_words):
                        logger.warning(f"[{self.name}] PNC word count mismatch; stopping alignment update.")
                        break
                    word["word"] = pnc_words[pnc_idx]
                    pnc_idx += 1

    def process_dataset_entry(self, data_entry: dict[str, Any]) -> list[AudioBatch]:
        if self.segments_key in data_entry:
            all_text: list[str] = []
            text_indices: list[int] = []
            for i, segment in enumerate(data_entry[self.segments_key]):
                if self.text_key not in segment or not segment[self.text_key]:
                    continue
                if self._should_skip(segment):
                    continue
                all_text.append(segment[self.text_key])
                text_indices.append(i)

            if all_text:
                text_pnc = self._pnc_model.add_punctuation_capitalization(all_text, batch_size=self.batch_size)
                for idx, pnc_text in zip(text_indices, text_pnc, strict=False):
                    data_entry[self.segments_key][idx][self.text_key] = pnc_text
                    self.update_segment_alignment(data_entry[self.segments_key][idx], pnc_text)

        elif data_entry.get(self.text_key, "") != "" and not self._should_skip(data_entry):
            text_pnc = self._pnc_model.add_punctuation_capitalization(
                [data_entry[self.text_key]], batch_size=self.batch_size
            )
            data_entry[self.text_key] = text_pnc[0]
            self.update_segment_alignment(data_entry, text_pnc[0])

        return [AudioBatch(data=[data_entry])]
