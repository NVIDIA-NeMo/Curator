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

"""Punctuation and Capitalization (PNC) with BERT stage.

.. note::
   ``PunctuationCapitalizationModel`` was removed in NeMo Toolkit >= 2.5.
   This stage requires ``nemo_toolkit <= 2.4.1``.  When the model cannot be
   imported the stage logs a warning at setup and becomes a pass-through
   (every task is returned unchanged).
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class PNCwithBERTStage(ProcessingStage[AudioTask, AudioTask]):
    """Punctuation and capitalisation using a BERT-based NeMo model.

    Supports two operating modes controlled by ``is_audio_entry``:

    * **audio entry** (``is_audio_entry=True``, default) -- iterates over the
      ``segments`` list inside each manifest entry (or falls back to the
      top-level ``text_key``) and applies PNC to each text.
    * **segmented entry** (``is_audio_entry=False``) -- treats each manifest
      row as an individual segment, reconstructs text from the ``alignment``
      word list, runs PNC, and writes the punctuated words back into
      ``alignment``.

    .. note::
       Requires ``nemo_toolkit < 2.4.1``.  When the installed version is
       2.4.1 or later the stage silently passes tasks through unchanged.

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

    # Stage metadata
    name: str = "PNCwithBERT"
    resources: Resources = field(default_factory=lambda: Resources(gpus=1))
    # Internal state
    _pnc_model: Any = field(default=None, repr=False)
    _skip: bool = field(default=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    @property
    def _device(self) -> str:
        """Derive device from resources configuration."""
        return "cuda" if self.resources.requires_gpu and torch.cuda.is_available() else "cpu"

    def load_model(self) -> None:
        try:
            from nemo.collections.nlp.models import PunctuationCapitalizationModel
        except (ImportError, ModuleNotFoundError):
            logger.warning(
                f"[{self.name}] Could not import PunctuationCapitalizationModel. "
                f"This model is only available in nemo_toolkit <= 2.4.1. "
                f"Skipping PNC — tasks will pass through unchanged."
            )
            self._skip = True
            return

        if self.model_path:
            self._pnc_model = PunctuationCapitalizationModel.restore_from(self.model_path)
        else:
            self._pnc_model = PunctuationCapitalizationModel.from_pretrained(self.model_name)

    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Setup stage on node."""
        if self._pnc_model is None and not self._skip:
            self.load_model()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup stage."""
        if self._pnc_model is None and not self._skip:
            self.load_model()

        if self._skip:
            return

        self._pnc_model.to(self._device)

        logger.info(f"[{self.name}] Initialized PNC model on {self._device}")

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

    def process(self, task: AudioTask) -> AudioTask:
        if self._skip:
            return task

        data_entry = task.data
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

        return task
