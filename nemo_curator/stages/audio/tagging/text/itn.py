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

"""Inverse Text Normalization stage."""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from nemo_text_processing.inverse_text_normalization.inverse_normalize import (
    InverseNormalizer,
)

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class InverseTextNormalizationStage(ProcessingStage[AudioTask, AudioTask]):
    """
    Stage that performs inverse text normalization on text data.

    Converts spoken text representations into written form
    (e.g., "the answer is forty two" -> "the answer is 42").

    Args:
        language: Language code for text normalization
        text_key: Key to use for the text
    """

    # Language
    language: str = "en"

    # Text key
    text_key: str = "text"

    # Stage metadata
    name: str = "InverseTextNormalization"

    # Internal state (lazy-loaded in setup() to avoid high RAM at pipeline build / worker spawn)
    _normalizer: Any = field(default=None, repr=False)
    _normalizer_initialized: bool = field(default=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["segments"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["segments"]

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002, ANN401
        """Lazy-load the inverse normalizer once per worker. Reduces RAM when many workers exist."""
        if self._normalizer_initialized:
            return
        self._normalizer = InverseNormalizer(lang=self.language)
        self._normalizer_initialized = True
        logger.info(f"[{self.name}] Initialized for language: {self.language}")

    def process(self, task: AudioTask) -> AudioTask:
        """Process entry for inverse text normalization."""
        if not self._normalizer_initialized:
            self.setup()
        data_entry = task.data
        segments = data_entry.get("segments", [])
        for segment in segments:
            if self.text_key in segment:
                text = segment[self.text_key]
                if text:
                    sentences = self._normalizer.split_text_into_sentences(text)
                    text_itn = " ".join(self._normalizer.normalize_list(sentences))
                    segment[f"{self.text_key}_ITN"] = text_itn

        return task
