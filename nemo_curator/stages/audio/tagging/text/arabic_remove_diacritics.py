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

"""Arabic diacritics removal stage."""

from dataclasses import dataclass

from pyarabic import araby

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class ArabicRemoveDiacriticsStage(ProcessingStage[AudioTask, AudioTask]):
    """Remove diacritics from Arabic text and alignment words.

    Only modifies entries that are final segments (no pending ``split_filepaths``)
    and whose ``text_key`` field is non-empty.  Diacritics are also stripped from
    every word in the ``alignment`` list when present.

    Args:
        text_key: Manifest key that holds the text to clean.
    """

    text_key: str = "text"
    alignment_key: str = "alignment"

    # Stage metadata
    name: str = "ArabicRemoveDiacritics"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.alignment_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.alignment_key]

    def strip_diacritics(self, text: str, alignment: list[dict]) -> tuple[str, list[dict]]:
        text = araby.strip_diacritics(text)
        for word in alignment:
            if word.get("word", "") != "":
                word["word"] = araby.strip_diacritics(word["word"])
        return text, alignment

    def _strip_entry(self, entry: dict) -> None:
        """Strip diacritics from a single entry if text and alignment are present."""
        text = entry.get(self.text_key, "")
        alignment = entry.get(self.alignment_key, [])
        if text:
            entry[self.text_key], alignment = self.strip_diacritics(text, alignment)
            if alignment:
                entry[self.alignment_key] = alignment

    def process(self, task: AudioTask) -> AudioTask:
        data_entry = task.data
        if "split_metadata" in data_entry:
            for entry in data_entry["split_metadata"]:
                self._strip_entry(entry)
        else:
            self._strip_entry(data_entry)

        return task
