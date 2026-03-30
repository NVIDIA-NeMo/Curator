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

"""Chinese text conversion stage (Traditional -> Simplified, etc.)."""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from opencc import OpenCC

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class ChineseConversionStage(ProcessingStage[AudioTask, AudioTask]):
    """Convert Traditional Chinese text to Simplified Chinese (or other OpenCC conversions).

    Iterates over the ``segments`` list of each entry and writes the converted
    text to ``{text_key}_simplified``.  If conversion fails for a segment the
    original text is kept as a fallback.

    Args:
        text_key:     Manifest key holding the text to convert.
        convert_type: OpenCC conversion type (e.g. ``"t2s"``, ``"s2t"``).
    """

    text_key: str = "text"
    convert_type: str = "t2s"

    # Stage metadata
    name: str = "ChineseConversion"

    # Internal state
    _converter: Any = field(default=None, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["segments"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["segments"]

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002, ANN401
        """Setup stage."""
        self._converter = OpenCC(self.convert_type)
        logger.info(f"[{self.name}] Using conversion type: {self.convert_type}")

    def process(self, task: AudioTask) -> AudioTask:
        data_entry = task.data
        output_key = f"{self.text_key}_simplified"
        for segment in data_entry.get("segments", []):
            if self.text_key in segment:
                try:
                    segment[output_key] = self._converter.convert(segment[self.text_key])
                except Exception:  # noqa: BLE001
                    logger.warning(f"[{self.name}] Chinese conversion failed, keeping original")
                    segment[output_key] = segment[self.text_key]

        return task
