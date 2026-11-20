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

"""Shared plumbing for the audio text-transform stages.

Two shapes exist, and each base owns only the loop and guards its stages had copied:

* :class:`RowTextTransform` -- one transcript field per row (``RegexSubstitutionStage``,
  ``AbbreviationConcatStage``). Rows already marked skipped, or whose text is not a string,
  get ``output_text_key`` defaulted to ``""`` and are otherwise left alone.
* :class:`SegmentTextTransform` -- ``text_key`` of every item in ``segments_key``, written to
  ``{text_key}{output_suffix}`` (``InverseTextNormalizationStage``, ``ChineseConversionStage``).

They are mixins with no dataclass fields, so each stage keeps its own field order.
"""

from typing import Any

from nemo_curator.tasks import AudioTask


class RowTextTransform:
    """Mixin: apply :meth:`transform_row` to ``task.data[text_key]``.

    The stage must define ``text_key``, ``output_text_key`` and ``skip_me_key``.
    """

    text_key: str
    output_text_key: str
    skip_me_key: str

    def transform_row(self, task: AudioTask, text: str) -> AudioTask:
        raise NotImplementedError

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key]

    def process(self, task: AudioTask) -> AudioTask:
        text = task.data.get(self.text_key, "")
        if task.data.get(self.skip_me_key, "") or not isinstance(text, str):
            task.data.setdefault(self.output_text_key, "")
            return task
        return self.transform_row(task, text)


class SegmentTextTransform:
    """Mixin: write ``transform_text(segment[text_key])`` to ``segment[text_key + output_suffix]``.

    The stage must define ``text_key``, ``segments_key`` and ``output_suffix``. Segments
    without ``text_key``, or whose text :meth:`accepts_text` rejects, are left unchanged.
    """

    text_key: str
    segments_key: str
    output_suffix: str

    def transform_text(self, text: Any) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def accepts_text(self, text: Any) -> bool:  # noqa: ANN401, ARG002
        return True

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.segments_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.segments_key]

    def process(self, task: AudioTask) -> AudioTask:
        output_key = f"{self.text_key}{self.output_suffix}"
        for segment in task.data.get(self.segments_key, []):
            if self.text_key in segment and self.accepts_text(segment[self.text_key]):
                segment[output_key] = self.transform_text(segment[self.text_key])
        return task
