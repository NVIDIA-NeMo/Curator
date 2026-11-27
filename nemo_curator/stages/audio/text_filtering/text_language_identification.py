# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Batched CPU language identification of transcript text with a pluggable adapter.

As in ``AudioTextLLMStage``'s language-ID mode, each checked row is verified, setting ``_skipme`` and the
``LLMLanguageVerification`` note.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import hydra
from loguru import logger

from nemo_curator.stages.audio.text_filtering.abbreviation_concat import _set_note
from nemo_curator.stages.audio.text_filtering.language_utils import verify_language
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.models.audio.text_lid import FastTextFormatLIDAdapter


def _normalize_language_code(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"source language must be a non-empty string, got {value!r}"
        raise ValueError(msg)
    return value.strip().lower()


@dataclass
class TextLanguageIDStage(ProcessingStage[AudioTask, AudioTask]):
    """Identify each transcript's language with a CPU model and verify it against ``source_lang_key``.

    ``adapter_target`` names the model's adapter class and ``adapter_kwargs`` its constructor arguments. ``languages``
    limits the stage to rows whose ``source_lang_key`` is listed; other rows pass through untouched, and an empty
    list loads no model. ``None`` checks every row.
    """

    adapter_target: str = ""
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)
    languages: list[str] | None = None
    text_key: str = "abbreviated_text"
    source_lang_key: str = "source_lang"
    output_text_key: str = "llm_language_prediction"
    skip_me_key: str = "_skipme"
    notes_key: str = "additional_notes"
    batch_size: int = 1024
    name: str = "LanguageID"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _adapter: FastTextFormatLIDAdapter | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.adapter_target:
            msg = "adapter_target is required for TextLanguageIDStage"
            raise ValueError(msg)
        if self.batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        if not self.output_text_key:
            msg = "output_text_key must be non-empty"
            raise ValueError(msg)
        if self.languages is not None:
            if isinstance(self.languages, str):
                msg = "languages must be a list of language codes"
                raise TypeError(msg)
            self.languages = sorted({_normalize_language_code(language) for language in self.languages})

    def setup(self, _worker_metadata: object | None = None) -> None:
        if self.languages == []:
            return
        self._adapter = hydra.utils.get_class(self.adapter_target)(**self.adapter_kwargs)
        self._adapter.setup()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.source_lang_key, self.skip_me_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_text_key, self.skip_me_key, self.notes_key]

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def _collect(self, tasks: list[AudioTask]) -> tuple[list[AudioTask], list[AudioTask], list[str]]:
        routed: list[AudioTask] = []
        eligible: list[AudioTask] = []
        texts: list[str] = []
        for task in tasks:
            language = _normalize_language_code(task.data.get(self.source_lang_key))
            if self.languages is not None and language not in self.languages:
                continue
            routed.append(task)
            if task.data.get(self.skip_me_key, ""):
                task.data[self.output_text_key] = ""
                _set_note(task.data, self.name, "skipped (flagged)", self.notes_key)
                continue
            text = task.data.get(self.text_key, "")
            if not isinstance(text, str) or not text.strip():
                task.data[self.output_text_key] = text
                _set_note(task.data, self.name, "skipped (empty)", self.notes_key)
                continue
            self._adapter.check_source_language(language)
            if "\x00" in text:
                msg = f"{type(self).__name__} does not accept NUL characters in {self.text_key!r}"
                raise ValueError(msg)
            eligible.append(task)
            texts.append(text)
        return routed, eligible, texts

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        # Ray Data may pass a NumPy array; a list keeps the emptiness check and the return type valid.
        task_list = list(tasks)
        if not task_list or self.languages == []:
            return task_list
        if self._adapter is None:
            logger.warning("{}: setup() was not called before inference; calling setup() now", self.name)
            self.setup()
        routed, eligible, texts = self._collect(task_list)
        if texts:
            for task, language in zip(eligible, self._adapter.predict(texts), strict=True):
                task.data[self.output_text_key] = language
                note = "applied (modified)" if language != task.data[self.text_key] else "applied (unchanged)"
                _set_note(task.data, self.name, note, self.notes_key)
        for task in routed:
            verify_language(task.data, self.output_text_key, self.source_lang_key, self.skip_me_key, self.notes_key)
        return task_list
