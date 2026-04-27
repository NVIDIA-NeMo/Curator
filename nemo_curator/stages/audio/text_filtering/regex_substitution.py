# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import re
from dataclasses import dataclass, field
from typing import Any

import yaml
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class RegexSubstitutionStage(ProcessingStage[AudioTask, AudioTask]):
    """Apply a sequence of regex substitutions to a text field in each AudioTask.

    Rules are loaded from a YAML file containing a list of dicts with
    ``pattern`` and ``repl`` keys (and an optional ``count`` key).

    Reads from ``text_key`` (default ``"pred_text"``), applies all rules,
    and writes the result to ``output_text_key`` (default ``"cleaned_text"``).
    The original ``text_key`` field is preserved unchanged.

    After all substitutions, if the result is empty and ``skip_me`` is
    still empty, ``skip_me`` is set to ``"Empty after regex cleaning"``.
    """

    regex_params_yaml: str = ""
    text_key: str = "pred_text"
    output_text_key: str = "cleaned_text"
    skip_me_key: str = "_skip_me"
    name: str = "RegexSubstitution"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _rules: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _setup_called: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.regex_params_yaml:
            msg = "regex_params_yaml is required for RegexSubstitutionStage"
            raise ValueError(msg)

    def setup(self, _worker_metadata: object | None = None) -> None:
        with open(self.regex_params_yaml, encoding="utf-8") as f:
            self._rules = yaml.safe_load(f)
        self._setup_called = True
        logger.info(f"RegexSubstitutionStage: loaded {len(self._rules)} rules from {self.regex_params_yaml}")

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_text_key, self.skip_me_key]

    def _process_single(self, task: AudioTask) -> AudioTask:
        if task.data.get(self.skip_me_key, ""):
            task.data.setdefault(self.output_text_key, task.data.get(self.text_key, ""))
            return task
        text = task.data[self.text_key]
        if not isinstance(text, str):
            return task
        text = " " + text + " "
        for rule in self._rules:
            text = re.sub(rule["pattern"], rule["repl"], text, count=rule.get("count", 0))
        text = re.sub(r"\s+", " ", text).strip()
        task.data[self.output_text_key] = text
        if not text and not task.data[self.skip_me_key]:
            task.data[self.skip_me_key] = "Empty after regex cleaning"
        return task

    def process(self, task: AudioTask) -> AudioTask:
        if not self._setup_called:
            logger.warning(
                f"RegexSubstitutionStage ({self.name}): setup() was not called before process(). "
                "Calling setup() now — check that your executor invokes setup() on each worker."
            )
            self.setup()
        return self._process_single(task)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if not self._setup_called:
            logger.warning(
                f"RegexSubstitutionStage ({self.name}): setup() was not called before process_batch(). "
                "Calling setup() now — check that your executor invokes setup() on each worker."
            )
            self.setup()
        return [self._process_single(task) for task in tasks]
