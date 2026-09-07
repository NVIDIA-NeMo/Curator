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

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class RegexSubstitutionStage(ProcessingStage[AudioTask, AudioTask]):
    """Apply an ordered YAML list of regex substitutions to one transcript field.

    The YAML document must be a list. Each rule must define string
    ``pattern`` and ``repl`` values and may define a non-negative integer
    ``count`` limit. Rules run in file order before whitespace is collapsed.
    Rows carrying an existing skip reason are not normalized. The empty-result
    skip is set only when cleaning removes non-whitespace input; already-empty
    input remains unskipped.
    """

    regex_params_yaml: str = ""
    text_key: str = "pred_text"
    output_text_key: str = "text"
    skip_me_key: str = "_skipme"
    name: str = "RegexSubstitution"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _rules: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.regex_params_yaml:
            msg = "regex_params_yaml is required for RegexSubstitutionStage"
            raise ValueError(msg)

    def setup(self, _worker_metadata: object | None = None) -> None:
        with Path(self.regex_params_yaml).open(encoding="utf-8") as stream:
            raw_rules = yaml.safe_load(stream)
        if not isinstance(raw_rules, list):
            msg = "Regex substitution YAML must contain a list of rules"
            raise TypeError(msg)
        for index, rule in enumerate(raw_rules):
            if not isinstance(rule, dict):
                msg = f"Regex rule {index} must be a mapping"
                raise TypeError(msg)
            if "pattern" not in rule or "repl" not in rule:
                msg = f"Regex rule {index} must define pattern and repl"
                raise ValueError(msg)
            pattern = rule["pattern"]
            repl = rule["repl"]
            count = rule.get("count", 0)
            if not isinstance(pattern, str):
                msg = f"Regex rule {index} pattern must be a string"
                raise TypeError(msg)
            if not isinstance(repl, str):
                msg = f"Regex rule {index} repl must be a string"
                raise TypeError(msg)
            if isinstance(count, bool) or not isinstance(count, int):
                msg = f"Regex rule {index} count must be an integer"
                raise TypeError(msg)
            if count < 0:
                msg = f"Regex rule {index} count must be non-negative"
                raise ValueError(msg)
            compiled = re.compile(pattern)
            compiled.sub(repl, "", count=count)
        self._rules = raw_rules

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_text_key, self.skip_me_key]

    def process(self, task: AudioTask) -> AudioTask:
        if task.data.get(self.skip_me_key, ""):
            task.data.setdefault(self.output_text_key, "")
            return task

        text = task.data.get(self.text_key, "")
        if not isinstance(text, str):
            task.data.setdefault(self.output_text_key, "")
            return task
        had_text = bool(text.strip())

        result = f" {text} "
        for rule in self._rules:
            result = re.sub(
                rule["pattern"],
                rule["repl"],
                result,
                count=rule.get("count", 0),
            )
        result = re.sub(r"\s+", " ", result).strip()
        task.data[self.output_text_key] = result
        if not result and had_text and not task.data.get(self.skip_me_key, ""):
            task.data[self.skip_me_key] = "Empty after regex cleaning"
        return task
