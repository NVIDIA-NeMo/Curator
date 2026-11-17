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
    """Apply an ordered list of regex substitutions to one transcript field.

    Provide exactly one of ``regex_params_yaml`` or inline ``rules``. Each must
    contain a list of rules. Each rule must define string
    ``pattern`` and ``repl`` values and may define a non-negative integer
    ``count`` limit. Rules run in list order before whitespace is collapsed.
    A rule may also define ``langs`` (apply only to those languages) or
    ``exclude_langs`` (apply to every language except those), each a list of
    language codes matched against ``language_key``; a region suffix such as
    ``el-GR`` matches ``el``. A row without a language gets every rule that has
    no ``langs`` list.
    Rows carrying an existing skip reason are not normalized. The empty-result
    skip is set only when cleaning removes non-whitespace input; already-empty
    input remains unskipped.
    """

    regex_params_yaml: str = ""
    text_key: str = "pred_text"
    output_text_key: str = "text"
    skip_me_key: str = "_skipme"
    language_key: str = "source_lang"
    name: str = "RegexSubstitution"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    rules: list[dict[str, Any]] | None = None

    _rules: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if bool(self.regex_params_yaml) == (self.rules is not None):
            msg = "Provide exactly one of regex_params_yaml or rules for RegexSubstitutionStage"
            raise ValueError(msg)

    def setup(self, _worker_metadata: object | None = None) -> None:
        raw_rules = self.rules
        if raw_rules is None:
            with Path(self.regex_params_yaml).open(encoding="utf-8") as stream:
                raw_rules = yaml.safe_load(stream)
        if not isinstance(raw_rules, list):
            msg = "Regex substitutions must contain a list of rules"
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
            self._validate_language_filter(index, rule)
        self._rules = raw_rules

    @staticmethod
    def _validate_language_filter(index: int, rule: dict[str, Any]) -> None:
        if "langs" in rule and "exclude_langs" in rule:
            msg = f"Regex rule {index} must not define both langs and exclude_langs"
            raise ValueError(msg)
        for key in ("langs", "exclude_langs"):
            if key not in rule:
                continue
            langs = rule[key]
            if not isinstance(langs, list) or not langs or not all(isinstance(x, str) and x for x in langs):
                msg = f"Regex rule {index} {key} must be a non-empty list of language codes"
                raise TypeError(msg)

    @staticmethod
    def _base_language(value: object) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        return re.split(r"[-_]", value.strip(), maxsplit=1)[0].lower()

    @staticmethod
    def _applies(rule: dict[str, Any], language: str | None) -> bool:
        if "langs" in rule:
            return language is not None and language in {x.lower() for x in rule["langs"]}
        if "exclude_langs" in rule:
            return language is None or language not in {x.lower() for x in rule["exclude_langs"]}
        return True

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

        language = self._base_language(task.data.get(self.language_key))
        result = f" {text} "
        for rule in self._rules:
            if not self._applies(rule, language):
                continue
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
