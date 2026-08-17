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
from pathlib import Path

import pytest
import yaml

from nemo_curator.stages.audio.text_filtering.abbreviation_concat import AbbreviationConcatStage
from nemo_curator.stages.audio.text_filtering.regex_substitution import RegexSubstitutionStage
from nemo_curator.tasks import AudioTask


def _rules(tmp_path: Path, rules: object) -> str:
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(rules, sort_keys=False), encoding="utf-8")
    return str(path)


def test_applies_ordered_rules_and_normalizes_whitespace(tmp_path: Path) -> None:
    stage = RegexSubstitutionStage(
        regex_params_yaml=_rules(
            tmp_path,
            [
                {"pattern": r"\bum\b", "repl": ""},
                {"pattern": r"\bN V I D I A\b", "repl": "NVIDIA"},
            ],
        )
    )
    stage.setup()
    task = AudioTask(data={"pred_text": "um  N V I D I A builds GPUs"})

    stage.process(task)

    assert task.data["text"] == "NVIDIA builds GPUs"


def test_preserves_skipped_rows(tmp_path: Path) -> None:
    stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, []))
    stage.setup()
    task = AudioTask(data={"pred_text": "ignored", "text": "preserved", "_skipme": "read_error"})

    stage.process(task)

    assert task.data["text"] == "preserved"
    assert task.data["_skipme"] == "read_error"


def test_marks_rows_that_clean_to_empty(tmp_path: Path) -> None:
    stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, [{"pattern": r"\S+", "repl": ""}]))
    stage.setup()
    task = AudioTask(data={"pred_text": "remove me"})

    stage.process(task)

    assert task.data["text"] == ""
    assert task.data["_skipme"] == "Empty after regex cleaning"


def test_rejects_invalid_rule_shape(tmp_path: Path) -> None:
    stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, [{"pattern": "foo"}]))

    with pytest.raises(ValueError, match="pattern and repl"):
        stage.setup()


def test_honors_rule_count_and_custom_fields(tmp_path: Path) -> None:
    stage = RegexSubstitutionStage(
        regex_params_yaml=_rules(tmp_path, [{"pattern": "bad", "repl": "good", "count": 1}]),
        text_key="transcript",
        output_text_key="normalized",
    )
    stage.setup()
    task = AudioTask(data={"transcript": "bad bad"})

    stage.process(task)

    assert task.data["normalized"] == "good bad"
    assert task.data["transcript"] == "bad bad"


def test_inherited_batch_processing_and_default_stage_chain(tmp_path: Path) -> None:
    regex_stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, [{"pattern": r"\bum\b", "repl": ""}]))
    regex_stage.setup()
    abbreviation_stage = AbbreviationConcatStage()
    tasks = [
        AudioTask(data={"pred_text": "um A P I on G P U", "source_lang": "en"}),
        AudioTask(data={"pred_text": "N V I D I A", "source_lang": "en"}),
    ]

    normalized = regex_stage.process_batch(tasks)
    results = abbreviation_stage.process_batch(normalized)

    assert [task.data["text"] for task in results] == ["API on GPU", "NVIDIA"]


def test_rejects_non_list_yaml(tmp_path: Path) -> None:
    stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, {"pattern": "foo", "repl": "bar"}))

    with pytest.raises(TypeError, match="must contain a list"):
        stage.setup()


def test_rejects_invalid_regex(tmp_path: Path) -> None:
    stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, [{"pattern": "[", "repl": ""}]))

    with pytest.raises(re.error):
        stage.setup()


def test_requires_rules_path() -> None:
    with pytest.raises(ValueError, match="regex_params_yaml is required"):
        RegexSubstitutionStage()
