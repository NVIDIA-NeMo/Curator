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

from pathlib import Path

import pytest
import yaml

from nemo_curator.stages.audio.text_filtering.regex_substitution import RegexSubstitutionStage
from nemo_curator.tasks import AudioTask


def _write_rules(tmp_path: Path, rules: list[dict]) -> str:
    p = tmp_path / "rules.yaml"
    p.write_text(yaml.dump(rules), encoding="utf-8")
    return str(p)


_TEXT_KEY = "cleaned_text"
_SKIP_KEY = "_skip_me"


def _make_stage(tmp_path: Path, rules: list[dict]) -> RegexSubstitutionStage:
    rules_path = _write_rules(tmp_path, rules)
    stage = RegexSubstitutionStage(
        regex_params_yaml=rules_path, text_key=_TEXT_KEY, skip_me_key=_SKIP_KEY,
    )
    stage.setup()
    return stage


def test_applies_substitution(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [{"pattern": "\u2019", "repl": "'"}])
    task = AudioTask(data={_TEXT_KEY: "it\u2019s fine", _SKIP_KEY: ""})
    result = stage.process(task)
    assert "'" in result.data[stage.output_text_key]
    assert result.data[_SKIP_KEY] == ""


def test_empty_text_after_rules_sets_skip_me(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [{"pattern": r"\w+", "repl": ""}])
    task = AudioTask(data={_TEXT_KEY: "hello", _SKIP_KEY: ""})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == "Empty after regex cleaning"


def test_whitespace_only_sets_skip_me(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [{"pattern": r"\S+", "repl": ""}])
    task = AudioTask(data={_TEXT_KEY: "hello world", _SKIP_KEY: ""})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == "Empty after regex cleaning"


def test_non_empty_text_preserves_skip_me_empty(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [{"pattern": r"bad", "repl": "good"}])
    task = AudioTask(data={_TEXT_KEY: "bad word", _SKIP_KEY: ""})
    result = stage.process(task)
    assert result.data[stage.output_text_key] == "good word"
    assert result.data[_SKIP_KEY] == ""


def test_strips_extra_whitespace(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [])
    task = AudioTask(data={_TEXT_KEY: "hello   world", _SKIP_KEY: ""})
    result = stage.process(task)
    assert result.data[stage.output_text_key] == "hello world"


def test_multiple_rules_applied_in_order(tmp_path: Path) -> None:
    stage = _make_stage(
        tmp_path,
        [
            {"pattern": "\u2014", "repl": "-"},
            {"pattern": r"\s+", "repl": " "},
        ],
    )
    task = AudioTask(data={_TEXT_KEY: "word\u2014word", _SKIP_KEY: ""})
    result = stage.process(task)
    assert result.data[stage.output_text_key] == "word-word"


def test_setup_called_lazily_when_skipped(tmp_path: Path) -> None:
    rules_path = _write_rules(tmp_path, [{"pattern": "\u2019", "repl": "'"}])
    stage = RegexSubstitutionStage(
        regex_params_yaml=rules_path, text_key=_TEXT_KEY, skip_me_key=_SKIP_KEY,
    )
    task = AudioTask(data={_TEXT_KEY: "it\u2019s fine", _SKIP_KEY: ""})
    result = stage.process(task)
    assert result.data[stage.output_text_key] == "it's fine"


def test_non_string_text_returns_task_unchanged(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [{"pattern": r"\w+", "repl": ""}])
    task = AudioTask(data={_TEXT_KEY: None, _SKIP_KEY: ""})
    result = stage.process(task)
    assert result.data[_TEXT_KEY] is None
    assert result.data[_SKIP_KEY] == ""


def test_requires_regex_params_yaml() -> None:
    with pytest.raises(ValueError, match="regex_params_yaml is required"):
        RegexSubstitutionStage(regex_params_yaml="")


def test_preserves_existing_skip_me_on_empty_result(tmp_path: Path) -> None:
    stage = _make_stage(tmp_path, [{"pattern": r"\w+", "repl": ""}])
    task = AudioTask(data={_TEXT_KEY: "hello", _SKIP_KEY: "Hallucination"})
    result = stage.process(task)
    assert result.data[_SKIP_KEY] == "Hallucination"
