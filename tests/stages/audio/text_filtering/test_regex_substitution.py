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

# ruff: noqa: INP001

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
                {"pattern": r"\bfoo\b", "repl": "A P I"},
                {"pattern": r"\bA P I\b", "repl": "G P U"},
                {"pattern": r"\bG P U\b", "repl": "GPU"},
            ],
        )
    )
    stage.setup()
    task = AudioTask(data={"pred_text": "  foo   acceleration  "})

    stage.process(task)

    assert task.data["text"] == "GPU acceleration"


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


@pytest.mark.parametrize("pred_text", ["", "   "])
def test_does_not_mark_rows_already_empty(tmp_path: Path, pred_text: str) -> None:
    stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, [{"pattern": r"\w+", "repl": ""}]))
    stage.setup()
    task = AudioTask(data={"pred_text": pred_text, "_skipme": ""})

    stage.process(task)

    assert task.data["text"] == ""
    assert task.data["_skipme"] == ""


@pytest.mark.parametrize("rule", [{"pattern": "foo"}, {"repl": "bar"}])
def test_rejects_invalid_rule_shape(tmp_path: Path, rule: dict[str, str]) -> None:
    stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, [rule]))

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


def test_stage_chain_preserves_contraction_pronoun(tmp_path: Path) -> None:
    regex_stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, [{"pattern": "\u2019", "repl": "'"}]))
    regex_stage.setup()
    task = AudioTask(data={"pred_text": "A B I’m ready"})  # noqa: RUF001

    regex_stage.process(task)
    AbbreviationConcatStage().process(task)

    assert task.data["text"] == "AB I'm ready"


# Golden outputs verified against nithinraok/Curator@a1df5ec9 with
# branch-neutral field names and the current-main stage lifecycle.
@pytest.mark.parametrize(
    ("rules", "raw", "language", "expected"),
    [
        ([{"pattern": "’", "repl": "'"}], "we’re  A P I", "en", ("we're A P I", "we're API")),  # noqa: RUF001
        (
            [
                {"pattern": r"\bfoo\b", "repl": "A P I"},
                {"pattern": r"\bA P I\b", "repl": "G P U"},
            ],
            "foo",
            "en",
            ("G P U", "GPU"),
        ),
        ([], "  А Б В  ", "ru", ("А Б В", "АБВ")),  # noqa: RUF001
    ],
)
def test_reference_compatible_stage_chain(
    tmp_path: Path,
    rules: list[dict[str, object]],
    raw: str,
    language: str,
    expected: tuple[str, str],
) -> None:
    regex_stage = RegexSubstitutionStage(
        regex_params_yaml=_rules(tmp_path, rules),
        text_key="raw",
        output_text_key="cleaned",
        skip_me_key="skip_reason",
    )
    abbreviation_stage = AbbreviationConcatStage(
        text_key="cleaned",
        output_text_key="normalized",
        skip_me_key="skip_reason",
        source_lang_key="language",
    )
    regex_stage.setup()
    tasks = [AudioTask(data={"raw": raw, "language": language, "skip_reason": ""})]

    cleaned = regex_stage.process_batch(tasks)
    results = abbreviation_stage.process_batch(cleaned)
    expected_cleaned, expected_normalized = expected

    assert results[0].data["raw"] == raw
    assert results[0].data["cleaned"] == expected_cleaned
    assert results[0].data["normalized"] == expected_normalized
    assert results[0].data["skip_reason"] == ""


def test_non_string_input_stabilizes_output_schema(tmp_path: Path) -> None:
    stage = RegexSubstitutionStage(
        regex_params_yaml=_rules(tmp_path, []),
        text_key="raw",
        output_text_key="cleaned",
    )
    stage.setup()
    task = AudioTask(data={"raw": None})

    stage.process(task)

    assert task.data["raw"] is None
    assert task.data["cleaned"] == ""


@pytest.mark.parametrize("document", [None, False, 0, {}, ""])
def test_rejects_falsey_non_list_yaml(tmp_path: Path, document: object) -> None:
    stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, document))

    with pytest.raises(TypeError, match="must contain a list"):
        stage.setup()


@pytest.mark.parametrize("document", [{"pattern": "foo", "repl": "bar"}, "not a list", 1])
def test_rejects_truthy_non_list_yaml(tmp_path: Path, document: object) -> None:
    stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, document))

    with pytest.raises(TypeError, match="must contain a list"):
        stage.setup()


def test_rejects_blank_yaml(tmp_path: Path) -> None:
    path = tmp_path / "blank.yaml"
    path.write_text("", encoding="utf-8")
    stage = RegexSubstitutionStage(regex_params_yaml=str(path))

    with pytest.raises(TypeError, match="must contain a list"):
        stage.setup()


def test_rejects_non_mapping_rule(tmp_path: Path) -> None:
    stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, ["not a rule"]))

    with pytest.raises(TypeError, match="must be a mapping"):
        stage.setup()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pattern", None),
        ("pattern", 1),
        ("repl", None),
        ("repl", ["replacement"]),
    ],
)
def test_rejects_non_string_pattern_or_repl(tmp_path: Path, field: str, value: object) -> None:
    rule: dict[str, object] = {"pattern": "foo", "repl": "bar"}
    rule[field] = value
    stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, [rule]))

    with pytest.raises(TypeError, match=rf"{field} must be a string"):
        stage.setup()


@pytest.mark.parametrize("count", [None, "1", 1.0, True])
def test_rejects_non_integer_count(tmp_path: Path, count: object) -> None:
    stage = RegexSubstitutionStage(
        regex_params_yaml=_rules(tmp_path, [{"pattern": "foo", "repl": "bar", "count": count}])
    )

    with pytest.raises(TypeError, match="count must be an integer"):
        stage.setup()


def test_rejects_negative_count(tmp_path: Path) -> None:
    stage = RegexSubstitutionStage(
        regex_params_yaml=_rules(tmp_path, [{"pattern": "foo", "repl": "bar", "count": -1}])
    )

    with pytest.raises(ValueError, match="count must be non-negative"):
        stage.setup()


def test_rejects_invalid_regex(tmp_path: Path) -> None:
    stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, [{"pattern": "[", "repl": ""}]))

    with pytest.raises(re.error):
        stage.setup()


def test_rejects_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "malformed.yaml"
    path.write_text("- pattern: [\n", encoding="utf-8")
    stage = RegexSubstitutionStage(regex_params_yaml=str(path))

    with pytest.raises(yaml.YAMLError):
        stage.setup()


def test_rejects_invalid_replacement(tmp_path: Path) -> None:
    stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, [{"pattern": "(foo)", "repl": r"\2"}]))

    with pytest.raises(re.error):
        stage.setup()


@pytest.mark.parametrize(
    "rules",
    [
        [],
        [
            {"pattern": r"\bfoo\b", "repl": "A P I", "count": 1},
            {"pattern": r"\bA P I\b", "repl": "GPU"},
        ],
        [{"pattern": r"\S+", "repl": ""}],
    ],
)
def test_inline_rules_match_file_results(tmp_path: Path, rules: list[dict[str, object]]) -> None:
    file_stage = RegexSubstitutionStage(regex_params_yaml=_rules(tmp_path, rules))
    inline_stage = RegexSubstitutionStage(rules=rules)
    rows = [
        {"pred_text": "  foo   foo  "},
        {"pred_text": "   ", "_skipme": ""},
        {"pred_text": "foo", "text": "preserved", "_skipme": "read_error"},
    ]
    file_stage.setup()
    inline_stage.setup()

    file_results = file_stage.process_batch([AudioTask(data=dict(row)) for row in rows])
    inline_results = inline_stage.process_batch([AudioTask(data=dict(row)) for row in rows])

    assert [task.data for task in inline_results] == [task.data for task in file_results]


@pytest.mark.parametrize(
    ("rules", "error", "message"),
    [
        ({}, TypeError, "must contain a list"),
        (["not a rule"], TypeError, "must be a mapping"),
        ([{"pattern": "foo"}], ValueError, "pattern and repl"),
        ([{"pattern": "foo", "repl": "bar", "count": True}], TypeError, "count must be an integer"),
        ([{"pattern": "[", "repl": ""}], re.error, "unterminated character set"),
    ],
)
def test_rejects_invalid_inline_rules(rules: object, error: type[Exception], message: str) -> None:
    stage = RegexSubstitutionStage(rules=rules)

    with pytest.raises(error, match=message):
        stage.setup()


@pytest.mark.parametrize(
    ("regex_params_yaml", "rules"), [("", None), ("rules.yaml", []), ("rules.yaml", [{"pattern": "x", "repl": ""}])]
)
def test_requires_exactly_one_rules_source(regex_params_yaml: str, rules: list[dict[str, str]] | None) -> None:
    with pytest.raises(ValueError, match="exactly one of regex_params_yaml or rules"):
        RegexSubstitutionStage(regex_params_yaml=regex_params_yaml, rules=rules)


_GRANARY_REGEX_YAML = (
    Path(__file__).resolve().parents[4] / "tutorials" / "audio" / "granary_v2" / "indic_asr" / "regex.yaml"
)


@pytest.mark.parametrize(
    "text",
    [
        "mình cũng biết mà",  # vi: ũ U+0169
        "nghĩ về chuyện cũ",  # vi: ĩ U+0129 and ũ
        "một bác sĩ giỏi",  # vi: ĩ
        "NOËL",  # fr: Ë U+00CB
        "AOÛT",  # fr: Û U+00DB
        "MAÏS",  # fr: Ï U+00CF
        "όλους τους ανθρώπους",  # el: final sigma ς U+03C2
        "Ίσως Ύστερα Ώρα",  # el: Ί U+038A, Ύ U+038E, Ώ U+038F
        "προϋπόθεση ταΐζω",  # el: ϋ U+03CB, ΐ U+0390
        "ΠΡΟΫΠΟΘΕΣΗ ΤΑΪΖΩ",  # el: Ϋ U+03AB, Ϊ U+03AA
        "naţiune Ţara",  # ro: legacy cedilla ţ U+0163, Ţ U+0162
        "Α΄ Παγκόσμιος Πόλεμος",  # el: numeral sign/keraia ΄ U+0384
        "тя ѝ каза",  # bg: ѝ U+045D
        "Ѝ и ѐ",  # bg: Ѝ U+040D, ѐ U+0450
        "ĳsje",  # nl: ĳ U+0133
        "人々は時々",  # ja: 々 U+3005
        "二〇二四年",  # zh/ja: 〇 U+3007
    ],
)
def test_production_whitelist_preserves_covered_language_letters(text: str) -> None:
    """Letters absent from the whitelist are silently replaced by a space.

    Ĩĩ/Ũũ and the uppercase Ë/Ï/Û were missing, so correct predictions lost them
    mid-word: "cũng" became "c ng" and "NOËL" became "NO L". The same gap hit the
    Greek final sigma ς ("τους" -> "του "), Romanian cedilla ţ, Bulgarian ѝ and the
    CJK iteration mark 々 ("人々" -> "人 "). Every one of these languages is in the
    rule file's declared coverage, so a covered letter must survive intact.
    """
    stage = RegexSubstitutionStage(
        regex_params_yaml=str(_GRANARY_REGEX_YAML),
        text_key="pred_text",
        output_text_key="cleaned_text",
    )
    stage.setup()
    task = AudioTask(data={"pred_text": text})

    stage.process(task)

    assert task.data["cleaned_text"] == text


def _whitelist_patterns(node: object) -> list[str]:
    """Collect every character-whitelist pattern (a negated class replaced by a space)."""
    if isinstance(node, dict):
        found = [p for v in node.values() for p in _whitelist_patterns(v)]
        pattern = node.get("pattern")
        if isinstance(pattern, str) and pattern.startswith("[^") and node.get("repl") == " ":
            found.append(pattern)
        return found
    if isinstance(node, list):
        return [p for v in node for p in _whitelist_patterns(v)]
    return []


def test_qwen_omni_whitelist_matches_production_rule_file() -> None:
    """qwen_omni_asr/pipeline.yaml carries an inline copy of the whitelist.

    The copy silently missed the letter fixes made to indic_asr/regex.yaml, so the
    Qwen-Omni path kept stripping them. Both copies must stay identical.
    """
    qwen_yaml = _GRANARY_REGEX_YAML.parents[1] / "qwen_omni_asr" / "pipeline.yaml"
    production = _whitelist_patterns(yaml.safe_load(_GRANARY_REGEX_YAML.read_text(encoding="utf-8")))
    inline = _whitelist_patterns(yaml.safe_load(qwen_yaml.read_text(encoding="utf-8")))

    assert len(production) == 1
    assert inline == production
