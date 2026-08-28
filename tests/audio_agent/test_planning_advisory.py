# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Soft curation-mode advice stays deterministic, separate, and non-blocking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemo_curator import audio_agent as aa

_INPUT = Path(__file__).resolve().parents[1] / "fixtures/audio/alm/sample_input.jsonl"
_REFINE = {
    "schema_version": 1,
    "curation_mode": "refine_later",
    "source": "explicit_user_choice",
}
_FAST = {
    "schema_version": 1,
    "curation_mode": "fast_first",
    "source": "explicit_user_choice",
}


def _recipe(
    tmp_path: Path,
    quality_stage: dict,
    *,
    preference: dict | None = _REFINE,
    selector: dict | None = None,
    source: Path = _INPUT,
) -> dict:
    stages = [
        {
            "ref": "ManifestReader",
            "params": {"manifest_path": str(source)},
        },
        quality_stage,
    ]
    if selector is not None:
        stages.append(selector)
    stages.append(
        {
            "ref": "ManifestWriterStage",
            "params": {"output_path": str(tmp_path / "out.jsonl")},
        }
    )
    result = {"stages": stages}
    if preference is not None:
        result["planning_preference"] = preference
    return result


@pytest.mark.parametrize(
    ("stage_ref", "params", "selector_ref"),
    [
        (
            "UTMOSFilterStage",
            {
                "action": "filter",
                "mode": "task",
                "input_residency": "file",
                "mos_threshold": 3.5,
                "score_key": "quality",
            },
            "PreserveByValueStage",
        ),
        (
            "SIGMOSFilterStage",
            {
                "action": "filter",
                "mode": "task",
                "input_residency": "file",
                "noise_threshold": 4.0,
                "ovrl_threshold": 3.5,
            },
            "PreserveByValueConditionsStage",
        ),
    ],
)
def test_refine_later_native_quality_filters_get_non_blocking_advice(
    tmp_path: Path,
    stage_ref: str,
    params: dict,
    selector_ref: str,
) -> None:
    authored = _recipe(
        tmp_path,
        {"ref": stage_ref, "params": params},
    )
    fast = aa.validate(
        {
            **authored,
            "planning_preference": _FAST,
        }
    )
    refined = aa.validate(authored)

    assert refined["runnable"] == fast["runnable"]
    assert refined["status"] == fast["status"]
    assert refined["planning_advisories"][0]["stage_index"] == 1
    assert refined["planning_advisories"][0]["stage"] == stage_ref
    assert "still valid" not in str(refined["issues"]).lower()
    suggestion = refined["planning_advisories"][0]["suggested_shape"]
    assert suggestion["ordering"] == ("annotate -> metadata checkpoint -> exact selector")
    assert suggestion["producer"]["params"]["mode"] == "task"
    assert suggestion["producer"]["params"]["input_residency"] == "file"
    assert suggestion["selector"]["ref"] == selector_ref
    if selector_ref == "PreserveByValueConditionsStage":
        assert suggestion["selector"]["params"]["condition_logic"] == "and"
    assert fast["planning_advisories"] == []


def test_provable_auto_scope_is_advised_as_explicit_without_becoming_an_issue(
    tmp_path: Path,
) -> None:
    source = tmp_path / "task-only.jsonl"
    source.write_text(
        json.dumps({"audio_filepath": str(tmp_path / "task-only.wav")}) + "\n",
        encoding="utf-8",
    )
    verdict = aa.validate(
        _recipe(
            tmp_path,
            {
                "ref": "UTMOSFilterStage",
                "params": {
                    "action": "filter",
                    "mode": "auto",
                    "input_residency": "file",
                    "mos_threshold": 3.5,
                },
            },
            source=source,
        )
    )

    advisory = verdict["planning_advisories"][0]
    assert "data_dependent_auto" in advisory["reasons"]
    assert advisory["suggested_shape"]["scope"] == "task"
    assert not any(
        issue.get("code") == advisory["code"]
        for pool in ("issues", "card_violations", "gate_flags")
        for issue in verdict[pool]
    )


def test_exact_task_and_segment_annotation_selector_forms_need_no_advice(
    tmp_path: Path,
) -> None:
    task = aa.validate(
        _recipe(
            tmp_path,
            {
                "ref": "UTMOSFilterStage",
                "params": {
                    "action": "annotate",
                    "mode": "task",
                    "input_residency": "file",
                    "mos_threshold": 3.5,
                    "score_key": "quality",
                },
            },
            selector={
                "ref": "PreserveByValueStage",
                "params": {
                    "input_value_key": "quality",
                    "target_value": 3.5,
                    "operator": "ge",
                    "missing_value_policy": "drop",
                },
            },
        )
    )
    segment_recipe = _recipe(
        tmp_path,
        {
            "ref": "UTMOSFilterStage",
            "params": {
                "action": "annotate",
                "mode": "segments",
                "input_residency": "file",
                "segments_key": "clips",
                "mos_threshold": 3.5,
                "score_key": "segment_quality",
            },
        },
        selector={
            "ref": "PreserveByValueConditionsStage",
            "params": {
                "conditions": [
                    {
                        "input_value_key": "segment_quality",
                        "target_value": 3.5,
                        "operator": "ge",
                    }
                ],
                "items_key": "clips",
                "missing_value_policy": "drop",
                "drop_parent_if_empty": True,
                "condition_logic": "and",
            },
        },
    )
    segment_recipe["stages"].insert(
        1,
        {
            "ref": "VADSegmentationStage",
            "params": {
                "nested": True,
                "segments_key": "clips",
                "input_residency": "file",
                "keep_segment_waveform_in_task": False,
            },
        },
    )
    segments = aa.validate(segment_recipe)

    assert task["planning_advisories"] == []
    assert segments["planning_advisories"] == []


@pytest.mark.parametrize("preference", [None, _FAST])
def test_absent_or_fast_first_preference_emits_no_preference_advice(
    tmp_path: Path,
    preference: dict | None,
) -> None:
    verdict = aa.validate(
        _recipe(
            tmp_path,
            {
                "ref": "SIGMOSFilterStage",
                "params": {
                    "action": "filter",
                    "mode": "task",
                    "input_residency": "file",
                },
            },
            preference=preference,
        )
    )

    assert verdict["planning_advisories"] == []


def test_ambiguous_auto_scope_does_not_invent_nested_equivalence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mixed.jsonl"
    first_audio = str(tmp_path / "a.wav")
    second_audio = str(tmp_path / "b.wav")
    source.write_text(
        "\n".join(
            [
                json.dumps({"audio_filepath": first_audio}),
                json.dumps(
                    {
                        "audio_filepath": second_audio,
                        "segments": [
                            {
                                "audio_filepath": second_audio,
                                "offset": 0.0,
                                "duration": 1.0,
                            }
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    verdict = aa.validate(
        _recipe(
            tmp_path,
            {
                "ref": "UTMOSFilterStage",
                "params": {
                    "action": "filter",
                    "mode": "auto",
                    "input_residency": "file",
                    "mos_threshold": 3.5,
                },
            },
            source=source,
        )
    )

    assert verdict["planning_advisories"] == []


def test_live_unpersisted_waveform_does_not_invent_a_file_equivalent_shape(
    tmp_path: Path,
) -> None:
    recipe = _recipe(
        tmp_path,
        {
            "ref": "UTMOSFilterStage",
            "params": {
                "action": "filter",
                "mode": "task",
                "input_residency": "waveform",
                "mos_threshold": 3.5,
            },
        },
    )
    recipe["stages"].insert(
        1,
        {
            "ref": "ChannelCountStage",
            "params": {
                "action": "convert",
                "target_channels": 1,
                "keep_waveform_in_task": True,
                "write_to_disk": False,
            },
        },
    )
    recipe["stages"][-1:] = [
        {"ref": "AudioToDocumentStage", "params": {}},
        {
            "ref": "DocumentBatchJsonlWriterStage",
            "params": {"output_path": str(tmp_path / "documents.jsonl")},
        },
    ]

    verdict = aa.validate(recipe)

    assert verdict["planning_advisories"] == []


def test_existing_persisted_transform_can_prove_a_file_backed_boundary(
    tmp_path: Path,
) -> None:
    recipe = _recipe(
        tmp_path,
        {
            "ref": "UTMOSFilterStage",
            "params": {
                "action": "filter",
                "mode": "task",
                "input_residency": "waveform",
                "mos_threshold": 3.5,
            },
        },
    )
    recipe["stages"].insert(
        1,
        {
            "ref": "ChannelCountStage",
            "params": {
                "action": "convert",
                "target_channels": 1,
                "keep_waveform_in_task": True,
                "write_to_disk": True,
                "update_audio_filepath": True,
                "output_dir": str(tmp_path / "converted"),
            },
        },
    )
    recipe["stages"][-1:] = [
        {"ref": "AudioToDocumentStage", "params": {}},
        {
            "ref": "DocumentBatchJsonlWriterStage",
            "params": {"output_path": str(tmp_path / "documents.jsonl")},
        },
    ]

    verdict = aa.validate(recipe)

    assert verdict["planning_advisories"][0]["reasons"] == [
        "native_filter",
        "live_waveform_residency",
    ]
    assert verdict["planning_advisories"][0]["suggested_shape"]["file_backed_boundary"] is True


# --------------------------------------------------------------- row independence
def _split_recipe(tmp_path: Path, *, output_dir: str | None, preference: dict | None = _REFINE) -> dict:
    """The who-said-what shape: split/ASR/join, then a terminal manifest a delta could merge."""
    split: dict = {
        "ref": "SplitASRAlignJoinStage",
        "params": {"segments_key": "diar_segments", "decoder_type": "rnnt"},
    }
    if output_dir is not None:
        split["params"]["output_dir"] = output_dir
    recipe = {
        "stages": [
            {"ref": "ManifestReader", "params": {"manifest_path": str(_INPUT)}},
            split,
            {"ref": "MergeAlignmentDiarizationStage", "params": {}},
            {"ref": "ManifestWriterStage", "params": {"output_path": str(tmp_path / "out.jsonl")}},
        ]
    }
    if preference is not None:
        recipe["planning_preference"] = preference
    return recipe


def _forfeited(result: dict) -> list[dict]:
    return [
        a for a in (result.get("planning_advisories") or []) if a["code"] == "refine_later_row_independence_forfeited"
    ]


def test_a_param_that_forfeits_incremental_reuse_is_surfaced_under_refine_later(
    tmp_path: Path,
) -> None:
    """One tidy-looking output directory costs every future delta; say so before it is chosen."""
    result = aa.validate(_split_recipe(tmp_path, output_dir=str(tmp_path / "chunks")))
    advisories = _forfeited(result)

    assert len(advisories) == 1
    assert advisories[0]["stage"] == "SplitASRAlignJoinStage"
    assert advisories[0]["params_responsible"] == ["output_dir"]
    assert advisories[0]["suggested_shape"]["drop_params"] == ["output_dir"]
    # Advice, never a blocker.
    assert result["runnable"] is True


def test_no_advice_when_the_recipe_already_keeps_its_rows_independent(tmp_path: Path) -> None:
    assert _forfeited(aa.validate(_split_recipe(tmp_path, output_dir=None))) == []


def test_row_independence_advice_is_refine_later_only(tmp_path: Path) -> None:
    """fast_first accepts repeating model work; the trade is not news there."""
    recipe = _split_recipe(tmp_path, output_dir=str(tmp_path / "chunks"), preference=_FAST)
    assert _forfeited(aa.validate(recipe)) == []


def test_nothing_is_said_when_no_reuse_is_lost(tmp_path: Path) -> None:
    """With nothing persisted below it, the truncated prefix costs a delta that never existed."""
    recipe = {
        "stages": [
            {"ref": "ManifestReader", "params": {"manifest_path": str(_INPUT)}},
            {
                "ref": "SplitASRAlignJoinStage",
                "params": {
                    "segments_key": "diar_segments",
                    "output_dir": str(tmp_path / "chunks"),
                },
            },
        ],
        "planning_preference": _REFINE,
    }
    assert _forfeited(aa.validate(recipe)) == []
