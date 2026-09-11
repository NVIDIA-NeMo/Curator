# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from nemo_curator import audio_agent as aa
from nemo_curator.audio_agent import (
    _safety,
    artifacts,
    checkpoint,
    cli,
    continuation,
    delta,
    reusable_pipeline,
    reuse,
    run_store,
    verbs,
)
from nemo_curator.audio_agent.contracts import RunRecord
from nemo_curator.audio_agent.profiler import profile_data
from nemo_curator.audio_agent.recipe import Recipe, StageRef, build_stages
from nemo_curator.stages.audio.common import (
    ManifestCheckpointStage,
    PreserveByValueConditionsStage,
    PreserveByValueStage,
)
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from collections.abc import Callable


def _duration_recipe(tmp_path: Path, *, checkpoint: bool = False, live_waveform: bool = False) -> Recipe:
    stages: list[dict] = [
        {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "input.jsonl")}},
    ]
    if live_waveform:
        stages.append(
            {
                "ref": "ChannelCountStage",
                "params": {
                    "action": "convert",
                    "target_channels": 1,
                    "keep_waveform_in_task": True,
                    "write_to_disk": False,
                },
            }
        )
    stages.append({"ref": "GetAudioDurationStage", "params": {"duration_key": "duration"}})
    if checkpoint:
        stages.append(
            {
                "ref": "ManifestCheckpointStage",
                "params": {
                    "output_path": str(tmp_path / "checkpoint.jsonl"),
                    "retention_sec": 600,
                    "owner": "user",
                },
            }
        )
    stages.extend(
        [
            {
                "ref": "PreserveByValueStage",
                "params": {
                    "input_value_key": "duration",
                    "target_value": 5.0,
                    "operator": "ge",
                },
            },
            {"ref": "ManifestWriterStage", "params": {"output_path": str(tmp_path / "kept.jsonl")}},
        ]
    )
    return Recipe.from_dict({"stages": stages}).freeze()


def _utmos_recipe(  # noqa: PLR0913 - test matrix controls each exactness constraint
    tmp_path: Path,
    *,
    action: str = "annotate",
    mode: str = "task",
    operator: str = "ge",
    missing_value_policy: str = "drop",
    live_waveform: bool = False,
) -> Recipe:
    stages: list[dict] = [
        {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "input.jsonl")}},
    ]
    if live_waveform:
        stages.append(
            {
                "ref": "ChannelCountStage",
                "params": {
                    "action": "convert",
                    "target_channels": 1,
                    "keep_waveform_in_task": True,
                    "write_to_disk": False,
                },
            }
        )
    stages.extend(
        [
            {
                "ref": "UTMOSFilterStage",
                "params": {
                    "action": action,
                    "mode": mode,
                    "input_residency": "waveform" if live_waveform else "file",
                    "score_key": "row_utmos",
                    "mos_threshold": 3.5,
                },
            },
            {
                "ref": "PreserveByValueStage",
                "params": {
                    "input_value_key": "row_utmos",
                    "target_value": 3.5,
                    "operator": operator,
                    "missing_value_policy": missing_value_policy,
                },
            },
            {"ref": "ManifestWriterStage", "params": {"output_path": str(tmp_path / "kept.jsonl")}},
        ]
    )
    return Recipe.from_dict({"stages": stages}).freeze()


def _utmos_segment_recipe(  # noqa: PLR0913 - exact segment selector matrix
    tmp_path: Path,
    *,
    action: str = "annotate",
    mode: str = "segments",
    segments_key: str = "clips",
    items_key: str | None = None,
    operator: str = "ge",
    missing_value_policy: str = "drop",
    drop_parent_if_empty: bool = True,
    condition_logic: str = "and",
    live_segment_waveforms: bool = False,
) -> Recipe:
    stages: list[dict] = [
        {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "input.jsonl")}},
    ]
    if live_segment_waveforms:
        stages.append(
            {
                "ref": "VADSegmentationStage",
                "params": {
                    "nested": True,
                    "segments_key": segments_key,
                    "keep_segment_waveform_in_task": True,
                    "input_residency": "file",
                },
            }
        )
    stages.extend(
        [
            {
                "ref": "UTMOSFilterStage",
                "params": {
                    "action": action,
                    "mode": mode,
                    "input_residency": "waveform" if live_segment_waveforms else "file",
                    "segments_key": segments_key,
                    "score_key": "segment_utmos",
                    "mos_threshold": 3.5,
                },
            },
            {
                "ref": "PreserveByValueConditionsStage",
                "params": {
                    "conditions": [
                        {
                            "input_value_key": "segment_utmos",
                            "target_value": 3.5,
                            "operator": operator,
                        }
                    ],
                    "missing_value_policy": missing_value_policy,
                    "items_key": segments_key if items_key is None else items_key,
                    "drop_parent_if_empty": drop_parent_if_empty,
                    "condition_logic": condition_logic,
                },
            },
            {"ref": "ManifestWriterStage", "params": {"output_path": str(tmp_path / "kept.jsonl")}},
        ]
    )
    return Recipe.from_dict({"stages": stages}).freeze()


def _sigmos_recipe(  # noqa: PLR0913 - test matrix controls each exactness constraint
    tmp_path: Path,
    *,
    action: str = "annotate",
    mode: str = "task",
    missing_value_policy: str = "drop",
    conditions: list[dict] | None = None,
    segments_key: str = "clips",
    items_key: str | None = None,
    drop_parent_if_empty: bool = True,
    condition_logic: str = "and",
) -> Recipe:
    exact_conditions = (
        conditions
        if conditions is not None
        else [
            {"input_value_key": "row_noise", "target_value": 4.0, "operator": "ge"},
            {"input_value_key": "row_ovrl", "target_value": 3.5, "operator": "ge"},
            {"input_value_key": "row_reverb", "target_value": 3.8, "operator": "ge"},
        ]
    )
    return Recipe.from_dict(
        {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "input.jsonl")}},
                {
                    "ref": "SIGMOSFilterStage",
                    "params": {
                        "action": action,
                        "mode": mode,
                        "input_residency": "file",
                        "segments_key": segments_key,
                        "noise_threshold": 4.0,
                        "ovrl_threshold": 3.5,
                        "sig_threshold": None,
                        "col_threshold": None,
                        "disc_threshold": None,
                        "loud_threshold": None,
                        "reverb_threshold": 3.8,
                        "noise_key": "row_noise",
                        "ovrl_key": "row_ovrl",
                        "reverb_key": "row_reverb",
                    },
                },
                {
                    "ref": "PreserveByValueConditionsStage",
                    "params": {
                        "conditions": exact_conditions,
                        "missing_value_policy": missing_value_policy,
                        "condition_logic": condition_logic,
                        **(
                            {
                                "items_key": segments_key if items_key is None else items_key,
                                "drop_parent_if_empty": drop_parent_if_empty,
                            }
                            if mode == "segments"
                            else {}
                        ),
                    },
                },
                {"ref": "ManifestWriterStage", "params": {"output_path": str(tmp_path / "kept.jsonl")}},
            ]
        }
    ).freeze()


def test_analysis_names_the_declared_pair_without_inventing_a_path(tmp_path: Path) -> None:
    result = aa.plan_checkpoint(_duration_recipe(tmp_path))

    assert result["status"] == "candidates"
    candidate = result["candidates"][0]
    assert candidate["status"] == "needs_output_path"
    assert candidate["producer_stage"] == "GetAudioDurationStage"
    assert candidate["selector_stage"] == "PreserveByValueStage"
    assert candidate["score_key"] == "duration"


def test_smoke_and_run_refuse_an_unresolved_recommended_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifacts,
        "stage_is_costly",
        lambda stage_ref: stage_ref == "GetAudioDurationStage",
    )
    recipe = _duration_recipe(tmp_path)

    smoke_result = aa.smoke(recipe, sample=1)
    run_result = aa.run(recipe, confirm=False)

    assert smoke_result["reason_code"] == "checkpoint_decision_required"
    assert run_result["reason_code"] == "checkpoint_decision_required"
    assert smoke_result["candidates"][0]["producer_stage"] == "GetAudioDurationStage"


def test_explicit_baseline_choice_is_bound_to_the_exact_recipe_and_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifacts,
        "stage_is_costly",
        lambda stage_ref: stage_ref == "GetAudioDurationStage",
    )
    result = aa.plan_checkpoint(_duration_recipe(tmp_path), choice="baseline")
    baseline = Recipe.from_dict(result["baseline"]["recipe"]).freeze()

    assert result["status"] == "baseline_selected"
    assert result["checkpoint_decision_required"] is False
    assert reusable_pipeline.checkpoint_decision_requirement(baseline) is None

    baseline.stages[2].params["target_value"] = 8.0
    baseline.freeze()
    requirement = reusable_pipeline.checkpoint_decision_requirement(baseline)
    assert requirement is not None
    assert requirement["reason_code"] == "checkpoint_decision_required"


def test_materialized_checkpoint_resolves_the_pre_smoke_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifacts,
        "stage_is_costly",
        lambda stage_ref: stage_ref == "GetAudioDurationStage",
    )
    result = aa.plan_checkpoint(
        _duration_recipe(tmp_path),
        output_path=str(tmp_path / "checkpoint.jsonl"),
        choice="checkpoint",
    )
    candidate = Recipe.from_dict(result["candidates"][0]["recipe"]).freeze()

    assert reusable_pipeline.checkpoint_decision_requirement(candidate) is None


def test_materialized_candidate_is_complete_and_keeps_the_gate_below_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.jsonl"
    result = aa.plan_checkpoint(
        _duration_recipe(tmp_path),
        output_path=str(checkpoint),
        retention_sec=600,
    )

    assert result["status"] == "candidates"
    candidate = result["candidates"][0]
    assert candidate["status"] == "ready"
    refs = [stage["ref"] for stage in candidate["recipe"]["stages"]]
    assert refs == [
        "ManifestReader",
        "GetAudioDurationStage",
        "ManifestCheckpointStage",
        "PreserveByValueStage",
        "ManifestWriterStage",
    ]
    assert candidate["recipe"]["stages"][2]["params"] == {
        "output_path": str(checkpoint),
        "retention_sec": 600,
        "owner": "user",
        "planning_provenance": "reusable_pipeline_v1",
    }
    assert candidate["cardinality"]["first_run"].startswith("unchanged")
    assert candidate["residency"]["waveform_persisted"] is False
    assert candidate["config_hash"] != result["baseline"]["config_hash"]


def test_planning_preference_survives_manual_recipe_transformations(
    tmp_path: Path,
) -> None:
    preference = {
        "schema_version": 1,
        "curation_mode": "refine_later",
        "source": "explicit_user_choice",
    }
    recipe = _duration_recipe(tmp_path)
    recipe.planning_preference = preference

    planned = aa.plan_checkpoint(
        recipe,
        output_path=str(tmp_path / "reusable.jsonl"),
    )
    reusable = Recipe.from_dict(planned["candidates"][0]["recipe"])
    checkpointed, checkpoint_error = checkpoint.insert(
        recipe,
        index=2,
        output_path=str(tmp_path / "legacy-checkpoint.jsonl"),
    )
    continued, continuation_error = continuation.materialize(
        recipe,
        uri=str(tmp_path / "prior.jsonl"),
        kind="manifest",
        prefix=2,
    )
    delta_recipe, _redirect, delta_error = delta.prefix_recipe(
        recipe,
        prefix=2,
        files=("changed.wav",),
        sandbox=str(tmp_path / "delta"),
        sinks_=[],
    )

    assert checkpoint_error == ""
    assert continuation_error == ""
    assert delta_error == ""
    assert checkpointed is not None
    assert continued is not None
    assert delta_recipe is not None
    assert reusable.planning_preference == preference
    assert checkpointed.planning_preference == preference
    assert continued.planning_preference == preference
    assert delta_recipe.planning_preference == preference


def test_from_run_adoption_inherits_planning_preference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    recipe = _duration_recipe(tmp_path)
    recipe.planning_preference = {
        "schema_version": 1,
        "curation_mode": "fast_first",
        "source": "explicit_user_choice",
    }
    run_store.save(
        RunRecord(
            run_id="preference-parent",
            recipe=recipe.to_dict(),
            config_hash=recipe.config_hash,
            semantic_hash=recipe.semantic_hash,
            status="completed",
        )
    )

    adopted, _summary, refusal = verbs._adopt_recipe("preference-parent")

    assert refusal is None
    assert adopted.planning_preference == recipe.planning_preference


def test_transformed_candidate_enforces_exact_hash_and_own_smoke_token(tmp_path: Path) -> None:
    result = aa.plan_checkpoint(
        _duration_recipe(tmp_path),
        output_path=str(tmp_path / "checkpoint.jsonl"),
    )
    baseline_hash = result["baseline"]["config_hash"]
    candidate_hash = result["candidates"][0]["config_hash"]

    assert _safety.verify_smoke_token(_safety.smoke_token(baseline_hash), baseline_hash)
    assert not _safety.verify_smoke_token(_safety.smoke_token(baseline_hash), candidate_hash)
    requirements = result["candidates"][0]["execution_requirements"]
    assert requirements["semantic_review"]["required_response"] == {
        "mechanically_runnable": True,
        "recipe_config_hash": candidate_hash,
        "intent_status": "pass",
    }
    assert requirements["approval"]["bare_true_allowed"] is False

    bare = aa.run(result["candidates"][0]["recipe"], confirm=True)
    wrong_smoke = aa.run(
        result["candidates"][0]["recipe"],
        confirm=candidate_hash,
        smoke_token=_safety.smoke_token(baseline_hash),
    )

    assert bare["status"] == "refused"
    assert "exact-hash" in bare["reason"]
    assert wrong_smoke["status"] == "refused"
    assert "authoritative smoke" in wrong_smoke["reason"]


def test_checkpoint_insertion_preserves_first_run_selector_verdicts_exactly(tmp_path: Path) -> None:
    rows = [
        AudioTask(dataset_name="d", data={"duration": 3.0, "id": "short"}),
        AudioTask(dataset_name="d", data={"duration": 7.0, "id": "long"}),
    ]
    baseline_gate = PreserveByValueStage(
        input_value_key="duration",
        target_value=5.0,
        operator="ge",
    )
    baseline = baseline_gate.process_batch(rows)

    checkpoint = ManifestCheckpointStage(output_path=str(tmp_path / "checkpoint.jsonl"))
    checkpoint.setup()
    checkpointed_rows = [checkpoint.process(row) for row in rows]
    candidate_gate = PreserveByValueStage(
        input_value_key="duration",
        target_value=5.0,
        operator="ge",
    )
    candidate = candidate_gate.process_batch(checkpointed_rows)

    assert [row.data["id"] for row in candidate] == [row.data["id"] for row in baseline]
    assert checkpointed_rows == rows


def test_configured_wer_key_is_the_declared_decision_identity(tmp_path: Path) -> None:
    recipe = Recipe.from_dict(
        {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "in.jsonl")}},
                {
                    "ref": "GetPairwiseWerStage",
                    "params": {"text_key": "reference", "pred_text_key": "prediction", "wer_key": "row_wer"},
                },
                {
                    "ref": "PreserveByValueStage",
                    "params": {"input_value_key": "row_wer", "target_value": 20.0, "operator": "le"},
                },
                {"ref": "ManifestWriterStage", "params": {"output_path": str(tmp_path / "out.jsonl")}},
            ]
        }
    ).freeze()

    result = aa.plan_checkpoint(recipe)

    assert result["candidates"][0]["score_key"] == "row_wer"
    assert result["candidates"][0]["operator"] == "le"


@pytest.mark.parametrize("with_output_path", [False, True])
def test_tensor_resident_boundary_is_refused_without_materializing_audio(
    tmp_path: Path, with_output_path: bool
) -> None:
    result = aa.plan_checkpoint(
        _duration_recipe(tmp_path, live_waveform=True),
        output_path=str(tmp_path / "checkpoint.jsonl") if with_output_path else None,
    )

    assert result["status"] == "no_candidate"
    assert "live waveform" in result["rejected"][0]["reason"]


@pytest.mark.parametrize("occupied", ["output", "marker"])
def test_occupied_or_stale_checkpoint_path_is_refused(tmp_path: Path, occupied: str) -> None:
    checkpoint = tmp_path / "checkpoint.jsonl"
    path = checkpoint if occupied == "output" else Path(f"{checkpoint}._COMPLETE")
    path.write_text("prior", encoding="utf-8")

    result = aa.plan_checkpoint(_duration_recipe(tmp_path), output_path=str(checkpoint))

    assert result["status"] == "no_candidate"
    assert "path" in result["rejected"][0]["reason"]


def test_existing_checkpoint_is_reused_when_only_the_declared_gate_changes(tmp_path: Path) -> None:
    result = aa.plan_checkpoint(
        _duration_recipe(tmp_path, checkpoint=True),
        decision_stage="GetAudioDurationStage",
        decision_value=8.0,
    )

    candidate = result["candidates"][0]
    assert candidate["status"] == "configured"
    assert candidate["decision_changed"] is True
    assert candidate["recipe"]["stages"][3]["params"]["target_value"] == 8.0
    assert candidate["recipe"]["stages"][1]["params"] == {"duration_key": "duration"}
    assert candidate["requires_reuse_scan"] is True


def test_downstream_threshold_change_scans_as_incremental_from_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    original = _duration_recipe(tmp_path, checkpoint=True)
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text('{"audio_filepath":"x.wav","duration":7.0}\n', encoding="utf-8")
    step = artifacts.plan_steps(original, "stat:data")[2]
    artifacts.publish(
        artifacts.Artifact(
            step_key=step.step_key,
            input_key=step.input_key,
            stage_ref=step.stage_ref,
            stage_index=step.index,
            semantic_params=step.semantic_params,
            uri=step.uri,
            kind="manifest",
            dataset_key="stat:data",
            fingerprint_tier="stat",
            impl_version=step.impl_version,
            deterministic=True,
        )
    )
    tuned = aa.plan_checkpoint(
        original,
        decision_stage="GetAudioDurationStage",
        decision_value=8.0,
    )["candidates"][0]["recipe"]

    scan = reuse.scan(Recipe.from_dict(tuned).freeze(), dataset_key="stat:data")

    assert scan["decision"] == "incremental"
    assert scan["reuse_point"]["stage"] == "ManifestCheckpointStage"
    assert len(scan["reuse_stages"]) == 3


def test_completed_checkpoint_validates_and_continues_but_full_run_cannot_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    source = tmp_path / "input.jsonl"
    source.write_text('{"audio_filepath":"clip.wav"}\n', encoding="utf-8")
    initial = Recipe.from_dict(
        aa.plan_checkpoint(
            _duration_recipe(tmp_path, checkpoint=True),
            decision_stage="GetAudioDurationStage",
        )["candidates"][0]["recipe"]
    ).freeze()
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    checkpoint_bytes = b'{"audio_filepath":"clip.wav","duration":7.0}\n'
    checkpoint_path.write_bytes(checkpoint_bytes)
    dataset_key = profile_data(str(source)).dataset_key()
    step = artifacts.plan_steps(initial, dataset_key)[2]
    artifacts.publish(
        artifacts.Artifact(
            step_key=step.step_key,
            input_key=step.input_key,
            stage_ref=step.stage_ref,
            stage_index=step.index,
            semantic_params=step.semantic_params,
            uri=step.uri,
            kind="manifest",
            dataset_key=dataset_key,
            fingerprint_tier="stat",
            impl_version=step.impl_version,
            deterministic=True,
        )
    )
    tuned = Recipe.from_dict(
        aa.plan_checkpoint(
            initial,
            decision_stage="GetAudioDurationStage",
            decision_value=8.0,
        )["candidates"][0]["recipe"]
    ).freeze()

    verdict = aa.validate(tuned, data=str(source))
    scan = aa.reuse_scan(tuned, data=str(source))
    continued = aa.plan_continuation(tuned, data=str(source))
    direct = aa.run(
        tuned,
        data=str(source),
        confirm=tuned.config_hash,
        smoke_token=_safety.smoke_token(tuned.config_hash),
        executor=lambda *_args, **_kwargs: pytest.fail("occupied checkpoint must refuse before execution"),
    )

    assert not any(issue["code"] == "checkpoint_output_occupied" for issue in verdict["issues"])
    assert scan["decision"] == "incremental"
    assert continued["mode"] == "incremental"
    assert direct["status"] == "refused"
    assert direct["reason"] == "dedicated checkpoint output is not safe to create"
    assert checkpoint_path.read_bytes() == checkpoint_bytes


def test_undeclared_native_filter_is_not_rewritten(tmp_path: Path) -> None:
    recipe = Recipe.from_dict(
        {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "in.jsonl")}},
                {
                    "ref": "UTMOSFilterStage",
                    "params": {"action": "filter", "mode": "task", "mos_threshold": 3.0},
                },
                {"ref": "ManifestWriterStage", "params": {"output_path": str(tmp_path / "out.jsonl")}},
            ]
        }
    ).freeze()

    result = aa.plan_checkpoint(recipe, output_path=str(tmp_path / "checkpoint.jsonl"))

    assert result["status"] == "no_candidate"
    assert result["candidates"] == []


def test_utmos_task_annotation_exact_selector_produces_checkpoint_candidate(
    tmp_path: Path,
) -> None:
    result = aa.plan_checkpoint(
        _utmos_recipe(tmp_path),
        output_path=str(tmp_path / "utmos-checkpoint.jsonl"),
    )

    assert result["status"] == "candidates"
    candidate = result["candidates"][0]
    assert candidate["status"] == "ready"
    assert candidate["decision_kind"] == "scalar"
    assert candidate["score_key"] == "row_utmos"
    assert candidate["operator"] == "ge"
    assert candidate["decision_contract"]["missing_score_policy"] == "selector_drop"
    assert [stage["ref"] for stage in candidate["recipe"]["stages"]] == [
        "ManifestReader",
        "UTMOSFilterStage",
        "ManifestCheckpointStage",
        "PreserveByValueStage",
        "ManifestWriterStage",
    ]


@pytest.mark.parametrize(
    ("stage_index", "param", "value", "reason"),
    [
        (1, "action", "filter", "action"),
        (1, "mode", "auto", "mode"),
        (1, "mode", "segments", "nested"),
        (2, "input_value_key", "wrong_score", "row_utmos"),
        (2, "operator", "le", "operator"),
        (2, "missing_value_policy", "error", "missing_value_policy"),
    ],
)
def test_utmos_planner_refuses_non_exact_separated_shapes(
    tmp_path: Path,
    stage_index: int,
    param: str,
    value: object,
    reason: str,
) -> None:
    recipe = _utmos_recipe(tmp_path)
    recipe.stages[stage_index].params[param] = value

    result = aa.plan_checkpoint(recipe)

    assert result["status"] == "no_candidate"
    assert reason in result["rejected"][0]["reason"]


def test_utmos_file_residency_is_legal_but_live_waveform_boundary_is_refused(
    tmp_path: Path,
) -> None:
    file_result = aa.plan_checkpoint(_utmos_recipe(tmp_path))
    waveform_result = aa.plan_checkpoint(_utmos_recipe(tmp_path, live_waveform=True))

    assert file_result["status"] == "candidates"
    assert waveform_result["status"] == "no_candidate"
    assert "live waveform" in waveform_result["rejected"][0]["reason"]


def test_utmos_segment_annotation_exact_selector_produces_checkpoint_candidate(
    tmp_path: Path,
) -> None:
    result = aa.plan_checkpoint(
        _utmos_segment_recipe(tmp_path, segments_key="utterances"),
        output_path=str(tmp_path / "utmos-segments-checkpoint.jsonl"),
    )

    assert result["status"] == "candidates"
    candidate = result["candidates"][0]
    assert candidate["decision_kind"] == "scalar"
    assert candidate["scope"] == "segments"
    assert candidate["items_key"] == "utterances"
    assert candidate["conditions"] == [
        {
            "input_value_key": "segment_utmos",
            "target_value": 3.5,
            "operator": "ge",
        }
    ]
    assert [stage["ref"] for stage in candidate["recipe"]["stages"]] == [
        "ManifestReader",
        "UTMOSFilterStage",
        "ManifestCheckpointStage",
        "PreserveByValueConditionsStage",
        "ManifestWriterStage",
    ]


@pytest.mark.parametrize(
    ("recipe_factory", "reason"),
    [
        (
            lambda path: _utmos_segment_recipe(path, items_key="wrong"),
            "must exactly match",
        ),
        (
            lambda path: _utmos_segment_recipe(path, drop_parent_if_empty=False),
            "drop_parent_if_empty",
        ),
        (
            lambda path: _utmos_segment_recipe(path, missing_value_policy="error"),
            "missing_value_policy",
        ),
        (
            lambda path: _utmos_segment_recipe(path, operator="le"),
            "required operator",
        ),
        (
            lambda path: _utmos_segment_recipe(path, condition_logic="or"),
            "condition_logic",
        ),
        (
            lambda path: _utmos_segment_recipe(path, action="filter"),
            "action",
        ),
        (
            lambda path: _utmos_segment_recipe(path, mode="auto"),
            "data-dependent",
        ),
    ],
)
def test_utmos_segment_planner_refuses_non_exact_selectors(
    tmp_path: Path,
    recipe_factory: Callable[[Path], Recipe],
    reason: str,
) -> None:
    result = aa.plan_checkpoint(recipe_factory(tmp_path))

    assert result["status"] == "no_candidate"
    assert reason in result["rejected"][0]["reason"]


def test_segment_checkpoint_accepts_file_metadata_and_refuses_nested_waveforms(
    tmp_path: Path,
) -> None:
    file_result = aa.plan_checkpoint(_utmos_segment_recipe(tmp_path))
    tensor_result = aa.plan_checkpoint(_utmos_segment_recipe(tmp_path, live_segment_waveforms=True))

    assert file_result["status"] == "candidates"
    assert tensor_result["status"] == "no_candidate"
    assert "live waveform" in tensor_result["rejected"][0]["reason"]


def test_sigmos_task_annotation_requires_complete_and_selector(
    tmp_path: Path,
) -> None:
    result = aa.plan_checkpoint(
        _sigmos_recipe(tmp_path),
        output_path=str(tmp_path / "sigmos-checkpoint.jsonl"),
    )

    assert result["status"] == "candidates"
    candidate = result["candidates"][0]
    assert candidate["decision_kind"] == "compound"
    assert candidate["operator"] == "and"
    assert candidate["score_keys"] == ["row_noise", "row_ovrl", "row_reverb"]
    assert candidate["conditions"] == [
        {"input_value_key": "row_noise", "target_value": 4.0, "operator": "ge"},
        {"input_value_key": "row_ovrl", "target_value": 3.5, "operator": "ge"},
        {"input_value_key": "row_reverb", "target_value": 3.8, "operator": "ge"},
    ]
    assert [stage["ref"] for stage in candidate["recipe"]["stages"]] == [
        "ManifestReader",
        "SIGMOSFilterStage",
        "ManifestCheckpointStage",
        "PreserveByValueConditionsStage",
        "ManifestWriterStage",
    ]


@pytest.mark.parametrize(
    ("recipe_factory", "reason"),
    [
        (
            lambda path: _sigmos_recipe(
                path,
                conditions=[
                    {"input_value_key": "row_noise", "target_value": 4.0, "operator": "ge"},
                    {"input_value_key": "row_ovrl", "target_value": 3.5, "operator": "ge"},
                ],
            ),
            "every enabled",
        ),
        (
            lambda path: _sigmos_recipe(
                path,
                conditions=[
                    {"input_value_key": "row_noise", "target_value": 4.0, "operator": "le"},
                    {"input_value_key": "row_ovrl", "target_value": 3.5, "operator": "ge"},
                    {"input_value_key": "row_reverb", "target_value": 3.8, "operator": "ge"},
                ],
            ),
            "required operator",
        ),
        (
            lambda path: _sigmos_recipe(path, missing_value_policy="error"),
            "missing_value_policy",
        ),
        (
            lambda path: _sigmos_recipe(path, condition_logic="or"),
            "condition_logic",
        ),
        (
            lambda path: _sigmos_recipe(path, action="filter"),
            "action",
        ),
        (
            lambda path: _sigmos_recipe(path, mode="auto"),
            "mode",
        ),
    ],
)
def test_sigmos_planner_refuses_unsafe_or_incomplete_shapes(
    tmp_path: Path,
    recipe_factory: Callable[[Path], Recipe],
    reason: str,
) -> None:
    result = aa.plan_checkpoint(recipe_factory(tmp_path))

    assert result["status"] == "no_candidate"
    assert reason in result["rejected"][0]["reason"]


def test_sigmos_segment_annotation_requires_exact_compound_selector(
    tmp_path: Path,
) -> None:
    result = aa.plan_checkpoint(
        _sigmos_recipe(tmp_path, mode="segments", segments_key="utterances"),
        output_path=str(tmp_path / "sigmos-segments-checkpoint.jsonl"),
    )

    assert result["status"] == "candidates"
    candidate = result["candidates"][0]
    assert candidate["decision_kind"] == "compound"
    assert candidate["scope"] == "segments"
    assert candidate["items_key"] == "utterances"
    assert candidate["score_keys"] == ["row_noise", "row_ovrl", "row_reverb"]
    assert [stage["ref"] for stage in candidate["recipe"]["stages"]] == [
        "ManifestReader",
        "SIGMOSFilterStage",
        "ManifestCheckpointStage",
        "PreserveByValueConditionsStage",
        "ManifestWriterStage",
    ]


@pytest.mark.parametrize(
    ("recipe_factory", "reason"),
    [
        (
            lambda path: _sigmos_recipe(
                path,
                mode="segments",
                items_key="wrong",
            ),
            "must exactly match",
        ),
        (
            lambda path: _sigmos_recipe(
                path,
                mode="segments",
                drop_parent_if_empty=False,
            ),
            "drop_parent_if_empty",
        ),
        (
            lambda path: _sigmos_recipe(
                path,
                mode="segments",
                missing_value_policy="error",
            ),
            "missing_value_policy",
        ),
        (
            lambda path: _sigmos_recipe(
                path,
                mode="segments",
                condition_logic="or",
            ),
            "condition_logic",
        ),
        (
            lambda path: _sigmos_recipe(
                path,
                mode="segments",
                conditions=[
                    {"input_value_key": "row_noise", "target_value": 4.0, "operator": "ge"},
                    {"input_value_key": "row_ovrl", "target_value": 3.5, "operator": "ge"},
                ],
            ),
            "every enabled",
        ),
        (
            lambda path: _sigmos_recipe(
                path,
                mode="segments",
                conditions=[
                    {"input_value_key": "row_noise", "target_value": 4.0, "operator": "le"},
                    {"input_value_key": "row_ovrl", "target_value": 3.5, "operator": "ge"},
                    {"input_value_key": "row_reverb", "target_value": 3.8, "operator": "ge"},
                ],
            ),
            "required operator",
        ),
        (
            lambda path: _sigmos_recipe(path, mode="segments", action="filter"),
            "action",
        ),
    ],
)
def test_sigmos_segment_planner_refuses_unsafe_or_incomplete_selectors(
    tmp_path: Path,
    recipe_factory: Callable[[Path], Recipe],
    reason: str,
) -> None:
    result = aa.plan_checkpoint(recipe_factory(tmp_path))

    assert result["status"] == "no_candidate"
    assert reason in result["rejected"][0]["reason"]


def test_sigmos_segment_planner_refuses_unsafe_score_lineage(tmp_path: Path) -> None:
    recipe = _sigmos_recipe(tmp_path, mode="segments")
    recipe.stages.insert(
        2,
        StageRef(
            ref="GetAudioDurationStage",
            params={"duration_key": "intervening_duration"},
        ),
    )

    result = aa.plan_checkpoint(recipe)

    assert result["status"] == "no_candidate"
    assert "exact score lineage" in result["rejected"][0]["reason"]


def test_sigmos_scalar_feedback_fails_closed_without_mutating_any_threshold(
    tmp_path: Path,
) -> None:
    recipe = _sigmos_recipe(tmp_path)

    result = aa.plan_checkpoint(
        recipe,
        decision_stage="SIGMOSFilterStage",
        decision_value=4.1,
    )

    assert result["status"] == "no_candidate"
    assert "compound decisions cannot be tuned with scalar decision_value" in result["rejected"][0]["reason"]
    assert recipe.stages[1].params["noise_threshold"] == 4.0
    assert recipe.stages[1].params["ovrl_threshold"] == 3.5
    assert recipe.stages[1].params["reverb_threshold"] == 3.8


def test_sigmos_compound_feedback_replaces_only_complete_selector_conditions(
    tmp_path: Path,
) -> None:
    recipe = _sigmos_recipe(tmp_path)

    result = aa.plan_checkpoint(
        recipe,
        output_path=str(tmp_path / "sigmos-checkpoint.jsonl"),
        decision_stage="SIGMOSFilterStage",
        decision_conditions={
            "row_ovrl": 3.9,
            "sigmos_sig": {"target_value": 3.7, "operator": "ge"},
        },
    )

    candidate = result["candidates"][0]
    stages = candidate["recipe"]["stages"]
    assert result["status"] == "candidates"
    assert candidate["score_keys"] == ["row_ovrl", "sigmos_sig"]
    assert candidate["conditions"] == [
        {"input_value_key": "row_ovrl", "target_value": 3.9, "operator": "ge"},
        {"input_value_key": "sigmos_sig", "target_value": 3.7, "operator": "ge"},
    ]
    assert stages[1]["params"] == recipe.stages[1].params
    assert stages[3]["params"]["conditions"] == candidate["conditions"]
    assert candidate["diff"]["changed"][0]["param"] == "conditions"


@pytest.mark.parametrize(
    ("decision_conditions", "reason"),
    [
        (
            [
                {"input_value_key": "row_noise", "target_value": 4.1, "operator": "ge"},
                {"input_value_key": "row_noise", "target_value": 4.2, "operator": "ge"},
            ],
            "duplicate",
        ),
        ({"unknown_score": 3.0}, "not a configured score key"),
        (
            [{"input_value_key": "row_noise", "target_value": 4.0, "operator": "le"}],
            "exactly 'ge'",
        ),
        (
            [{"input_value_key": "row_noise", "target_value": float("nan"), "operator": "ge"}],
            "finite",
        ),
        ({}, "at least one enabled dimension"),
    ],
)
def test_sigmos_compound_feedback_rejects_incomplete_or_ambiguous_conditions(
    tmp_path: Path,
    decision_conditions: object,
    reason: str,
) -> None:
    result = aa.plan_checkpoint(
        _sigmos_recipe(tmp_path),
        decision_stage="SIGMOSFilterStage",
        decision_conditions=decision_conditions,
    )

    assert result["status"] == "no_candidate"
    assert reason in result["rejected"][0]["reason"]


def test_compound_feedback_requires_scores_in_completed_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "sigmos-checkpoint.jsonl"
    initial = Recipe.from_dict(
        aa.plan_checkpoint(
            _sigmos_recipe(tmp_path),
            output_path=str(checkpoint_path),
        )["candidates"][0]["recipe"]
    ).freeze()
    checkpoint_path.write_text('{"row_ovrl":4.0}\n', encoding="utf-8")
    Path(f"{checkpoint_path}._COMPLETE").write_text("complete", encoding="utf-8")

    result = aa.plan_checkpoint(
        initial,
        decision_stage="SIGMOSFilterStage",
        decision_conditions={"row_ovrl": 3.8, "sigmos_sig": 3.6},
    )

    assert result["status"] == "no_candidate"
    assert "only part of the requested compound score set" in result["rejected"][0]["reason"]


def test_compound_feedback_keeps_checkpoint_prefix_reusable(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "sigmos-checkpoint.jsonl"
    initial = Recipe.from_dict(
        aa.plan_checkpoint(
            _sigmos_recipe(tmp_path),
            output_path=str(checkpoint_path),
        )["candidates"][0]["recipe"]
    ).freeze()
    checkpoint_path.write_text(
        ('{"row_noise":4.5,"row_ovrl":4.0,"sigmos_sig":4.1,"row_reverb":4.2}\n'),
        encoding="utf-8",
    )
    step = artifacts.plan_steps(initial, "stat:sigmos-data")[2]
    artifacts.publish(
        artifacts.Artifact(
            step_key=step.step_key,
            input_key=step.input_key,
            stage_ref=step.stage_ref,
            stage_index=step.index,
            semantic_params=step.semantic_params,
            uri=step.uri,
            kind="manifest",
            dataset_key="stat:sigmos-data",
            fingerprint_tier="stat",
            impl_version=step.impl_version,
            deterministic=True,
        )
    )

    tuned = Recipe.from_dict(
        aa.plan_checkpoint(
            initial,
            decision_stage="SIGMOSFilterStage",
            decision_conditions={"row_ovrl": 3.9, "sigmos_sig": 3.7},
        )["candidates"][0]["recipe"]
    ).freeze()
    scan = reuse.scan(tuned, dataset_key="stat:sigmos-data")

    assert tuned.stages[1].params == initial.stages[1].params
    assert scan["decision"] == "incremental"
    assert scan["reuse_point"]["stage"] == "ManifestCheckpointStage"


@pytest.mark.parametrize("selector_kind", ["utmos", "sigmos"])
def test_model_checkpoint_preserves_first_run_verdicts(
    tmp_path: Path,
    selector_kind: str,
) -> None:
    if selector_kind == "utmos":
        rows = [
            AudioTask(data={"id": "pass", "row_utmos": 4.0}),
            AudioTask(data={"id": "fail", "row_utmos": 3.0}),
            AudioTask(data={"id": "missing"}),
        ]
        baseline_selector = PreserveByValueStage(
            "row_utmos",
            3.5,
            "ge",
            missing_value_policy="drop",
        )
        candidate_selector = PreserveByValueStage(
            "row_utmos",
            3.5,
            "ge",
            missing_value_policy="drop",
        )
    else:
        rows = [
            AudioTask(data={"id": "pass", "row_noise": 4.2, "row_ovrl": 3.7}),
            AudioTask(data={"id": "fail", "row_noise": 4.2, "row_ovrl": 3.2}),
            AudioTask(data={"id": "missing", "row_noise": 4.2}),
        ]
        conditions = [
            {"input_value_key": "row_noise", "target_value": 4.0, "operator": "ge"},
            {"input_value_key": "row_ovrl", "target_value": 3.5, "operator": "ge"},
        ]
        baseline_selector = PreserveByValueConditionsStage(
            conditions,
            missing_value_policy="drop",
        )
        candidate_selector = PreserveByValueConditionsStage(
            conditions,
            missing_value_policy="drop",
        )

    baseline = baseline_selector.process_batch(rows)
    checkpoint = ManifestCheckpointStage(output_path=str(tmp_path / f"{selector_kind}.jsonl"))
    checkpoint.setup()
    checkpointed = [checkpoint.process(row) for row in rows]
    candidate = candidate_selector.process_batch(checkpointed)

    assert [row.data["id"] for row in candidate] == [row.data["id"] for row in baseline]


def test_changed_dataset_is_routed_to_existing_delta_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    recipe = _duration_recipe(tmp_path, checkpoint=True)
    run_store.save(
        RunRecord(
            run_id="prior-run",
            recipe=recipe.to_dict(),
            config_hash=recipe.config_hash,
            semantic_hash=recipe.semantic_hash,
            dataset_key="stat:prior",
            status="completed",
        )
    )
    monkeypatch.setattr(verbs, "_dataset_key_arg", lambda _data: "stat:current")

    result = aa.plan_checkpoint(
        from_run="prior-run",
        data=str(tmp_path / "data"),
        decision_stage="GetAudioDurationStage",
        decision_value=8.0,
    )

    assert result["status"] == "changed_dataset"
    assert result["route"] == "delta_run"


def test_checkpoint_artifact_uses_cumulative_prefix_trust(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = _duration_recipe(tmp_path, checkpoint=True)
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text('{"audio_filepath":"x.wav","duration":7.0}\n', encoding="utf-8")
    stages, issues = __import__("nemo_curator.audio_agent.recipe", fromlist=["build_stages"]).build_stages(recipe)
    assert stages is not None, issues

    real_stage_trust = artifacts.stage_trust

    def trust(stage_ref: str) -> tuple[bool, int]:
        if stage_ref == "GetAudioDurationStage":
            return False, 120
        return real_stage_trust(stage_ref)

    monkeypatch.setattr(artifacts, "stage_trust", trust)
    published: list[artifacts.Artifact] = []
    monkeypatch.setattr(artifacts, "publish", lambda artifact: published.append(artifact) or artifact)

    verbs._publish_artifacts(
        recipe,
        stages,
        dataset_key="stat:data",
        fingerprint_tier="stat",
        per_stage={},
        run_id="run",
        input_count=1,
        data_profile={"manifest_keys": ["audio_filepath"]},
        started_at="",
        ended_at="",
    )

    checkpoint_artifact = next(a for a in published if a.stage_ref == "ManifestCheckpointStage")
    assert checkpoint_artifact.deterministic is False
    assert checkpoint_artifact.ttl_sec == 120
    policy = checkpoint_artifact.metrics["checkpoint_policy"]
    assert policy["owner"] == "user"
    assert policy["retention_sec"] == 600
    assert "max_bytes" not in policy
    assert policy["expires_at"]
    assert policy["automatic_deletion"] is False


def test_validation_refuses_checkpoint_collision_even_when_authored_by_hand(tmp_path: Path) -> None:
    recipe = _duration_recipe(tmp_path, checkpoint=True)
    recipe.stages[-1].params["output_path"] = recipe.stages[2].params["output_path"]

    reasons = verbs._checkpoint_output_refusals(recipe)

    assert reasons
    assert "collides" in reasons[0]


def test_checkpoint_worker_override_fails_build_and_validation(tmp_path: Path) -> None:
    recipe = _duration_recipe(tmp_path, checkpoint=True)
    recipe.stages[2].params["num_workers"] = 2

    stages, issues = build_stages(recipe)
    verdict = aa.validate(recipe)

    assert stages is None
    assert any(issue["code"] == "checkpoint_single_worker_required" for issue in issues)
    assert any(issue["code"] == "checkpoint_single_worker_required" for issue in verdict["issues"])


def test_configured_checkpoint_still_runs_boundary_simulation(tmp_path: Path) -> None:
    result = aa.plan_checkpoint(
        _duration_recipe(tmp_path, checkpoint=True, live_waveform=True),
        decision_stage="GetAudioDurationStage",
    )

    assert result["status"] == "no_candidate"
    assert "live waveform" in result["rejected"][0]["reason"]


def test_checkpoint_candidate_refuses_task_id_dependent_suffix(tmp_path: Path) -> None:
    recipe = _duration_recipe(tmp_path)
    recipe.stages.insert(
        -1,
        StageRef(
            ref="SnippetExtractionStage",
            params={
                "output_dir": str(tmp_path / "snippets"),
                "output_audio_tar_path": str(tmp_path / "snippets.tar"),
                "dry_run": True,
            },
        ),
    )

    result = aa.plan_checkpoint(recipe)

    assert result["status"] == "no_candidate"
    assert "stable framework task.task_id" in result["rejected"][0]["reason"]
    assert "SnippetExtractionStage" in result["rejected"][0]["reason"]


@pytest.mark.parametrize("occupied", ["partial", "stale_marker"])
def test_configured_checkpoint_refuses_partial_or_stale_artifact(
    tmp_path: Path,
    occupied: str,
) -> None:
    checkpoint = tmp_path / "checkpoint.jsonl"
    path = checkpoint if occupied == "partial" else Path(f"{checkpoint}._COMPLETE")
    path.write_text("unproven", encoding="utf-8")

    result = aa.plan_checkpoint(
        _duration_recipe(tmp_path, checkpoint=True),
        decision_stage="GetAudioDurationStage",
    )

    assert result["status"] == "no_candidate"
    assert "completion marker" in result["rejected"][0]["reason"]


def test_configured_checkpoint_collision_is_refused_by_planner(tmp_path: Path) -> None:
    recipe = _duration_recipe(tmp_path, checkpoint=True)
    recipe.stages[-1].params["output_path"] = recipe.stages[2].params["output_path"]

    result = aa.plan_checkpoint(
        recipe,
        decision_stage="GetAudioDurationStage",
    )

    assert result["status"] == "no_candidate"
    assert "collides" in result["rejected"][0]["reason"]


def test_non_adjacent_selector_without_proven_lineage_is_refused(tmp_path: Path) -> None:
    recipe = Recipe.from_dict(
        {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "in.jsonl")}},
                {"ref": "GetAudioDurationStage", "params": {"duration_key": "duration"}},
                {
                    "ref": "ManifestWriterStage",
                    "params": {"output_path": str(tmp_path / "intermediate.jsonl")},
                },
                {
                    "ref": "PreserveByValueStage",
                    "params": {
                        "input_value_key": "duration",
                        "target_value": 5.0,
                        "operator": "ge",
                    },
                },
                {
                    "ref": "ManifestWriterStage",
                    "params": {"output_path": str(tmp_path / "out.jsonl")},
                },
            ]
        }
    ).freeze()

    result = aa.plan_checkpoint(recipe)

    assert result["status"] == "no_candidate"
    assert "exact score lineage" in result["rejected"][0]["reason"]


@pytest.mark.parametrize("decision_value", [[1], {"threshold": 1}])
def test_python_surface_rejects_non_scalar_decision_values(
    tmp_path: Path,
    decision_value: object,
) -> None:
    result = aa.plan_checkpoint(
        _duration_recipe(tmp_path),
        decision_value=decision_value,
    )

    assert result["status"] == "refused"
    assert "JSON scalar" in result["reason"]


@pytest.mark.parametrize("decision_value", [True, "5", float("nan")])
def test_numeric_decision_rejects_incompatible_scalar(
    tmp_path: Path,
    decision_value: object,
) -> None:
    result = aa.plan_checkpoint(
        _duration_recipe(tmp_path),
        decision_value=decision_value,
    )

    expected_status = "refused" if isinstance(decision_value, float) else "no_candidate"
    assert result["status"] == expected_status
    reason = result.get("reason") or result["rejected"][0]["reason"]
    assert "number" in reason


def test_configured_selector_with_container_target_is_refused(tmp_path: Path) -> None:
    recipe = _duration_recipe(tmp_path, checkpoint=True)
    recipe.stages[3].params["target_value"] = [5.0]

    result = aa.plan_checkpoint(recipe)

    assert result["status"] == "no_candidate"
    assert "JSON scalar" in result["rejected"][0]["reason"]


@pytest.mark.parametrize("raw", ["[]", '{"threshold": 1}', "null", "NaN"])
def test_cli_decision_parser_rejects_non_scalar_or_non_finite_values(raw: str) -> None:
    with pytest.raises(ValueError, match="decision-value"):
        cli._parse_scalar(raw)


def test_cli_condition_parser_accepts_complete_list_or_mapping() -> None:
    assert cli._parse_conditions('{"sigmos_noise": 4.1}') == {"sigmos_noise": 4.1}
    assert cli._parse_conditions('[{"input_value_key":"sigmos_noise","target_value":4.1,"operator":"ge"}]') == [
        {
            "input_value_key": "sigmos_noise",
            "target_value": 4.1,
            "operator": "ge",
        }
    ]


@pytest.mark.parametrize("raw", ["4.0", '"sigmos_noise"', "null"])
def test_cli_condition_parser_rejects_non_container_values(raw: str) -> None:
    with pytest.raises(ValueError, match="decision-conditions"):
        cli._parse_conditions(raw)


def test_file_uri_checkpoint_path_is_rejected_consistently(tmp_path: Path) -> None:
    uri = (tmp_path / "checkpoint.jsonl").as_uri()

    planned = aa.plan_checkpoint(_duration_recipe(tmp_path), output_path=uri)

    assert planned["status"] == "no_candidate"
    assert "plain local path" in planned["rejected"][0]["reason"]
    with pytest.raises(ValueError, match="plain local path"):
        ManifestCheckpointStage(output_path=uri)


# --------------------------------------------------------------------------- managed location
_KEY = "stat:0123456789abcdef"


def _managed_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the run store at a scratch tree so tests never touch the real one."""
    root = tmp_path / "runs"
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(root))
    return root


def _derived(recipe: Recipe, **kwargs: object) -> dict:
    return reusable_pipeline.plan(recipe, dataset_key=_KEY, **kwargs)["candidates"][0]


def test_a_derived_checkpoint_is_addressed_by_its_step_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _managed_runs(monkeypatch, tmp_path)
    candidate = _derived(_duration_recipe(tmp_path), accept=True)
    materialized = Recipe.from_dict(candidate["recipe"]).freeze()

    index = [s.ref for s in materialized.stages].index("ManifestCheckpointStage")
    step_key = artifacts.plan_steps(materialized, _KEY)[index].step_key

    assert candidate["checkpoint"]["output_path"] == run_store.checkpoint_path(step_key)
    assert candidate["checkpoint"]["path_source"] == "derived"


def test_a_derived_path_survives_a_downstream_threshold_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of a checkpoint: retuning the selector must still find it.

    A path named after the recipe hash cannot do this -- inserting the checkpoint changes
    that hash, and so does every later threshold edit. The step key of everything ABOVE the
    selector does not move, which is why the checkpoint is addressed by that instead.
    """
    _managed_runs(monkeypatch, tmp_path)
    loose = _duration_recipe(tmp_path)
    strict = Recipe.from_dict(loose.to_dict()).freeze()
    strict.stages[2].params["target_value"] = 9.0
    strict = strict.freeze()

    assert loose.config_hash != strict.config_hash
    assert (
        _derived(loose, accept=True)["checkpoint"]["output_path"]
        == _derived(strict, accept=True)["checkpoint"]["output_path"]
    )


def test_a_derived_path_does_not_collide_across_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _managed_runs(monkeypatch, tmp_path)
    recipe = _duration_recipe(tmp_path)
    mine = _derived(recipe, accept=True)["checkpoint"]["output_path"]
    theirs = reusable_pipeline.plan(recipe, dataset_key="stat:ffffffffffffffff", accept=True)
    assert theirs["candidates"][0]["checkpoint"]["output_path"] != mine


def test_a_derived_checkpoint_still_needs_the_users_yes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deriving the location answers WHERE, never WHETHER."""
    _managed_runs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        artifacts,
        "stage_is_costly",
        lambda stage_ref: stage_ref == "GetAudioDurationStage",
    )
    result = reusable_pipeline.plan(_duration_recipe(tmp_path), dataset_key=_KEY)

    assert result["candidates"][0]["status"] == "needs_decision"
    assert result["checkpoint_decision_required"] is True
    assert reusable_pipeline.recommended_candidate_ids(result) == ["checkpoint-after-1"]


def test_accepting_a_derived_checkpoint_clears_the_pre_smoke_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _managed_runs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        artifacts,
        "stage_is_costly",
        lambda stage_ref: stage_ref == "GetAudioDurationStage",
    )
    result = reusable_pipeline.plan(_duration_recipe(tmp_path), dataset_key=_KEY, accept=True)
    accepted = Recipe.from_dict(result["candidates"][0]["recipe"]).freeze()

    assert result["candidates"][0]["status"] == "ready"
    assert result["checkpoint_decision_required"] is False
    assert reusable_pipeline.checkpoint_decision_requirement(accepted) is None


def test_an_unkeyable_source_still_asks_for_a_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No dataset key means no step key, so the old question is the honest fallback."""
    _managed_runs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        artifacts,
        "stage_is_costly",
        lambda stage_ref: stage_ref == "GetAudioDurationStage",
    )
    result = reusable_pipeline.plan(_duration_recipe(tmp_path))

    assert result["candidates"][0]["status"] == "needs_output_path"
    assert result["checkpoint_decision_required"] is True


def test_an_explicit_path_overrides_the_managed_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _managed_runs(monkeypatch, tmp_path)
    chosen = str(tmp_path / "mine.jsonl")
    candidate = _derived(_duration_recipe(tmp_path), output_path=chosen)

    assert candidate["checkpoint"]["output_path"] == chosen
    assert candidate["checkpoint"]["path_source"] == "explicit"
    # Naming a path IS the acceptance, so it needs no separate yes.
    assert candidate["status"] == "ready"


def test_an_existing_file_refuses_a_named_path_but_not_a_derived_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At a content address an existing file is a cache hit, not a collision."""
    _managed_runs(monkeypatch, tmp_path)
    recipe = _duration_recipe(tmp_path)
    derived_path = Path(_derived(recipe, accept=True)["checkpoint"]["output_path"])
    derived_path.write_text('{"duration": 1.0}\n', encoding="utf-8")

    assert _derived(recipe, accept=True)["checkpoint"]["output_path"] == str(derived_path)

    named = tmp_path / "mine.jsonl"
    named.write_text("", encoding="utf-8")
    refused = reusable_pipeline.plan(recipe, dataset_key=_KEY, output_path=str(named))
    assert refused["candidates"] == []
    assert "already exists" in refused["rejected"][0]["reason"]


def test_the_managed_checkpoint_directory_is_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent state names source paths and per-file scores; other accounts must not read it."""
    _managed_runs(monkeypatch, tmp_path)
    assert Path(run_store.checkpoints_dir()).stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("step_key", ["", "..", "a/b", "../escape", "x" * 65])
def test_an_unusable_step_key_names_no_file(
    step_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _managed_runs(monkeypatch, tmp_path)
    assert run_store.checkpoint_path(step_key) is None


# --------------------------------------------------------------------------- lifecycle
def _publish_managed(recipe: Recipe, *, rows: str, ttl_sec: int = 0) -> tuple[str, Path]:
    """Write a checkpoint at its managed path and publish the artifact that addresses it."""
    index = [s.ref for s in recipe.stages].index("ManifestCheckpointStage")
    step = artifacts.plan_steps(recipe, _KEY)[index]
    path = Path(run_store.checkpoint_path(step.step_key))
    path.write_text(rows, encoding="utf-8")
    artifacts.publish(
        artifacts.Artifact(
            step_key=step.step_key,
            input_key=step.input_key,
            stage_ref=step.stage_ref,
            stage_index=step.index,
            semantic_params=step.semantic_params,
            uri=str(path),
            kind="manifest",
            dataset_key=_KEY,
            fingerprint_tier="stat",
            impl_version=step.impl_version,
            deterministic=True,
            ttl_sec=ttl_sec,
        )
    )
    return step.step_key, path


def test_a_published_checkpoint_lists_as_reusable_and_is_never_collected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _managed_runs(monkeypatch, tmp_path)
    recipe = Recipe.from_dict(_derived(_duration_recipe(tmp_path), accept=True)["recipe"]).freeze()
    step_key, path = _publish_managed(recipe, rows='{"duration": 7.0}\n')

    listed = aa.checkpoints()
    entry = next(e for e in listed["checkpoints"] if e["step_key"] == step_key)
    assert entry["status"] == "reusable"
    assert entry["collectable"] is False
    assert entry["reasons"] == []

    assert aa.checkpoints(gc=True)["removed"] == []
    assert path.exists()


def test_gc_collects_bytes_no_step_key_can_address(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An orphan is a file no run can ever find: no record means no step key resolves to it."""
    _managed_runs(monkeypatch, tmp_path)
    orphan = Path(run_store.checkpoints_dir()) / ("a" * 24 + ".jsonl")
    orphan.write_text('{"duration": 1.0}\n', encoding="utf-8")
    marker = Path(f"{orphan}._COMPLETE")
    marker.write_text("{}", encoding="utf-8")

    assert aa.checkpoints()["checkpoints"][0]["status"] == "orphan"

    collected = aa.checkpoints(gc=True)
    assert [r["why"] for r in collected["removed"]] == ["orphan"]
    assert collected["reclaimed_bytes"] > 0
    assert not orphan.exists()
    assert not marker.exists()


def test_gc_collects_a_checkpoint_past_its_declared_ttl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _managed_runs(monkeypatch, tmp_path)
    recipe = Recipe.from_dict(_derived(_duration_recipe(tmp_path), accept=True)["recipe"]).freeze()
    _, path = _publish_managed(recipe, rows='{"duration": 7.0}\n', ttl_sec=1)
    monkeypatch.setattr(artifacts, "_age_sec", lambda _artifact: 3600.0)

    collected = aa.checkpoints(gc=True)

    assert [r["why"] for r in collected["removed"]] == ["expired"]
    assert not path.exists()


def test_gc_keeps_a_checkpoint_only_this_checkout_cannot_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reverting the code makes it valid again; re-running the model is the only way back."""
    _managed_runs(monkeypatch, tmp_path)
    recipe = Recipe.from_dict(_derived(_duration_recipe(tmp_path), accept=True)["recipe"]).freeze()
    step_key, path = _publish_managed(recipe, rows='{"duration": 7.0}\n')
    monkeypatch.setattr(artifacts, "impl_version", lambda _stage_ref: "moved-on")

    listed = aa.checkpoints(gc=True)
    entry = next(e for e in listed["checkpoints"] if e["step_key"] == step_key)

    assert entry["status"] == "stale"
    assert any("implementation changed" in reason for reason in entry["reasons"])
    assert listed["removed"] == []
    assert path.exists()


def test_gc_never_leaves_the_managed_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkpoint the user pointed somewhere of their own is theirs, not cache."""
    _managed_runs(monkeypatch, tmp_path)
    theirs = tmp_path / "their_scores.jsonl"
    theirs.write_text('{"duration": 7.0}\n', encoding="utf-8")
    link = Path(run_store.checkpoints_dir()) / ("b" * 24 + ".jsonl")
    link.symlink_to(theirs)

    listed = aa.checkpoints(gc=True)

    assert listed["checkpoints"] == []
    assert listed["removed"] == []
    assert theirs.exists()
    assert link.is_symlink()


# --------------------------------------------------------------------------- provenance + scope
def test_a_workspace_id_is_minted_once_and_stays_put(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _managed_runs(monkeypatch, tmp_path)
    first = run_store.workspace_id()

    assert first
    assert run_store.workspace_id() == first
    assert (root / "workspace.json").is_file()

    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "elsewhere"))
    assert run_store.workspace_id() != first


def test_an_origin_recipe_is_addressed_by_its_config_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _managed_runs(monkeypatch, tmp_path)
    recipe = _duration_recipe(tmp_path)

    saved = run_store.save_origin_recipe(recipe.config_hash, recipe.to_dict())

    assert saved is not None
    assert recipe.config_hash in saved
    assert run_store.load_origin_recipe(recipe.config_hash) == recipe.to_dict()
    assert run_store.load_origin_recipe("never-published") is None


@pytest.mark.parametrize("config_hash", ["", "..", "a/b", "x" * 65])
def test_an_unusable_config_hash_names_no_origin_recipe(
    config_hash: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _managed_runs(monkeypatch, tmp_path)
    assert run_store.origin_recipe_path(config_hash) is None
    assert run_store.save_origin_recipe(config_hash, {"stages": []}) is None


def test_a_published_artifact_records_what_produced_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _managed_runs(monkeypatch, tmp_path)
    recipe = _duration_recipe(tmp_path, checkpoint=True)
    (tmp_path / "checkpoint.jsonl").write_text('{"duration":7.0}\n', encoding="utf-8")
    stages, issues = build_stages(recipe)
    assert stages is not None, issues
    published: list[artifacts.Artifact] = []
    monkeypatch.setattr(artifacts, "publish", lambda artifact: published.append(artifact) or artifact)

    verbs._publish_artifacts(
        recipe,
        stages,
        dataset_key="stat:data",
        fingerprint_tier="stat",
        per_stage={},
        run_id="run",
        input_count=1,
        data_profile={"manifest_keys": ["audio_filepath"]},
        started_at="",
        ended_at="",
    )

    artifact = next(a for a in published if a.stage_ref == "ManifestCheckpointStage")
    assert artifact.origin_config_hash == recipe.config_hash
    assert artifact.workspace_id == run_store.workspace_id()
    assert run_store.load_origin_recipe(recipe.config_hash) is not None
    assert artifact.origin_recipe_uri.endswith(f"{recipe.config_hash}.json")


def test_an_artifact_from_another_workspace_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local work is local: bytes another tree produced are not ours to trust."""
    _managed_runs(monkeypatch, tmp_path)
    foreign = artifacts.Artifact(step_key="k", uri=str(tmp_path / "x.jsonl"), workspace_id="someone-else")

    assert any("different workspace" in r for r in artifacts.invalid_reasons(foreign))


def test_an_artifact_predating_workspace_ids_is_not_refused_for_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The field is guarded, not fail-closed: older records simply do not carry it."""
    _managed_runs(monkeypatch, tmp_path)
    legacy = artifacts.Artifact(step_key="k", uri=str(tmp_path / "x.jsonl"))

    assert not any("different workspace" in r for r in artifacts.invalid_reasons(legacy))


def test_provenance_never_becomes_a_reuse_precondition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retuning a threshold changes the recipe hash; the checkpoint above it must still match.

    Binding reuse to ``origin_config_hash`` would break exactly the case checkpoints exist
    for, so the artifact records where it came from and matches on the step key regardless.
    """
    _managed_runs(monkeypatch, tmp_path)
    loose = _duration_recipe(tmp_path, checkpoint=True)
    (tmp_path / "checkpoint.jsonl").write_text('{"audio_filepath":"x.wav","duration":7.0}\n', encoding="utf-8")
    step = artifacts.plan_steps(loose, "stat:data")[2]
    artifacts.publish(
        artifacts.Artifact(
            step_key=step.step_key,
            input_key=step.input_key,
            stage_ref=step.stage_ref,
            stage_index=step.index,
            semantic_params=step.semantic_params,
            uri=step.uri,
            kind="manifest",
            dataset_key="stat:data",
            fingerprint_tier="stat",
            impl_version=step.impl_version,
            deterministic=True,
            origin_config_hash=loose.config_hash,
            workspace_id=run_store.workspace_id(),
        )
    )
    strict = Recipe.from_dict(loose.to_dict()).freeze()
    strict.stages[3].params["target_value"] = 9.0
    strict = strict.freeze()

    assert strict.config_hash != loose.config_hash
    found, reasons = artifacts.lookup(artifacts.plan_steps(strict, "stat:data")[2].step_key)
    assert found is not None
    assert reasons == []


def test_an_index_built_before_the_provenance_columns_migrates_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An older index must keep caching, not silently stop until someone runs reindex."""
    import sqlite3

    from nemo_curator.audio_agent import run_index

    root = _managed_runs(monkeypatch, tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    old = sqlite3.connect(root / "index.db")
    old.executescript(
        "CREATE TABLE artifacts (step_key TEXT PRIMARY KEY, stage_ref TEXT, stage_index INTEGER,"
        " dataset_key TEXT, uri TEXT, kind TEXT, rows_out INTEGER, duration_sec REAL,"
        " status TEXT, run_id TEXT, created_at TEXT);"
        " INSERT INTO artifacts (step_key, stage_ref) VALUES ('older', 'GetAudioDurationStage');"
    )
    old.commit()
    old.close()

    assert run_index.index_artifact(
        artifacts.Artifact(step_key="newer", stage_ref="X", origin_config_hash="abc", workspace_id="ws")
    )

    check = sqlite3.connect(root / "index.db")
    columns = {row[1] for row in check.execute("PRAGMA table_info(artifacts)")}
    rows = dict(check.execute("SELECT step_key, origin_config_hash FROM artifacts"))
    check.close()
    assert {"origin_config_hash", "workspace_id"} <= columns
    assert rows == {"older": None, "newer": "abc"}


def test_retention_policy_does_not_move_the_checkpoint_address(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """How long the scores are kept says nothing about what they are.

    While retention was part of the reuse identity it moved the step key, and since a
    checkpoint is addressed by that key, asking to keep the same scores for a week wrote
    them somewhere else and missed the cache that already held them.
    """
    _managed_runs(monkeypatch, tmp_path)
    recipe = _duration_recipe(tmp_path)
    addresses = {
        _derived(recipe, accept=True, **policy)["checkpoint"]["output_path"]
        for policy in (
            {"retention_sec": 0},
            {"retention_sec": 3600},
            {"owner": "project"},
            {"retention_sec": 86400, "owner": "project"},
        )
    }

    assert len(addresses) == 1


def test_retention_policy_still_reaches_the_artifact_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropped from the reuse identity, not from the record: ttl_sec is what GC reads."""
    _managed_runs(monkeypatch, tmp_path)
    recipe = _duration_recipe(tmp_path, checkpoint=True)
    (tmp_path / "checkpoint.jsonl").write_text('{"duration":7.0}\n', encoding="utf-8")
    stages, issues = build_stages(recipe)
    assert stages is not None, issues
    published: list[artifacts.Artifact] = []
    monkeypatch.setattr(artifacts, "publish", lambda artifact: published.append(artifact) or artifact)

    verbs._publish_artifacts(
        recipe,
        stages,
        dataset_key="stat:data",
        fingerprint_tier="stat",
        per_stage={},
        run_id="run",
        input_count=1,
        data_profile={"manifest_keys": ["audio_filepath"]},
        started_at="",
        ended_at="",
    )

    artifact = next(a for a in published if a.stage_ref == "ManifestCheckpointStage")
    assert artifact.metrics["checkpoint_policy"]["retention_sec"] == 600
    assert artifact.metrics["checkpoint_policy"]["owner"] == "user"
    assert "retention_sec" not in artifact.semantic_params
    assert "owner" not in artifact.semantic_params
