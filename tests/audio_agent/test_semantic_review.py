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

from __future__ import annotations

import json

from nemo_curator.audio_agent.recipe import Recipe, build_stages
from nemo_curator.audio_agent.semantic_review import (
    build_semantic_review,
    semantic_response_contract,
)
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, StageContract
from nemo_curator.stages.audio.common import (
    GetAudioDurationStage,
    PreserveByValueStage,
)
from nemo_curator.stages.audio.postprocessing.timestamp_mapper import TimestampMapperStage
from nemo_curator.stages.audio.preprocessing.concatenation import SegmentConcatenationStage
from nemo_curator.stages.base import _STAGE_REGISTRY, CompositeStage, ProcessingStage
from nemo_curator.tasks import AudioTask


class _InnerComposite(AgentReady, CompositeStage[AudioTask, AudioTask]):
    name = "inner_composite"

    def __init__(self, duration_key: str) -> None:
        super().__init__()
        self.duration_key = duration_key

    def describe(self) -> StageContract:
        return StageContract(wrappable=False)

    def decompose(self) -> list[ProcessingStage]:
        return [
            GetAudioDurationStage(duration_key=self.duration_key),
            PreserveByValueStage(
                input_value_key=self.duration_key,
                target_value=1,
                operator="ge",
            ),
        ]


class _OuterComposite(AgentReady, CompositeStage[AudioTask, AudioTask]):
    name = "outer_composite"

    def __init__(self, duration_key: str = "duration") -> None:
        super().__init__()
        self.duration_key = duration_key

    def describe(self) -> StageContract:
        return StageContract(wrappable=False)

    def decompose(self) -> list[ProcessingStage]:
        return [_InnerComposite(self.duration_key)]


# StageMeta registers every ProcessingStage subclass at class-definition time.
# These fixtures are direct-only test doubles, not public catalog entries.
_STAGE_REGISTRY.pop("_InnerComposite", None)
_STAGE_REGISTRY.pop("_OuterComposite", None)


def _build(stage_specs: list[dict]) -> tuple[Recipe, list]:
    recipe = Recipe.from_dict({"stages": stage_specs})
    stages, issues = build_stages(recipe)
    assert stages is not None, issues
    return recipe, stages


def _edge(packet: dict, *, consumer_index: int, key: str) -> dict:
    return next(
        edge
        for edge in packet["lineage"]
        if edge["consumer"]["stage_index"] == consumer_index and edge["read"]["key"] == key
    )


def test_generic_literal_consumer_is_linked_to_latest_producer_with_full_card_material() -> None:
    recipe, stages = _build(
        [
            {"ref": "SpeakerSeparationStage", "params": {}},
            {
                "ref": "PreserveByValueStage",
                "params": {"input_value_key": "num_speakers", "target_value": 1},
            },
        ]
    )

    packet = build_semantic_review(stages, recipe=recipe)
    edge = _edge(packet, consumer_index=1, key="num_speakers")

    assert packet["status"] == "complete"
    assert packet["review_required"] is True
    assert packet["advisory_only"] is True
    assert packet["intent_interpretation_performed"] is False
    assert packet["required_response"]["intent_status"] == ["pass", "revise", "ask"]
    assert packet["required_response"]["mechanically_runnable"]["required"] is True
    consumer_stage = packet["stages"][1]
    assert consumer_stage["configured_params"]["target_value"] == {
        "value": 1,
        "source": "recipe",
    }
    assert consumer_stage["configured_params"]["operator"] == {
        "value": "eq",
        "source": "default",
    }
    assert edge["read"]["configured_by"] == ["input_value_key"]
    assert edge["read"]["role"] == "unknown"
    assert edge["latest_upstream_producer"]["stage"] == "SpeakerSeparationStage"
    assert edge["latest_upstream_producer"]["stage_index"] == 0
    assert edge["latest_upstream_producer"]["semantic_material"]["notes"]
    assert "num_speakers" in edge["latest_upstream_producer"]["semantic_material"]["semantic_facts"]
    assert edge["consumer"]["semantic_material"]["notes"]
    assert "input_value_key" in edge["consumer"]["semantic_material"]["semantic_facts"]
    assert edge["crossed_cardinality_seams"][0]["kind"] == "fan_out"
    assert edge["crossed_cardinality_seams"][0]["stage"] == "SpeakerSeparationStage"
    json.dumps(packet)


def test_unresolved_generic_literal_key_is_evidence_not_a_rejection() -> None:
    recipe, stages = _build(
        [
            {
                "ref": "PreserveByValueStage",
                "params": {"input_value_key": "definitely_missing", "target_value": 1},
            }
        ]
    )

    packet = build_semantic_review(stages, recipe=recipe)
    edge = _edge(packet, consumer_index=0, key="definitely_missing")

    assert packet["status"] == "complete"
    assert edge["latest_upstream_producer"]["kind"] == "unresolved"
    assert packet["unresolved_lineage"] == [edge]
    assert all("severity" not in item for item in packet["contract_issues"])
    assert "runnable" not in packet


def test_latest_contract_writer_wins_and_earlier_history_is_retained() -> None:
    recipe, stages = _build(
        [
            {"ref": "GetAudioDurationStage", "params": {"duration_key": "duration"}},
            {"ref": "GetAudioDurationStage", "params": {"duration_key": "duration"}},
            {
                "ref": "PreserveByValueStage",
                "params": {"input_value_key": "duration", "target_value": 1},
            },
        ]
    )

    packet = build_semantic_review(stages, initial_keys=["audio_filepath"], recipe=recipe)
    edge = _edge(packet, consumer_index=2, key="duration")

    assert edge["latest_upstream_producer"]["stage_index"] == 1
    assert [writer["stage_index"] for writer in edge["earlier_contract_writers"]] == [0]


def test_cardinality_packet_reports_aggregation_and_nested_collection_generically() -> None:
    recipe, stages = _build(
        [
            {"ref": "VADSegmentationStage", "params": {"nested": True}},
            {"ref": "SegmentConcatenationStage", "params": {}},
        ]
    )

    packet = build_semantic_review(stages, initial_keys=["audio_filepath"], recipe=recipe)

    assert [(seam["stage_index"], seam["kind"]) for seam in packet["cardinality_seams"]] == [
        (0, "nested_collection"),
        (1, "aggregation"),
    ]
    assert all(seam["semantic_material"]["available"] for seam in packet["cardinality_seams"])


def test_optional_semantic_facts_are_copied_without_interpretation(monkeypatch) -> None:  # noqa: ANN001
    recipe, stages = _build(
        [
            {"ref": "GetAudioDurationStage", "params": {}},
            {
                "ref": "PreserveByValueStage",
                "params": {"input_value_key": "duration", "target_value": 1},
            },
        ]
    )
    from nemo_curator.audio_agent.index import get_index

    card = get_index().card("GetAudioDurationStage")
    assert card is not None
    monkeypatch.setitem(card, "semantic_facts", {"duration": "verbatim test fact"})

    packet = build_semantic_review(stages, initial_keys=["audio_filepath"], recipe=recipe)
    edge = _edge(packet, consumer_index=1, key="duration")

    assert edge["latest_upstream_producer"]["semantic_material"]["semantic_facts"] == {
        "duration": "verbatim test fact"
    }


def test_metrics_comparison_notes_and_caveats_are_attached_verbatim() -> None:
    recipe, stages = _build([{"ref": "UTMOSFilterStage", "params": {"action": "annotate"}}])

    packet = build_semantic_review(stages, initial_keys=["audio_filepath"], recipe=recipe)
    material = packet["stages"][0]["semantic_material"]

    assert material["metrics"]["utmos_mos"]["scale"]["direction"] == "higher_better"
    assert material["comparison"]["known_limitations"]
    assert material["notes"]
    assert material["caveats"]


def test_every_nonempty_candidate_requires_review_and_has_intent_checklist() -> None:
    recipe, stages = _build([{"ref": "GetAudioDurationStage", "params": {}}])

    packet = build_semantic_review(stages, initial_keys=["audio_filepath"], recipe=recipe)
    checklist = {item["id"]: item for item in packet["checklist"]}

    assert packet["review_required"] is True
    assert checklist["per_stage_justification"]["required"] is True
    assert checklist["transform_output_consumption"]["required"] is True
    assert "model_domain_metric_caveats" in checklist


def test_profiled_initial_keys_survive_an_opaque_source_with_uncertainty() -> None:
    recipe, stages = _build(
        [
            {"ref": "ManifestReader", "params": {"manifest_path": "/tmp/input.jsonl"}},  # noqa: S108
            {
                "ref": "PreserveByValueStage",
                "params": {"input_value_key": "source_label", "target_value": "keep"},
            },
        ]
    )

    packet = build_semantic_review(
        stages,
        initial_keys=["audio_filepath", "source_label"],
        recipe=recipe,
    )
    edge = _edge(packet, consumer_index=2, key="source_label")
    origin = edge["latest_upstream_producer"]

    assert packet["status"] == "partial"
    assert [stage["stage"] for stage in packet["stages"]] == [
        "FilePartitioningStage",
        "ManifestReaderStage",
        "PreserveByValueStage",
    ]
    assert packet["recipe_stages"][0]["execution_leaf_indices"] == [0, 1]
    assert packet["composites"][0]["expansion_status"] == "complete"
    assert origin["kind"] == "initial_input"
    assert origin["basis"] == "declared_post_source_initial_key"
    assert origin["visibility_uncertainty"]["reason"] == "profiled_source_key_without_declared_semantic_schema"
    assert edge["semantic_provenance"]["status"] == "unresolved_source_schema"
    assert packet["semantic_evidence_gaps"][0]["code"] == ("initial_key_semantics_unresolved")
    assert {gap["reason"] for gap in packet["missing_card_semantics"]} == {"card_absent", "semantic_facts_absent"}


def test_a_nested_composite_is_reported_as_unsupported_not_expanded(
    monkeypatch,  # noqa: ANN001
) -> None:
    """Semantic review must describe a plan the backend can actually run.

    ``Pipeline._decompose_stages`` expands each stage once and raises TypeError
    ("Nested composition is not supported") when a child decomposes further, so a
    recursively-expanded review would document an execution plan that cannot start.
    Reporting the executor's own limit is the honest answer -- and it is why no depth
    bound or cycle check is needed here: neither is reachable once nesting is refused.
    """
    from nemo_curator.audio_agent.index import get_index

    index = get_index()
    for stage_id in ("_OuterComposite", "_InnerComposite"):
        monkeypatch.setitem(
            index._cards_by_stage,
            stage_id,
            {
                "stage_id": stage_id,
                "category": "test",
                "summary": "test composite",
                "semantic_facts": {"behavior": "test-only composite semantics"},
                "verified": {"semantic_facts": "mechanical"},
                "provenance": {"card_version": 1},
            },
        )

    packet = build_semantic_review(
        [_OuterComposite(duration_key="clip_seconds")],
        recipe={
            "stages": [{"ref": "_OuterComposite", "params": {"duration_key": "clip_seconds"}}],
            "rationale": "keep clips of at least one second",
        },
    )

    nested = [i for i in packet["contract_issues"] if i["code"] == "nested_composite_unsupported"]
    assert nested, "the inner composite the executor would reject must be reported"
    assert nested[0]["execution_path"] == [0], "and located at the offending child"

    # No leaf is fabricated for a branch that cannot run.
    assert packet["status"] == "partial", "a plan that cannot run is not a complete review"
    assert packet["recipe"]["execution_leaf_count"] == 0
    assert packet["stages"] == []
    # The outer composite is still described, marked with why expansion stopped.
    assert packet["composites"][0]["authored_params"] == {"duration_key": "clip_seconds"}
    assert packet["composites"][1]["expansion_error"] == "nested_composite"


def test_recipe_free_instance_values_are_not_labeled_as_defaults() -> None:
    packet = build_semantic_review(
        [
            PreserveByValueStage(
                input_value_key="quality_score",
                target_value=0.8,
                operator="ge",
            )
        ]
    )

    params = packet["stages"][0]["configured_params"]
    assert params["operator"] == {"value": "ge", "source": "configured_instance"}
    assert params["target_value"] == {
        "value": 0.8,
        "source": "configured_instance",
    }


def test_recipe_and_profile_context_is_bounded_and_redacted() -> None:
    recipe, stages = _build([{"ref": "GetAudioDurationStage", "params": {}}])
    recipe.rationale = "token=do-not-leak"
    recipe.preset = "quality"
    recipe.acceptance_criteria = [{"id": "duration", "type": "output_completeness", "field": "duration"}]
    recipe.config_strategy = [{"stage": "GetAudioDurationStage", "reason": "goal"}]

    packet = build_semantic_review(
        stages,
        recipe=recipe,
        data_profile={
            "source": "/private/input.jsonl",
            "kind": "manifest",
            "num_files": 3,
            "manifest_keys": ["audio_filepath", "duration"],
            "has_transcripts": False,
        },
    )

    assert packet["recipe"]["rationale"] == "token=<redacted-secret>"
    assert packet["recipe"]["preset"] == "quality"
    assert packet["recipe"]["acceptance_criteria"]
    assert packet["recipe"]["config_strategy"]
    assert packet["data_profile"] == {
        "kind": "manifest",
        "num_files": 3,
        "has_transcripts": False,
        "manifest_keys": ["audio_filepath", "duration"],
    }
    assert "source" not in packet["data_profile"]
    assert semantic_response_contract() == packet["required_response"]


def test_metadata_lineage_connects_current_producer_and_consumer_contracts() -> None:
    recipe = {
        "stages": [
            {"ref": "SegmentConcatenationStage", "params": {}},
            {"ref": "TimestampMapperStage", "params": {}},
        ]
    }
    packet = build_semantic_review(
        [SegmentConcatenationStage(), TimestampMapperStage()],
        initial_keys=["segments"],
        recipe=recipe,
    )
    edge = next(
        edge
        for edge in packet["lineage"]
        if edge["consumer"]["stage_index"] == 1
        and edge["read"]["scope"] == "metadata"
        and edge["read"]["key"] == "segment_mappings"
    )

    assert edge["latest_upstream_producer"]["stage"] == "SegmentConcatenationStage"
    assert edge["latest_upstream_producer"]["write"]["scope"] == "metadata"
    assert edge["latest_upstream_producer"]["write"]["certainty"] == "definite"


def test_review_packet_is_bound_to_canonical_recipe_hash_without_mutating_recipe() -> None:
    recipe, stages = _build([{"ref": "GetAudioDurationStage", "params": {}}])
    expected = recipe.compute_hash()

    packet = build_semantic_review(
        stages,
        initial_keys=["audio_filepath"],
        recipe=recipe,
    )

    assert recipe.config_hash is None
    assert packet["recipe"]["config_hash"] == expected
    assert packet["recipe"]["config_hash_source"] == "computed_canonical_recipe"
    assert packet["required_response"]["schema_version"] == 2
    assert packet["required_response"]["recipe_config_hash"] == {
        "type": "string",
        "source": "semantic_review.recipe.config_hash",
        "required": True,
        "rule": ("Copy exactly; a different recipe requires another validation and critique."),
    }


def test_canonical_hash_wins_over_stale_authored_hash() -> None:
    recipe, stages = _build([{"ref": "GetAudioDurationStage", "params": {}}])
    expected = recipe.compute_hash()
    recipe.config_hash = "stale"

    packet = build_semantic_review(stages, recipe=recipe)

    assert packet["recipe"]["config_hash"] == expected
    assert packet["recipe"]["authored_config_hash_mismatch"] is True
