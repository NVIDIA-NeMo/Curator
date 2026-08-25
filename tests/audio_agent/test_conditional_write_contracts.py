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

import pytest

from nemo_curator.audio_agent.semantic_review import build_semantic_review
from nemo_curator.stages.audio._agent._agent_ready import (
    AgentReady,
    ConditionalWrite,
    IOSpec,
    StageContract,
)
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.filtering.band import BandFilterStage
from nemo_curator.stages.audio.filtering.sigmos import SIGMOSFilterStage
from nemo_curator.stages.audio.filtering.utmos import UTMOSFilterStage
from nemo_curator.stages.audio.metrics.bandwidth import BandwidthEstimationStage
from nemo_curator.stages.audio.metrics.squim import TorchSquimQualityMetricsStage
from nemo_curator.stages.audio.metrics.wer import ComputeWERStage, GetPairwiseWerStage
from nemo_curator.stages.audio.postprocessing.timestamp_mapper import TimestampMapperStage
from nemo_curator.stages.audio.preprocessing.concatenation import SegmentConcatenationStage
from nemo_curator.stages.base import _STAGE_REGISTRY, ProcessingStage
from nemo_curator.tasks import AudioTask


def _scoped_keys(contract: StageContract) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for item in contract.conditional_writes:
        keys.update(("task", key) for key in item.writes.data_keys)
        keys.update(("segment", key) for key in item.writes.segment_data_keys)
        keys.update(("metadata", key) for key in item.metadata_writes)
    return keys


def test_conditional_write_is_additive_and_json_safe() -> None:
    contract = StageContract(
        IOSpec(data_keys=["source"]),
        IOSpec(data_keys=["score"]),
        conditional_writes=[
            ConditionalWrite(
                writes=IOSpec(data_keys=["score"]),
                condition="'source' is non-null",
            )
        ],
    )

    assert contract.contract_resolution == "configured"
    assert contract.writes.data_keys == ["score"]
    payload = contract.to_dict()
    json.dumps(payload)
    assert payload["conditional_writes"] == [
        {
            "writes": {
                "data_keys": ["score"],
                "segment_data_keys": [],
                "accepts": [],
                "produces": [],
            },
            "condition": "'source' is non-null",
            "value_origin": "stage_generated",
            "metadata_writes": [],
        }
    ]


@pytest.mark.parametrize(
    ("stage_cls", "key_param"),
    [
        pytest.param(UTMOSFilterStage, "score_key", id="utmos"),
        pytest.param(SIGMOSFilterStage, "noise_key", id="sigmos"),
        pytest.param(BandFilterStage, "prediction_key", id="band"),
    ],
)
def test_auto_quality_scope_writes_are_conditional(stage_cls: type, key_param: str) -> None:
    stage = stage_cls(mode="auto", **{key_param: "quality"})
    contract = stage.describe()

    assert {("task", "quality"), ("segment", "quality")} <= _scoped_keys(contract)
    assert all(item.value_origin == "stage_generated" for item in contract.conditional_writes)
    assert any("'segments' is absent" in item.condition for item in contract.conditional_writes)
    assert any("'segments' is present" in item.condition for item in contract.conditional_writes)


@pytest.mark.parametrize("mode", ["task", "segments"])
def test_explicit_quality_scope_still_labels_assignment_as_conditional(mode: str) -> None:
    contract = UTMOSFilterStage(mode=mode).describe()
    assert len(contract.conditional_writes) == 1
    expected_scope = "task" if mode == "task" else "segment"
    assert _scoped_keys(contract) == {(expected_scope, "utmos_mos")}
    assert "numeric MOS" in contract.conditional_writes[0].condition


def test_wer_contracts_label_actual_data_dependent_writes() -> None:
    compute = ComputeWERStage(
        segments_key="clips",
        hypothesis_text_key="hyp",
        reference_text_key="ref",
        metrics_key="quality",
    ).describe()
    assert _scoped_keys(compute) == {("task", "quality"), ("segment", "quality")}
    assert any("'clips' is absent" in item.condition for item in compute.conditional_writes)
    assert any("'clips' is present" in item.condition for item in compute.conditional_writes)

    pairwise = GetPairwiseWerStage(text_key="ref", pred_text_key="hyp", wer_key="wer").describe()
    assert _scoped_keys(pairwise) == {("task", "wer")}
    assert "non-null" in pairwise.conditional_writes[0].condition


@pytest.mark.parametrize(
    "contract",
    [
        pytest.param(BandwidthEstimationStage(metrics_key="quality").describe(), id="bandwidth"),
        pytest.param(TorchSquimQualityMetricsStage(metrics_key="quality").describe(), id="squim"),
    ],
)
def test_optional_segment_metric_stages_label_both_runtime_scopes(contract: StageContract) -> None:
    assert _scoped_keys(contract) == {("task", "quality"), ("segment", "quality")}
    assert all(item.value_origin == "augments_upstream_same_key" for item in contract.conditional_writes)


def test_timestamp_mapper_declares_conditional_same_key_passthrough() -> None:
    contract = TimestampMapperStage(
        passthrough_keys=["quality", "sample_rate", "waveform", "original_file"],
    ).describe()
    passthrough = next(item for item in contract.conditional_writes if item.value_origin == "upstream_same_key")

    assert passthrough.writes.data_keys == ["quality", "sample_rate"]
    assert "present, non-null" in passthrough.condition
    assert "waveform" not in passthrough.writes.data_keys
    assert "original_file" not in passthrough.writes.data_keys
    assert contract.preserves_upstream_keys is False


def test_semantic_review_labels_auto_scope_writes_as_conditional() -> None:
    packet = build_semantic_review(
        [UTMOSFilterStage(mode="auto", action="annotate", score_key="quality")],
        initial_keys=["audio_filepath", "segments"],
    )
    quality_writes = [write for write in packet["stages"][0]["writes"] if write["key"] == "quality"]

    assert {(write["scope"], write["certainty"]) for write in quality_writes} == {
        ("task", "conditional"),
        ("segment", "conditional"),
    }
    assert all(write["legacy_mechanical_write"] is True for write in quality_writes)
    assert all(
        condition["value_origin"] == "stage_generated" for write in quality_writes for condition in write["conditions"]
    )


def test_timestamp_passthrough_keeps_original_producer_lineage() -> None:
    packet = build_semantic_review(
        [
            GetAudioDurationStage(duration_key="quality"),
            TimestampMapperStage(passthrough_keys=["quality"]),
            PreserveByValueStage(input_value_key="quality", target_value=4),
        ],
        initial_keys=["audio_filepath"],
    )
    mapper_write = next(write for write in packet["stages"][1]["writes"] if write["key"] == "quality")
    filter_edge = next(
        edge for edge in packet["lineage"] if edge["consumer"]["stage_index"] == 2 and edge["read"]["key"] == "quality"
    )
    producer = filter_edge["latest_upstream_producer"]

    assert mapper_write["certainty"] == "conditional"
    assert mapper_write["legacy_mechanical_write"] is False
    assert mapper_write["conditions"][0]["value_origin"] == "upstream_same_key"
    assert producer["stage_index"] == 0
    assert producer["stage"] == "GetAudioDurationStage"
    assert producer["write"]["key"] == "quality"
    assert producer["conditional_passthroughs"][0]["through"]["stage_index"] == 1
    assert producer["conditional_passthroughs"][0]["through"]["stage"] == "TimestampMapperStage"


def test_mapping_augmentation_keeps_same_key_upstream_evidence() -> None:
    packet = build_semantic_review(
        [
            TorchSquimQualityMetricsStage(
                audio_filepath_key="audio_filepath",
                metrics_key="metrics",
            ),
            BandwidthEstimationStage(metrics_key="metrics"),
            PreserveByValueStage(input_value_key="metrics", target_value=4),
        ],
        initial_keys=["audio_filepath", "duration"],
    )
    filter_edge = next(
        edge for edge in packet["lineage"] if edge["consumer"]["stage_index"] == 2 and edge["read"]["key"] == "metrics"
    )
    producer = filter_edge["latest_upstream_producer"]

    assert producer["stage"] == "BandwidthEstimationStage"
    assert producer["write"]["certainty"] == "conditional"
    assert producer["same_key_upstream"]["producer"]["stage"] == "TorchSquimQualityMetricsStage"
    assert {item["value_origin"] for item in producer["same_key_upstream"]["relationships"]} == {
        "augments_upstream_same_key"
    }


def test_conditional_metadata_write_keeps_upstream_fallback_across_data_rebuild() -> None:
    class _ConditionalMetadataRewriter(
        AgentReady,
        ProcessingStage[AudioTask, AudioTask],
    ):
        def describe(self) -> StageContract:
            return StageContract(
                metadata_writes=["segment_mappings"],
                preserves_upstream_keys=False,
                conditional_writes=[
                    ConditionalWrite(
                        metadata_writes=["segment_mappings"],
                        condition="a replacement mapping is available",
                    )
                ],
            )

        def process(self, task: AudioTask) -> AudioTask:
            return task

    try:
        packet = build_semantic_review(
            [
                SegmentConcatenationStage(),
                _ConditionalMetadataRewriter(),
                TimestampMapperStage(mappings_key="segment_mappings"),
            ],
            initial_keys=["segments"],
        )
    finally:
        _STAGE_REGISTRY.pop("_ConditionalMetadataRewriter", None)

    metadata_edge = next(
        edge
        for edge in packet["lineage"]
        if edge["consumer"]["stage_index"] == 2
        and edge["read"]["scope"] == "metadata"
        and edge["read"]["key"] == "segment_mappings"
    )
    producer = metadata_edge["latest_upstream_producer"]

    assert producer["stage"] == "_ConditionalMetadataRewriter"
    assert producer["write"]["certainty"] == "conditional"
    assert producer["write"]["scope"] == "metadata"
    assert producer["when_condition_not_met"]["producer"]["stage"] == "SegmentConcatenationStage"
