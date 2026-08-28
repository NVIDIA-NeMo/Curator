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

import pytest

from nemo_curator.audio_agent.recipe import Recipe, build_stages
from nemo_curator.audio_agent.semantic_review import build_semantic_review
from nemo_curator.stages.audio.filtering.band import BandFilterStage
from nemo_curator.stages.audio.filtering.sigmos import SIGMOSFilterStage
from nemo_curator.stages.audio.filtering.utmos import UTMOSFilterStage

_STAGE_CASES = [
    pytest.param(UTMOSFilterStage, "score_key", id="utmos"),
    pytest.param(SIGMOSFilterStage, "noise_key", id="sigmos"),
    pytest.param(BandFilterStage, "prediction_key", id="band"),
]


def _read_shapes(stage: object) -> set[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
    contract = stage.describe()  # type: ignore[attr-defined]
    return {
        (
            tuple(spec.data_keys),
            tuple(spec.segment_data_keys),
            tuple(spec.accepts),
        )
        for spec in contract.reads_one_of
    }


@pytest.mark.parametrize(("stage_cls", "output_key_param"), _STAGE_CASES)
def test_task_mode_contract_exposes_only_task_residency_and_outputs(stage_cls: type, output_key_param: str) -> None:
    stage = stage_cls(
        mode="task",
        input_residency="auto",
        audio_filepath_key="path",
        waveform_key="samples",
        sample_rate_key="rate",
        **{output_key_param: "quality"},
    )

    contract = stage.describe()
    expected_outputs = stage.outputs()[1]

    assert contract.reads.data_keys == []
    assert contract.reads.segment_data_keys == []
    assert _read_shapes(stage) == {
        (("samples", "rate"), (), ("waveform",)),
        (("path",), (), ("file",)),
    }
    assert contract.writes.data_keys == expected_outputs
    assert contract.writes.segment_data_keys == []


@pytest.mark.parametrize(("stage_cls", "output_key_param"), _STAGE_CASES)
def test_segments_mode_contract_requires_container_and_segment_residency(
    stage_cls: type,
    output_key_param: str,
) -> None:
    stage = stage_cls(
        mode="segments",
        input_residency="auto",
        audio_filepath_key="path",
        waveform_key="samples",
        sample_rate_key="rate",
        segments_key="clips",
        **{output_key_param: "quality"},
    )

    contract = stage.describe()
    expected_outputs = stage.outputs()[1]

    assert contract.reads.data_keys == ["clips"]
    assert contract.reads.segment_data_keys == []
    assert _read_shapes(stage) == {
        ((), ("samples", "rate"), ("waveform",)),
        ((), ("path",), ("file",)),
    }
    assert contract.writes.data_keys == []
    assert contract.writes.segment_data_keys == expected_outputs


@pytest.mark.parametrize(("stage_cls", "output_key_param"), _STAGE_CASES)
def test_auto_mode_contract_conservatively_exposes_both_scoped_branches(
    stage_cls: type,
    output_key_param: str,
) -> None:
    stage = stage_cls(
        mode="auto",
        input_residency="file",
        audio_filepath_key="path",
        segments_key="clips",
        **{output_key_param: "quality"},
    )

    contract = stage.describe()
    expected_outputs = stage.outputs()[1]

    assert contract.reads.data_keys == []
    assert contract.reads.segment_data_keys == []
    assert _read_shapes(stage) == {
        (("path",), (), ("file",)),
        (("clips",), ("path",), ("file",)),
    }
    assert contract.writes.data_keys == expected_outputs
    assert contract.writes.segment_data_keys == expected_outputs


@pytest.mark.parametrize(
    ("stage_ref", "output_key_param"),
    [
        pytest.param("UTMOSFilterStage", "score_key", id="utmos"),
        pytest.param("SIGMOSFilterStage", "noise_key", id="sigmos"),
        pytest.param("BandFilterStage", "prediction_key", id="band"),
    ],
)
def test_segment_only_annotation_does_not_claim_to_feed_top_level_value_filter(
    stage_ref: str,
    output_key_param: str,
) -> None:
    recipe = Recipe.from_dict(
        {
            "stages": [
                {
                    "ref": stage_ref,
                    "params": {
                        "mode": "segments",
                        "action": "annotate",
                        output_key_param: "quality",
                    },
                },
                {
                    "ref": "PreserveByValueStage",
                    "params": {
                        "input_value_key": "quality",
                        "target_value": 4.0,
                    },
                },
            ]
        }
    )
    stages, issues = build_stages(recipe)
    assert stages is not None, issues

    packet = build_semantic_review(
        stages,
        initial_keys=["audio_filepath", "segments"],
        recipe=recipe,
    )
    filter_read = next(
        edge for edge in packet["lineage"] if edge["consumer"]["stage_index"] == 1 and edge["read"]["key"] == "quality"
    )

    quality_write = next(write for write in packet["stages"][0]["writes"] if write["key"] == "quality")
    assert quality_write["scope"] == "segment"
    assert all(write["scope"] != "task" for write in packet["stages"][0]["writes"] if write["key"] == "quality")
    assert filter_read["read"]["scope"] == "task"
    assert filter_read["latest_upstream_producer"]["kind"] == "unresolved"
    assert filter_read in packet["unresolved_lineage"]
