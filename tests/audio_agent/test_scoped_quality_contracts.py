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

from typing import TYPE_CHECKING

import pytest

from nemo_curator.audio_agent.recipe import Recipe, build_stages
from nemo_curator.audio_agent.semantic_review import build_semantic_review
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.audio.common import ManifestWriterStage
from nemo_curator.stages.audio.filtering.band import BandFilterStage
from nemo_curator.stages.audio.filtering.sigmos import SIGMOSFilterStage
from nemo_curator.stages.audio.filtering.utmos import UTMOSFilterStage

if TYPE_CHECKING:
    from pathlib import Path

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
    expected_reads = {
        (("samples", "rate"), (), ("waveform",)),
        (("path",), (), ("file",)),
    }
    if stage_cls is BandFilterStage:
        expected_reads.add((("samples", "path"), (), ("waveform",)))
    assert _read_shapes(stage) == expected_reads
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
    expected_reads = {
        ((), ("samples", "rate"), ("waveform",)),
        ((), ("path",), ("file",)),
    }
    if stage_cls is BandFilterStage:
        expected_reads.add(((), ("samples", "path"), ("waveform",)))
    assert _read_shapes(stage) == expected_reads
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
    assert contract.reads.data_keys == []
    assert contract.reads.segment_data_keys == []
    assert contract.reads_one_of == []
    assert len(contract.conditional_reads) == 2
    task_branch, segment_branch = contract.conditional_reads
    assert task_branch.forbids_keys == ["clips"]
    assert task_branch.requires_keys == []
    assert {
        (tuple(spec.data_keys), tuple(spec.segment_data_keys), tuple(spec.accepts))
        for spec in task_branch.reads_one_of
    } == {(("path",), (), ("file",))}
    assert segment_branch.requires_keys == ["clips"]
    assert segment_branch.forbids_keys == []
    assert {
        (tuple(spec.data_keys), tuple(spec.segment_data_keys), tuple(spec.accepts))
        for spec in segment_branch.reads_one_of
    } == {((), ("path",), ("file",))}
    assert contract.writes.data_keys == []
    assert contract.writes.segment_data_keys == []


@pytest.mark.parametrize(("stage_cls", "output_key_param"), _STAGE_CASES)
@pytest.mark.parametrize("mode", ["task", "segments"])
def test_annotation_outputs_are_conditional_not_guaranteed(
    stage_cls: type,
    output_key_param: str,
    mode: str,
) -> None:
    stage = stage_cls(mode=mode, action="annotate", **{output_key_param: "quality"})

    contract = stage.describe()

    assert contract.writes.data_keys == []
    assert contract.writes.segment_data_keys == []
    score_writes = [
        conditional
        for conditional in contract.conditional_writes
        if "quality" in conditional.writes.data_keys or "quality" in conditional.writes.segment_data_keys
    ]
    assert len(score_writes) == 1


@pytest.mark.parametrize(
    ("stage_cls", "output_key_param", "input_residency"),
    [
        pytest.param(BandFilterStage, "prediction_key", "file", id="band-file"),
        pytest.param(BandFilterStage, "prediction_key", "auto", id="band-auto"),
        pytest.param(SIGMOSFilterStage, "noise_key", "auto", id="sigmos-auto-partial"),
        pytest.param(UTMOSFilterStage, "score_key", "auto", id="utmos-auto-partial"),
    ],
)
@pytest.mark.parametrize(
    ("mode", "expected_scopes"),
    [
        pytest.param("task", {"task"}, id="task"),
        pytest.param("segments", {"segment"}, id="segments"),
        pytest.param("auto", {"task", "segment"}, id="auto"),
    ],
)
def test_possible_file_hydration_is_conditional_and_mode_scoped(
    stage_cls: type,
    output_key_param: str,
    input_residency: str,
    mode: str,
    expected_scopes: set[str],
) -> None:
    contract = stage_cls(
        mode=mode,
        input_residency=input_residency,
        waveform_key="samples",
        sample_rate_key="rate",
        segments_key="clips",
        **{output_key_param: "quality"},
    ).describe()

    assert "samples" not in contract.writes.data_keys
    assert "rate" not in contract.writes.data_keys
    assert "samples" not in contract.writes.segment_data_keys
    assert "rate" not in contract.writes.segment_data_keys

    hydration = [
        conditional
        for conditional in contract.conditional_writes
        if set(conditional.writes.data_keys or conditional.writes.segment_data_keys) == {"samples", "rate"}
    ]
    scopes = {"task" if conditional.writes.data_keys else "segment" for conditional in hydration}
    assert scopes == expected_scopes
    assert all(conditional.writes.produces == ["tensor"] for conditional in hydration)
    assert all(conditional.value_origin == "stage_generated" for conditional in hydration)
    assert all("decoded successfully" in conditional.condition for conditional in hydration)
    assert all("assigned together" in conditional.condition for conditional in hydration)
    if stage_cls in {SIGMOSFilterStage, UTMOSFilterStage}:
        # ``auto_partial`` only REPLACES an incomplete resident pair, so each scope declares the
        # two reachable halves separately, each gated on the key that must already be present.
        assert all("is present without" in conditional.condition for conditional in hydration)
        for scope in expected_scopes:
            in_scope = [
                conditional for conditional in hydration if bool(conditional.writes.data_keys) == (scope == "task")
            ]
            assert sorted(tuple(conditional.requires_keys) for conditional in in_scope) == [("rate",), ("samples",)]
    else:
        assert all(conditional.requires_keys == [] for conditional in hydration)


@pytest.mark.parametrize(
    ("stage_cls", "output_key_param"),
    [
        pytest.param(SIGMOSFilterStage, "noise_key", id="sigmos"),
        pytest.param(UTMOSFilterStage, "score_key", id="utmos"),
    ],
)
def test_explicit_file_residency_does_not_advertise_pair_hydration(
    stage_cls: type,
    output_key_param: str,
) -> None:
    contract = stage_cls(
        mode="auto",
        input_residency="file",
        waveform_key="samples",
        sample_rate_key="rate",
        **{output_key_param: "quality"},
    ).describe()

    assert not any(
        conditional.writes.produces == ["tensor"]
        and {"samples", "rate"} <= set(conditional.writes.data_keys + conditional.writes.segment_data_keys)
        for conditional in contract.conditional_writes
    )


@pytest.mark.parametrize(("stage_cls", "output_key_param"), _STAGE_CASES)
def test_waveform_only_residency_does_not_advertise_file_hydration(stage_cls: type, output_key_param: str) -> None:
    contract = stage_cls(
        mode="auto",
        input_residency="waveform",
        waveform_key="samples",
        sample_rate_key="rate",
        **{output_key_param: "quality"},
    ).describe()

    assert not any(
        conditional.writes.produces == ["tensor"]
        and {"samples", "rate"} <= set(conditional.writes.data_keys + conditional.writes.segment_data_keys)
        for conditional in contract.conditional_writes
    )


@pytest.mark.parametrize(
    ("mode", "expected_scopes"),
    [
        pytest.param("task", {"task"}, id="task"),
        pytest.param("segments", {"segment"}, id="segments"),
        pytest.param("auto", {"task", "segment"}, id="auto"),
    ],
)
def test_band_header_completion_is_a_separate_sample_rate_only_write(
    mode: str,
    expected_scopes: set[str],
) -> None:
    contract = BandFilterStage(
        mode=mode,
        input_residency="auto",
        waveform_key="samples",
        sample_rate_key="rate",
        segments_key="clips",
    ).describe()

    header_writes = [
        conditional for conditional in contract.conditional_writes if "file header" in conditional.condition
    ]
    scopes = {"task" if conditional.writes.data_keys else "segment" for conditional in header_writes}
    assert scopes == expected_scopes
    assert all(
        (conditional.writes.data_keys or conditional.writes.segment_data_keys) == ["rate"]
        for conditional in header_writes
    )
    assert all(conditional.writes.produces == [] for conditional in header_writes)
    assert all("resident waveform is retained" in conditional.condition for conditional in header_writes)


@pytest.mark.parametrize(("stage_cls", "output_key_param"), _STAGE_CASES)
def test_conditional_hydration_does_not_mechanically_feed_waveform_consumer(
    stage_cls: type,
    output_key_param: str,
) -> None:
    producer = stage_cls(
        mode="task",
        action="annotate",
        input_residency="auto",
        **{output_key_param: "quality"},
    )
    consumer = UTMOSFilterStage(mode="task", input_residency="waveform")

    report = validate_pipeline(
        [producer, consumer],
        initial_roles={"audio_filepath"},
        initial_keys={"audio_filepath"},
    )

    # Hydration is never a GUARANTEED write, so the pair is absent from produced_keys either way.
    assert "waveform" not in report.produced_keys
    assert "sample_rate" not in report.produced_keys
    if stage_cls is BandFilterStage:
        # Band always hydrates from a decoded file, so the consumer's read is POSSIBLY met:
        # a conditional_read warning, not a clean pass and not a hard refusal.
        assert report.ok
        assert any(issue.code == "conditional_read" and issue.stage_index == 1 for issue in report.issues)
    else:
        # SIGMOS/UTMOS ``auto_partial`` only replaces an incomplete resident pair; on a
        # file-only manifest that branch is unreachable, so nothing can feed the consumer.
        assert not report.ok
        assert any(issue.code == "unsatisfied_reads" and issue.stage_index == 1 for issue in report.issues)


@pytest.mark.parametrize(("stage_cls", "output_key_param"), _STAGE_CASES)
def test_possible_file_hydration_blocks_unsanitized_json_sink(
    stage_cls: type,
    output_key_param: str,
    tmp_path: Path,
) -> None:
    stage = stage_cls(
        mode="task",
        action="annotate",
        input_residency="auto",
        **{output_key_param: "quality"},
    )
    writer = ManifestWriterStage(output_path=str(tmp_path / "out.jsonl"))

    file_only = validate_pipeline(
        [stage, writer],
        initial_roles={"audio_filepath"},
        initial_keys={"audio_filepath"},
    )
    partial_pair = validate_pipeline(
        [stage, writer],
        initial_roles={"audio_filepath", "sample_rate"},
        initial_keys={"audio_filepath", "sample_rate"},
    )

    def blocked(report: object) -> bool:
        return any(issue.code == "tensor_into_sink" and issue.severity == "error" for issue in report.issues)

    # A resident sample_rate without a waveform is completed from the file at runtime, so the
    # sink is (correctly) refused for every scorer.
    assert blocked(partial_pair)
    if stage_cls is BandFilterStage:
        # Band hydrates from any decoded file: a file-only row DOES gain a tensor.
        assert blocked(file_only)
    else:
        # SIGMOS/UTMOS ``auto_partial`` cannot fire without one half of the pair already
        # resident (``requires_keys``), so the default scorer -> manifest chain composes.
        assert file_only.ok, file_only.summary()


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
