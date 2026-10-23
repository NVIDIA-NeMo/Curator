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

import pickle
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from nemo_curator.stages.audio.common import (
    GetAudioDurationStage,
    ManifestReader,
    ManifestReaderStage,
    ManifestWriterStage,
    PreserveByValueConditionsStage,
    PreserveByValueStage,
)
from nemo_curator.tasks import AudioTask


@pytest.mark.parametrize("policy", ["error", "drop"])
def test_nested_condition_contract_declares_child_reads(policy: str) -> None:
    from nemo_curator.stages.audio.agent import validate_pipeline

    stage = PreserveByValueConditionsStage({"mos": 3.0}, items_key="segments", missing_value_policy=policy)
    contract = stage.describe()
    if policy == "error":
        assert contract.reads.segment_data_keys == ["mos"]
        report = validate_pipeline([stage], initial_keys={"segments"})
        assert not report.ok
    else:
        assert contract.optional_reads.segment_data_keys == ["mos"]
        assert validate_pipeline([stage], initial_keys={"segments"}).ok
        assert stage.process_batch([AudioTask(data={"segments": [{}]})]) == []


@pytest.mark.parametrize("policy", ["error", "drop"])
@pytest.mark.parametrize("key", ["mos", "quality_score"])
def test_nested_condition_planning_matches_seeded_and_missing_children(policy: str, key: str) -> None:
    from nemo_curator.stages.audio.agent import validate_pipeline

    stage = PreserveByValueConditionsStage({key: 3.0}, items_key="segments", missing_value_policy=policy)
    report = validate_pipeline([stage], initial_keys={"segments"}, initial_segment_keys={key})
    assert report.ok
    kept = stage.process_batch([AudioTask(data={"segments": [{key: 3.0}, {key: 2.0}]})])
    assert kept[0].data["segments"] == [{key: 3.0}]
    assert validate_pipeline([stage], initial_keys={"segments"}).ok is (policy == "drop")
    if policy == "error":
        with pytest.raises(ValueError, match="missing"):
            stage.process_batch([AudioTask(data={"segments": [{}]})])
    else:
        assert stage.process_batch([AudioTask(data={"segments": [{}]})]) == []


def test_preserve_by_value_validate_input_valid() -> None:
    stage = PreserveByValueStage(input_value_key="wer", target_value=50, operator="le")
    assert stage.validate_input(AudioTask(data={"wer": 30})) is True


def test_preserve_by_value_validate_input_missing_column() -> None:
    stage = PreserveByValueStage(input_value_key="wer", target_value=50, operator="le")
    assert stage.validate_input(AudioTask(data={"text": "hello"})) is False


def test_preserve_by_value_process_raises_not_implemented() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=3, operator="eq")
    with pytest.raises(NotImplementedError, match="only supports process_batch"):
        stage.process(AudioTask(data={"v": 3}))


def test_preserve_by_value_process_batch_raises_on_missing_column() -> None:
    stage = PreserveByValueStage(input_value_key="wer", target_value=50, operator="le")
    assert stage.missing_value_policy == "error"
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([AudioTask(data={"text": "hello"})])


def test_preserve_by_value_eq_keeps_match() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=3, operator="eq")
    result = stage.process_batch([AudioTask(data={"v": 3})])
    assert len(result) == 1
    assert isinstance(result[0], AudioTask)
    assert result[0].data["v"] == 3


def test_preserve_by_value_eq_filters_non_match() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=3, operator="eq")
    result = stage.process_batch([AudioTask(data={"v": 1})])
    assert len(result) == 0


def test_preserve_by_value_lt() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=5, operator="lt")
    assert len(stage.process_batch([AudioTask(data={"v": 2})])) == 1
    assert len(stage.process_batch([AudioTask(data={"v": 7})])) == 0


def test_preserve_by_value_ge() -> None:
    stage = PreserveByValueStage(input_value_key="v", target_value=10.0, operator="ge")
    assert len(stage.process_batch([AudioTask(data={"v": 9})])) == 0
    assert len(stage.process_batch([AudioTask(data={"v": 10})])) == 1
    assert len(stage.process_batch([AudioTask(data={"v": 11})])) == 1


def test_preserve_by_value_contract_accepts_float_targets_and_exposes_policy() -> None:
    from nemo_curator.stages.audio._agent._agent_registry import stage_params

    params = {param.name: param for param in stage_params(PreserveByValueStage)}

    assert params["target_value"].type == "float | str"
    assert params["missing_value_policy"].default == "error"
    assert params["missing_value_policy"].choices == ["error", "drop"]


def test_preserve_by_value_drop_policy_drops_only_missing_or_failing_rows() -> None:
    stage = PreserveByValueStage(
        input_value_key="score",
        target_value=3.5,
        operator="ge",
        missing_value_policy="drop",
    )
    tasks = [
        AudioTask(data={"id": "pass", "score": 4.0}),
        AudioTask(data={"id": "fail", "score": 3.0}),
        AudioTask(data={"id": "missing"}),
    ]

    assert [task.data["id"] for task in stage.process_batch(tasks)] == ["pass"]


def test_compound_preserve_uses_and_semantics_and_drops_missing() -> None:
    stage = PreserveByValueConditionsStage(
        conditions=[
            {"input_value_key": "noise", "target_value": 4.0, "operator": "ge"},
            {"input_value_key": "ovrl", "target_value": 3.5, "operator": "ge"},
        ],
        missing_value_policy="drop",
    )
    tasks = [
        AudioTask(data={"id": "pass", "noise": 4.1, "ovrl": 3.6}),
        AudioTask(data={"id": "noise_fail", "noise": 3.9, "ovrl": 4.0}),
        AudioTask(data={"id": "ovrl_fail", "noise": 4.5, "ovrl": 3.4}),
        AudioTask(data={"id": "missing", "noise": 4.5}),
    ]

    assert [task.data["id"] for task in stage.process_batch(tasks)] == ["pass"]
    assert stage.normalized_conditions == (
        {"input_value_key": "noise", "target_value": 4.0, "operator": "ge"},
        {"input_value_key": "ovrl", "target_value": 3.5, "operator": "ge"},
    )


@pytest.mark.parametrize("condition_count", [1, 2, 4])
@pytest.mark.parametrize("condition_logic", ["and", "or"])
def test_compound_preserve_top_level_truth_tables(
    condition_count: int,
    condition_logic: str,
) -> None:
    conditions = [
        {"input_value_key": f"c{index}", "target_value": True, "operator": "eq"} for index in range(condition_count)
    ]
    combinations = list(product([False, True], repeat=condition_count))
    tasks = [
        AudioTask(
            data={
                "id": combination,
                **{f"c{index}": value for index, value in enumerate(combination)},
            }
        )
        for combination in combinations
    ]
    expected = [
        combination
        for combination in combinations
        if (all(combination) if condition_logic == "and" else any(combination))
    ]

    result = PreserveByValueConditionsStage(
        conditions,
        condition_logic=condition_logic,
    ).process_batch(tasks)

    assert [task.data["id"] for task in result] == expected


@pytest.mark.parametrize("condition_count", [1, 2, 4])
@pytest.mark.parametrize("condition_logic", ["and", "or"])
def test_compound_preserve_nested_truth_tables_with_arbitrary_items_key(
    condition_count: int,
    condition_logic: str,
) -> None:
    conditions = [
        {"input_value_key": f"c{index}", "target_value": True, "operator": "eq"} for index in range(condition_count)
    ]
    combinations = list(product([False, True], repeat=condition_count))
    children = [
        {
            "id": combination,
            **{f"c{index}": value for index, value in enumerate(combination)},
        }
        for combination in combinations
    ]
    parent = AudioTask(data={"custom_children": children})
    expected = [
        combination
        for combination in combinations
        if (all(combination) if condition_logic == "and" else any(combination))
    ]

    result = PreserveByValueConditionsStage(
        conditions,
        items_key="custom_children",
        condition_logic=condition_logic,
        drop_parent_if_empty=False,
    ).process_batch([parent])

    assert result == [parent]
    assert [child["id"] for child in parent.data["custom_children"]] == expected


def test_compound_preserve_condition_logic_defaults_to_and_and_rejects_invalid() -> None:
    conditions = [
        {"input_value_key": "left", "target_value": True, "operator": "eq"},
        {"input_value_key": "right", "target_value": True, "operator": "eq"},
    ]
    stage = PreserveByValueConditionsStage(conditions)

    assert stage.condition_logic == "and"
    assert stage.process_batch([AudioTask(data={"left": True, "right": False})]) == []
    with pytest.raises(ValueError, match="condition_logic must be 'and' or 'or'"):
        PreserveByValueConditionsStage(conditions, condition_logic="xor")


@pytest.mark.parametrize("missing_value_policy", ["error", "drop"])
def test_compound_preserve_or_never_skips_a_missing_top_level_condition(
    missing_value_policy: str,
) -> None:
    stage = PreserveByValueConditionsStage(
        [
            {"input_value_key": "present", "target_value": True, "operator": "eq"},
            {"input_value_key": "missing", "target_value": True, "operator": "eq"},
        ],
        missing_value_policy=missing_value_policy,
        condition_logic="or",
    )
    task = AudioTask(data={"present": True})

    if missing_value_policy == "error":
        with pytest.raises(ValueError, match="failed validation"):
            stage.process_batch([task])
    else:
        assert stage.process_batch([task]) == []


@pytest.mark.parametrize("missing_value_policy", ["error", "drop"])
def test_compound_preserve_or_never_skips_a_missing_nested_condition(
    missing_value_policy: str,
) -> None:
    stage = PreserveByValueConditionsStage(
        [
            {"input_value_key": "present", "target_value": True, "operator": "eq"},
            {"input_value_key": "missing", "target_value": True, "operator": "eq"},
        ],
        items_key="children",
        missing_value_policy=missing_value_policy,
        condition_logic="or",
    )
    parent = AudioTask(data={"children": [{"present": True}]})

    if missing_value_policy == "error":
        with pytest.raises(ValueError, match="missing condition key 'missing'"):
            stage.process_batch([parent])
    else:
        assert stage.process_batch([parent]) == []
        assert parent.data["children"] == []


def test_compound_preserve_mapping_form_and_default_missing_error() -> None:
    stage = PreserveByValueConditionsStage(
        conditions={
            "noise": {"target_value": 4.0, "operator": "ge"},
            "kind": "speech",
        }
    )

    assert stage.process_batch([AudioTask(data={"noise": 4.2, "kind": "speech"})])
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([AudioTask(data={"noise": 4.2})])


def test_compound_preserve_filters_arbitrary_one_level_items_key_by_reference() -> None:
    passing = {"id": "pass", "quality": 4.2, "metadata": {"speaker": "a"}}
    failing = {"id": "fail", "quality": 2.0, "metadata": {"speaker": "b"}}
    parent = AudioTask(data={"recording": "r1", "clips": [passing, failing]})
    stage = PreserveByValueConditionsStage(
        [{"input_value_key": "quality", "target_value": 3.5, "operator": "ge"}],
        items_key="clips",
    )

    result = stage.process_batch([parent])

    assert result == [parent]
    assert parent.data["recording"] == "r1"
    assert parent.data["clips"] == [passing]
    assert parent.data["clips"][0] is passing
    assert parent.data["clips"][0]["metadata"] is passing["metadata"]


@pytest.mark.parametrize("condition_logic", ["and", "or"])
@pytest.mark.parametrize(
    ("drop_parent_if_empty", "expected_count"),
    [(True, 0), (False, 1)],
)
def test_compound_preserve_nested_empty_parent_policy(
    drop_parent_if_empty: bool,
    expected_count: int,
    condition_logic: str,
) -> None:
    parent = AudioTask(data={"windows": [{"score": 1.0}]})
    stage = PreserveByValueConditionsStage(
        [{"input_value_key": "score", "target_value": 2.0, "operator": "ge"}],
        items_key="windows",
        drop_parent_if_empty=drop_parent_if_empty,
        condition_logic=condition_logic,
    )

    result = stage.process_batch([parent])

    assert len(result) == expected_count
    assert parent.data["windows"] == []


@pytest.mark.parametrize("condition_logic", ["and", "or"])
@pytest.mark.parametrize(
    ("data", "error_type", "message"),
    [
        ({"clips": {}}, TypeError, "must contain a list"),
        ({"clips": [{"score": 4.0}, "not-a-mapping"]}, TypeError, "child 1 must be mapping-like"),
    ],
)
def test_compound_preserve_rejects_malformed_nested_structure_without_mutation(
    data: dict,
    error_type: type[Exception],
    message: str,
    condition_logic: str,
) -> None:
    original_items = data.get("clips")
    stage = PreserveByValueConditionsStage(
        [{"input_value_key": "score", "target_value": 3.5, "operator": "ge"}],
        items_key="clips",
        missing_value_policy="drop",
        condition_logic=condition_logic,
    )

    with pytest.raises(error_type, match=message):
        stage.process_batch([AudioTask(data=data)])

    assert data.get("clips") is original_items


@pytest.mark.parametrize("missing_value_policy", ["error", "drop"])
@pytest.mark.parametrize("condition_logic", ["and", "or"])
def test_compound_preserve_missing_nested_container_is_always_structural_error(
    missing_value_policy: str,
    condition_logic: str,
) -> None:
    stage = PreserveByValueConditionsStage(
        [{"input_value_key": "score", "target_value": 3.5, "operator": "ge"}],
        items_key="clips",
        missing_value_policy=missing_value_policy,
        condition_logic=condition_logic,
    )

    with pytest.raises(ValueError, match="missing nested items_key 'clips'"):
        stage.process_batch([AudioTask(data={"other": []})])


def test_compound_preserve_nested_missing_condition_key_error_vs_drop() -> None:
    condition = [{"input_value_key": "score", "target_value": 3.5, "operator": "ge"}]
    missing = {"id": "missing", "nested": {"score": 5.0}}
    passing = {"id": "pass", "score": 4.0}

    with pytest.raises(ValueError, match="child 0 is missing condition key 'score'"):
        PreserveByValueConditionsStage(
            condition,
            items_key="candidates",
        ).process_batch([AudioTask(data={"candidates": [missing, passing]})])

    parent = AudioTask(data={"candidates": [missing, passing]})
    result = PreserveByValueConditionsStage(
        condition,
        items_key="candidates",
        missing_value_policy="drop",
    ).process_batch([parent])
    assert result == [parent]
    assert parent.data["candidates"] == [passing]


def test_compound_preserve_nested_contract_includes_child_condition_keys() -> None:
    from nemo_curator.stages.audio._agent._agent_registry import stage_params

    stage = PreserveByValueConditionsStage(
        [{"input_value_key": "score", "target_value": 3.5, "operator": "ge"}],
        items_key="candidates",
        drop_parent_if_empty=False,
    )
    contract = stage.describe()
    params = {param.name: param for param in stage_params(PreserveByValueConditionsStage)}

    assert contract.reads.data_keys == ["candidates"]
    assert contract.writes.data_keys == ["candidates"]
    assert contract.reads.segment_data_keys == ["score"]
    assert contract.writes.segment_data_keys == []
    assert contract.iteration_key == "candidates"
    assert contract.cardinality == "1:1 nested-list"
    assert contract.gates.per_row_independent is True
    assert params["items_key"].default is None
    assert params["drop_parent_if_empty"].default is True
    assert params["condition_logic"].default == "and"
    assert params["condition_logic"].choices == ["and", "or"]

    dropping_contract = PreserveByValueConditionsStage(
        [{"input_value_key": "score", "target_value": 3.5, "operator": "ge"}],
        items_key="candidates",
    ).describe()
    assert dropping_contract.cardinality == "filter"
    assert dropping_contract.iteration_key is None
    assert "one-level" in dropping_contract.description
    assert "AND" in dropping_contract.description

    or_contract = PreserveByValueConditionsStage(
        [{"input_value_key": "score", "target_value": 3.5, "operator": "ge"}],
        condition_logic="or",
    ).describe()
    assert "OR" in or_contract.description


@pytest.mark.parametrize(
    ("base", "args"),
    [
        (GetAudioDurationStage, ("duration", "audio_filepath", "duration")),
        (ManifestReaderStage, ("reader",)),
        (ManifestReader, ("manifest.jsonl", "reader", 1, None, [".jsonl"], None)),
    ],
)
def test_additive_base_fields_preserve_subclass_positionals(base: type, args: tuple[Any, ...]) -> None:
    @dataclass
    class Extended(base):
        marker: str = "default"

    assert Extended(*args, "legacy").marker == "legacy"


@pytest.mark.parametrize("policy", ["error", "drop"])
def test_nested_condition_dependency_is_visible_to_planning(policy: str) -> None:
    from nemo_curator.stages.audio.agent import validate_pipeline

    stage = PreserveByValueConditionsStage({"mos": 3.0}, items_key="segments", missing_value_policy=policy)
    missing = validate_pipeline([stage], initial_keys={"segments"}, initial_segment_keys=set())
    seeded = validate_pipeline([stage], initial_keys={"segments"}, initial_segment_keys={"mos"})
    assert seeded.ok
    assert seeded.keys_ok
    if policy == "error":
        assert not missing.ok or not missing.keys_ok
    else:
        assert missing.ok
        assert missing.keys_ok
        assert stage.describe().optional_reads.segment_data_keys == ["mos"]


@pytest.mark.parametrize("residency", ["file", "waveform", "auto"])
def test_duration_input_spec_preserves_subclass_requirements(residency: str) -> None:
    class TenantDuration(GetAudioDurationStage):
        def inputs(self) -> tuple[list[str], list[str]]:
            attrs, keys = super().inputs()
            return attrs, [*keys, "tenant_id"]

    stage = TenantDuration(input_residency=residency)
    data = {"audio_filepath": "source.wav", "waveform": torch.zeros(1, 16000), "sample_rate": 16000}
    assert not stage.validate_input(AudioTask(data=data))
    data["tenant_id"] = "tenant"
    assert stage.validate_input(AudioTask(data=data))


def test_compound_preserve_nested_contract_declares_child_condition_keys() -> None:
    from nemo_curator.stages.audio._agent._agent_registry import stage_params

    stage = PreserveByValueConditionsStage(
        [{"input_value_key": "score", "target_value": 3.5, "operator": "ge"}],
        items_key="candidates",
        drop_parent_if_empty=False,
    )
    contract = stage.describe()
    params = {param.name: param for param in stage_params(PreserveByValueConditionsStage)}

    assert contract.reads.data_keys == ["candidates"]
    assert contract.writes.data_keys == ["candidates"]
    assert contract.reads.segment_data_keys == ["score"]
    assert contract.writes.segment_data_keys == []
    assert contract.iteration_key == "candidates"
    assert contract.cardinality == "1:1 nested-list"
    assert contract.gates.per_row_independent is True
    assert params["items_key"].default is None
    assert params["drop_parent_if_empty"].default is True
    assert params["condition_logic"].default == "and"
    assert params["condition_logic"].choices == ["and", "or"]

    dropping_contract = PreserveByValueConditionsStage(
        [{"input_value_key": "score", "target_value": 3.5, "operator": "ge"}],
        items_key="candidates",
    ).describe()
    assert dropping_contract.cardinality == "filter"
    assert dropping_contract.iteration_key is None
    assert "one-level" in dropping_contract.description
    assert "AND" in dropping_contract.description

    or_contract = PreserveByValueConditionsStage(
        [{"input_value_key": "score", "target_value": 3.5, "operator": "ge"}],
        condition_logic="or",
    ).describe()
    assert "OR" in or_contract.description


def test_duration_preserves_legacy_file_key_alias(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    import soundfile

    soundfile.write(source, np.zeros(16000, dtype=np.float32), 16000)
    result = GetAudioDurationStage(audio_filepath_key="waveform").process_batch(
        [AudioTask(data={"waveform": str(source)})]
    )
    assert result[0].data["duration"] == 1.0


def test_writer_worker_missing_run_reservation_preserves_output(tmp_path: Path) -> None:
    writer = ManifestWriterStage(str(tmp_path / "output.jsonl"))
    writer.prepare_on_driver()
    worker = pickle.loads(pickle.dumps(writer))  # noqa: S301 - only locally serialized stage objects
    writer.process(AudioTask(data={"row": 1}))
    Path(f"{writer.output_path}._RUN").unlink()
    with pytest.raises(RuntimeError, match="driver-prepared"):
        worker.setup_on_node()
    assert Path(writer.output_path).read_text() == '{"row": 1}\n'


def test_condition_configuration_identity_cannot_follow_external_mutation() -> None:
    from nemo_curator.stages.audio.agent import pipeline_identity
    from nemo_curator.stages.audio.common import PreserveByValueConditionsStage

    config = {"duration": {"target_value": 1, "operator": "gt"}}
    first = PreserveByValueConditionsStage(config)
    identity = pipeline_identity([first])
    config["duration"]["target_value"] = 2
    second = PreserveByValueConditionsStage(config)
    inspected = first.conditions
    inspected[0]["target_value"] = 3
    assert pipeline_identity([first]) == identity
    assert pipeline_identity([second]) != identity
    assert len(first.process_batch([AudioTask(data={"duration": 1.5})])) == 1
    assert second.process_batch([AudioTask(data={"duration": 1.5})]) == []
