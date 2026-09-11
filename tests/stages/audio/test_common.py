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

"""Tests for common audio stages: GetAudioDurationStage, PreserveByValueStage,
ManifestReaderStage, ManifestReader, ManifestWriterStage, and ManifestCheckpointStage."""

import json
from itertools import product
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm import ALMDataBuilderStage, ALMDataOverlapStage
from nemo_curator.stages.audio.common import (
    GetAudioDurationStage,
    ManifestCheckpointStage,
    ManifestReader,
    ManifestReaderStage,
    ManifestWriterStage,
    PreserveByValueConditionsStage,
    PreserveByValueStage,
    ensure_mono,
    ensure_waveform_2d,
    load_audio_file,
    resolve_model_path,
    resolve_waveform_from_item,
)
from nemo_curator.tasks import AudioTask, FileGroupTask
from tests import FIXTURES_DIR

ALM_FIXTURES_DIR = FIXTURES_DIR / "audio" / "alm"


def _make_file_group_task(paths: list[str]) -> FileGroupTask:
    return FileGroupTask(dataset_name="test", data=paths)


# ---------------------------------------------------------------------------
# PreserveByValueStage
# ---------------------------------------------------------------------------


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


def test_compound_preserve_nested_contract_uses_only_top_level_container_key() -> None:
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
    assert contract.reads.segment_data_keys == []
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


# ---------------------------------------------------------------------------
# GetAudioDurationStage
# ---------------------------------------------------------------------------


def test_get_audio_duration_validate_input_valid() -> None:
    stage = GetAudioDurationStage()
    assert stage.validate_input(AudioTask(data={"audio_filepath": "/a.wav"})) is True


def test_get_audio_duration_validate_input_missing_column() -> None:
    stage = GetAudioDurationStage()
    assert stage.validate_input(AudioTask(data={"text": "hello"})) is False


def test_get_audio_duration_process_batch_raises_on_missing_column() -> None:
    stage = GetAudioDurationStage()
    stage.setup()
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([AudioTask(data={"text": "hello"})])


def test_get_audio_duration_success(tmp_path: Path) -> None:
    class FakeInfo:
        def __init__(self, frames: int, samplerate: int):
            self.frames = frames
            self.samplerate = samplerate

    fake_info = FakeInfo(frames=16000 * 2, samplerate=16000)
    with mock.patch("soundfile.info", return_value=fake_info):
        stage = GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration")
        stage.setup()
        entry = AudioTask(data={"audio_filepath": (tmp_path / "fake.wav").as_posix()})
        result = stage.process(entry)
        assert isinstance(result, AudioTask)
        assert result.data["duration"] == 2.0


def test_get_audio_duration_error_sets_minus_one(tmp_path: Path) -> None:
    with mock.patch("soundfile.info", side_effect=RuntimeError("bad file")):
        stage = GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration")
        stage.setup()
        entry = AudioTask(data={"audio_filepath": (tmp_path / "missing.wav").as_posix()})
        result = stage.process(entry)
        assert result.data["duration"] == -1.0


def test_get_audio_duration_waveform_residency() -> None:
    """input_residency='waveform' computes duration from samples/sample_rate (no file)."""
    import torch

    stage = GetAudioDurationStage(input_residency="waveform")
    stage.setup()
    result = stage.process(AudioTask(data={"waveform": torch.zeros(1, 16000 * 3), "sample_rate": 16000}))
    assert result.data["duration"] == 3.0


def test_get_audio_duration_auto_prefers_waveform() -> None:
    import torch

    stage = GetAudioDurationStage(input_residency="auto")
    stage.setup()
    result = stage.process(AudioTask(data={"waveform": torch.zeros(1, 16000), "sample_rate": 16000}))
    assert result.data["duration"] == 1.0


def test_get_audio_duration_default_rejects_waveform_only() -> None:
    """Regression: default residency is 'file'; a waveform-only task is not valid input."""
    import torch

    stage = GetAudioDurationStage()
    assert stage.input_residency == "file"
    assert stage.validate_input(AudioTask(data={"waveform": torch.zeros(1, 16000), "sample_rate": 16000})) is False
    assert stage.validate_input(AudioTask(data={"audio_filepath": "/a.wav"})) is True


def test_get_audio_duration_waveform_validate() -> None:
    import torch

    stage = GetAudioDurationStage(input_residency="waveform")
    assert stage.validate_input(AudioTask(data={"waveform": torch.zeros(1, 16000), "sample_rate": 16000})) is True
    assert stage.validate_input(AudioTask(data={"audio_filepath": "/a.wav"})) is False


# ---------------------------------------------------------------------------
# ManifestReaderStage
# ---------------------------------------------------------------------------


class TestManifestReaderStage:
    """Unit tests for ManifestReaderStage (low-level stage)."""

    def test_reads_single_manifest(self, tmp_path: Path) -> None:
        entries = [
            {"audio_filepath": "a.wav", "audio_sample_rate": 16000, "segments": []},
            {"audio_filepath": "b.wav", "audio_sample_rate": 22050, "segments": []},
        ]
        manifest = tmp_path / "input.jsonl"
        manifest.write_text("\n".join(json.dumps(e) for e in entries))

        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)]))

        assert len(result) == 2
        assert all(isinstance(r, AudioTask) for r in result)
        assert result[0].data["audio_filepath"] == "a.wav"
        assert result[1].data["audio_filepath"] == "b.wav"

    def test_worker_defaults(self) -> None:
        stage = ManifestReaderStage()
        assert stage.num_workers() == 1
        assert stage.ray_stage_spec()["is_fanout_stage"] is True
        assert stage.xenna_stage_spec() == {}

    def test_reads_multiple_manifests(self, tmp_path: Path) -> None:
        m1 = tmp_path / "m1.jsonl"
        m2 = tmp_path / "m2.jsonl"
        m1.write_text(json.dumps({"audio_filepath": "a.wav", "segments": []}))
        m2.write_text(json.dumps({"audio_filepath": "b.wav", "segments": []}))

        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(m1), str(m2)]))

        assert len(result) == 2
        paths = [r.data["audio_filepath"] for r in result]
        assert paths == ["a.wav", "b.wav"]

    def test_one_audio_entry_per_line(self, tmp_path: Path) -> None:
        entries = [{"audio_filepath": f"{i}.wav", "segments": []} for i in range(5)]
        manifest = tmp_path / "input.jsonl"
        manifest.write_text("\n".join(json.dumps(e) for e in entries))

        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)]))

        assert len(result) == 5
        for i, audio_entry in enumerate(result):
            assert isinstance(audio_entry, AudioTask)
            assert audio_entry.data["audio_filepath"] == f"{i}.wav"

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        manifest = tmp_path / "input.jsonl"
        manifest.write_text(
            json.dumps({"audio_filepath": "a.wav", "segments": []})
            + "\n\n  \n"
            + json.dumps({"audio_filepath": "b.wav", "segments": []})
            + "\n"
        )

        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)]))

        assert len(result) == 2

    def test_empty_manifest(self, tmp_path: Path) -> None:
        manifest = tmp_path / "empty.jsonl"
        manifest.write_text("")

        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)]))

        assert result == []

    def test_preserves_nested_data(self, tmp_path: Path) -> None:
        entry = {
            "audio_filepath": "a.wav",
            "audio_sample_rate": 16000,
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.2,
                    "speaker": "spk_0",
                    "metrics": {"bandwidth": 8000},
                }
            ],
        }
        manifest = tmp_path / "input.jsonl"
        manifest.write_text(json.dumps(entry))

        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)]))

        loaded = result[0].data
        assert loaded["segments"][0]["metrics"]["bandwidth"] == 8000
        assert loaded["segments"][0]["speaker"] == "spk_0"

    def test_duplicate_manifests_for_repeat(self, tmp_path: Path) -> None:
        manifest = tmp_path / "input.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": "a.wav", "segments": []}))

        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(manifest)] * 3))

        assert len(result) == 3
        assert all(r.data["audio_filepath"] == "a.wav" for r in result)


class TestManifestReaderDirectory:
    """Tests for directory-based manifest discovery."""

    @staticmethod
    def _nested_dir() -> Path:
        return ALM_FIXTURES_DIR / "nested_manifests"

    def test_reads_all_jsonl_from_directory(self) -> None:
        nested = self._nested_dir()
        all_files = sorted(str(p) for p in nested.rglob("*.jsonl"))
        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task(all_files))

        assert len(result) == 20  # 4 files x 5 entries each
        assert all(isinstance(r, AudioTask) for r in result)

    def test_reads_from_subdirectory_a(self) -> None:
        subdir = self._nested_dir() / "subdir_a"
        files = sorted(str(p) for p in subdir.glob("*.jsonl"))
        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task(files))

        assert len(result) == 10  # 2 files x 5 entries each

    def test_reads_from_subdirectory_b(self) -> None:
        subdir = self._nested_dir() / "subdir_b"
        files = sorted(str(p) for p in subdir.glob("*.jsonl"))
        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task(files))

        assert len(result) == 10  # 2 files x 5 entries each

    def test_composite_discovers_nested_directory(self) -> None:
        nested = self._nested_dir()
        composite = ManifestReader(manifest_path=str(nested))
        stages = composite.decompose()

        partitioner = stages[0]
        assert partitioner.file_paths == str(nested)
        assert partitioner.file_extensions == [".jsonl", ".json"]

    def test_ignores_non_jsonl_files(self) -> None:
        nested = self._nested_dir()
        txt_files = list(nested.rglob("*.txt"))
        assert len(txt_files) > 0, "Test setup: .txt file should exist"

        jsonl_files = sorted(str(p) for p in nested.rglob("*.jsonl"))
        for f in jsonl_files:
            assert not f.endswith(".txt")


class TestManifestReaderIntegration:
    """Integration tests using real sample fixtures."""

    def test_reads_sample_fixture(self) -> None:
        fixture = ALM_FIXTURES_DIR / "sample_input.jsonl"
        stage = ManifestReaderStage()
        result = stage.process(_make_file_group_task([str(fixture)]))

        assert len(result) == 5
        for audio_entry in result:
            assert isinstance(audio_entry, AudioTask)
            entry_data = audio_entry.data
            assert "audio_filepath" in entry_data
            assert "segments" in entry_data
            assert len(entry_data["segments"]) > 0

    def test_composite_end_to_end_with_directory(self) -> None:
        """End-to-end: ManifestReader composite with directory input through full pipeline."""
        nested = ALM_FIXTURES_DIR / "nested_manifests"

        pipeline = Pipeline(name="test_dir_e2e", description="Directory discovery end-to-end test")
        pipeline.add_stage(ManifestReader(manifest_path=str(nested)))
        pipeline.add_stage(
            ALMDataBuilderStage(
                target_window_duration=120.0,
                tolerance=0.1,
                min_sample_rate=16000,
                min_bandwidth=8000,
                min_speakers=2,
                max_speakers=5,
            )
        )
        pipeline.add_stage(ALMDataOverlapStage(overlap_percentage=50, target_duration=120.0))

        executor = XennaExecutor()
        results = pipeline.run(executor)

        output_entries = []
        for task in results or []:
            output_entries.append(task.data)

        assert len(output_entries) == 20  # 4 files x 5 entries
        total_windows = sum(len(e.get("filtered_windows", [])) for e in output_entries)
        assert total_windows == 100  # 25 per file x 4 files
        total_dur = sum(e.get("filtered_dur", 0) for e in output_entries)
        assert abs(total_dur - 12142.0) < 1.0


# ---------------------------------------------------------------------------
# ManifestWriterStage
# ---------------------------------------------------------------------------


class TestManifestWriterStage:
    """Unit tests for ManifestWriterStage."""

    def test_writes_entry_to_jsonl(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        task = AudioTask(
            data={"audio_filepath": "a.wav", "duration": 1.0},
            dataset_name="ds",
        )
        writer.process(task)

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["audio_filepath"] == "a.wav"

    def test_returns_audio_task(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        task = AudioTask(data={"x": 1}, dataset_name="ds")
        result = writer.process(task)

        assert isinstance(result, AudioTask)
        assert result.data == {"x": 1}
        assert result.dataset_name == "ds"

    def test_propagates_metadata_and_stage_perf(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        metadata = {"source_files": ["manifest.jsonl"]}
        stage_perf = [{"stage": "some_stage", "process_time": 0.5}]
        task = AudioTask(
            data={"x": 1},
            dataset_name="ds",
            _metadata=metadata,
            _stage_perf=stage_perf,
        )
        result = writer.process(task)

        assert result._metadata == metadata
        assert result._stage_perf == stage_perf

    def test_appends_across_multiple_process_calls(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        writer.process(AudioTask(data={"entry": 1}))
        writer.process(AudioTask(data={"entry": 2}))
        writer.process(AudioTask(data={"entry": 3}))

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3
        assert [json.loads(line)["entry"] for line in lines] == [1, 2, 3]

    def test_setup_truncates_existing_file(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        out.write_text('{"old": "data"}\n')

        writer = ManifestWriterStage(output_path=str(out))
        writer.setup()

        assert out.read_text() == ""

    def test_setup_on_node_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "deep" / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()

        assert out.parent.exists()

    def test_handles_unicode_content(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        task = AudioTask(data={"text": "日本語テスト", "speaker": "Ñoño"})
        writer.process(task)

        loaded = json.loads(out.read_text().strip())
        assert loaded["text"] == "日本語テスト"
        assert loaded["speaker"] == "Ñoño"

    def test_preserves_nested_structures(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()

        entry = {
            "audio_filepath": "a.wav",
            "windows": [
                {"segments": [{"start": 0.0, "end": 5.0, "speaker": "spk_0"}]},
            ],
            "stats": {"lost_bw": 3, "lost_sr": 0},
        }
        task = AudioTask(data=entry)
        writer.process(task)

        loaded = json.loads(out.read_text().strip())
        assert loaded["windows"][0]["segments"][0]["speaker"] == "spk_0"
        assert loaded["stats"]["lost_bw"] == 3

    def test_num_workers_returns_one(self, tmp_path: Path) -> None:
        writer = ManifestWriterStage(output_path=str(tmp_path / "out.jsonl"))
        assert writer.num_workers() == 1

    def test_xenna_stage_spec(self, tmp_path: Path) -> None:
        writer = ManifestWriterStage(output_path=str(tmp_path / "out.jsonl"))
        assert writer.xenna_stage_spec() == {}


class TestManifestCheckpointStage:
    """Focused unit tests for the reusable metadata checkpoint."""

    def test_setup_atomically_refuses_to_overwrite_existing_checkpoint(self, tmp_path: Path) -> None:
        out = tmp_path / "checkpoint.jsonl"
        out.write_bytes(b"retained artifact\n")
        checkpoint = ManifestCheckpointStage(output_path=str(out))

        with pytest.raises(FileExistsError, match="refuses to overwrite"):
            checkpoint.setup()

        assert out.read_bytes() == b"retained artifact\n"
        assert not Path(f"{out}._RETRY_OWNER").exists()

    def test_setup_refuses_stale_completion_marker_without_leaving_output(self, tmp_path: Path) -> None:
        out = tmp_path / "checkpoint.jsonl"
        Path(f"{out}._COMPLETE").write_text("stale", encoding="utf-8")
        checkpoint = ManifestCheckpointStage(output_path=str(out))

        with pytest.raises(FileExistsError, match="completion marker"):
            checkpoint.setup()

        assert not out.exists()

    def test_retry_reset_removes_only_owned_partial_and_reserves_cleanly(
        self,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "checkpoint.jsonl"
        checkpoint = ManifestCheckpointStage(output_path=str(out))
        checkpoint.setup()
        checkpoint.process(AudioTask(data={"attempt": 1}))

        checkpoint.reset_for_retry()

        assert not out.exists()
        assert checkpoint._checkpoint_rows_written == 0
        assert checkpoint._checkpoint_bytes_written == 0
        checkpoint.setup()
        checkpoint.process(AudioTask(data={"attempt": 2}))
        assert out.read_text(encoding="utf-8") == '{"attempt": 2}\n'

    def test_retry_reset_refuses_completed_checkpoint(self, tmp_path: Path) -> None:
        out = tmp_path / "checkpoint.jsonl"
        checkpoint = ManifestCheckpointStage(output_path=str(out))
        checkpoint.setup()
        checkpoint.process(AudioTask(data={"retained": True}))
        before = out.read_bytes()
        Path(f"{out}._COMPLETE").write_text("complete", encoding="utf-8")

        with pytest.raises(FileExistsError, match="completion marker"):
            checkpoint.reset_for_retry()

        assert out.read_bytes() == before

    def test_retry_reset_refuses_preexisting_unowned_checkpoint(
        self,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "checkpoint.jsonl"
        out.write_text("user file\n", encoding="utf-8")
        checkpoint = ManifestCheckpointStage(output_path=str(out))

        with pytest.raises(FileExistsError, match="did not reserve"):
            checkpoint.reset_for_retry()

        assert out.read_text(encoding="utf-8") == "user file\n"

    def test_retry_reset_refuses_replaced_reservation(self, tmp_path: Path) -> None:
        out = tmp_path / "checkpoint.jsonl"
        checkpoint = ManifestCheckpointStage(output_path=str(out))
        checkpoint.setup()
        out.unlink()
        out.write_text("replacement\n", encoding="utf-8")

        with pytest.raises(FileExistsError, match="no longer its exact reservation"):
            checkpoint.reset_for_retry()

        assert out.read_text(encoding="utf-8") == "replacement\n"

    def test_configured_contract_is_audio_pass_through_with_checkpoint_gates(self, tmp_path: Path) -> None:
        from nemo_curator.stages.audio._agent._agent_registry import build_contract

        checkpoint = ManifestCheckpointStage(output_path=str(tmp_path / "checkpoint.jsonl"))
        contract = build_contract(checkpoint)
        params = {parameter.name: parameter for parameter in contract.params}

        assert checkpoint.name == "manifest_checkpoint"
        assert checkpoint.name != ManifestWriterStage(output_path=str(tmp_path / "manifest.jsonl")).name
        assert checkpoint.num_workers() == 1
        assert contract.accepts_task_type == "AudioTask"
        assert contract.produces_task_type == "AudioTask"
        assert contract.gates.writes_to_disk is True
        assert contract.gates.output_path_params == ["output_path"]
        assert contract.gates.requires_serializable_input is True
        assert contract.gates.per_row_independent is True
        assert contract.gates.lifecycle_side_effects is True
        assert params["output_path"].required is True
        assert "max_bytes" not in params
        assert params["retention_sec"].default == 0
        assert params["owner"].choices == ["user", "project"]
        assert params["planning_provenance"].default is None

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"retention_sec": -1}, "retention_sec"),
            ({"owner": "nobody"}, "owner"),
            ({"output_path": "s3://bucket/checkpoint.jsonl"}, "plain local path"),
            ({"output_path": "file:///tmp/checkpoint.jsonl"}, "plain local path"),
        ],
    )
    def test_rejects_invalid_checkpoint_policy(
        self,
        tmp_path: Path,
        kwargs: dict[str, object],
        message: str,
    ) -> None:
        params = {"output_path": str(tmp_path / "checkpoint.jsonl"), **kwargs}
        with pytest.raises(ValueError, match=message):
            ManifestCheckpointStage(**params)


class TestManifestWriterRoundTrip:
    """Round-trip test: write with writer, read back and verify."""

    def test_reader_writer_round_trip(self, sample_entries: list[dict], tmp_path: Path) -> None:
        out = tmp_path / "round_trip.jsonl"

        writer = ManifestWriterStage(output_path=str(out))
        writer.setup_on_node()
        writer.setup()
        for _i, entry in enumerate(sample_entries):
            task = AudioTask(data=entry)
            writer.process(task)

        reader = ManifestReaderStage()
        result = reader.process(FileGroupTask(dataset_name="rt", data=[str(out)]))

        assert len(result) == len(sample_entries)
        for orig, audio_entry in zip(sample_entries, result, strict=True):
            loaded = audio_entry.data
            assert loaded["audio_filepath"] == orig["audio_filepath"]
            assert len(loaded["segments"]) == len(orig["segments"])


def test_ensure_waveform_2d_from_tensor() -> None:
    assert ensure_waveform_2d(torch.randn(16000)).shape == (1, 16000)


def test_ensure_waveform_2d_from_numpy() -> None:
    assert ensure_waveform_2d(np.random.default_rng(0).standard_normal(16000).astype(np.float32)).dim() == 2


def test_ensure_mono() -> None:
    assert ensure_mono(torch.randn(2, 16000)).shape == (1, 16000)


def test_load_audio_file(tmp_path: Path) -> None:
    fake_data = np.random.default_rng(0).standard_normal(32000).astype(np.float32)
    with mock.patch("nemo_curator.stages.audio.common.soundfile.read", return_value=(fake_data, 16000)):
        waveform, sr = load_audio_file(str(tmp_path / "test.wav"), mono=True)
        assert sr == 16000
        assert waveform.shape == (1, 32000)


def test_resolve_waveform_with_data() -> None:
    item = {"waveform": torch.randn(1, 16000), "sample_rate": 16000}
    result = resolve_waveform_from_item(item, "test")
    assert result is not None
    assert result[1] == 16000


def test_resolve_waveform_from_file(tmp_path: Path) -> None:
    wav_path = str(tmp_path / "audio.wav")
    Path(wav_path).write_bytes(b"\x00")
    with mock.patch("nemo_curator.stages.audio.common.load_audio_file", return_value=(torch.randn(1, 16000), 16000)):
        item = {"audio_filepath": wav_path}
        result = resolve_waveform_from_item(item, "test")
        assert result is not None
        assert item["waveform"] is not None


def test_resolve_waveform_returns_none_when_missing() -> None:
    assert resolve_waveform_from_item({}, "test") is None
    assert resolve_waveform_from_item({"audio_filepath": "/nonexistent.wav"}, "test") is None
    assert resolve_waveform_from_item({"waveform": torch.randn(16000)}, "test") is None


def test_resolve_model_path(tmp_path: Path) -> None:
    assert resolve_model_path("/abs/model.bin", __file__, "sub") == "/abs/model.bin"

    module_dir = tmp_path / "sub"
    module_dir.mkdir()
    (module_dir / "model.bin").write_bytes(b"\x00")
    result = resolve_model_path("model.bin", str(tmp_path / "ref.py"), "sub")
    assert result == str(module_dir / "model.bin")


# Lifted from tests/stages/audio/test_agent_simulation_pipelines.py: ManifestWriterStage
# lives in common.py, and this was its only truncate-on-rerun coverage.
def test_agent_manifest_writer_truncates_on_setup(tmp_path: Path) -> None:
    """A fresh run (setup) truncates the output so reruns do not accumulate duplicates."""
    out_path = tmp_path / "manifest.jsonl"
    writer = ManifestWriterStage(output_path=str(out_path))
    task = AudioTask(dataset_name="t", data={"audio_filepath": "src.wav", "text": "row"})

    writer.setup()
    writer.process(task)
    writer.process(task)
    assert len(out_path.read_text(encoding="utf-8").strip().splitlines()) == 2  # appends within a run

    writer.setup()  # new run truncates
    writer.process(task)
    assert len(out_path.read_text(encoding="utf-8").strip().splitlines()) == 1
