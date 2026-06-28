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

from pathlib import Path

import pytest

from nemo_curator.stages.audio import common as common_module
from nemo_curator.stages.audio.common import (
    GlobalSegmentAssemblerStage,
    _GlobalSegmentAssemblyState,
    _GlobalSegmentParentDataStore,
)
from nemo_curator.tasks import AudioTask


def _segment_data(idx: int, text: str, *, parent_source_index: int = 0, parent_id: str = "manifest:0:0") -> dict:
    return {
        "audio_filepath": "/local/audio.wav",
        "duration": 10.0,
        "num_samples": 160000,
        "waveform_ref": f"ref-{idx}",
        "qwen3_prediction_s1": text,
        "qwen3_prediction_s2": f"s2-{idx}",
        "segment_start_s": float(idx * 10),
        "segment_duration_s": 10.0,
        "_curator_payload_bytes": 640000,
        "_curator_payload_producer_node_id": f"node-{idx}",
        "_curator_segment_parent_id": parent_id,
        "_curator_segment_idx": idx,
        "_curator_segment_count": 2,
        "_curator_segment_parent_duration_s": 20.0,
        "_curator_segment_parent_num_samples": 320000,
        "_curator_segment_parent_source_index": parent_source_index,
    }


def _parent_data() -> dict:
    return {
        "audio_filepath": "/local/audio.wav",
        "duration": 20.0,
        "num_samples": 320000,
        "source_lang": "en",
        "text_prompt": "original prompt",
        "speaker_id": "speaker-a",
        "audio_item_id": "audio",
    }


def test_parent_store_delete_many_releases_completed_parents() -> None:
    store = _GlobalSegmentParentDataStore()
    store.put_many({"parent-a": {"value": 1}, "parent-b": {"value": 2}})

    assert store.delete_many(["parent-a", "missing"]) == 1
    assert store.get_parent("parent-a") is None
    assert store.get_parent("parent-b") == {"value": 2}


def test_segment_assembly_reports_parent_completion_without_changing_default_contract() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1"],
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
    )

    ready, completed_parent_id = state.add_segment(
        parent_id="manifest:0:0",
        segment_idx=0,
        segment_count=1,
        data={**_segment_data(0, "hello"), "_curator_segment_count": 1},
        metadata={},
        stage_perf=[],
        include_completion=True,
    )

    assert completed_parent_id == "manifest:0:0"
    assert ready[0]["data"]["qwen3_prediction_s1"] == "hello"


def test_segment_assembly_waits_for_all_segments_and_strips_payload_refs() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1", "qwen3_prediction_s2"],
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
    )

    assert (
        state.add_segment(
            parent_id="manifest:0:0",
            segment_idx=1,
            segment_count=2,
            data=_segment_data(1, "world"),
            metadata={"m": 1},
            stage_perf=[],
        )
        == []
    )
    assembled_items = state.add_segment(
        parent_id="manifest:0:0",
        segment_idx=0,
        segment_count=2,
        data=_segment_data(0, "hello"),
        metadata={"m": 1},
        stage_perf=[],
    )

    assert len(assembled_items) == 1
    assembled = assembled_items[0]
    data = assembled["data"]
    assert data["qwen3_prediction_s1"] == "hello world"
    assert data["qwen3_prediction_s2"] == "s2-0 s2-1"
    assert data["duration"] == 20.0
    assert data["num_samples"] == 320000
    assert "waveform_ref" not in data
    assert "segment_start_s" not in data
    assert "_curator_payload_bytes" not in data
    assert "_curator_payload_producer_node_id" not in data
    assert not any(key.startswith("_curator_segment_") for key in data)


def test_segment_assembly_restores_parent_fields_from_parent_store_data() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1", "qwen3_prediction_s2"],
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
    )

    assert (
        state.add_segment(
            parent_id="manifest:0:0",
            segment_idx=1,
            segment_count=2,
            data={**_segment_data(1, "world"), "text_prompt": "original prompt", "speaker_id": "speaker-a"},
            metadata={},
            stage_perf=[],
            parent_data=_parent_data(),
        )
        == []
    )
    assembled_items = state.add_segment(
        parent_id="manifest:0:0",
        segment_idx=0,
        segment_count=2,
        data={**_segment_data(0, "hello"), "speaker_id": "speaker-a"},
        metadata={},
        stage_perf=[],
        parent_data=_parent_data(),
    )

    data = assembled_items[0]["data"]
    assert data["qwen3_prediction_s1"] == "hello world"
    assert data["text_prompt"] == "original prompt"
    assert data["speaker_id"] == "speaker-a"


def test_segment_assembly_rejects_parent_key_collision_by_default() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1"],
        field_merge_strategies={"qwen3_prediction_s2": "drop"},
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
    )

    with pytest.raises(ValueError, match="overwrite"):
        state.add_segment(
            parent_id="manifest:0:0",
            segment_idx=0,
            segment_count=1,
            data={**_segment_data(0, "hello"), "speaker_id": "segment-speaker"},
            metadata={},
            stage_perf=[],
            parent_data=_parent_data(),
        )


def test_segment_assembly_allows_declared_parent_key_overwrite() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1", "qwen3_prediction_s2"],
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
        overwrite_keys=["speaker_id"],
    )

    assert (
        state.add_segment(
            parent_id="manifest:0:0",
            segment_idx=0,
            segment_count=2,
            data={**_segment_data(0, "hello"), "speaker_id": "segment-speaker"},
            metadata={},
            stage_perf=[],
            parent_data=_parent_data(),
        )
        == []
    )
    assembled_items = state.add_segment(
        parent_id="manifest:0:0",
        segment_idx=1,
        segment_count=2,
        data={**_segment_data(1, "world"), "speaker_id": "segment-speaker"},
        metadata={},
        stage_perf=[],
        parent_data=_parent_data(),
    )

    data = assembled_items[0]["data"]
    assert data["speaker_id"] == "segment-speaker"
    assert data["qwen3_prediction_s1"] == "hello world"


def test_segment_assembly_rejects_conflicting_declared_parent_key_overwrites() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1", "qwen3_prediction_s2"],
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
        overwrite_keys=["speaker_id"],
    )

    assert (
        state.add_segment(
            parent_id="manifest:0:0",
            segment_idx=0,
            segment_count=2,
            data={**_segment_data(0, "hello"), "speaker_id": "speaker-a"},
            metadata={},
            stage_perf=[],
            parent_data=_parent_data(),
        )
        == []
    )
    with pytest.raises(ValueError, match="conflicting segment values"):
        state.add_segment(
            parent_id="manifest:0:0",
            segment_idx=1,
            segment_count=2,
            data={**_segment_data(1, "world"), "speaker_id": "speaker-b"},
            metadata={},
            stage_perf=[],
            parent_data=_parent_data(),
        )


def test_segment_assembly_can_require_parent_store_data() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1"],
        field_merge_strategies={"qwen3_prediction_s2": "drop"},
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
        require_parent_data=True,
    )

    with pytest.raises(RuntimeError, match="requires original parent data"):
        state.add_segment(
            parent_id="manifest:0:0",
            segment_idx=0,
            segment_count=1,
            data=_segment_data(0, "hello"),
            metadata={},
            stage_perf=[],
        )


def test_segment_assembly_passes_through_consistent_generated_segment_field() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1", "qwen3_prediction_s2"],
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
    )

    assembled_items = state.add_segment(
        parent_id="manifest:0:0",
        segment_idx=0,
        segment_count=1,
        data={**_segment_data(0, "hello"), "unconfigured_stage_output": 0.9},
        metadata={},
        stage_perf=[],
        parent_data=_parent_data(),
    )

    assert assembled_items[0]["data"]["unconfigured_stage_output"] == 0.9


def test_segment_assembly_rejects_conflicting_unconfigured_generated_field() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1", "qwen3_prediction_s2"],
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
    )

    assert (
        state.add_segment(
            parent_id="manifest:0:0",
            segment_idx=0,
            segment_count=2,
            data={**_segment_data(0, "hello"), "unconfigured_stage_output": 0.9},
            metadata={},
            stage_perf=[],
            parent_data=_parent_data(),
        )
        == []
    )
    with pytest.raises(ValueError, match="unconfigured generated field 'unconfigured_stage_output'"):
        state.add_segment(
            parent_id="manifest:0:0",
            segment_idx=1,
            segment_count=2,
            data={**_segment_data(1, "world"), "unconfigured_stage_output": 0.8},
            metadata={},
            stage_perf=[],
            parent_data=_parent_data(),
        )


def test_segment_assembly_allows_reader_helper_fields_from_segment_processing() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1"],
        field_merge_strategies={"qwen3_prediction_s2": "drop"},
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
    )

    assembled_items = state.add_segment(
        parent_id="manifest:0:0",
        segment_idx=0,
        segment_count=1,
        data={
            **_segment_data(0, "hello"),
            "audio_item_id": "audio",
            "sample_rate": 16000,
            "is_mono": True,
        },
        metadata={},
        stage_perf=[],
        parent_data=_parent_data(),
    )

    data = assembled_items[0]["data"]
    assert data["audio_item_id"] == "audio"
    assert data["sample_rate"] == 16000
    assert data["is_mono"] is True


def test_segment_assembly_allows_parent_key_modified_when_sent_to_segment_consumer() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1"],
        field_merge_strategies={"qwen3_prediction_s2": "drop"},
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
    )

    assembled_items = state.add_segment(
        parent_id="manifest:0:0",
        segment_idx=0,
        segment_count=1,
        data={
            **_segment_data(0, "hello"),
            "text_prompt": "normalized prompt",
            _GlobalSegmentAssemblyState._SEGMENT_INPUT_KEYS_FIELD: ("audio_filepath", "text_prompt"),
        },
        metadata={},
        stage_perf=[],
        parent_data=_parent_data(),
    )

    assert assembled_items[0]["data"]["text_prompt"] == "normalized prompt"


def test_segment_assembly_rejects_duplicate_segment() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1"],
        field_merge_strategies={"qwen3_prediction_s2": "drop"},
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
    )

    kwargs = {
        "parent_id": "manifest:0:0",
        "segment_idx": 0,
        "segment_count": 2,
        "data": _segment_data(0, "hello"),
        "metadata": {},
        "stage_perf": [],
    }
    state.add_segment(**kwargs)
    with pytest.raises(ValueError, match="Duplicate segment"):
        state.add_segment(**kwargs)


def test_segment_assembly_preserves_original_parent_order() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1"],
        field_merge_strategies={"qwen3_prediction_s2": "drop"},
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
    )

    assert (
        state.add_segment(
            parent_id="manifest:1:1",
            segment_idx=0,
            segment_count=2,
            data=_segment_data(0, "parent one first", parent_source_index=1, parent_id="manifest:1:1"),
            metadata={},
            stage_perf=[],
        )
        == []
    )
    assert (
        state.add_segment(
            parent_id="manifest:1:1",
            segment_idx=1,
            segment_count=2,
            data=_segment_data(1, "parent one second", parent_source_index=1, parent_id="manifest:1:1"),
            metadata={},
            stage_perf=[],
        )
        == []
    )

    assembled_items = state.add_segment(
        parent_id="manifest:0:0",
        segment_idx=0,
        segment_count=1,
        data={
            **_segment_data(0, "parent zero", parent_source_index=0, parent_id="manifest:0:0"),
            "_curator_segment_count": 1,
        },
        metadata={},
        stage_perf=[],
    )

    assert [item["source_index"] for item in assembled_items] == [0, 1]
    assert [item["data"]["qwen3_prediction_s1"] for item in assembled_items] == [
        "parent zero",
        "parent one first parent one second",
    ]


def test_segment_assembly_spills_ready_parents_without_losing_order(tmp_path: Path) -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1"],
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
        max_ready_parents_in_memory=0,
        spill_dir=str(tmp_path),
    )

    assert (
        state.add_segment(
            parent_id="manifest:1:1",
            segment_idx=0,
            segment_count=1,
            data={
                **_segment_data(0, "parent one", parent_source_index=1, parent_id="manifest:1:1"),
                "_curator_segment_count": 1,
            },
            metadata={},
            stage_perf=[],
        )
        == []
    )

    assembled_items = state.add_segment(
        parent_id="manifest:0:0",
        segment_idx=0,
        segment_count=1,
        data={
            **_segment_data(0, "parent zero", parent_source_index=0, parent_id="manifest:0:0"),
            "_curator_segment_count": 1,
        },
        metadata={},
        stage_perf=[],
    )

    assert [item["source_index"] for item in assembled_items] == [0, 1]
    assert [item["data"]["qwen3_prediction_s1"] for item in assembled_items] == ["parent zero", "parent one"]


def test_segment_assembly_completes_parent_with_dropped_segment_tombstone() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1"],
        field_merge_strategies={"qwen3_prediction_s2": "drop"},
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
    )

    assert (
        state.add_segment(
            parent_id="manifest:0:0",
            segment_idx=0,
            segment_count=2,
            data=_segment_data(0, "kept text"),
            metadata={},
            stage_perf=[],
        )
        == []
    )
    assembled_items = state.add_segment(
        parent_id="manifest:0:0",
        segment_idx=1,
        segment_count=2,
        data={
            **_segment_data(1, ""),
            "_skip_me": "dropped_segment_row",
            "_curator_segment_dropped": True,
            "_curator_segment_dropped_by_stage": "ASRStage",
        },
        metadata={},
        stage_perf=[],
    )

    assert len(assembled_items) == 1
    assert assembled_items[0]["data"]["qwen3_prediction_s1"] == "one or more intermediate segments dropped by ASRStage"
    assert "_skip_me" not in assembled_items[0]["data"]


def test_segment_assembly_marks_outputs_when_all_segments_drop_with_consumer_keys() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=["qwen3_prediction_s1"],
        skip_me_key="_skip_me",
        skip_me_keys=["stage_two_skip"],
        field_merge_strategies={"qwen3_prediction_s2": "drop"},
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
    )

    assert (
        state.add_segment(
            parent_id="manifest:0:0",
            segment_idx=0,
            segment_count=2,
            data={**_segment_data(0, ""), "stage_two_skip": "filtered"},
            metadata={},
            stage_perf=[],
        )
        == []
    )
    assembled_items = state.add_segment(
        parent_id="manifest:0:0",
        segment_idx=1,
        segment_count=2,
        data={
            **_segment_data(1, ""),
            "_curator_segment_dropped": True,
            "_curator_segment_dropped_by_stage": "ASRStage",
        },
        metadata={},
        stage_perf=[],
    )

    assert len(assembled_items) == 1
    assert (
        assembled_items[0]["data"]["qwen3_prediction_s1"]
        == "one or more intermediate segments dropped by stage_two_skip, ASRStage"
    )
    assert "_skip_me" not in assembled_items[0]["data"]


def test_segment_assembly_supports_structured_field_merge_strategies() -> None:
    state = _GlobalSegmentAssemblyState(
        text_keys_to_join=[],
        field_merge_strategies={
            "qwen3_prediction_s1": "join_text",
            "confidence": "list",
            "token_count": "sum",
            "metadata": "dict_merge",
            "temporary": "drop",
            "qwen3_prediction_s2": "drop",
        },
        skip_me_key="_skip_me",
        waveform_key="waveform",
        waveform_ref_key="waveform_ref",
        duration_key="duration",
        num_samples_key="num_samples",
        segment_start_key="segment_start_s",
        segment_duration_key="segment_duration_s",
    )

    first = {
        **_segment_data(0, "hello"),
        "confidence": 0.7,
        "token_count": 3,
        "metadata": {"a": 1},
        "temporary": "remove-me",
    }
    second = {
        **_segment_data(1, "world"),
        "confidence": 0.8,
        "token_count": 4,
        "metadata": {"b": 2},
        "temporary": "remove-me-too",
    }

    assert (
        state.add_segment(
            parent_id="manifest:0:0",
            segment_idx=0,
            segment_count=2,
            data=first,
            metadata={},
            stage_perf=[],
        )
        == []
    )
    assembled_items = state.add_segment(
        parent_id="manifest:0:0",
        segment_idx=1,
        segment_count=2,
        data=second,
        metadata={},
        stage_perf=[],
    )

    data = assembled_items[0]["data"]
    assert data["qwen3_prediction_s1"] == "hello world"
    assert data["confidence"] == [0.7, 0.8]
    assert data["token_count"] == 7
    assert data["metadata"] == {"a": 1, "b": 2}
    assert "temporary" not in data


def test_segment_assembler_passes_non_segment_rows_through() -> None:
    task = AudioTask(dataset_name="ds", data={"audio_filepath": "/local/audio.wav"})
    stage = GlobalSegmentAssemblerStage(text_keys_to_join=["qwen3_prediction_s1"])

    assert stage.process(task) is task


def test_segment_assembler_releases_completed_parent_data() -> None:
    deleted: list[list[str]] = []

    class _RemoteMethod:
        def remote(self, parent_ids: list[str]) -> int:
            deleted.append(parent_ids)
            return len(parent_ids)

    class _ParentStore:
        delete_many = _RemoteMethod()

    stage = GlobalSegmentAssemblerStage(text_keys_to_join=["qwen3_prediction_s1"])
    stage._parent_store_actor = _ParentStore()
    stage._parent_data_cache = {
        "parent-a": {"value": 1},
        "parent-b": {"value": 2},
    }
    stage._ray_get = lambda value: value  # type: ignore[method-assign]

    stage._release_parent_data(["parent-a", "parent-a"])

    assert deleted == [["parent-a"]]
    assert "parent-a" not in stage._parent_data_cache
    assert stage._parent_data_cache["parent-b"] == {"value": 2}


def test_global_segment_assembler_cleanup_kills_run_scoped_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[tuple[str, str | None]] = []
    monkeypatch.setattr(common_module, "_current_ray_namespace", lambda: "assembler-ns")
    monkeypatch.setattr(
        common_module,
        "_kill_named_ray_actor",
        lambda name, namespace=None: killed.append((name, namespace)) or True,
    )

    stage = GlobalSegmentAssemblerStage(actor_name_prefix="assembler", run_id="run/id")
    stage._actor = object()

    stage.cleanup_run_resources()

    assert killed == [
        ("assembler_run_id", "assembler-ns"),
        ("curator_global_segment_parent_store_run_id", "assembler-ns"),
    ]
    assert stage._actor is None
