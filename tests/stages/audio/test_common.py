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
# ruff: noqa: ANN001, ANN202, C901

"""Tests for common audio stages: GetAudioDurationStage, PreserveByValueStage,
ManifestReaderStage, ManifestReader, and ManifestWriterStage."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.pipeline.payload_refs import PayloadRef
from nemo_curator.stages.audio.alm import ALMDataBuilderStage, ALMDataOverlapStage
from nemo_curator.stages.audio.common import (
    GetAudioDurationStage,
    ManifestReader,
    ManifestReaderStage,
    ManifestWriterStage,
    PreserveByValueStage,
    _GlobalSegmentAssemblyState,
    ensure_mono,
    ensure_waveform_2d,
    load_audio_file,
    resolve_model_path,
    resolve_waveform_from_item,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.payload_lifecycle import PayloadReleaseStage
from nemo_curator.tasks import AudioTask, DispatchBatchTask, EmptyTask, FileGroupTask
from nemo_curator.tasks.task_terminals import (
    TERMINAL_COUNT_KEY,
    TERMINAL_GROUP_ID_KEY,
    TERMINAL_INDEX_KEY,
    TERMINAL_SOURCE_INDEX_KEY,
)
from tests import FIXTURES_DIR

ALM_FIXTURES_DIR = FIXTURES_DIR / "audio" / "alm"


def _make_file_group_task(paths: list[str]) -> FileGroupTask:
    return FileGroupTask(dataset_name="test", data=paths)


def _audio_task_with_id(task_id: str, **kwargs) -> AudioTask:
    task = AudioTask(**kwargs)
    task.task_id = task_id
    return task


@dataclass
class _DropTerminalIndexStage(ProcessingStage[AudioTask, AudioTask]):
    name: str = "drop_terminal_index"
    drop_index: int = 1
    skip_me_key: str = "_skip_me"
    _curator_preserves_terminal_tasks: bool = True
    _curator_tracks_payload_refs: bool = True
    _curator_payload_ref_key: str = "waveform_ref"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["qwen3_prediction_s1"]

    def process(self, task: AudioTask) -> AudioTask | None:
        if int(task.data[TERMINAL_INDEX_KEY]) == self.drop_index:
            return None
        task.data["qwen3_prediction_s1"] = "kept text"
        return task


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
    stage = PreserveByValueStage(input_value_key="v", target_value=10, operator="ge")
    assert len(stage.process_batch([AudioTask(data={"v": 9})])) == 0
    assert len(stage.process_batch([AudioTask(data={"v": 10})])) == 1
    assert len(stage.process_batch([AudioTask(data={"v": 11})])) == 1


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
        assert "_shard_total" not in result[0]._metadata
        assert "_shard_key" not in result[0]._metadata

    def test_uses_storage_options_when_opening_manifest(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        manifest = tmp_path / "input.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": "a.wav", "segments": []}))
        seen_kwargs: list[dict] = []

        class _LocalFS:
            def open(self, path: str, mode: str, encoding: str | None = None):
                return open(path, mode, encoding=encoding)

        def fake_url_to_fs(path: str, **kwargs):
            seen_kwargs.append(kwargs)
            return _LocalFS(), path

        monkeypatch.setattr("nemo_curator.stages.audio.common.url_to_fs", fake_url_to_fs)

        stage = ManifestReaderStage(storage_options={"profile": "private"})
        result = stage.process(_make_file_group_task([str(manifest)]))

        assert len(result) == 1
        assert seen_kwargs == [{"profile": "private"}]

    def test_worker_defaults(self) -> None:
        stage = ManifestReaderStage()
        assert stage.num_workers() == 1
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
        assert all("_shard_total" not in r._metadata for r in result)
        assert all("_shard_key" not in r._metadata for r in result)

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
        assert all("_shard_total" not in r._metadata for r in result)
        assert all("_shard_key" not in r._metadata for r in result)


class TestManifestReaderGlobalBucketing:
    """Unit tests for the metadata-global segment manifest planner."""

    @pytest.fixture(autouse=True)
    def _fake_parent_store_ray(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _RemoteMethod:
            def __init__(self, fn):
                self._fn = fn

            def remote(self, *args, **kwargs):
                return self._fn(*args, **kwargs)

        class _Actor:
            def __init__(self):
                self._parents = {}
                self.put_many = _RemoteMethod(self._put_many)

            def _put_many(self, items):
                self._parents.update(items)
                return len(items)

        class _RemoteFactory:
            def __init__(self, ray_module):
                self._ray_module = ray_module

            def options(self, **_kwargs):
                return self

            def remote(self):
                self._ray_module.actor = _Actor()
                return self._ray_module.actor

        class _RayModule:
            def __init__(self):
                self.actor = None

            @staticmethod
            def is_initialized():  # noqa: ANN205
                return True

            def get_actor(self, *_args, **_kwargs):
                if self.actor is None:
                    raise ValueError
                return self.actor

            def remote(self, _cls):
                return _RemoteFactory(self)

            @staticmethod
            def get(value):  # noqa: ANN205
                return value

        monkeypatch.setitem(sys.modules, "ray", _RayModule())

    @staticmethod
    def _write_manifest(tmp_path: Path, entries: list[dict]) -> Path:
        manifest = tmp_path / "input.jsonl"
        manifest.write_text("\n".join(json.dumps(entry) for entry in entries))
        return manifest

    @staticmethod
    def _flatten_dispatch_batches(batches: list[DispatchBatchTask]) -> list[AudioTask]:
        return [item for batch in batches for item in batch.items if isinstance(item, AudioTask)]

    def test_disabled_uses_regular_streaming_reader_decomposition(self, tmp_path: Path) -> None:
        manifest = self._write_manifest(
            tmp_path,
            [
                {"audio_filepath": "a.wav", "duration": 10.0},
                {"audio_filepath": "b.wav", "duration": 130.0},
                {"audio_filepath": "c.wav", "duration": 40.0},
            ],
        )
        reader = ManifestReader(
            manifest_path=str(manifest),
            enable_global_bucketing=False,
            max_inference_duration_s=60.0,
            buckets_sec=[0.0, 30.0, 60.0],
            max_items_per_batch_by_bucket=[4, 4, 4],
        )

        stages = reader.decompose()
        partitioned = stages[0].process(EmptyTask())
        result = stages[1].process(partitioned[0])

        assert len(stages) == 2
        assert isinstance(stages[1], ManifestReaderStage)
        assert [task.data["audio_filepath"] for task in result] == ["a.wav", "b.wav", "c.wav"]
        assert all("_curator_global_plan_order" not in task._metadata for task in result)
        assert "_curator_global_plan_order" not in result[0].data

    def test_enabled_uses_full_manifest_segment_plan(self, tmp_path: Path) -> None:
        manifest = self._write_manifest(
            tmp_path,
            [
                {"audio_filepath": "short.wav", "duration": 10.0},
                {"audio_filepath": "long_a.wav", "duration": 70.0},
                {"audio_filepath": "long_b.wav", "duration": 130.0},
                {"audio_filepath": "mid.wav", "duration": 40.0},
            ],
        )
        reader = ManifestReader(
            manifest_path=str(manifest),
            enable_global_bucketing=True,
            owner_stage="qwen_omni",
            max_inference_duration_s=60.0,
            buckets_sec=[0.0, 30.0, 60.0],
            max_items_per_batch_by_bucket=[4, 4, 4],
            annotate_segment_plan=True,
        )

        [stage] = reader.decompose()
        batches = stage.process(EmptyTask())
        result = self._flatten_dispatch_batches(batches)

        assert all(isinstance(batch, DispatchBatchTask) for batch in batches)
        assert all(batch.owner_stage == "qwen_omni" for batch in batches)
        assert all(batch.validate() for batch in batches)
        assert len(result) == 7
        assert sum(batch.num_items for batch in batches) == 7
        assert [batch.sequence_index for batch in batches] == list(range(len(batches)))
        assert all(batch.total_cost == pytest.approx(sum(batch.item_costs)) for batch in batches)
        assert [task._metadata["_curator_global_plan_order"] for task in result] == list(range(7))
        assert result[0]._metadata["_curator_global_owner_stage"] == "qwen_omni"
        long_b_segments = [task for task in result if task.data["audio_filepath"] == "long_b.wav"]
        assert len(long_b_segments) == 3
        assert [
            task.data["_curator_segment_idx"]
            for task in sorted(long_b_segments, key=lambda t: t.data["_curator_segment_idx"])
        ] == [0, 1, 2]
        assert [
            task.data["segment_duration_s"]
            for task in sorted(long_b_segments, key=lambda t: t.data["_curator_segment_idx"])
        ] == [60.0, 60.0, 10.0]
        assert [
            task.data["segment_start_s"]
            for task in sorted(long_b_segments, key=lambda t: t.data["_curator_segment_idx"])
        ] == [0.0, 60.0, 120.0]
        assert all(task.data["duration"] == task.data["segment_duration_s"] for task in result)
        assert all("_curator_segment_parent_id" in task.data for task in result)
        segment_plan = long_b_segments[0]._metadata["_curator_global_plan_segment_boundaries"]
        assert [segment["duration_s"] for segment in segment_plan] == [60.0, 60.0, 10.0]
        assert [segment["bucket_id"] for segment in segment_plan] == [2, 2, 0]

    def test_enabled_emits_slim_segment_rows_with_configured_inputs(self, tmp_path: Path) -> None:
        manifest = self._write_manifest(
            tmp_path,
            [
                {
                    "audio_filepath": "long.wav",
                    "duration": 70.0,
                    "source_lang": "en",
                    "text_prompt": "transcribe this",
                    "speaker_id": "speaker-a",
                },
            ],
        )
        reader = ManifestReader(
            manifest_path=str(manifest),
            enable_global_bucketing=True,
            owner_stage="qwen_omni",
            max_inference_duration_s=60.0,
            buckets_sec=[0.0, 30.0, 60.0],
            max_items_per_batch_by_bucket=[4, 4, 4],
            segment_input_keys=["audio_filepath", "text_prompt"],
        )

        [stage] = reader.decompose()
        batches = stage.process(EmptyTask())
        result = self._flatten_dispatch_batches(batches)

        assert len(result) == 2
        for task in result:
            assert task.data["audio_filepath"] == "long.wav"
            assert task.data["text_prompt"] == "transcribe this"
            assert task.data["_curator_segment_input_keys"] == ("audio_filepath", "text_prompt")
            assert "source_lang" not in task.data
            assert "speaker_id" not in task.data
            assert task.data["duration"] == task.data["segment_duration_s"]
        assert stage._parent_store_actor._parents["0:0:0"]["speaker_id"] == "speaker-a"

    def test_planner_drop_release_and_assembly_complete_with_tombstone(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manifest = self._write_manifest(
            tmp_path,
            [{"audio_filepath": "long.wav", "duration": 70.0, "source_lang": "en"}],
        )
        reader = ManifestReader(
            manifest_path=str(manifest),
            enable_global_bucketing=True,
            owner_stage="qwen_omni",
            max_inference_duration_s=60.0,
            buckets_sec=[0.0, 30.0, 60.0],
            max_items_per_batch_by_bucket=[4, 4, 4],
        )
        [planner] = reader.decompose()
        segments = self._flatten_dispatch_batches(planner.process(EmptyTask()))
        segments.sort(key=lambda task: int(task.data[TERMINAL_INDEX_KEY]))
        released: list[str] = []
        for index, task in enumerate(segments):
            task.data["waveform_ref"] = PayloadRef(
                payload_id=f"payload-{index}",
                owner_node_id="node",
                store_actor_name="store",
                admission_actor_name="admission",
                amount_bytes=1,
                sample_rate=16_000,
                num_samples=1,
            )

        def _record_release(payload_ref: PayloadRef) -> None:
            released.append(payload_ref.payload_id)

        monkeypatch.setattr("nemo_curator.pipeline.payload_refs.release_payload_ref", _record_release)
        monkeypatch.setattr("nemo_curator.stages.payload_lifecycle.release_payload_ref", _record_release)

        processed = BaseStageAdapter(_DropTerminalIndexStage()).process_batch(segments)
        released_tasks = PayloadReleaseStage().process_batch(processed)

        parent_id = str(released_tasks[0].data[TERMINAL_GROUP_ID_KEY])
        parent_data = planner._parent_store_actor._parents[parent_id]
        state = _GlobalSegmentAssemblyState(
            text_keys_to_join=["qwen3_prediction_s1"],
            skip_me_key="_skip_me",
            waveform_key="waveform",
            waveform_ref_key="waveform_ref",
            duration_key="duration",
            num_samples_key="num_samples",
            segment_start_key="segment_start_s",
            segment_duration_key="segment_duration_s",
            require_parent_data=True,
        )
        assembled: list[dict] = []
        for index, task in enumerate(released_tasks):
            assembled.extend(
                state.add_segment(
                    parent_id=parent_id,
                    segment_idx=int(task.data[TERMINAL_INDEX_KEY]),
                    segment_count=int(task.data[TERMINAL_COUNT_KEY]),
                    data=dict(task.data),
                    metadata=dict(task._metadata),
                    stage_perf=list(task._stage_perf),
                    parent_data=parent_data if index == 0 else None,
                )
            )

        assert released == ["payload-1", "payload-0"]
        assert len(assembled) == 1
        assert assembled[0]["source_index"] == int(released_tasks[0].data[TERMINAL_SOURCE_INDEX_KEY])
        assert (
            assembled[0]["data"]["qwen3_prediction_s1"]
            == "one or more intermediate segments dropped by drop_terminal_index"
        )

    def test_enabled_propagates_storage_options_to_global_manifest_open(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manifest = self._write_manifest(tmp_path, [{"audio_filepath": "a.wav", "duration": 10.0}])
        seen_kwargs: list[dict] = []

        class _LocalFS:
            def open(self, path: str, mode: str, encoding: str | None = None):
                return open(path, mode, encoding=encoding)

        def fake_url_to_fs(path: str, **kwargs):
            seen_kwargs.append(kwargs)
            return _LocalFS(), path

        monkeypatch.setattr("nemo_curator.stages.audio.common.url_to_fs", fake_url_to_fs)
        reader = ManifestReader(
            manifest_path=str(manifest),
            storage_options={"profile": "private"},
            enable_global_bucketing=True,
            owner_stage="qwen_omni",
            buckets_sec=[0.0, 30.0],
            max_items_per_batch_by_bucket=[1, 1],
        )

        [stage] = reader.decompose()
        result = stage.process(EmptyTask())

        assert len(result) == 1
        assert isinstance(result[0], DispatchBatchTask)
        assert len(result[0].items) == 1
        assert isinstance(result[0].items[0], AudioTask)
        assert seen_kwargs == [{"profile": "private"}]

    def test_enabled_preserves_each_ready_batch_as_one_atomic_task(self, tmp_path: Path) -> None:
        manifest = self._write_manifest(
            tmp_path,
            [
                {"audio_filepath": "a.wav", "duration": 10.0},
                {"audio_filepath": "b.wav", "duration": 11.0},
                {"audio_filepath": "c.wav", "duration": 12.0},
                {"audio_filepath": "d.wav", "duration": 13.0},
                {"audio_filepath": "e.wav", "duration": 14.0},
            ],
        )
        reader = ManifestReader(
            manifest_path=str(manifest),
            enable_global_bucketing=True,
            owner_stage="qwen_omni",
            max_inference_duration_s=60.0,
            buckets_sec=[0.0, 30.0, 60.0],
            max_items_per_batch_by_bucket=[2, 1, 1],
            max_audio_sec_per_batch=60.0,
        )

        [stage] = reader.decompose()
        batches = stage.process(EmptyTask())

        assert [len(batch.items) for batch in batches] == [2, 2, 1]
        assert [batch.total_cost for batch in batches] == pytest.approx([25.0, 21.0, 14.0])
        assert [[item.data["audio_filepath"] for item in batch.items] for batch in batches] == [
            ["c.wav", "d.wav"],
            ["a.wav", "b.wav"],
            ["e.wav"],
        ]

    def test_enabled_rejects_non_positive_ready_batch_count(self, tmp_path: Path) -> None:
        manifest = self._write_manifest(
            tmp_path,
            [{"audio_filepath": "a.wav", "duration": 60.0}],
        )
        reader = ManifestReader(
            manifest_path=str(manifest),
            enable_global_bucketing=True,
            owner_stage="qwen_omni",
            buckets_sec=[0.0, 30.0],
            max_items_per_batch_by_bucket=[1, 1],
            target_ready_batches_per_bucket=0,
        )

        with pytest.raises(ValueError, match="target_ready_batches_per_bucket must be > 0"):
            reader.decompose()


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
        writer = ManifestWriterStage(output_path=str(out), write_perf_stats=False)
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
        writer = ManifestWriterStage(output_path=str(out), write_perf_stats=False)
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

    def test_process_batch_drops_waveform_and_array_like_keys(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(
            output_path=str(out),
            write_perf_stats=False,
            drop_manifest_keys=("waveform",),
            drop_array_like_values=True,
        )
        writer.setup_on_node()
        writer.setup()

        returned = writer.process_batch(
            [
                _audio_task_with_id(
                    "t1",
                    data={
                        "audio_filepath": "a.wav",
                        "duration": 1.0,
                        "waveform": torch.zeros(1, 16000),
                        "embedding": np.zeros(4, dtype=np.float32),
                        "text": "hello",
                    },
                ),
                _audio_task_with_id("t2", data={"audio_filepath": "b.wav", "duration": 2.0}),
            ]
        )

        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert [row["audio_filepath"] for row in rows] == ["a.wav", "b.wav"]
        assert "waveform" not in rows[0]
        assert "embedding" not in rows[0]
        assert rows[0]["text"] == "hello"
        assert [task.task_id for task in returned] == ["", ""]

    def test_writes_perf_summary_during_process_batch(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        writer = ManifestWriterStage(output_path=str(out), write_perf_stats=True)
        writer.setup_on_node()
        writer.setup()

        writer.process_batch(
            [
                _audio_task_with_id("t1", data={"audio_filepath": "a.wav", "duration": 1.0}),
                _audio_task_with_id("t2", data={"audio_filepath": "b.wav", "duration": 2.0}),
            ]
        )

        summary = json.loads((tmp_path / "perf_summary.json").read_text(encoding="utf-8"))
        assert summary["total_utterances"] == 2
        assert summary["total_audio_seconds"] == 3.0
        writer_summary = summary["stages"]["manifest_writer"]
        assert writer_summary["total_items_processed"] == 2.0
        assert writer_summary["invocation_count"] == 1.0
        assert writer_summary["custom_metrics_sum"]["writer_items_processed"] == 2.0

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
