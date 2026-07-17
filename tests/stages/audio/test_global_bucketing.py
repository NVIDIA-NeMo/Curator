# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import json
from pathlib import Path

import numpy as np
import pytest

from nemo_curator.stages.audio.global_bucketing import (
    PARENT_ID_KEY,
    PARENT_REPOSITORY_KEY,
    SEGMENT_COUNT_KEY,
    SEGMENT_DESCRIPTOR_KEY,
    SEGMENT_INDEX_KEY,
    AudioSegmentDescriptor,
    GlobalAudioManifestPlannerStage,
    GlobalAudioParentAssemblerStage,
    SoundFileSegmentDecoder,
    _local_parent_repository,
    _ParentAssemblyState,
    _ParentMetadataRepository,
)
from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy
from nemo_curator.tasks import AudioTask, EmptyTask


def _policy() -> BatchPolicy:
    return BatchPolicy(
        buckets_sec=[0.0, 2.0, 5.0],
        max_items_per_batch_by_bucket=[4, 2, 1],
        max_audio_sec_per_batch=5.0,
    )


def test_global_planner_reads_only_metadata_and_segments_full_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    rows = [
        {"audio_filepath": "/audio/a.wav", "duration": 7.0, "text": "a"},
        {"audio_filepath": "/audio/b.wav", "duration": 2.0, "text": "b"},
    ]
    manifest.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    stage = GlobalAudioManifestPlannerStage(
        manifest_path=str(manifest),
        max_model_input_duration_s=3.0,
        batch_policy=_policy(),
    )

    batches = stage.process(EmptyTask())
    segments = [item for batch in batches for item in batch.items]

    assert len(segments) == 4
    assert sum(batch.num_items for batch in batches) == 4
    assert all(isinstance(item.data[SEGMENT_DESCRIPTOR_KEY], AudioSegmentDescriptor) for item in segments)
    for segment in segments:
        descriptor = segment.data[SEGMENT_DESCRIPTOR_KEY]
        assert segment.data["audio_filepath"] == descriptor.audio_filepath
        assert segment.data["duration"] == descriptor.duration_s
        assert segment.data["segment_start_s"] == descriptor.start_s
        assert segment.data["segment_duration_s"] == descriptor.duration_s
    first_parent_segments = sorted(
        (segment for segment in segments if segment.data["audio_filepath"] == "/audio/a.wav"),
        key=lambda segment: segment.data["segment_start_s"],
    )
    assert [
        (
            segment.data["segment_start_s"],
            segment.data["segment_duration_s"],
        )
        for segment in first_parent_segments
    ] == [(0.0, 3.0), (3.0, 3.0), (6.0, 1.0)]
    assert all("waveform" not in item.data for item in segments)
    assert all("_curator_parent_data" not in item.data for item in segments)
    repository = _local_parent_repository(stage.repository_id)
    assert repository.parents == {"0:0": rows[0], "0:1": rows[1]}
    assert sorted(batch.sequence_index for batch in batches) == list(range(len(batches)))
    assert all(batch.validate() for batch in batches)


def test_soundfile_decoder_reads_only_requested_segment(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    path = tmp_path / "audio.wav"
    sample_rate = 10
    soundfile.write(path, np.arange(100, dtype=np.float32) / 100, sample_rate)
    descriptor = AudioSegmentDescriptor(audio_filepath=str(path), start_s=2.0, duration_s=3.0)

    waveform, decoded_rate = SoundFileSegmentDecoder().decode(descriptor)

    assert decoded_rate == sample_rate
    assert waveform.shape == (30,)


def _segment(
    *,
    repository_id: str,
    index: int,
    count: int = 2,
    text: str,
    parent_id: str = "0:0",
) -> AudioTask:
    return AudioTask(
        data={
            PARENT_ID_KEY: parent_id,
            PARENT_REPOSITORY_KEY: repository_id,
            SEGMENT_INDEX_KEY: index,
            SEGMENT_COUNT_KEY: count,
            "pred_text": text,
        }
    )


def test_parent_assembler_handles_reordered_batches_and_restores_parent_metadata() -> None:
    repository_id = "reordered-parent-test"
    repository = _local_parent_repository(repository_id)
    repository.parents.clear()
    parent = {
        "audio_filepath": "/audio/a.wav",
        "duration": 6.0,
        "speaker": "one",
        "nested": {"preserved": [1, 2, 3]},
    }
    repository.put_many({"0:0": parent})
    stage = GlobalAudioParentAssemblerStage(pred_text_key="pred_text")
    second = _segment(repository_id=repository_id, index=1, text="world")
    first = _segment(repository_id=repository_id, index=0, text="hello")

    assert stage.process_batch([second]) == []
    results = stage.process_batch([first])

    assert len(results) == 1
    assert results[0].data == {**parent, "pred_text": "hello world"}


def test_ray_repository_restores_parent_after_distributed_reordering(
    tmp_path: Path,
    shared_ray_client: None,  # noqa: ARG001
) -> None:
    manifest = tmp_path / "distributed.jsonl"
    parent = {
        "audio_filepath": "/audio/distributed.wav",
        "duration": 6.0,
        "speaker": "distributed",
        "nested": {"preserved": True},
    }
    manifest.write_text(f"{json.dumps(parent)}\n", encoding="utf-8")
    planner = GlobalAudioManifestPlannerStage(
        manifest_path=str(manifest),
        max_model_input_duration_s=3.0,
        batch_policy=_policy(),
    )
    segments = [item for batch in planner.process(EmptyTask()) for item in batch.items]
    for segment, text in zip(segments, ("hello", "world"), strict=True):
        segment.data["pred_text"] = text

    results = GlobalAudioParentAssemblerStage().process_batch(list(reversed(segments)))

    assert len(results) == 1
    assert results[0].data == {**parent, "pred_text": "hello world"}
    assert all("_curator_parent_data" not in segment.data for segment in segments)


def test_parent_assembler_duplicate_delivery_emits_parent_exactly_once() -> None:
    repository_id = "duplicate-parent-test"
    repository = _local_parent_repository(repository_id)
    repository.parents.clear()
    repository.put_many({"0:0": {"audio_filepath": "/audio/a.wav", "duration": 6.0}})
    stage = GlobalAudioParentAssemblerStage()
    first = _segment(repository_id=repository_id, index=0, text="hello")
    second = _segment(repository_id=repository_id, index=1, text="world")

    assert stage.process_batch([first]) == []
    results = stage.process_batch([second, second])
    assert len(results) == 1
    assert stage.process_batch([second]) == []
    assert stage.process_batch([first]) == []


def test_parent_assembly_replays_unacknowledged_retry_then_honors_ack() -> None:
    repository = _ParentMetadataRepository()
    repository.put_many({"0:0": {"audio_filepath": "/audio/a.wav", "duration": 6.0}})
    state = _ParentAssemblyState("pred_text", repository)
    first = _segment(repository_id="unused", index=0, text="hello")
    second = _segment(repository_id="unused", index=1, text="world")

    assert state.add_many([first]) == []
    initial = state.add_many([second])
    replay = state.add_many([second])
    operation_id = state.operation_id(second)
    state.add_many([], acknowledge_operation_ids=(operation_id,))

    assert len(initial) == 1
    assert replay[0].data == initial[0].data
    assert state.add_many([second]) == []


def test_parent_assembly_reports_missing_segments() -> None:
    repository = _ParentMetadataRepository()
    repository.put_many({"0:0": {"audio_filepath": "/audio/a.wav", "duration": 9.0}})
    state = _ParentAssemblyState("pred_text", repository)
    state.add_many([_segment(repository_id="unused", index=0, count=3, text="hello")])

    assert state.missing_segments() == {"0:0": (1, 2)}
    with pytest.raises(RuntimeError, match="Incomplete parent segment sets"):
        state.assert_complete()


def test_parent_assembly_rejects_conflicting_segment_retries_and_counts() -> None:
    repository = _ParentMetadataRepository()
    repository.put_many({"0:0": {"audio_filepath": "/audio/a.wav", "duration": 9.0}})
    state = _ParentAssemblyState("pred_text", repository)
    state.add_many([_segment(repository_id="unused", index=0, text="hello")])

    with pytest.raises(ValueError, match="Conflicting retry"):
        state.add_many([_segment(repository_id="unused", index=0, text="different")])
    with pytest.raises(ValueError, match="Conflicting segment counts"):
        state.add_many([_segment(repository_id="unused", index=1, count=3, text="world")])
