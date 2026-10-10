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

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import torch

from nemo_curator.backends.ray_data.adapter import RayDataStageAdapter
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.models.audio.vad.base import VADResult, VADSegment
from nemo_curator.stages.audio.inference.base import AdapterInferenceStage
from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class _RecordingAdapter:
    results: list[VADResult] | None = None
    items: list[dict[str, Any]] = field(default_factory=list)

    def detect_batch(self, items: list[dict[str, Any]]) -> list[VADResult]:
        self.items = items
        if self.results is not None:
            return self.results
        return [VADResult([VADSegment(0.25, 0.75)]) for _ in items]


def _task(
    task_id: str = "recording",
    *,
    samples: int = 16000,
    sample_rate: int = 16000,
    channels: int = 1,
) -> AudioTask:
    waveform = torch.arange(channels * samples, dtype=torch.float32).reshape(channels, samples)
    return AudioTask(
        task_id=task_id,
        dataset_name="dataset",
        data={"waveform": waveform, "sample_rate": sample_rate, "audio_filepath": f"/{task_id}.wav"},
        _metadata={"_shard_key": "catalog/shard"},
    )


def _stage(
    *, results: list[VADResult] | None = None, **kwargs: object
) -> tuple[VADSegmentationStage, _RecordingAdapter]:
    stage = VADSegmentationStage(**kwargs)
    adapter = _RecordingAdapter(results=results)
    stage._adapter = adapter
    return stage, adapter


def test_stage_uses_shared_adapter_lifecycle_and_preserves_defaults() -> None:
    stage = VADSegmentationStage()

    assert isinstance(stage, AdapterInferenceStage)
    assert stage.adapter_target == "nemo_curator.models.audio.vad.silero.SileroVADAdapter"
    assert stage.min_duration_sec == 2.0
    assert stage.max_duration_sec == 60.0
    assert stage.min_interval_ms == 500
    assert stage.threshold == 0.5
    assert stage.speech_pad_ms == 300
    assert stage.duration_key == "duration"
    assert stage.batch_size == 1
    assert stage.resources.gpus == 0.0


def test_current_main_positional_constructor_order_is_preserved() -> None:
    resources = Resources(cpus=2.0)
    stage = VADSegmentationStage(
        250,
        1.0,
        30.0,
        0.65,
        120,
        "samples",
        "rate",
        True,
        "LegacyVAD",
        2,
        resources,
    )

    assert stage.min_interval_ms == 250
    assert stage.min_duration_sec == 1.0
    assert stage.max_duration_sec == 30.0
    assert stage.threshold == 0.65
    assert stage.speech_pad_ms == 120
    assert stage.waveform_key == "samples"
    assert stage.sample_rate_key == "rate"
    assert stage.nested is True
    assert stage.name == "LegacyVAD"
    assert stage.batch_size == 2
    assert stage.resources is resources


def test_create_adapter_forwards_stage_options_and_adapter_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Factory:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    stage = VADSegmentationStage(adapter_kwargs={"backend": "onnx"}, threshold=0.65)
    monkeypatch.setattr(stage, "_adapter_class", lambda: _Factory)

    assert isinstance(stage._create_adapter(), _Factory)
    assert captured == {
        "backend": "onnx",
        "threshold": 0.65,
        "min_duration_sec": 2.0,
        "max_duration_sec": 60.0,
        "min_interval_ms": 500,
        "speech_pad_ms": 300,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": 1.1}, "threshold"),
        ({"min_duration_sec": -1}, "min_duration_sec"),
        ({"min_duration_sec": 2, "max_duration_sec": 2}, "max_duration_sec"),
        ({"max_duration_sec": float("nan")}, "max_duration_sec"),
        ({"max_duration_sec": float("-inf")}, "max_duration_sec"),
        ({"min_interval_ms": -1}, "min_interval_ms"),
        ({"speech_pad_ms": -1}, "speech_pad_ms"),
        ({"batch_size": 0}, "batch_size"),
        ({"adapter_kwargs": {"threshold": 0.4}}, "must not also appear"),
    ],
)
def test_invalid_options_are_rejected(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VADSegmentationStage(**kwargs)


def test_unlimited_max_duration_is_supported() -> None:
    stage = VADSegmentationStage(max_duration_sec=float("inf"))

    assert stage.max_duration_sec == float("inf")


def test_fanout_contract_and_ray_spec() -> None:
    stage = VADSegmentationStage()

    assert stage.inputs() == ([], [])
    assert stage.outputs()[1][:7] == [
        "waveform",
        "sample_rate",
        "start_ms",
        "end_ms",
        "segment_num",
        "duration",
        "original_file",
    ]
    assert stage.ray_stage_spec()[RayStageSpecKeys.IS_FANOUT_STAGE] is True


def test_nested_contract_is_not_fanout() -> None:
    stage = VADSegmentationStage(nested=True)

    assert stage.outputs() == ([], ["segments", "vad_empty", "read_error"])
    assert stage.ray_stage_spec() == {}


def test_fanout_rejects_multi_recording_batch_configuration() -> None:
    with pytest.raises(ValueError, match="fan-out mode requires batch_size=1"):
        VADSegmentationStage(batch_size=2)

    with pytest.raises(ValueError, match="fan-out mode requires batch_size=1"):
        VADSegmentationStage().with_(batch_size=2)


def test_fanout_rejects_ambiguous_manual_multi_recording_batch() -> None:
    stage, _ = _stage(results=[])

    with pytest.raises(ValueError, match="one input per process_batch"):
        stage.process_batch([_task("first"), _task("second")])


def test_nested_batch_delegates_mono_contiguous_source_rate_audio_and_preserves_task_order() -> None:
    stage, adapter = _stage(
        nested=True,
        results=[
            VADResult([VADSegment(0.0, 0.5)]),
            VADResult([VADSegment(0.25, 0.75)]),
        ],
    )
    first = _task("first", channels=2)
    second = _task("second", sample_rate=48000, samples=48000)
    expected_first_mono = first.data["waveform"].mean(0).numpy()

    result = stage.process_batch([first, second])

    assert [item["sample_rate"] for item in adapter.items] == [16000, 48000]
    assert [item["waveform"].ndim for item in adapter.items] == [1, 1]
    assert all(item["waveform"].dtype == np.float32 for item in adapter.items)
    assert all(item["waveform"].flags.c_contiguous for item in adapter.items)
    np.testing.assert_allclose(adapter.items[0]["waveform"], expected_first_mono)
    assert [task.data["segments"][0]["original_file"] for task in result] == ["/first.wav", "/second.wav"]
    assert [task.data["segments"][0]["start_ms"] for task in result] == [0, 250]


def test_ray_data_adapter_accepts_two_item_numpy_object_batch() -> None:
    stage, adapter = _stage(
        nested=True,
        batch_size=2,
        results=[
            VADResult([VADSegment(0.0, 0.5)]),
            VADResult([VADSegment(0.25, 0.75)]),
        ],
    )
    tasks = [_task("first"), _task("second")]
    ray_batch = np.empty(2, dtype=object)
    ray_batch[:] = tasks

    result = RayDataStageAdapter(stage)._process_batch_internal({"item": ray_batch})

    assert result["item"] == tasks
    assert len(adapter.items) == 2
    assert [task.data["segments"][0]["start_ms"] for task in result["item"]] == [0, 250]


def test_batch_validates_public_subclass_inputs_before_inference() -> None:
    class TenantVADSegmentationStage(VADSegmentationStage):
        def inputs(self) -> tuple[list[str], list[str]]:
            return [], ["tenant_id"]

    stage = TenantVADSegmentationStage()
    adapter = _RecordingAdapter()
    stage._adapter = adapter
    task = _task()

    assert not stage.validate_input(task)
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([task])
    assert adapter.items == []


def test_batch_dispatches_public_process_override_without_recursing() -> None:
    class CustomVADSegmentationStage(VADSegmentationStage):
        process_calls = 0

        def process(self, task: AudioTask) -> AudioTask | list[AudioTask]:
            self.process_calls += 1
            result = super().process(task)
            assert isinstance(result, AudioTask)
            result.data["custom_process"] = True
            return result

    stage = CustomVADSegmentationStage(nested=True)
    adapter = _RecordingAdapter(results=[VADResult([VADSegment(0.0, 0.5)])])
    stage._adapter = adapter
    task = _task()

    result = stage.process_batch([task])

    assert result == [task]
    assert stage.process_calls == 1
    assert result[0].data["custom_process"] is True
    assert len(adapter.items) == 1


def test_fanout_builds_segment_metadata_and_slices_original_channels() -> None:
    stage, _ = _stage(results=[VADResult([VADSegment(0.25, 0.75)])])
    task = _task(channels=2)
    task.data["language"] = "en"
    task.data["diar_segments"] = [
        {"start": 0.0, "end": 0.5, "speaker": "speaker_0"},
        {"start": 0.5, "end": 1.0, "speaker": "speaker_1"},
    ]

    result = stage.process(task)

    assert isinstance(result, list)
    assert len(result) == 1
    child = result[0]
    assert child.task_id == ""
    assert child.dataset_name == "dataset"
    assert child._metadata == task._metadata
    assert child._metadata is not task._metadata
    assert child.data["waveform"].shape == (2, 8000)
    assert child.data["start_ms"] == 250
    assert child.data["end_ms"] == 750
    assert child.data["duration"] == 0.5
    assert child.data["segment_num"] == 0
    assert child.data["num_speakers"] == 2
    assert child.data["language"] == "en"


def test_fanout_preserves_speaker_separation_count_for_interval_segments() -> None:
    stage, _ = _stage(results=[VADResult([VADSegment(0.25, 0.75)])])
    task = _task()
    task.data.update({"diar_segments": [(0.0, 0.5), (0.5, 1.0)], "num_speakers": 2})

    child = stage.process(task)[0]

    assert child.data["num_speakers"] == 2


def test_custom_duration_key_is_used_without_leaking_parent_duration() -> None:
    stage, _ = _stage(
        results=[VADResult([VADSegment(0.0, 0.5)])],
        duration_key="duration_sec",
    )
    task = _task()
    task.data["duration"] = 99.0

    child = stage.process(task)[0]

    assert child.data["duration_sec"] == 0.5
    assert "duration" not in child.data


def test_file_input_is_loaded_without_storing_parent_waveform(monkeypatch: pytest.MonkeyPatch) -> None:
    stage, adapter = _stage(results=[VADResult([VADSegment(0.0, 0.5)])])
    monkeypatch.setattr(
        stage,
        "_load_audio",
        lambda _path: (np.stack([np.ones(16000), np.zeros(16000)]), 16000),
    )
    task = AudioTask(task_id="file", data={"audio_filepath": "/audio/file.wav"})

    child = stage.process(task)[0]

    np.testing.assert_allclose(adapter.items[0]["waveform"], 0.5)
    assert child.data["waveform"].shape == (2, 8000)
    assert "waveform" not in task.data


@pytest.mark.parametrize(
    ("segments", "expect_empty"),
    [
        ([VADSegment(0.0, 0.5)], False),
        ([], True),
    ],
)
def test_nested_file_input_preserves_discovered_parent_sample_rate(
    monkeypatch: pytest.MonkeyPatch,
    segments: list[VADSegment],
    expect_empty: bool,
) -> None:
    stage, _ = _stage(nested=True, results=[VADResult(segments)])
    monkeypatch.setattr(stage, "_load_audio", lambda _path: (np.ones(22050, dtype=np.float32), 22050))
    task = AudioTask(task_id="file", data={"audio_filepath": "/audio/file.wav"})

    result = stage.process(task)

    assert result is task
    assert result.data["sample_rate"] == 22050
    assert "waveform" not in result.data
    assert (result.data.get("vad_empty") is True) is expect_empty
    if expect_empty:
        assert result.data["segments"] == []
    else:
        assert result.data["segments"][0]["sample_rate"] == 22050


def test_nested_mode_keeps_one_parent_and_stores_segment_dicts() -> None:
    stage, _ = _stage(
        nested=True,
        results=[VADResult([VADSegment(0.0, 0.25), VADSegment(0.5, 0.75)])],
    )
    task = _task()

    result = stage.process(task)

    assert result is task
    assert [segment["segment_num"] for segment in result.data["segments"]] == [0, 1]
    assert [segment["duration"] for segment in result.data["segments"]] == [0.25, 0.25]
    assert "waveform" not in result.data


def test_nested_no_speech_is_an_auditable_placeholder() -> None:
    stage, _ = _stage(nested=True, results=[VADResult([])])
    task = _task()

    result = stage.process(task)

    assert result is task
    assert result.data["vad_empty"] is True
    assert result.data["duration"] == 1.0
    assert "waveform" not in result.data
    assert result.data["segments"] == []


def test_fanout_no_speech_preserves_legacy_drop_behavior() -> None:
    stage, _ = _stage(results=[VADResult([])])

    assert stage.process(_task()) == []


def test_fanout_no_speech_can_emit_an_auditable_placeholder() -> None:
    stage, _ = _stage(emit_audit_placeholders=True, results=[VADResult([])])
    task = _task()

    result = stage.process(task)

    assert result == [task]
    assert task.data["vad_empty"] is True
    assert task.data["duration"] == 1.0
    assert "waveform" not in task.data


def test_read_error_passes_through_without_adapter_call() -> None:
    stage, adapter = _stage(emit_audit_placeholders=True)
    task = AudioTask(task_id="failed", data={"read_error": True, "waveform": torch.zeros(10)})

    result = stage.process(task)

    assert result == [task]
    assert adapter.items == []
    assert "waveform" not in task.data


def test_audio_preparation_error_is_auditable_when_requested() -> None:
    stage, adapter = _stage(emit_audit_placeholders=True)
    task = AudioTask(task_id="missing", data={"some_key": "value"})

    result = stage.process(task)

    assert result == [task]
    assert task.data["read_error"] is True
    assert adapter.items == []


def test_audio_preparation_error_can_fail_fast() -> None:
    stage, _ = _stage(fail_on_audio_error=True)

    with pytest.raises(RuntimeError, match="Failed to prepare VAD audio"):
        stage.process(AudioTask(task_id="missing", data={}))


def test_adapter_failures_propagate() -> None:
    class _BrokenAdapter:
        def detect_batch(self, _items: list[dict[str, Any]]) -> list[VADResult]:
            message = "provider failed"
            raise RuntimeError(message)

    stage = VADSegmentationStage()
    stage._adapter = _BrokenAdapter()

    with pytest.raises(RuntimeError, match="provider failed"):
        stage.process(_task())


def test_nested_batch_isolates_per_recording_adapter_errors() -> None:
    stage, _ = _stage(
        nested=True,
        results=[
            VADResult([VADSegment(0.0, 0.25)]),
            VADResult([], error="RuntimeError: provider rejected this recording"),
            VADResult([VADSegment(0.5, 0.75)]),
        ],
    )
    tasks = [_task("good-first"), _task("bad"), _task("good-last")]

    results = stage.process_batch(tasks)

    assert results == tasks
    assert results[0].data["segments"][0]["start_ms"] == 0
    assert results[2].data["segments"][0]["start_ms"] == 500
    assert results[1].data["read_error"] is True
    assert results[1].data["segments"] == []
    assert "waveform" not in results[1].data
    assert "read_error" not in results[0].data
    assert "read_error" not in results[2].data


def test_adapter_cardinality_must_match() -> None:
    stage, _ = _stage(results=[])

    with pytest.raises(RuntimeError, match="must match 1:1"):
        stage.process(_task())


def test_invalid_adapter_segments_become_an_auditable_placeholder() -> None:
    stage, _ = _stage(
        emit_audit_placeholders=True,
        results=[VADResult([VADSegment(0.5, 0.75), VADSegment(float("nan"), 0.9)])],
    )

    result = stage.process(_task())

    assert result[0].data["read_error"] is True
    assert "waveform" not in result[0].data


def test_degenerate_segments_are_skipped() -> None:
    stage, _ = _stage(
        emit_audit_placeholders=True,
        results=[VADResult([VADSegment(0.5, 0.5), VADSegment(0.75, 0.5)])],
    )

    result = stage.process(_task())

    assert result[0].data["vad_empty"] is True


def test_empty_batch_and_setup_requirement() -> None:
    stage = VADSegmentationStage()
    assert stage.process_batch([]) == []
    with pytest.raises(RuntimeError, match="not initialized"):
        stage.process_batch([_task()])


def test_stage_is_picklable_without_loaded_adapter() -> None:
    stage = VADSegmentationStage(min_duration_sec=1.0, adapter_kwargs={"backend": "onnx"})

    restored = pickle.loads(pickle.dumps(stage))  # noqa: S301

    assert restored.min_duration_sec == 1.0
    assert restored.adapter_kwargs == {"backend": "onnx"}
    assert restored._adapter is None
