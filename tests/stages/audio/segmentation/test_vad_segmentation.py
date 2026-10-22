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

import json
import pickle
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch

from nemo_curator.backends.ray_data.adapter import RayDataStageAdapter
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.models.audio.vad.base import VADResult, VADSegment
from nemo_curator.stages.audio import agent
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio.common import ManifestWriterStage
from nemo_curator.stages.audio.inference.base import AdapterInferenceStage
from nemo_curator.stages.audio.inference.sed.stage import SEDInferenceStage
from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path


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
    assert stage.BATCH_ONLY is True


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


def test_process_is_unsupported() -> None:
    with pytest.raises(NotImplementedError, match="only supports process_batch"):
        VADSegmentationStage().process(_task())


def test_fanout_builds_segment_metadata_and_slices_original_channels() -> None:
    stage, _ = _stage(results=[VADResult([VADSegment(0.25, 0.75)])])
    task = _task(channels=2)
    task.data["language"] = "en"
    task.data["diar_segments"] = [
        {"start": 0.0, "end": 0.5, "speaker": "speaker_0"},
        {"start": 0.5, "end": 1.0, "speaker": "speaker_1"},
    ]

    result = stage.process_batch([task])

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

    child = stage.process_batch([task])[0]

    assert child.data["num_speakers"] == 2


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "invalid_interval",
    [
        {"start": 0.6, "end": 0.4, "speaker": "invalid"},
        {"start": 0.5, "end": 0.5, "speaker": "invalid"},
        {"start": 0.0, "end": 1.0, "speaker": None},
        {"start": 0.0, "end": "bad", "speaker": "invalid"},
        {"start": float("nan"), "end": 1.0, "speaker": "invalid"},
        {"start": 0.0, "end": float("inf"), "speaker": "invalid"},
        {"start": 0.0, "end": 10**1000, "speaker": "invalid"},
        {"start": 0.0, "speaker": "invalid"},
    ],
)
def test_invalid_diarization_metadata_does_not_drop_speech_or_inflate_count(
    nested: bool, invalid_interval: dict[str, Any]
) -> None:
    stage, _ = _stage(nested=nested)
    task = _task()
    task.data["diar_segments"] = [
        {"start": 0.0, "end": 1.0, "speaker": "valid"},
        invalid_interval,
    ]

    result = stage.process_batch([task])[0]

    child = result.data["segments"][0] if nested else result.data
    assert child["num_speakers"] == 1
    assert child["waveform"].shape == (1, 8000)


def test_custom_duration_key_is_used_without_leaking_parent_duration() -> None:
    stage, _ = _stage(
        results=[VADResult([VADSegment(0.0, 0.5)])],
        duration_key="duration_sec",
    )
    task = _task()
    task.data["duration"] = 99.0

    child = stage.process_batch([task])[0]

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

    child = stage.process_batch([task])[0]

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

    result = stage.process_batch([task])[0]

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

    result = stage.process_batch([task])[0]

    assert result is task
    assert [segment["segment_num"] for segment in result.data["segments"]] == [0, 1]
    assert [segment["duration"] for segment in result.data["segments"]] == [0.25, 0.25]
    assert "waveform" not in result.data


def test_nested_no_speech_is_an_auditable_placeholder() -> None:
    stage, _ = _stage(nested=True, results=[VADResult([])])
    task = _task()

    result = stage.process_batch([task])[0]

    assert result is task
    assert result.data["vad_empty"] is True
    assert result.data["duration"] == 1.0
    assert "waveform" not in result.data
    assert result.data["segments"] == []


def test_fanout_no_speech_preserves_legacy_drop_behavior() -> None:
    stage, _ = _stage(results=[VADResult([])])

    assert stage.process_batch([_task()]) == []


def test_fanout_no_speech_can_emit_an_auditable_placeholder() -> None:
    stage, _ = _stage(emit_audit_placeholders=True, results=[VADResult([])])
    task = _task()

    result = stage.process_batch([task])

    assert result == [task]
    assert task.data["vad_empty"] is True
    assert task.data["duration"] == 1.0
    assert "waveform" not in task.data


def test_read_error_passes_through_without_adapter_call() -> None:
    stage, adapter = _stage(emit_audit_placeholders=True)
    task = AudioTask(task_id="failed", data={"read_error": True, "waveform": torch.zeros(10)})

    result = stage.process_batch([task])

    assert result == [task]
    assert adapter.items == []
    assert "waveform" not in task.data


@pytest.mark.parametrize("skip_key", [None, "_skipme", "reason"])
@pytest.mark.parametrize("prior_reason", ["", "upstream_rejection"])
def test_empty_parent_can_skip_downstream_inference(skip_key: str | None, prior_reason: str) -> None:
    stage, _ = _stage(emit_audit_placeholders=True, skip_me_key=skip_key, results=[VADResult([])])
    task = _task()
    if skip_key is not None:
        task.data[skip_key] = prior_reason

    assert stage.process_batch([task]) == [task]
    assert task.data["vad_empty"] is True
    if skip_key is None:
        assert "_skipme" not in task.data
    else:
        assert task.data[skip_key] == (prior_reason or "vad_empty")


def test_failed_parent_has_distinct_skip_reason() -> None:
    stage, _ = _stage(emit_audit_placeholders=True, skip_me_key="_skipme")
    task = AudioTask(data={"read_error": True})
    assert stage.process_batch([task]) == [task]
    assert task.data["_skipme"] == "read_error"


@pytest.mark.parametrize("key", ["", "waveform", "start_ms", "duration_sec"])
def test_skip_reason_cannot_overwrite_vad_fields(key: str) -> None:
    with pytest.raises(ValueError, match="skip_me_key"):
        VADSegmentationStage(skip_me_key=key)


def test_audio_preparation_error_is_auditable_when_requested() -> None:
    stage, adapter = _stage(emit_audit_placeholders=True)
    task = AudioTask(task_id="missing", data={"some_key": "value"})

    result = stage.process_batch([task])

    assert result == [task]
    assert task.data["read_error"] is True
    assert adapter.items == []


def test_audio_preparation_error_can_fail_fast() -> None:
    stage, _ = _stage(fail_on_audio_error=True)

    with pytest.raises(RuntimeError, match="Failed to prepare VAD audio"):
        stage.process_batch([AudioTask(task_id="missing", data={})])


@pytest.mark.parametrize("error", [MemoryError("out of memory"), torch.cuda.OutOfMemoryError("out of memory")])
def test_audio_preparation_memory_failures_propagate(error: Exception, monkeypatch: pytest.MonkeyPatch) -> None:
    stage, _ = _stage(emit_audit_placeholders=True)

    def fail(_task: AudioTask) -> tuple[torch.Tensor, int]:
        raise error

    monkeypatch.setattr(stage, "_resolve_audio", fail)

    with pytest.raises(type(error), match="out of memory"):
        stage.process_batch([_task()])


def test_adapter_failures_propagate() -> None:
    class _BrokenAdapter:
        def detect_batch(self, _items: list[dict[str, Any]]) -> list[VADResult]:
            message = "provider failed"
            raise RuntimeError(message)

    stage = VADSegmentationStage()
    stage._adapter = _BrokenAdapter()

    with pytest.raises(RuntimeError, match="provider failed"):
        stage.process_batch([_task()])


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
        stage.process_batch([_task()])


def test_invalid_adapter_segments_become_an_auditable_placeholder() -> None:
    stage, _ = _stage(
        emit_audit_placeholders=True,
        results=[VADResult([VADSegment(0.5, 0.75), VADSegment(float("nan"), 0.9)])],
    )

    result = stage.process_batch([_task()])

    assert result[0].data["read_error"] is True
    assert "waveform" not in result[0].data


@pytest.mark.parametrize("error", [MemoryError("out of memory"), torch.cuda.OutOfMemoryError("out of memory")])
def test_result_emission_memory_failures_propagate(error: Exception, monkeypatch: pytest.MonkeyPatch) -> None:
    stage, _ = _stage(results=[VADResult([VADSegment(0.0, 0.5)])])

    def fail(*_args: object) -> AudioTask | list[AudioTask]:
        raise error

    monkeypatch.setattr(stage, "_emit_segments", fail)

    with pytest.raises(type(error), match="out of memory"):
        stage.process_batch([_task()])


def test_degenerate_segments_are_skipped() -> None:
    stage, _ = _stage(
        emit_audit_placeholders=True,
        results=[VADResult([VADSegment(0.5, 0.5), VADSegment(0.75, 0.5)])],
    )

    result = stage.process_batch([_task()])

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


@pytest.mark.parametrize(
    ("nested", "cardinality"),
    [(False, "1:N fan-out"), (True, "1:1 nested-list")],
)
def test_stage_is_agent_ready_for_each_output_shape(nested: bool, cardinality: str) -> None:
    stage, _ = _stage(
        nested=nested,
        results=[VADResult([VADSegment(0.25, 0.75)])],
    )

    contract = assert_agent_ready(
        stage,
        _task,
        expected_cardinality=cardinality,
        available_keys={"waveform", "sample_rate", "audio_filepath"},
        segments_key="segments",
    )

    assert contract.key_roles["waveform"] == "waveform"
    assert contract.key_roles["sample_rate"] == "sample_rate"
    assert contract.gates.lifecycle_side_effects is True
    assert contract.gates.per_row_independent is True
    if not nested:
        assert {"num_samples", "segments", "vad_empty", "duration_sec"} <= set(contract.removes_keys)
        assert "duration" not in contract.removes_keys


def test_agent_contract_models_audit_placeholders_as_conditional() -> None:
    stage, _ = _stage(
        emit_audit_placeholders=True,
        results=[VADResult([])],
    )

    contract = assert_agent_ready(
        stage,
        _task,
        expected_cardinality="1:N fan-out",
        available_keys={"waveform", "sample_rate", "audio_filepath"},
    )

    assert contract.writes.data_keys == []
    conditional_keys = {key for write in contract.conditional_writes for key in write.writes.data_keys}
    assert {"waveform", "vad_empty", "read_error", "duration"} <= conditional_keys
    assert contract.iteration_key is None
    assert len(contract.conditional_reads) == 3


def test_agent_contract_accepts_a_read_error_only_audit_row() -> None:
    stage, adapter = _stage(emit_audit_placeholders=True, read_error_key="failed")

    contract = assert_agent_ready(
        stage,
        lambda: AudioTask(task_id="failed", data={"failed": True}),
        expected_cardinality="1:N fan-out",
        available_keys={"failed"},
    )

    assert contract.writes.data_keys == []
    assert adapter.items == []


def test_static_catalog_is_conservative_for_selectable_vad_adapters() -> None:
    contract = VADSegmentationStage.describe_static()

    assert contract.cardinality == "1:N fan-out"
    assert contract.cardinality_options == ["1:N fan-out", "1:1 nested-list"]
    assert contract.gates.lifecycle_side_effects is True
    assert contract.gates.per_row_independent is False
    assert contract.batch_only is True


def test_nested_segment_task_keys_are_configurable() -> None:
    stage, _ = _stage(
        nested=True,
        waveform_key="samples",
        sample_rate_key="rate",
        audio_filepath_key="path",
        segments_key="speech",
        output_waveform_key="pcm",
        output_sample_rate_key="hz",
        start_ms_key="begin_ms",
        end_ms_key="finish_ms",
        segment_num_key="index",
        duration_key="seconds",
        original_file_key="source",
        diar_segments_key="turns",
        diar_start_key="turn_begin",
        diar_end_key="turn_end",
        diar_speaker_key="who",
        num_speakers_key="speaker_count",
        num_samples_key="sample_count",
        legacy_duration_keys=("old_duration",),
        results=[VADResult([VADSegment(0.25, 0.75)])],
    )
    task = AudioTask(
        task_id="custom",
        data={
            "samples": torch.zeros(1, 16000),
            "rate": 16000,
            "path": "/custom.wav",
            "turns": [{"turn_begin": 0.0, "turn_end": 1.0, "who": "speaker_0"}],
            "old_duration": 99.0,
        },
    )

    result = stage.process_batch([task])[0]

    assert result is task
    assert "samples" not in result.data
    assert set(result.data["speech"][0]) >= {
        "pcm",
        "hz",
        "begin_ms",
        "finish_ms",
        "index",
        "seconds",
        "source",
        "speaker_count",
    }
    assert not ({"waveform", "sample_rate", "start_ms", "end_ms", "segment_num"} & set(result.data["speech"][0]))
    assert "old_duration" not in result.data["speech"][0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_key": "audio_filepath"},
        {"nested": True, "segments_key": "waveform"},
        {"output_waveform_key": "duration"},
    ],
)
def test_cross_role_task_key_collisions_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"only same-role|must be distinct"):
        VADSegmentationStage(**kwargs)


@pytest.mark.parametrize("residency", ["auto", "file", "waveform"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("keep_waveform", [False, True])
def test_residency_and_segment_audio_match_agent_contract(
    residency: str, nested: bool, keep_waveform: bool, tmp_path: Path
) -> None:
    import soundfile as sf

    path = tmp_path / "source.wav"
    sf.write(path, np.full((44100, 2), 0.25, dtype=np.float32), 44100, subtype="FLOAT")
    stage, adapter = _stage(
        input_residency=residency,
        nested=nested,
        keep_segment_waveform_in_task=keep_waveform,
    )

    def fixture() -> AudioTask:
        return AudioTask(
            dataset_name="dataset",
            data={
                "waveform": torch.full((2, 8000), 0.75),
                "sample_rate": 8000,
                "audio_filepath": str(path),
                "original_file": "archive/source.flac",
                "label": "kept",
            },
        )

    contract = assert_agent_ready(
        stage,
        fixture,
        expected_cardinality="1:1 nested-list" if nested else "1:N fan-out",
        available_keys=set(fixture().data),
    )
    result = stage.process_batch([fixture()])[0]
    child = result.data["segments"][0] if nested else result.data
    expected_rate, expected_value = (44100, 0.25) if residency == "file" else (8000, 0.75)
    assert adapter.items[0]["sample_rate"] == expected_rate
    np.testing.assert_allclose(adapter.items[0]["waveform"], expected_value)
    assert child["sample_rate"] == expected_rate
    assert child["original_file"] == "archive/source.flac"
    assert child["audio_filepath"] == str(path)
    assert child["label"] == "kept"
    assert (child["start_ms"], child["end_ms"], child["duration"]) == (250, 750, 0.5)
    if keep_waveform:
        torch.testing.assert_close(child["waveform"], torch.full((2, expected_rate // 2), expected_value))
    else:
        assert "waveform" not in child
        assert contract.writes.produces == []
        json.dumps(result.data, allow_nan=False)


@pytest.mark.parametrize("sample_rate", [0, -1, True, 16000.5, "invalid", None])
@pytest.mark.parametrize("residency", ["auto", "waveform"])
def test_invalid_resident_rate_does_not_fall_back_to_file(
    sample_rate: object, residency: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, adapter = _stage(input_residency=residency)

    def unexpected_read(_path: str) -> None:
        pytest.fail("Invalid resident audio must not be replaced by file audio")

    monkeypatch.setattr(stage, "_load_audio", unexpected_read)
    task = _task()
    task.data["sample_rate"] = sample_rate

    assert stage.process_batch([task]) == []
    assert adapter.items == []


def test_waveform_mode_does_not_read_file_only_input(monkeypatch: pytest.MonkeyPatch) -> None:
    stage, adapter = _stage(input_residency="waveform")

    def unexpected_read(_path: str) -> None:
        pytest.fail("Waveform mode must not load files")

    monkeypatch.setattr(stage, "_load_audio", unexpected_read)

    assert stage.process_batch([AudioTask(data={"audio_filepath": "/source.wav"})]) == []
    assert adapter.items == []


@pytest.mark.parametrize("nested", [False, True])
def test_metadata_only_output_removes_stale_audio_aliases(nested: bool) -> None:
    stage, _ = _stage(
        nested=nested,
        waveform_key="samples",
        sample_rate_key="rate",
        output_waveform_key="pcm",
        keep_segment_waveform_in_task=False,
    )
    task = AudioTask(
        data={
            "samples": torch.ones(1, 16000),
            "rate": 16000,
            "waveform": torch.zeros(1, 16000),
            "sample_rate": 48000,
        }
    )

    result = stage.process_batch([task])[0]
    child = result.data["segments"][0] if nested else result.data

    assert not ({"samples", "waveform", "pcm"} & child.keys())
    assert child["sample_rate"] == 16000
    json.dumps(result.data, allow_nan=False)


@pytest.mark.parametrize("keep_waveform", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_nested_empty_or_failed_result_cleans_parent_audio(keep_waveform: bool, failed: bool) -> None:
    stage, _ = _stage(
        nested=True,
        waveform_key="samples",
        keep_segment_waveform_in_task=keep_waveform,
        results=[VADResult([], error="failed" if failed else None)],
    )
    task = AudioTask(data={"samples": torch.ones(1, 16000), "waveform": torch.zeros(1, 10), "sample_rate": 16000})

    result = stage.process_batch([task])[0]

    assert result is task
    assert result.data["segments"] == []
    assert not ({"samples", "waveform"} & result.data.keys())
    assert result.data["read_error" if failed else "vad_empty"] is True


def test_nested_container_cannot_alias_canonical_waveform_cleanup() -> None:
    with pytest.raises(ValueError, match="waveform cleanup"):
        VADSegmentationStage(nested=True, segments_key="waveform", waveform_key="samples", output_waveform_key="clip")


@pytest.mark.parametrize("source", [None, ""])
def test_fanout_preserves_falsey_provenance(source: object) -> None:
    stage, _ = _stage(original_file_key="source", audio_filepath_key="path")
    task = _task()
    task.data["source"] = source

    child = stage.process_batch([task])[0]

    assert child.data["source"] is source
    assert {"path", "audio_filepath"} <= set(stage.describe().invalidates_keys)


@pytest.mark.parametrize("nested", [False, True])
def test_metadata_only_remapped_waveform_does_not_leak_into_provenance(nested: bool, tmp_path: Path) -> None:
    stage, _ = _stage(
        nested=nested,
        input_residency="waveform",
        waveform_key="audio_filepath",
        audio_filepath_key="source_path",
        keep_segment_waveform_in_task=False,
    )
    parent = AudioTask(data={"audio_filepath": torch.zeros(1, 16000), "sample_rate": 16000})
    writer = ManifestWriterStage(output_path=str(tmp_path / "manifest.jsonl"))
    report = agent.validate_pipeline(
        [stage, writer],
        initial_keys=set(parent.data),
        initial_roles={"waveform", "sample_rate"},
        initial_tensor_keys={"audio_filepath"},
    )

    result = stage.process_batch([parent])[0]
    writer.setup()
    writer.process(result)

    assert report.ok, report.issues
    row = json.loads((tmp_path / "manifest.jsonl").read_text())
    child = row["segments"][0] if nested else row
    assert child["original_file"] == "unknown"
    assert "audio_filepath" not in row


@pytest.mark.parametrize("metadata_field", ["read_error_key", "vad_empty_key", "duration_key"])
def test_nested_cleanup_preserves_remapped_audit_metadata(metadata_field: str) -> None:
    stage, _ = _stage(
        nested=True,
        waveform_key="samples",
        output_waveform_key="pcm",
        keep_segment_waveform_in_task=False,
        results=[VADResult([])],
        **{metadata_field: "waveform"},
    )
    parent = AudioTask(data={"samples": torch.zeros(1, 16000), "sample_rate": 16000})
    if metadata_field == "read_error_key":
        parent.data["waveform"] = True

    result = stage.process_batch([parent])[0]

    assert result.data["waveform"] == (1.0 if metadata_field == "duration_key" else True)
    assert result.data["segments"] == []
    assert "samples" not in result.data
    assert "waveform" not in stage.describe().removes_keys
    json.dumps(result.data)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("metadata_key", ["waveform", "sample_rate"])
@pytest.mark.parametrize("metadata_field", ["num_speakers_key", "audio_filepath_key"])
def test_child_cleanup_preserves_remapped_parent_metadata(
    nested: bool, metadata_key: str, metadata_field: str
) -> None:
    stage, _ = _stage(
        nested=nested,
        input_residency="waveform",
        waveform_key="samples",
        sample_rate_key="rate",
        output_waveform_key="pcm",
        output_sample_rate_key="hz",
        keep_segment_waveform_in_task=False,
        **{metadata_field: metadata_key},
    )
    value = 3 if metadata_field == "num_speakers_key" else "/parent.wav"
    parent = AudioTask(data={"samples": torch.zeros(1, 16000), "rate": 16000, metadata_key: value})

    result = stage.process_batch([parent])[0]

    child = result.data["segments"][0] if nested else result.data
    assert child[metadata_key] == value
    assert metadata_key not in stage.describe().removes_keys
    json.dumps(result.data)


@pytest.mark.parametrize("nested", [False, True])
def test_legacy_path_fallback_remains_available(nested: bool) -> None:
    stage, _ = _stage(nested=nested, input_residency="waveform", audio_filepath_key="source_path")
    parent = _task()

    result = stage.process_batch([parent])[0]

    child = result.data["segments"][0] if nested else result.data
    assert child["original_file"] == "/recording.wav"


@pytest.mark.parametrize(
    ("keep_waveform", "consumer_residency", "expected_ok"),
    [(True, "waveform", True), (True, "file", False), (False, "waveform", False), (False, "file", False)],
)
def test_fanout_planner_matches_runtime_audio_carriers(
    keep_waveform: bool, consumer_residency: str, expected_ok: bool
) -> None:
    stages = [
        VADSegmentationStage(input_residency="waveform", keep_segment_waveform_in_task=keep_waveform),
        VADSegmentationStage(input_residency=consumer_residency),
    ]

    report = agent.validate_pipeline(
        stages,
        initial_keys={"waveform", "sample_rate"},
        initial_roles={"waveform", "sample_rate"},
    )

    assert report.ok is expected_ok, report.issues


@pytest.mark.parametrize("sample_rate", [8000, 16000, 44100, 48000])
@pytest.mark.parametrize("nested", [False, True])
def test_bundled_silero_silence_smoke(sample_rate: int, nested: bool) -> None:
    stage = VADSegmentationStage(nested=nested, keep_segment_waveform_in_task=False)
    task = AudioTask(data={"waveform": torch.zeros(1, sample_rate), "sample_rate": sample_rate})

    stage.setup()
    try:
        result = stage.process_batch([task])
    finally:
        stage.teardown()

    if nested:
        assert result == [task]
        assert result[0].data["segments"] == []
        assert result[0].data["vad_empty"] is True
        assert "waveform" not in result[0].data
    else:
        assert result == []


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("prior_reason", ["", "upstream_rejection"])
def test_audit_skip_reason_survives_remapped_waveform_cleanup(nested: bool, failed: bool, prior_reason: str) -> None:
    stage, _ = _stage(
        nested=nested,
        emit_audit_placeholders=True,
        waveform_key="samples",
        output_waveform_key="pcm",
        skip_me_key="reason",
        results=[VADResult([], error="failed" if failed else None)],
    )
    task = AudioTask(
        data={
            "samples": torch.zeros(1, 16000),
            "waveform": torch.zeros(1, 10),
            "sample_rate": 16000,
            "reason": prior_reason,
        }
    )

    contract = assert_agent_ready(
        stage,
        lambda: AudioTask(data=dict(task.data)),
        expected_cardinality="1:1 nested-list" if nested else "1:N fan-out",
        available_keys=set(task.data),
    )
    result = stage.process_batch([task])

    assert result == [task]
    assert task.data["reason"] == (prior_reason or ("read_error" if failed else "vad_empty"))
    assert not {"samples", "waveform", "pcm"} & task.data.keys()
    assert any("reason" in write.writes.data_keys for write in contract.conditional_writes)
    if nested:
        assert task.data["segments"] == []
    downstream = SEDInferenceStage(
        adapter_target="nemo_curator.models.audio.sed.tensorrt.TensorRTPANNsSEDAdapter",
        skip_me_key="reason",
        save_npz=False,
    )
    assert downstream.process_batch(result) == result
    json.dumps(task.data)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("keep_waveform", [False, True])
def test_skip_signaling_preserves_successful_segment_output_and_contract(nested: bool, keep_waveform: bool) -> None:
    stage, _ = _stage(nested=nested, keep_segment_waveform_in_task=keep_waveform, skip_me_key="_skipme")
    baseline, _ = _stage(nested=nested, keep_segment_waveform_in_task=keep_waveform)

    result = stage.process_batch([_task()])[0]
    expected = baseline.process_batch([_task()])[0]
    child = result.data["segments"][0] if nested else result.data
    expected_child = expected.data["segments"][0] if nested else expected.data

    assert "_skipme" not in result.data
    assert "_skipme" not in child
    assert child.keys() == expected_child.keys()
    if keep_waveform:
        torch.testing.assert_close(child.pop("waveform"), expected_child.pop("waveform"))
    assert child == expected_child
    contract = stage.describe()
    assert contract.writes.produces == (["tensor"] if keep_waveform else [])
    assert (
        "waveform" in (contract.writes.segment_data_keys if nested else contract.writes.data_keys)
    ) == keep_waveform


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_skip_reason_can_use_a_remapped_legacy_waveform_key(nested: bool, failed: bool) -> None:
    stage, _ = _stage(
        nested=nested,
        emit_audit_placeholders=True,
        waveform_key="samples",
        output_waveform_key="pcm",
        skip_me_key="waveform",
        results=[VADResult([], error="failed" if failed else None)],
    )
    task = AudioTask(data={"samples": torch.zeros(1, 16000), "sample_rate": 16000})

    result = stage.process_batch([task])

    assert result == [task]
    assert task.data["waveform"] == ("read_error" if failed else "vad_empty")
    assert "samples" not in task.data
    assert "waveform" not in stage.describe().removes_keys
