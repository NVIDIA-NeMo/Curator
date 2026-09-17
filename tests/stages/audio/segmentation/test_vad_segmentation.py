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

import json
import pickle
from unittest.mock import MagicMock, patch

import pytest
import soundfile as sf
import torch

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.audio.common import ManifestWriterStage
from nemo_curator.stages.audio.preprocessing.concatenation import SegmentConcatenationStage
from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@pytest.mark.gpu
class TestVADSegmentationStage:
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_process_returns_segments(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_model = MagicMock()
        mock_load_vad.return_value = mock_model

        sr = 48000
        mock_get_ts.return_value = [
            {"start": 0, "end": sr * 3},
            {"start": sr * 5, "end": sr * 8},
        ]

        waveform = torch.randn(1, sr * 10)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": sr},
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=1.0, max_duration_sec=30.0)
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, list)
        assert len(result) == 2
        for seg in result:
            assert isinstance(seg, AudioTask)
            assert "waveform" in seg.data
            assert "start_ms" in seg.data
            assert "end_ms" in seg.data
            assert "segment_num" in seg.data
            assert "duration" in seg.data

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_process_output_keys(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()

        sr = 48000
        mock_get_ts.return_value = [{"start": 0, "end": sr * 5}]

        waveform = torch.randn(1, sr * 10)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": sr},
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=1.0)
        stage.setup()
        result = stage.process(task)

        assert result[0].data["start_ms"] == 0
        assert result[0].data["segment_num"] == 0
        assert result[0].data["duration"] > 0
        assert result[0].data["sample_rate"] == sr

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_empty_speech_returns_empty(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()
        mock_get_ts.return_value = []

        waveform = torch.randn(1, 48000 * 5)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": 48000},
            dataset_name="test",
        )

        stage = VADSegmentationStage()
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_segment_numbering(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()

        sr = 48000
        mock_get_ts.return_value = [
            {"start": 0, "end": sr * 2},
            {"start": sr * 3, "end": sr * 5},
            {"start": sr * 6, "end": sr * 8},
        ]

        waveform = torch.randn(1, sr * 10)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": sr},
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=0.5)
        stage.setup()
        result = stage.process(task)

        assert len(result) == 3
        for i, seg in enumerate(result):
            assert seg.data["segment_num"] == i

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_missing_waveform_and_filepath_skipped(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()

        task = AudioTask(
            data={"some_key": "value"},
            dataset_name="test",
        )

        stage = VADSegmentationStage()
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_nested_mode_returns_single_task(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        mock_load_vad.return_value = MagicMock()

        sr = 48000
        mock_get_ts.return_value = [
            {"start": 0, "end": sr * 3},
            {"start": sr * 5, "end": sr * 8},
        ]

        waveform = torch.randn(1, sr * 10)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": sr},
            dataset_name="test",
        )

        stage = VADSegmentationStage(min_duration_sec=1.0, nested=True)
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert "segments" in result.data
        assert len(result.data["segments"]) == 2
        for seg in result.data["segments"]:
            assert "waveform" in seg
            assert "start_ms" in seg
            assert "end_ms" in seg
            assert "segment_num" in seg
            assert "duration" in seg
            assert "original_file" in seg

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_nested_mode_no_speech_returns_task_with_empty_segments(
        self, mock_load_vad: MagicMock, mock_get_ts: MagicMock
    ) -> None:
        mock_load_vad.return_value = MagicMock()
        mock_get_ts.return_value = []

        waveform = torch.randn(1, 48000 * 5)
        task = AudioTask(
            data={"waveform": waveform, "sample_rate": 48000},
            dataset_name="test",
        )

        stage = VADSegmentationStage(nested=True)
        stage.setup()
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["segments"] == []

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_nested_mode_ray_stage_spec_no_fanout(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        stage = VADSegmentationStage(nested=True)
        spec = stage.ray_stage_spec()
        assert spec == {}

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_non_nested_mode_ray_stage_spec_has_fanout(self, mock_load_vad: MagicMock, mock_get_ts: MagicMock) -> None:
        stage = VADSegmentationStage(nested=False)
        spec = stage.ray_stage_spec()
        assert spec[RayStageSpecKeys.IS_FANOUT_STAGE] is True

    def test_pickling(self) -> None:
        stage = VADSegmentationStage(min_duration_sec=2.0, threshold=0.6)
        pickled = pickle.dumps(stage)
        restored = pickle.loads(pickled)  # noqa: S301
        assert restored.min_duration_sec == 2.0
        assert restored.threshold == 0.6
        assert restored._vad_model is None


class TestNestedAndFanoutAgree:
    """The two packagings of a VAD result must describe the same speech.

    Deliberately outside ``TestVADSegmentationStage``, which is marked ``gpu``: this drives no
    model -- both timestamps and the loader are patched -- so gating it behind a GPU would mean
    the CPU suite never checks that the two modes agree. Lifted from
    tests/stages/audio/test_agent_simulation_pipelines.py, its only previous home.
    """

    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.get_speech_timestamps")
    @patch("nemo_curator.stages.audio.segmentation.vad_segmentation.load_silero_vad")
    def test_nested_and_fanout_produce_the_same_boundaries(
        self, mock_load_vad: MagicMock, mock_get_ts: MagicMock
    ) -> None:
        sr = 16000
        mock_load_vad.return_value = MagicMock()
        mock_get_ts.return_value = [{"start": 0, "end": int(sr * 0.4)}, {"start": int(sr * 0.4), "end": int(sr * 0.9)}]
        waveform = torch.randn(1, sr)

        def _task() -> AudioTask:
            return AudioTask(dataset_name="t", data={"waveform": waveform.clone(), "sample_rate": sr})

        nested = VADSegmentationStage(nested=True, input_residency="waveform")
        nested.setup()
        nested_segments = nested.process(_task()).data["segments"]

        fanout = VADSegmentationStage(nested=False, input_residency="waveform")
        fanout.setup()
        children = fanout.process(_task())

        assert len(nested_segments) == len(children) == 2, "both modes must find the same speech"
        nested_bounds = [(seg["start_ms"], seg["end_ms"]) for seg in nested_segments]
        fanout_bounds = [(child.data["start_ms"], child.data["end_ms"]) for child in children]
        assert nested_bounds == fanout_bounds == [(0, 400), (400, 900)], "only the packaging may differ"


def _stubbed_vad_stage(
    *,
    segments: list[dict[str, float]] | None = None,
    **kwargs,
) -> VADSegmentationStage:
    stage = VADSegmentationStage(resources=Resources(cpus=1.0, gpus=0.0), **kwargs)
    stage._vad_model = object()
    stage._get_vad_segments = MagicMock(return_value=[{"start": 0.1, "end": 0.3}] if segments is None else segments)
    return stage


def test_legacy_positional_constructor_order_is_preserved() -> None:
    resources = Resources(cpus=2.0, gpus=0.0)
    stage = VADSegmentationStage(
        100,
        1.5,
        20.0,
        0.7,
        50,
        "samples",
        "hz",
        True,
        "legacy-name",
        3,
        resources,
    )

    assert stage.min_interval_ms == 100
    assert stage.min_duration_sec == 1.5
    assert stage.max_duration_sec == 20.0
    assert stage.threshold == 0.7
    assert stage.speech_pad_ms == 50
    assert stage.waveform_key == "samples"
    assert stage.sample_rate_key == "hz"
    assert stage.nested is True
    assert stage.name == "legacy-name"
    assert stage.batch_size == 3
    assert stage.resources is resources
    assert stage.audio_filepath_key == "audio_filepath"


def test_custom_input_keys_preserve_legacy_canonical_output_schema() -> None:
    waveform = torch.arange(16000).reshape(1, -1)
    stage = _stubbed_vad_stage(
        input_residency="waveform",
        waveform_key="samples",
        sample_rate_key="hz",
    )

    child = stage.process(AudioTask(dataset_name="t", data={"samples": waveform, "hz": 16000}))[0].data

    assert stage.outputs()[1] == ["waveform", "sample_rate", "start_ms", "end_ms", "segment_num", "duration"]
    assert "samples" not in child
    assert "hz" not in child
    torch.testing.assert_close(child["waveform"], waveform[:, 1600:4800])
    assert child["sample_rate"] == 16000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_residency": "typo"},
        {"segments_key": ""},
        {"segments_key": "start_ms"},
        {"start_ms_key": "waveform"},
        {"output_waveform_key": "duration"},
    ],
)
def test_invalid_residency_and_destructive_key_collisions_are_rejected(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="must"):
        VADSegmentationStage(**kwargs)


def test_resident_waveform_without_sample_rate_does_not_fall_back_to_file(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "different.wav"
    sf.write(path, torch.ones(1600).numpy(), 16000)
    stage = _stubbed_vad_stage(input_residency="auto")
    resident = torch.zeros(1, 3200)

    result = stage.process(AudioTask(dataset_name="t", data={"waveform": resident, "audio_filepath": str(path)}))

    assert result == []
    stage._get_vad_segments.assert_not_called()


def test_empty_nested_result_preserves_default_parent_waveform() -> None:
    waveform = torch.zeros(1, 1600)
    stage = _stubbed_vad_stage(segments=[], nested=True, input_residency="waveform")

    result = stage.process(AudioTask(dataset_name="t", data={"waveform": waveform, "sample_rate": 16000}))

    assert result.data["waveform"] is waveform
    assert result.data["segments"] == []


def test_file_only_nested_result_retains_discovered_sample_rate(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "source.wav"
    sf.write(path, torch.zeros(1600).numpy(), 8000)
    stage = _stubbed_vad_stage(nested=True, input_residency="auto")

    result = stage.process(AudioTask(dataset_name="t", data={"audio_filepath": str(path)}))

    assert result.data["sample_rate"] == 8000


def test_falsey_original_file_is_preserved() -> None:
    stage = _stubbed_vad_stage(input_residency="waveform")
    task = AudioTask(
        dataset_name="t",
        data={
            "waveform": torch.zeros(1, 1600),
            "sample_rate": 16000,
            "audio_filepath": "/parent.wav",
            "original_file": "",
        },
    )

    child = stage.process(task)[0].data

    assert child["original_file"] == ""
    assert child["audio_filepath"] == "/parent.wav"


@pytest.mark.parametrize("residency", ["file", "waveform", "auto"])
def test_agent_ready_residency_modes_are_model_free(residency: str, tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / f"{residency}.wav"
    sf.write(path, torch.zeros(1600).numpy(), 16000)
    stage = _stubbed_vad_stage(input_residency=residency)

    def fixture() -> AudioTask:
        if residency == "file":
            data = {"audio_filepath": str(path), "label": "kept"}
        else:
            data = {"waveform": torch.zeros(1, 1600), "sample_rate": 16000, "label": "kept"}
            if residency == "auto":
                data["audio_filepath"] = str(path)
        return AudioTask(dataset_name="t", data=data)

    assert_agent_ready(
        stage,
        fixture,
        expected_cardinality="1:N fan-out",
        available_keys=set(fixture().data),
    )


@pytest.mark.parametrize("segments", [[], [{"start": 0.1, "end": 0.3}]])
@pytest.mark.parametrize("keep_segment_waveform", [False, True])
def test_nested_agent_ready_empty_and_populated_cleanup(
    segments: list[dict[str, float]],
    keep_segment_waveform: bool,
) -> None:
    stage = _stubbed_vad_stage(
        segments=segments,
        nested=True,
        input_residency="waveform",
        keep_segment_waveform_in_task=keep_segment_waveform,
    )

    def fixture() -> AudioTask:
        return AudioTask(
            dataset_name="t",
            data={"waveform": torch.arange(1600).reshape(1, -1), "sample_rate": 16000},
        )

    assert_agent_ready(
        stage,
        fixture,
        expected_cardinality="1:1 nested-list",
        available_keys={"waveform", "sample_rate"},
        segments_key="segments",
    )
    result = stage.process(fixture())
    assert isinstance(result, AudioTask)
    if segments or not keep_segment_waveform:
        assert "waveform" not in result.data
    else:
        assert "waveform" in result.data
    assert len(result.data["segments"]) == len(segments)
    if not keep_segment_waveform:
        json.dumps(result.data)


def test_fanout_children_retain_parent_paths_as_provenance() -> None:
    waveform = torch.arange(16000).reshape(1, -1)
    task = AudioTask(
        dataset_name="t",
        data={
            "samples": waveform,
            "hz": 16000,
            "recording_path": "/inputs/parent.wav",
            "source_path": "/archive/original.flac",
            "num_samples": 16000,
            "label": "kept",
        },
    )
    stage = _stubbed_vad_stage(
        input_residency="waveform",
        waveform_key="samples",
        sample_rate_key="hz",
        audio_filepath_key="recording_path",
        original_file_key="source_path",
    )

    contract = assert_agent_ready(
        stage,
        lambda: AudioTask(dataset_name="t", data=dict(task.data)),
        expected_cardinality="1:N fan-out",
        available_keys=set(task.data),
    )
    child = stage.process(task)[0].data

    assert contract.iteration_key == stage.segment_num_key
    assert contract.preserves_upstream_keys is True
    assert {"recording_path", "audio_filepath"}.issubset(contract.invalidates_keys)
    assert "num_samples" in contract.removes_keys
    assert child["source_path"] == "/archive/original.flac"
    assert child["label"] == "kept"
    assert child["recording_path"] == "/inputs/parent.wav"
    assert "num_samples" not in child
    assert "samples" not in child
    assert "hz" not in child
    torch.testing.assert_close(child["waveform"], waveform[:, 1600:4800])
    assert child["sample_rate"] == 16000
    assert (child["start_ms"], child["end_ms"]) == (100, 300)


def test_nested_segments_retain_parent_paths_as_provenance() -> None:
    task = AudioTask(
        dataset_name="t",
        data={
            "waveform": torch.zeros(1, 16000),
            "sample_rate": 16000,
            "audio_filepath": "/inputs/parent.wav",
            "num_samples": 16000,
        },
    )
    stage = _stubbed_vad_stage(nested=True, input_residency="waveform")

    result = stage.process(task)
    segment = result.data["segments"][0]

    assert result.data["audio_filepath"] == "/inputs/parent.wav"
    assert result.data["num_samples"] == 16000
    assert segment["original_file"] == "/inputs/parent.wav"
    assert segment["audio_filepath"] == "/inputs/parent.wav"
    assert "num_samples" not in segment


def test_nested_contract_declares_container_and_segment_fields() -> None:
    stage = VADSegmentationStage(nested=True, segments_key="speech_chunks")
    contract = stage.describe()

    assert contract.writes.data_keys == ["speech_chunks"]
    assert contract.writes.segment_data_keys == [
        "sample_rate",
        "start_ms",
        "end_ms",
        "segment_num",
        "duration",
        "original_file",
        "waveform",
    ]
    assert contract.iteration_key == "speech_chunks"


def test_contract_declares_mode_specific_parent_carrier_removals() -> None:
    nested = VADSegmentationStage(nested=True).describe()
    fanout_memory = VADSegmentationStage(nested=False).describe()
    fanout_metadata = VADSegmentationStage(
        nested=False,
        keep_segment_waveform_in_task=False,
    ).describe()

    assert nested.removes_keys == []
    assert {"audio_filepath"}.issubset(fanout_memory.invalidates_keys)
    assert "num_samples" in fanout_memory.removes_keys
    assert "waveform" not in fanout_memory.removes_keys
    assert "audio_filepath" in fanout_metadata.invalidates_keys
    assert {"num_samples", "waveform"}.issubset(fanout_metadata.removes_keys)


@pytest.mark.parametrize(
    ("keep_segment_waveform", "consumer_residency", "expected_ok"),
    [
        (True, "waveform", True),
        (True, "file", False),
        (False, "waveform", False),
        (False, "file", False),
    ],
)
def test_fanout_planner_matches_runtime_audio_carriers(
    keep_segment_waveform: bool,
    consumer_residency: str,
    expected_ok: bool,
) -> None:
    stages = [
        VADSegmentationStage(
            input_residency="waveform",
            keep_segment_waveform_in_task=keep_segment_waveform,
        ),
        VADSegmentationStage(input_residency=consumer_residency),
    ]

    report = validate_pipeline(
        stages,
        initial_keys={"waveform", "sample_rate"},
        initial_roles={"waveform", "sample_rate"},
    )

    assert report.ok is expected_ok, report.issues


def test_nested_vad_to_disk_concat_to_manifest_is_serialization_safe(tmp_path) -> None:  # noqa: ANN001
    stages = [
        VADSegmentationStage(nested=True, input_residency="waveform"),
        SegmentConcatenationStage(
            keep_waveform_in_task=False,
            write_to_disk=True,
            output_dir=str(tmp_path / "concat"),
        ),
        ManifestWriterStage(output_path=str(tmp_path / "manifest.jsonl")),
    ]

    report = validate_pipeline(
        stages,
        initial_keys={"waveform", "sample_rate"},
        initial_roles={"waveform", "sample_rate"},
    )

    assert report.ok, report.issues
    assert stages[1].describe().gates.sanitizes_output is True


def test_nested_vad_to_memory_concat_is_rejected_before_manifest(tmp_path) -> None:  # noqa: ANN001
    stages = [
        VADSegmentationStage(nested=True, input_residency="waveform"),
        SegmentConcatenationStage(),
        ManifestWriterStage(output_path=str(tmp_path / "manifest.jsonl")),
    ]

    report = validate_pipeline(
        stages,
        initial_keys={"waveform", "sample_rate"},
        initial_roles={"waveform", "sample_rate"},
    )

    assert not report.ok
    assert any(issue.code == "tensor_into_sink" for issue in report.issues)
    assert stages[1].describe().gates.sanitizes_output is False
