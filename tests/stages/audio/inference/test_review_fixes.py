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

"""Model-free regressions for the inference agent-readiness review."""

from __future__ import annotations

import inspect
from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.stages.audio._agent._agent_registry import build_contract, static_contract
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.audio._agent._residency import resolve_audio_path
from nemo_curator.stages.audio.inference import base as inference_base
from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.stages.audio.inference.base import _channel_first_waveform
from nemo_curator.stages.audio.inference.speaker_diarization import pyannote as pyannote_module
from nemo_curator.stages.audio.inference.speaker_diarization.pyannote import PyAnnoteDiarizationStage
from nemo_curator.stages.audio.inference.speaker_diarization.sortformer import InferenceSortformerStage
from nemo_curator.stages.audio.inference.vad.whisperx_vad import WhisperXVADStage
from nemo_curator.stages.audio.tagging.utils import add_non_speaker_segments
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_ASR_TARGET = "nemo_curator.models.asr.qwen_omni.QwenOmniASRAdapter"
_SAMPLE_RATE = 10
_SEGMENTS = [
    {"start": 0.2, "end": 0.6, "speaker": "speaker_0"},
    {"start": 0.7, "end": 1.0, "speaker": "speaker_0"},
]


@pytest.fixture(scope="session", autouse=True)
def shared_ray_cluster() -> None:
    """These direct stage tests do not need the repository-wide Ray fixture."""


class _ProgressHook(AbstractContextManager):
    def __exit__(self, *_args: object) -> None:
        return None


class _FakeAnnotation:
    def __init__(self, segments: list[dict[str, Any]], *, fail_after_rttm: bool = False) -> None:
        self._segments = segments
        self._tracks = list(segments)
        self._fail_after_rttm = fail_after_rttm

    def get_overlap(self) -> SimpleNamespace:
        return SimpleNamespace(segments_list_=[])

    def crop(self, _segment: object) -> _FakeAnnotation:
        return self

    def write_rttm(self, stream: Any) -> None:  # noqa: ANN401
        stream.write("SPEAKER fake 1 0.000 0.100 <NA> <NA> speaker_0 <NA> <NA>\n")

    def itertracks(self, *, yield_label: bool) -> Iterator[tuple[SimpleNamespace, None, str]]:
        assert yield_label
        if self._fail_after_rttm:
            msg = "failure after RTTM write"
            raise RuntimeError(msg)
        for segment in self._segments:
            yield SimpleNamespace(start=segment["start"], end=segment["end"]), None, segment["speaker"]


def _make_stage(  # noqa: PLR0913
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    input_residency: str = "file",
    fanout: bool = False,
    write_rttm: bool = False,
    rttm_out_dir: str | None = None,
    fail_after_rttm: bool = False,
) -> tuple[Any, list[int]]:
    seen_lengths: list[int] = []
    if kind == "pyannote":
        annotation = _FakeAnnotation(_SEGMENTS, fail_after_rttm=fail_after_rttm)

        def infer(payload: dict[str, Any], *, hook: object) -> _FakeAnnotation:
            assert hook is not None
            seen_lengths.append(int(payload["waveform"].shape[-1]))
            return annotation

        monkeypatch.setattr(pyannote_module, "ProgressHook", _ProgressHook)
        monkeypatch.setattr(pyannote_module, "add_non_speaker_segments", lambda *_args: None)
        stage = PyAnnoteDiarizationStage(
            input_residency=input_residency,
            fanout=fanout,
            write_rttm=write_rttm,
            min_length=0.0,
            resources=Resources(gpus=0),
        )
        stage._pipeline = infer
        return stage, seen_lengths

    if kind == "whisperx":
        model = MagicMock()

        def vad(audio: np.ndarray, _max_length: float, *, sample_rate: int) -> list[dict[str, Any]]:
            assert sample_rate == _SAMPLE_RATE
            seen_lengths.append(int(audio.shape[-1]))
            return [dict(segment) for segment in _SEGMENTS]

        model.get_vad_segments.side_effect = vad
        stage = WhisperXVADStage(
            input_residency=input_residency,
            fanout=fanout,
            min_length=0.0,
            resources=Resources(gpus=0),
        )
        stage._vad_model = model
        return stage, seen_lengths

    if kind == "sortformer":
        model = MagicMock()

        def diarize(*, audio: list[str], batch_size: int) -> list[list[str]]:
            assert batch_size == 1
            seen_lengths.append(sf.info(audio[0]).frames)
            return [[f"{segment['start']} {segment['end']} {segment['speaker']}" for segment in _SEGMENTS]]

        model.diarize.side_effect = diarize
        stage = InferenceSortformerStage(
            diar_model=model,
            input_residency=input_residency,
            fanout=fanout,
            rttm_out_dir=rttm_out_dir,
            resources=Resources(gpus=0),
        )
        return stage, seen_lengths

    raise AssertionError(kind)


def _write_audio(path: Path, waveform: np.ndarray) -> None:
    sf.write(path, waveform.T, _SAMPLE_RATE, subtype="FLOAT")


@pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
def test_process_batch_accepts_file_waveform_and_auto_residency(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_waveform = np.arange(12, dtype=np.float32)[None, :]
    resident_waveform = np.arange(20, dtype=np.float32)[None, :] + 100
    audio_path = tmp_path / "source.wav"
    _write_audio(audio_path, file_waveform)

    file_stage, file_seen = _make_stage(kind, monkeypatch, input_residency="file")
    file_result = file_stage.process_batch([AudioTask(data={"audio_filepath": str(audio_path)})])
    assert len(file_result) == 1
    assert file_seen == [len(file_waveform[0])]

    waveform_stage, waveform_seen = _make_stage(kind, monkeypatch, input_residency="waveform")
    waveform_result = waveform_stage.process_batch(
        [AudioTask(data={"waveform": resident_waveform, "sample_rate": _SAMPLE_RATE})]
    )
    assert len(waveform_result) == 1
    assert waveform_seen == [len(resident_waveform[0])]

    auto_stage, auto_seen = _make_stage(kind, monkeypatch, input_residency="auto")
    auto_result = auto_stage.process_batch(
        [
            AudioTask(
                data={
                    "audio_filepath": str(audio_path),
                    "waveform": resident_waveform,
                    "sample_rate": _SAMPLE_RATE,
                }
            )
        ]
    )
    assert len(auto_result) == 1
    assert auto_seen == [len(resident_waveform[0])], "auto must prefer the complete resident pair"

    auto_file_stage, auto_file_seen = _make_stage(kind, monkeypatch, input_residency="auto")
    auto_file_result = auto_file_stage.process_batch([AudioTask(data={"audio_filepath": str(audio_path)})])
    assert len(auto_file_result) == 1
    assert auto_file_seen == [len(file_waveform[0])]


@pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
@pytest.mark.parametrize("residency", ["waveform", "auto"])
@pytest.mark.parametrize(
    "partial",
    [
        {"waveform": np.ones((1, 10), dtype=np.float32)},
        {"sample_rate": _SAMPLE_RATE},
    ],
)
def test_process_batch_rejects_partial_resident_pairs(
    kind: str,
    residency: str,
    partial: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, _seen = _make_stage(kind, monkeypatch, input_residency=residency)
    # The valid file alternative is deliberate: auto must not fall back while
    # an orphaned resident key could describe different audio.
    partial = {"audio_filepath": "/data/source.wav", **partial}

    with pytest.raises(ValueError, match=r"incomplete resident audio.*must be provided together"):
        stage.process_batch([AudioTask(data=partial)])


@pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
@pytest.mark.parametrize(
    "partial",
    [
        {"waveform": np.ones((1, 10), dtype=np.float32)},
        {"sample_rate": _SAMPLE_RATE},
    ],
)
def test_file_residency_ignores_partial_resident_fragments(
    kind: str,
    partial: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waveform = np.arange(12, dtype=np.float32)[None, :]
    audio_path = tmp_path / f"{kind}.wav"
    _write_audio(audio_path, waveform)
    stage, seen = _make_stage(kind, monkeypatch, input_residency="file")

    result = stage.process_batch([AudioTask(data={"audio_filepath": str(audio_path), **partial})])

    assert len(result) == 1
    assert seen == [waveform.shape[-1]]


@pytest.mark.parametrize("residency", ["waveform", "auto"])
def test_whisperx_resident_duration_controls_short_input_decisions(
    residency: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "fallback.wav"
    _write_audio(audio_path, np.zeros((1, 2), dtype=np.float32))
    stage, seen = _make_stage("whisperx", monkeypatch, input_residency=residency)
    stage.min_length = 0.5
    common = {"audio_filepath": str(audio_path)} if residency == "auto" else {}

    long_task = AudioTask(
        data={
            **common,
            "waveform": np.ones((1, 12), dtype=np.float32),
            "sample_rate": _SAMPLE_RATE,
            "duration": 0.1,
        }
    )
    long_result = stage.process_batch([long_task])
    assert long_result == [long_task]
    assert long_task.data["vad_segments"]
    assert seen == [12]

    short_task = AudioTask(
        data={
            **common,
            "waveform": np.ones((1, 2), dtype=np.float32),
            "sample_rate": _SAMPLE_RATE,
            "duration": 99.0,
        }
    )
    short_result = stage.process_batch([short_task])
    assert short_result == [short_task]
    assert short_task.data["vad_segments"] == []
    assert seen == [12], "the VAD model must not run for the selected 0.2-second waveform"


def test_whisperx_file_mode_keeps_manifest_duration_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "long.wav"
    _write_audio(audio_path, np.ones((1, 12), dtype=np.float32))
    stage, seen = _make_stage("whisperx", monkeypatch, input_residency="file")
    stage.min_length = 0.5
    task = AudioTask(data={"audio_filepath": str(audio_path), "duration": 0.1})

    result = stage.process_batch([task])

    assert result == [task]
    assert task.data["vad_segments"] == []
    assert seen == []


@pytest.mark.parametrize("residency", ["waveform", "auto"])
def test_pyannote_non_speaker_bounds_use_selected_resident_duration(
    residency: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "fallback.wav"
    _write_audio(audio_path, np.zeros((1, 2), dtype=np.float32))
    stage, _seen = _make_stage("pyannote", monkeypatch, input_residency=residency)
    monkeypatch.setattr(pyannote_module, "add_non_speaker_segments", add_non_speaker_segments)
    common = {"audio_filepath": str(audio_path)} if residency == "auto" else {}
    task = AudioTask(
        data={
            **common,
            "waveform": np.ones((1, 12), dtype=np.float32),
            "sample_rate": _SAMPLE_RATE,
            "duration": 99.0,
        }
    )

    result = stage.process_batch([task])[0]

    assert max(segment["end"] for segment in result.data["segments"]) == pytest.approx(1.2)


@pytest.mark.parametrize("cls", [PyAnnoteDiarizationStage, WhisperXVADStage, InferenceSortformerStage])
def test_inference_stages_validate_input_residency_at_construction(cls: type) -> None:
    with pytest.raises(ValueError, match="input_residency must be one of"):
        cls(input_residency="wavefrom")  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
@pytest.mark.parametrize("resident", [False, True], ids=["file", "waveform"])
def test_fanout_children_feed_asr_exact_slices(
    kind: str,
    resident: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_channel = np.arange(12, dtype=np.float32) / 10
    waveform = np.stack([first_channel, first_channel + 2])
    audio_path = tmp_path / f"{kind}.wav"
    _write_audio(audio_path, waveform)
    residency = "waveform" if resident else "file"
    stage, _seen = _make_stage(kind, monkeypatch, input_residency=residency, fanout=True)
    parent_data: dict[str, Any] = (
        {
            "waveform": waveform,
            "sample_rate": _SAMPLE_RATE,
            "audio_filepath": str(audio_path),
            "resampled_audio_filepath": str(audio_path),
        }
        if resident
        else {
            "audio_filepath": str(audio_path),
            "resampled_audio_filepath": str(audio_path),
        }
    )

    children = stage.process_batch([AudioTask(dataset_name="d", data=parent_data)])

    assert len(children) == 2
    expected = [waveform[:, 2:6], waveform[:, 7:10]]
    dropped_containers = {
        "pyannote": {"segments", "overlap_segments"},
        "whisperx": {"vad_segments"},
        "sortformer": {"diar_segments"},
    }
    for child, expected_slice in zip(children, expected, strict=True):
        np.testing.assert_array_equal(child.data["waveform"], expected_slice)
        assert child.data["waveform"].shape[0] == 2
        assert child.data["sample_rate"] == _SAMPLE_RATE
        assert not np.shares_memory(child.data["waveform"], waveform)
        assert {"audio_filepath", "resampled_audio_filepath"}.isdisjoint(child.data)
        assert dropped_containers[kind].isdisjoint(child.data)
        assert child.data["original_file"]
    if kind == "pyannote":
        assert [child.data["num_speakers"] for child in children] == [1, 1]

    asr = ASRStage(
        adapter_target=_ASR_TARGET,
        model_id="mock/model",
        waveform_key="waveform",
        sample_rate_key="sample_rate",
        target_sample_rate=_SAMPLE_RATE,
        keep_waveform=True,
    )
    asr._adapter = MagicMock()
    asr._adapter.transcribe_batch.return_value = [ASRResult(text="one"), ASRResult(text="two")]
    asr.process_batch(children)
    assert all("waveform" in child.data for child in children)
    asr_items = asr._adapter.transcribe_batch.call_args.args[0]
    np.testing.assert_array_equal(asr_items[0]["waveform"], expected[0].mean(axis=0))
    np.testing.assert_array_equal(asr_items[1]["waveform"], expected[1].mean(axis=0))


@pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
def test_fanout_contract_is_waveform_only_and_blocks_file_consumers(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, _seen = _make_stage(kind, monkeypatch, fanout=True)
    contract = build_contract(stage)
    assert {"waveform", "sample_rate"}.issubset(contract.writes.data_keys)
    assert contract.writes.produces == ["tensor"]
    assert {"audio_filepath", "resampled_audio_filepath"}.issubset(contract.removes_keys)
    removed_containers = {
        "pyannote": {"segments", "overlap_segments"},
        "whisperx": {"vad_segments"},
        "sortformer": {"diar_segments"},
    }
    assert removed_containers[kind].issubset(contract.removes_keys)
    assert contract.cardinality == "1:N fan-out"
    assert contract.iteration_key is not None
    after_fanout = validate_pipeline(
        [stage],
        initial_roles={"audio_filepath"},
        initial_keys={"audio_filepath", "resampled_audio_filepath", *removed_containers[kind]},
        initial_task_type="AudioTask",
    )
    assert removed_containers[kind].isdisjoint(after_fanout.produced_keys)

    file_asr = ASRStage(adapter_target=_ASR_TARGET, model_id="mock/model")
    rejected = validate_pipeline(
        [stage, file_asr],
        initial_roles={"audio_filepath"},
        initial_keys={"audio_filepath", "resampled_audio_filepath"},
        initial_task_type="AudioTask",
    )
    assert not rejected.ok
    assert any(issue.stage_index == 1 and issue.code == "key_removed_upstream" for issue in rejected.issues)

    waveform_asr = ASRStage(
        adapter_target=_ASR_TARGET,
        model_id="mock/model",
        waveform_key="waveform",
        sample_rate_key="sample_rate",
    )
    accepted = validate_pipeline(
        [stage, waveform_asr],
        initial_roles={"audio_filepath"},
        initial_keys={"audio_filepath", "resampled_audio_filepath"},
        initial_task_type="AudioTask",
    )
    assert accepted.ok

    setattr(stage, "filepath_key" if kind == "sortformer" else "audio_filepath_key", "recording_path")
    custom_contract = build_contract(stage)
    assert {"recording_path", "audio_filepath", "resampled_audio_filepath"}.issubset(custom_contract.removes_keys)


@pytest.mark.parametrize(
    ("cls", "path_key", "container_key"),
    [
        (PyAnnoteDiarizationStage, "audio_filepath_key", "segments_key"),
        (WhisperXVADStage, "audio_filepath_key", "segments_key"),
        (InferenceSortformerStage, "filepath_key", "diar_segments_key"),
    ],
)
def test_fanout_rejects_output_and_path_key_collisions(cls: type, path_key: str, container_key: str) -> None:
    with pytest.raises(ValueError, match="collide with removed full-recording path keys"):
        cls(fanout=True, waveform_key="audio_filepath")
    with pytest.raises(ValueError, match="fan-out output keys must be distinct"):
        cls(fanout=True, waveform_key="samples", sample_rate_key="samples")
    with pytest.raises(ValueError, match="fan-out output keys must be non-empty"):
        cls(fanout=True, original_file_key="")
    with pytest.raises(ValueError, match="collide with removed full-recording path keys"):
        cls(fanout=True, original_file_key="recording_path", **{path_key: "recording_path"})
    with pytest.raises(ValueError, match="collide with removed parent container keys"):
        cls(fanout=True, waveform_key="parent_segments", **{container_key: "parent_segments"})

    stage = cls(
        fanout=True,
        waveform_key="segment_samples",
        sample_rate_key="segment_rate",
        **{path_key: "recording_path"},
    )
    contract = build_contract(stage)
    assert "recording_path" in contract.removes_keys
    assert "recording_path" not in contract.writes.data_keys


@pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
def test_assert_agent_ready_for_fanout_inference_stages(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waveform = np.arange(12, dtype=np.float32)[None, :]
    audio_path = tmp_path / f"{kind}.wav"
    _write_audio(audio_path, waveform)
    stage, _seen = _make_stage(kind, monkeypatch, fanout=True)

    assert_agent_ready(
        stage,
        lambda: AudioTask(data={"audio_filepath": str(audio_path)}),
        expected_cardinality="1:N fan-out",
        available_keys={"audio_filepath"},
    )


def test_asr_waveform_removal_contract_matches_runtime_including_skips() -> None:
    removing = ASRStage(
        adapter_target=_ASR_TARGET,
        model_id="mock/model",
        waveform_key="waveform",
        sample_rate_key="sample_rate",
        skip_if_output_exists=True,
        keep_waveform=False,
    )
    retaining = ASRStage(
        adapter_target=_ASR_TARGET,
        model_id="mock/model",
        waveform_key="waveform",
        sample_rate_key="sample_rate",
        keep_waveform=True,
    )
    assert build_contract(removing).removes_keys == ["waveform"]
    assert build_contract(retaining).removes_keys == []
    assert (
        build_contract(ASRStage(adapter_target=_ASR_TARGET, model_id="mock/model", waveform_key="")).removes_keys == []
    )
    planner_seed = {
        "initial_roles": {"waveform", "sample_rate"},
        "initial_keys": {"waveform", "sample_rate"},
        "initial_task_type": "AudioTask",
    }
    after_removal = validate_pipeline([removing, retaining], **planner_seed)
    after_retention = validate_pipeline([retaining, retaining], **planner_seed)
    assert not after_removal.ok
    assert any(issue.code == "key_removed_upstream" for issue in after_removal.issues)
    assert after_retention.ok

    removing._adapter = MagicMock()
    skipped = AudioTask(
        data={
            "waveform": np.ones((1, 10), dtype=np.float32),
            "sample_rate": _SAMPLE_RATE,
            "pred_text": "existing",
        }
    )
    assert removing.process_batch([skipped]) == [skipped]
    assert "waveform" not in skipped.data
    assert skipped.data["sample_rate"] == _SAMPLE_RATE
    removing._adapter.transcribe_batch.assert_not_called()

    runtime = ASRStage(
        adapter_target=_ASR_TARGET,
        model_id="mock/model",
        waveform_key="waveform",
        sample_rate_key="sample_rate",
        target_sample_rate=_SAMPLE_RATE,
    )
    runtime._adapter = MagicMock()
    runtime._adapter.transcribe_batch.return_value = [ASRResult(text="ok")]
    assert_agent_ready(
        runtime,
        lambda: AudioTask(data={"waveform": np.ones((1, 10), dtype=np.float32), "sample_rate": _SAMPLE_RATE}),
        expected_cardinality="1:1",
        available_keys={"waveform", "sample_rate"},
    )


@pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
@pytest.mark.parametrize("resident_type", ["numpy", "torch"])
def test_pcm16_resident_fanout_matches_file_amplitude(
    kind: str,
    resident_type: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pcm = np.array(
        [[-32768, -24576, -16384, -8192, 0, 8192, 16384, 24576, 32767, -32768, 0, 32767]],
        dtype=np.int16,
    )
    normalized = pcm.astype(np.float32) / np.float32(32768)
    audio_path = tmp_path / f"{kind}-{resident_type}.wav"
    sf.write(audio_path, normalized.T, _SAMPLE_RATE, subtype="PCM_16")
    resident = torch.from_numpy(pcm.copy()) if resident_type == "torch" else pcm

    resident_stage, _seen = _make_stage(kind, monkeypatch, input_residency="waveform", fanout=True)
    file_stage, _seen = _make_stage(kind, monkeypatch, input_residency="file", fanout=True)
    resident_children = resident_stage.process_batch(
        [AudioTask(data={"waveform": resident, "sample_rate": _SAMPLE_RATE})]
    )
    file_children = file_stage.process_batch([AudioTask(data={"audio_filepath": str(audio_path)})])

    assert len(resident_children) == len(file_children) == 2
    for resident_child, file_child in zip(resident_children, file_children, strict=True):
        assert resident_child.data["waveform"].dtype == np.float32
        np.testing.assert_array_equal(resident_child.data["waveform"], file_child.data["waveform"])

    asr = ASRStage(
        adapter_target=_ASR_TARGET,
        model_id="mock/model",
        waveform_key="waveform",
        sample_rate_key="sample_rate",
        target_sample_rate=_SAMPLE_RATE,
        keep_waveform=True,
    )
    asr._adapter = MagicMock()
    asr._adapter.transcribe_batch.return_value = [ASRResult(text="one"), ASRResult(text="two")]
    asr.process_batch(resident_children)
    asr_items = asr._adapter.transcribe_batch.call_args.args[0]
    np.testing.assert_array_equal(asr_items[0]["waveform"], normalized[0, 2:6])
    np.testing.assert_array_equal(asr_items[1]["waveform"], normalized[0, 7:10])


def test_pcm32_and_floating_waveforms_are_canonical_float32() -> None:
    pcm32 = np.array([[-2147483648, -1073741824, 0, 1073741824, 2147483647]], dtype=np.int32)
    expected = pcm32.astype(np.float32) / np.float32(2147483648)
    normalized = _channel_first_waveform(pcm32)
    np.testing.assert_array_equal(normalized, expected)
    assert normalized.dtype == np.float32

    floating = _channel_first_waveform(np.array([[0.25, -0.5]], dtype=np.float64))
    np.testing.assert_array_equal(floating, np.array([[0.25, -0.5]], dtype=np.float32))
    assert floating.dtype == np.float32


@pytest.mark.parametrize(
    "waveform",
    [
        np.array([[0, 1]], dtype=np.uint16),
        np.array([[0, 1]], dtype=np.int8),
        np.array([[0, 1]], dtype=np.int64),
        np.array([[0, 1]], dtype=np.complex64),
        np.array([["0", "1"]]),
    ],
)
def test_unsupported_resident_waveform_dtypes_are_rejected(waveform: np.ndarray) -> None:
    with pytest.raises(ValueError, match="unsupported resident waveform"):
        _channel_first_waveform(waveform)


def test_waveform_identity_hash_streams_a_memoryview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[bytes | memoryview] = []
    real_sha256 = inference_base.hashlib.sha256

    class RecordingDigest:
        def __init__(self) -> None:
            self.delegate = real_sha256()

        def update(self, payload: bytes | memoryview) -> None:
            payloads.append(payload)
            self.delegate.update(payload)

        def hexdigest(self) -> str:
            return self.delegate.hexdigest()

    monkeypatch.setattr(inference_base.hashlib, "sha256", lambda: RecordingDigest())
    waveform = np.arange(12, dtype=np.float32).reshape(1, -1)

    identity = inference_base._stable_audio_identity(
        {},
        waveform,
        _SAMPLE_RATE,
        source_path=None,
    )

    assert identity.startswith("audio_")
    assert isinstance(payloads[0], memoryview)
    assert payloads[0].nbytes == waveform.nbytes
    assert isinstance(payloads[1], bytes)


@pytest.mark.parametrize("kind", ["pyannote", "whisperx", "sortformer"])
def test_unsigned_resident_waveforms_are_rejected_by_fanout_stages(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, _seen = _make_stage(kind, monkeypatch, input_residency="waveform", fanout=True)

    with pytest.raises(ValueError, match="unsupported resident waveform integer dtype"):
        stage.process_batch(
            [
                AudioTask(
                    data={
                        "waveform": np.array([[0, 32768]], dtype=np.uint16),
                        "sample_rate": _SAMPLE_RATE,
                    }
                )
            ]
        )


def test_legacy_positional_signatures_are_exact() -> None:
    expected = {
        PyAnnoteDiarizationStage: [
            "hf_token",
            "model_name",
            "segmentation_batch_size",
            "embedding_batch_size",
            "min_length",
            "max_length",
            "audio_filepath_key",
            "segments_key",
            "overlap_segments_key",
            "name",
            "resources",
            "xenna_num_workers",
            "_pipeline",
            "_vad_model",
            "_rng",
        ],
        InferenceSortformerStage: [
            "model_name",
            "model_path",
            "cache_dir",
            "diar_model",
            "filepath_key",
            "diar_segments_key",
            "rttm_out_dir",
            "chunk_len",
            "chunk_left_context",
            "chunk_right_context",
            "fifo_len",
            "spkcache_update_period",
            "spkcache_len",
            "inference_batch_size",
            "name",
            "batch_size",
            "resources",
        ],
        WhisperXVADStage: [
            "min_length",
            "max_length",
            "vad_onset",
            "vad_offset",
            "segments_key",
            "audio_filepath_key",
            "name",
            "resources",
            "_vad_model",
        ],
    }
    for cls, expected_names in expected.items():
        signature = inspect.signature(cls.__init__)
        parameters = list(signature.parameters.values())[1:]
        positional = [
            parameter.name for parameter in parameters if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        assert positional == expected_names
        keyword_only = {parameter.name for parameter in parameters if parameter.kind is inspect.Parameter.KEYWORD_ONLY}
        assert {"waveform_key", "sample_rate_key", "input_residency", "fanout"}.issubset(keyword_only)
        bound = signature.bind(None, *range(len(expected_names)))
        assert list(bound.arguments)[1:] == expected_names


@pytest.mark.parametrize("fail_after_rttm", [False, True], ids=["success", "failure"])
def test_pyannote_cleans_temporary_rttm_siblings(
    fail_after_rttm: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, _seen = _make_stage(
        "pyannote",
        monkeypatch,
        input_residency="waveform",
        write_rttm=True,
        fail_after_rttm=fail_after_rttm,
    )

    def temp_resolver(item: dict[str, Any], **kwargs: object) -> str | None:
        return resolve_audio_path(item, temp_dir=str(tmp_path), **kwargs)

    monkeypatch.setattr(pyannote_module, "resolve_audio_path", temp_resolver)
    task = AudioTask(data={"waveform": np.ones((1, 12), dtype=np.float32), "sample_rate": _SAMPLE_RATE})
    if fail_after_rttm:
        with pytest.raises(RuntimeError, match="failure after RTTM"):
            stage.process(task)
    else:
        stage.process(task)
    assert list(tmp_path.glob("*.wav")) == []
    assert list(tmp_path.glob("*.rttm")) == []


def test_pyannote_preserves_durable_file_rttm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage, _seen = _make_stage("pyannote", monkeypatch, write_rttm=True)
    audio_path = tmp_path / "durable.wav"
    _write_audio(audio_path, np.ones((1, 12), dtype=np.float32))

    stage.process(AudioTask(data={"audio_filepath": str(audio_path)}))

    assert audio_path.with_suffix(".rttm").exists()


def test_static_and_configured_inference_hints_are_truthful(
    tmp_path: Path,
) -> None:
    pyannote_static = static_contract(PyAnnoteDiarizationStage)
    assert pyannote_static.gates.requires_internet_first_run
    assert pyannote_static.gates.runtime_secrets == ["HF_TOKEN"]
    assert pyannote_static.gates.writes_to_disk
    assert pyannote_static.gates.output_path_params == []
    assert pyannote_static.gates.per_row_independent is False
    assert {"1:1", "1:N fan-out"}.issubset(pyannote_static.cardinality_options)

    local_pipeline = tmp_path / "pyannote-pipeline"
    local_pipeline.mkdir()
    local_pyannote = build_contract(PyAnnoteDiarizationStage(model_name=str(local_pipeline), write_rttm=False))
    assert local_pyannote.gates.requires_internet_first_run
    assert local_pyannote.gates.runtime_secrets == []
    assert not local_pyannote.gates.writes_to_disk

    whisperx = build_contract(WhisperXVADStage(resources=Resources(gpus=0)))
    assert whisperx.gates.requires_internet_first_run
    assert static_contract(WhisperXVADStage).gates.requires_internet_first_run

    asr_static = static_contract(ASRStage)
    asr_configured = build_contract(
        ASRStage(
            adapter_target=_ASR_TARGET,
            model_id="mock/model",
            resources=Resources(gpus=0),
        )
    )
    assert asr_static.gates.requires_gpu
    assert asr_static.gates.requires_internet_first_run
    assert asr_static.gates.per_row_independent is True
    assert asr_static.dispatch == "process_batch"
    assert not asr_configured.gates.requires_gpu
    assert asr_configured.gates.requires_internet_first_run
    assert asr_configured.gates.per_row_independent is True

    default_sortformer = build_contract(InferenceSortformerStage(resources=Resources(gpus=0)))
    local_sortformer = build_contract(
        InferenceSortformerStage(model_path="/models/local.nemo", resources=Resources(gpus=0))
    )
    injected_sortformer = build_contract(InferenceSortformerStage(diar_model=MagicMock(), resources=Resources(gpus=0)))
    assert default_sortformer.gates.requires_internet_first_run
    assert not local_sortformer.gates.requires_internet_first_run
    assert not injected_sortformer.gates.requires_internet_first_run
    assert static_contract(InferenceSortformerStage).gates.requires_internet_first_run
    assert static_contract(InferenceSortformerStage).gates.output_path_params == ["rttm_out_dir"]
    assert static_contract(InferenceSortformerStage).gates.per_row_independent is False
    with patch("nemo_curator.stages.audio.inference.speaker_diarization.sortformer.snapshot_download") as download:
        InferenceSortformerStage(diar_model=MagicMock()).setup_on_node()
    download.assert_not_called()


def test_waveform_only_identities_are_stable_and_explicit_ids_win(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waveform = np.arange(12, dtype=np.float32)[None, :]
    changed = waveform + 1

    pyannote, _seen = _make_stage(
        "pyannote",
        monkeypatch,
        input_residency="waveform",
        fanout=True,
    )

    def pyannote_identity(audio: np.ndarray, **extra: object) -> tuple[str, str]:
        child = pyannote.process(AudioTask(data={"waveform": audio, "sample_rate": _SAMPLE_RATE, **extra}))[0]
        return child.data["original_file"], child.data["speaker"]

    first_pyannote = pyannote_identity(waveform)
    assert pyannote_identity(waveform) == first_pyannote
    assert pyannote_identity(changed) != first_pyannote
    explicit_original, explicit_speaker = pyannote_identity(waveform, audio_item_id="explicit-id")
    assert explicit_original == "explicit-id"
    assert explicit_speaker.startswith("explicit-id_")

    whisperx, _seen = _make_stage(
        "whisperx",
        monkeypatch,
        input_residency="waveform",
        fanout=True,
    )

    def whisperx_identity(audio: np.ndarray, **extra: object) -> str:
        child = whisperx.process(AudioTask(data={"waveform": audio, "sample_rate": _SAMPLE_RATE, **extra}))[0]
        return child.data["original_file"]

    first_whisperx = whisperx_identity(waveform)
    assert whisperx_identity(waveform) == first_whisperx
    assert whisperx_identity(changed) != first_whisperx
    assert whisperx_identity(waveform, audio_item_id="explicit-id") == "explicit-id"

    sortformer, _seen = _make_stage(
        "sortformer",
        monkeypatch,
        input_residency="waveform",
        rttm_out_dir=str(tmp_path),
    )
    task_data = {"waveform": waveform, "sample_rate": _SAMPLE_RATE}
    sortformer.process(AudioTask(data=dict(task_data)))
    first_names = {path.name for path in tmp_path.glob("*.rttm")}
    sortformer.process(AudioTask(data=dict(task_data)))
    assert {path.name for path in tmp_path.glob("*.rttm")} == first_names
    sortformer.process(AudioTask(data={"waveform": changed, "sample_rate": _SAMPLE_RATE}))
    assert len(list(tmp_path.glob("*.rttm"))) == 2
    sortformer.process(
        AudioTask(
            data={
                "waveform": waveform,
                "sample_rate": _SAMPLE_RATE,
                "session_name": "explicit-session",
            }
        )
    )
    assert (tmp_path / "explicit-session.rttm").exists()


def test_file_identity_precedence_preserves_existing_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waveform = np.arange(12, dtype=np.float32)[None, :]
    audio_path = tmp_path / "real-source.wav"
    _write_audio(audio_path, waveform)

    pyannote, _seen = _make_stage("pyannote", monkeypatch, fanout=True)
    pyannote_child = pyannote.process(
        AudioTask(
            data={
                "audio_filepath": str(audio_path),
                "original_file": "/stale/provenance.wav",
            }
        )
    )[0]
    assert pyannote_child.data["speaker"].startswith("real-source_")
    assert pyannote_child.data["original_file"] == "/stale/provenance.wav"

    pyannote_with_id = pyannote.process(
        AudioTask(
            data={
                "audio_filepath": str(audio_path),
                "audio_item_id": "legacy-item",
                "original_file": "/stale/provenance.wav",
            }
        )
    )[0]
    assert pyannote_with_id.data["speaker"].startswith("legacy-item_")
    assert pyannote_with_id.data["original_file"] == "/stale/provenance.wav"

    pyannote_with_speaker = pyannote.process(
        AudioTask(
            data={
                "audio_filepath": str(audio_path),
                "speaker_id": "legacy-speaker",
                "original_file": "/stale/provenance.wav",
            }
        )
    )[0]
    assert pyannote_with_speaker.data["speaker"].startswith("legacy-speaker_")

    sortformer, _seen = _make_stage(
        "sortformer",
        monkeypatch,
        fanout=True,
        rttm_out_dir=str(tmp_path / "rttm"),
    )
    sortformer_child = sortformer.process(
        AudioTask(
            data={
                "audio_filepath": str(audio_path),
                "audio_item_id": "new-item-id",
                "original_file": "/stale/provenance.wav",
            }
        )
    )[0]
    assert (tmp_path / "rttm" / "real-source.rttm").exists()
    assert sortformer_child.data["original_file"] == "/stale/provenance.wav"

    sortformer.process(
        AudioTask(
            data={
                "audio_filepath": str(audio_path),
                "session_name": "explicit-session",
                "audio_item_id": "ignored-item-id",
            }
        )
    )
    assert (tmp_path / "rttm" / "explicit-session.rttm").exists()

    whisperx, _seen = _make_stage("whisperx", monkeypatch, fanout=True)
    whisperx_child = whisperx.process(
        AudioTask(
            data={
                "audio_filepath": str(audio_path),
                "original_file": "/stale/provenance.wav",
            }
        )
    )[0]
    assert whisperx_child.data["original_file"] == "/stale/provenance.wav"
