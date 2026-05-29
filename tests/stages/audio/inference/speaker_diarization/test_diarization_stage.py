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

"""Unit tests for the generic DiarizationStage.

The stage is tested with a fake adapter so these tests never touch
PyAnnote and never need a GPU. The PyAnnote-specific code is covered by
``tests/adapters/diarization/test_pyannote_adapter.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from nemo_curator.adapters.diarization import DiarizationResult, DiarSegment
from nemo_curator.stages.audio.inference.speaker_diarization import DiarizationStage
from nemo_curator.tasks import AudioTask


# ---------------------------------------------------------------------------
# Fake adapter used as the swap target
# ---------------------------------------------------------------------------


@dataclass
class _FakeDiarAdapter:
    """In-process diarization adapter used as DiarizationStage.adapter_target."""

    model_id: str = "fake/diar"
    revision: str | None = None
    device: str = "cpu"
    fixed_result: DiarizationResult | None = None
    setup_called: int = 0
    teardown_called: int = 0
    last_batch: list[dict[str, Any]] | None = None
    last_metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def prefetch_weights(cls, model_id: str, revision: str | None = None) -> None:
        del model_id, revision  # no-op for the fake

    def setup(self) -> None:
        self.setup_called += 1

    def teardown(self) -> None:
        self.teardown_called += 1

    def diarize_batch(self, items: list[dict[str, Any]]) -> list[DiarizationResult]:
        self.last_batch = list(items)
        self.last_metrics = {"batch_size": float(len(items))}
        if self.fixed_result is not None:
            return [self.fixed_result for _ in items]
        return [
            DiarizationResult(
                diar_segments=[DiarSegment(start=0.0, end=1.0, speaker="spk_0")],
                overlap_segments=[],
                model_id=self.model_id,
            )
            for _ in items
        ]


_ADAPTER_TARGET = f"{__name__}._FakeDiarAdapter"


def _audio_task(**data: Any) -> AudioTask:  # noqa: ANN401
    base = {"resampled_audio_filepath": "/tmp/fake.wav", "audio_item_id": "id_1", "duration": 10.0}
    base.update(data)
    return AudioTask(data=base)


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestDiarizationStageConstruction:
    def test_requires_adapter_target(self) -> None:
        with pytest.raises(ValueError, match="adapter_target is required"):
            DiarizationStage()

    def test_default_io_keys(self) -> None:
        stage = DiarizationStage(adapter_target=_ADAPTER_TARGET)
        assert stage.audio_filepath_key == "resampled_audio_filepath"
        assert stage.segments_key == "segments"
        assert stage.overlap_segments_key == "overlap_segments"
        assert stage.non_speaker_max_length == 40.0

    def test_inputs_outputs(self) -> None:
        stage = DiarizationStage(adapter_target=_ADAPTER_TARGET)
        assert stage.inputs() == ([], ["resampled_audio_filepath"])
        assert stage.outputs() == ([], ["resampled_audio_filepath", "segments", "overlap_segments"])

    def test_outputs_skips_overlap_key_when_disabled(self) -> None:
        stage = DiarizationStage(adapter_target=_ADAPTER_TARGET, overlap_segments_key=None)
        assert stage.outputs() == ([], ["resampled_audio_filepath", "segments"])


# ---------------------------------------------------------------------------
# Adapter lifecycle (setup_on_node + setup + teardown)
# ---------------------------------------------------------------------------


class TestDiarizationStageLifecycle:
    def test_setup_instantiates_adapter_with_tier1_and_tier2_kwargs(self) -> None:
        stage = DiarizationStage(
            adapter_target=_ADAPTER_TARGET,
            model_id="pyannote/x",
            revision="rev-1",
            adapter_kwargs={"fixed_result": None},
        )
        stage.setup()
        assert isinstance(stage._adapter, _FakeDiarAdapter)
        assert stage._adapter.model_id == "pyannote/x"
        assert stage._adapter.revision == "rev-1"
        assert stage._adapter.setup_called == 1

    def test_setup_forwards_device(self) -> None:
        stage = DiarizationStage(adapter_target=_ADAPTER_TARGET)
        stage.setup()
        # Default Resources(gpus=1) -> requires_gpu -> "cuda"
        assert stage._adapter.device == "cuda"

    def test_teardown_releases_adapter(self) -> None:
        stage = DiarizationStage(adapter_target=_ADAPTER_TARGET)
        stage.setup()
        adapter = stage._adapter
        stage.teardown()
        assert stage._adapter is None
        assert adapter.teardown_called == 1

    def test_setup_on_node_calls_prefetch_weights(self) -> None:
        with patch.object(_FakeDiarAdapter, "prefetch_weights") as mock_prefetch:
            stage = DiarizationStage(adapter_target=_ADAPTER_TARGET, model_id="m", revision="r")
            stage.setup_on_node()
        mock_prefetch.assert_called_once_with("m", "r")

    def test_setup_on_node_propagates_when_fail_on_error(self) -> None:
        with patch.object(_FakeDiarAdapter, "prefetch_weights", side_effect=RuntimeError("boom")):
            stage = DiarizationStage(adapter_target=_ADAPTER_TARGET, prefetch_fail_on_error=True)
            with pytest.raises(RuntimeError, match="prefetch_weights failed"):
                stage.setup_on_node()

    def test_setup_on_node_swallows_when_fail_disabled(self) -> None:
        with patch.object(_FakeDiarAdapter, "prefetch_weights", side_effect=RuntimeError("boom")):
            stage = DiarizationStage(adapter_target=_ADAPTER_TARGET, prefetch_fail_on_error=False)
            stage.setup_on_node()  # must not raise

    def test_xenna_stage_spec_default_empty(self) -> None:
        stage = DiarizationStage(adapter_target=_ADAPTER_TARGET)
        assert stage.xenna_stage_spec() == {}

    def test_xenna_stage_spec_with_num_workers(self) -> None:
        stage = DiarizationStage(adapter_target=_ADAPTER_TARGET, xenna_num_workers=4)
        assert stage.xenna_stage_spec() == {"num_workers": 4}


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


class TestDiarizationStageProcess:
    def test_process_requires_setup_first(self) -> None:
        stage = DiarizationStage(adapter_target=_ADAPTER_TARGET)
        with pytest.raises(RuntimeError, match="setup\\(\\) was not called"):
            stage.process(_audio_task())

    def test_process_requires_audio_filepath_key(self) -> None:
        stage = DiarizationStage(adapter_target=_ADAPTER_TARGET)
        stage.setup()
        task = AudioTask(data={"audio_item_id": "id_1"})
        with pytest.raises(ValueError, match="Missing key 'resampled_audio_filepath'"):
            stage.process(task)

    def test_process_writes_segments_and_overlap_keys(self) -> None:
        stage = DiarizationStage(adapter_target=_ADAPTER_TARGET)
        stage.setup()
        stage._adapter.fixed_result = DiarizationResult(
            diar_segments=[
                DiarSegment(start=0.0, end=2.0, speaker="id_1_spk_0"),
                DiarSegment(start=4.0, end=6.0, speaker="id_1_spk_1"),
            ],
            overlap_segments=[DiarSegment(start=3.0, end=4.0, speaker="id_1_spk_0")],
            model_id="pyannote/x",
        )
        task = stage.process(_audio_task(duration=10.0))
        segs = task.data["segments"]
        # 2 real turns + 3 no-speaker gap fills (0-0, 2-4, 6-10)
        speaker_turns = [s for s in segs if s["speaker"] != "no-speaker"]
        assert len(speaker_turns) == 2
        no_speakers = [s for s in segs if s["speaker"] == "no-speaker"]
        assert len(no_speakers) >= 2
        assert task.data["overlap_segments"] == [
            {"speaker": "id_1_spk_0", "start": 3.0, "end": 4.0}
        ]

    def test_process_skips_overlap_key_when_disabled(self) -> None:
        stage = DiarizationStage(adapter_target=_ADAPTER_TARGET, overlap_segments_key=None)
        stage.setup()
        stage._adapter.fixed_result = DiarizationResult(
            diar_segments=[DiarSegment(start=0.0, end=2.0, speaker="spk_0")],
            overlap_segments=[DiarSegment(start=5.0, end=6.0, speaker="spk_0")],
            model_id="x",
        )
        task = stage.process(_audio_task(duration=10.0))
        assert "overlap_segments" not in task.data
        assert "segments" in task.data

    def test_process_non_speaker_max_length_chunks_long_gaps(self) -> None:
        stage = DiarizationStage(
            adapter_target=_ADAPTER_TARGET,
            non_speaker_max_length=5.0,
        )
        stage.setup()
        stage._adapter.fixed_result = DiarizationResult(
            diar_segments=[DiarSegment(start=0.0, end=1.0, speaker="spk_0")],
            overlap_segments=[],
            model_id="x",
        )
        task = stage.process(_audio_task(duration=30.0))
        no_speakers = [s for s in task.data["segments"] if s["speaker"] == "no-speaker"]
        # 1s..30s = 29s gap, chunked at 5s -> ceil(29/5)=6 chunks.
        assert len(no_speakers) == 6
        for ns in no_speakers:
            assert ns["end"] - ns["start"] <= 5.0 + 1e-9

    def test_process_forwards_item_dict_to_adapter(self) -> None:
        stage = DiarizationStage(
            adapter_target=_ADAPTER_TARGET,
            waveform_key="waveform",
            sample_rate_key="sample_rate",
        )
        stage.setup()
        task = _audio_task(
            duration=8.0,
            waveform=[0.0, 1.0],
            sample_rate=16000,
        )
        stage.process(task)
        batch = stage._adapter.last_batch
        assert batch is not None and len(batch) == 1
        item = batch[0]
        assert item["audio_filepath"] == "/tmp/fake.wav"
        assert item["audio_item_id"] == "id_1"
        assert item["duration"] == 8.0
        assert item["waveform"] == [0.0, 1.0]
        assert item["sample_rate"] == 16000

    def test_process_uses_get_audio_duration_when_missing(self) -> None:
        stage = DiarizationStage(adapter_target=_ADAPTER_TARGET)
        stage.setup()
        with patch(
            "nemo_curator.stages.audio.inference.speaker_diarization.stage.get_audio_duration",
            return_value=12.0,
        ) as mock_dur:
            task = AudioTask(
                data={"resampled_audio_filepath": "/tmp/fake.wav", "audio_item_id": "id_1"}
            )
            stage.process(task)
        mock_dur.assert_called_once_with("/tmp/fake.wav")

    def test_process_uses_default_diar_segment_when_adapter_returns_empty(self) -> None:
        stage = DiarizationStage(adapter_target=_ADAPTER_TARGET)
        stage.setup()
        stage._adapter.fixed_result = DiarizationResult(
            diar_segments=[], overlap_segments=[], model_id="x"
        )
        task = stage.process(_audio_task(duration=4.0))
        # Whole clip is one big no-speaker block, chunked at default 40.0s.
        no_speakers = [s for s in task.data["segments"] if s["speaker"] == "no-speaker"]
        assert len(no_speakers) == 1
        assert no_speakers[0]["start"] == 0.0
        assert no_speakers[0]["end"] == 4.0


# ---------------------------------------------------------------------------
# Metric logging
# ---------------------------------------------------------------------------


class TestDiarizationStageMetrics:
    def test_log_metrics_includes_adapter_aliases(self) -> None:
        stage = DiarizationStage(adapter_target=_ADAPTER_TARGET)
        stage.setup()
        observed: list[dict[str, float]] = []

        def capture(metrics: dict[str, float]) -> None:
            observed.append(metrics)

        stage._log_metrics = capture  # type: ignore[assignment]
        stage._adapter.fixed_result = DiarizationResult(
            diar_segments=[
                DiarSegment(start=0.0, end=1.0, speaker="A"),
                DiarSegment(start=2.0, end=3.0, speaker="B"),
            ],
            overlap_segments=[DiarSegment(start=1.5, end=2.0, speaker="A")],
            model_id="x",
        )
        stage.process(_audio_task(duration=5.0))

        assert observed, "log_metrics must be called"
        m = observed[-1]
        assert m["speakers_detected"] == 2.0
        assert m["overlap_segments_detected"] == 1.0
        assert m["audio_duration"] == 5.0
        # Adapter-side last_metrics keys must be prefixed with "model_".
        assert m["model_batch_size"] == 1.0
        assert "process_time" in m
