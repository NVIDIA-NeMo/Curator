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

"""Unit tests for UTMOSFilterStage."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch
from nemo_curator.stages.audio._agent._agent_registry import static_contract
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready, assert_residency_consumption

from nemo_curator.stages.audio.filtering.utmos import UTMOSFilterStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path


def _make_task(duration_s: float = 1.0, sample_rate: int = 16000) -> AudioTask:
    num_samples = int(duration_s * sample_rate)
    return AudioTask(
        data={"waveform": torch.randn(1, num_samples), "sample_rate": sample_rate},
        dataset_name="test",
    )


def _sequential_model(scores: list[float]) -> MagicMock:
    """A model returning ``scores`` in order, so one call per segment can differ."""
    remaining = list(scores)
    model = MagicMock()
    model.side_effect = lambda *_args, **_kwargs: torch.tensor([remaining.pop(0)])
    model.parameters = lambda: iter([torch.tensor([0.0])])
    return model


def _mock_model(score: float) -> MagicMock:
    model = MagicMock()
    model.return_value = torch.tensor([score])
    model.parameters = lambda: iter([torch.tensor([0.0])])
    return model


def _write_wav(path: Path, *, samples: int = 32, sample_rate: int = 16000) -> str:
    sf.write(path, np.linspace(-0.25, 0.25, samples, dtype=np.float32), sample_rate)
    return str(path)


def _stage(*, mode: str = "task", scorable: bool = True, input_residency: str = "auto") -> UTMOSFilterStage:
    stage = UTMOSFilterStage(mode=mode, action="annotate", input_residency=input_residency)
    if scorable:
        stage._model = _mock_model(4.0)
    return stage


class TestUTMOSFilterStage:
    def test_legacy_positional_constructor_order_is_preserved(self) -> None:
        resources = Resources(cpus=2)

        stage = UTMOSFilterStage(4.1, 22050, "legacy", 8, resources)

        assert stage.mos_threshold == 4.1
        assert stage.sample_rate == 22050
        assert stage.name == "legacy"
        assert stage.batch_size == 8
        assert stage.resources is resources

    @pytest.mark.parametrize("residency", ["wavefrom", "", "FILE"])
    def test_invalid_input_residency_is_rejected(self, residency: str) -> None:
        with pytest.raises(ValueError, match="input_residency"):
            UTMOSFilterStage(input_residency=residency)  # type: ignore[arg-type]

    def test_score_key_cannot_overwrite_audio_input(self) -> None:
        with (
            patch("nemo_curator.stages.audio.filtering.utmos.torch.hub.load") as load_model,
            pytest.raises(ValueError, match="must not collide"),
        ):
            UTMOSFilterStage(score_key="waveform")

        load_model.assert_not_called()

    def test_static_contract_conservatively_reports_download_and_row_independence(self) -> None:
        contract = static_contract(UTMOSFilterStage)

        assert contract.gates.requires_internet_first_run is True
        assert contract.gates.per_row_independent is True

    def test_native_defaults_remain_filter_and_auto(self) -> None:
        stage = UTMOSFilterStage()

        assert stage.action == "filter"
        assert stage.mode == "auto"
        assert stage.mos_threshold == 3.5

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_process_passes_above_threshold(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        stage._model = _mock_model(4.5)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert abs(result.data["utmos_mos"] - 4.5) < 1e-3

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_auto_incomplete_waveform_pair_falls_back_to_file(self, mock_ensure: MagicMock, tmp_path: Path) -> None:
        stage = UTMOSFilterStage(action="annotate", input_residency="auto")
        stage._model = _mock_model(4.0)
        task = AudioTask(
            dataset_name="test",
            data={
                "waveform": torch.ones(1, 7),
                "audio_filepath": _write_wav(tmp_path / "fallback.wav", samples=32),
            },
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        inferred_waveform = stage._model.call_args.args[0]
        assert inferred_waveform.shape == (1, 32)
        assert task.data["sample_rate"] == 16000
        assert torch.allclose(inferred_waveform, task.data["waveform"])

    @pytest.mark.parametrize(
        "orphan",
        [
            pytest.param({"waveform": torch.ones(1, 7)}, id="waveform-without-rate"),
            pytest.param({"sample_rate": 8000}, id="rate-without-waveform"),
        ],
    )
    def test_auto_partial_pair_is_replaced_with_file_audio(self, orphan: dict, tmp_path: Path) -> None:
        stage = _stage(input_residency="auto")
        task = AudioTask(
            dataset_name="test",
            data={**orphan, "audio_filepath": _write_wav(tmp_path / "fallback.wav", samples=31)},
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        inferred_waveform = stage._model.call_args.args[0]
        assert stage._model.call_args.kwargs["sr"] == task.data["sample_rate"] == 16000
        assert inferred_waveform.shape == task.data["waveform"].shape == (1, 31)
        assert torch.allclose(inferred_waveform, task.data["waveform"])

    @pytest.mark.parametrize("input_residency", ["file", "auto"])
    def test_ordinary_file_only_input_does_not_persist_residency(
        self,
        input_residency: str,
        tmp_path: Path,
    ) -> None:
        stage = _stage(input_residency=input_residency)
        task = AudioTask(
            dataset_name="test",
            data={"audio_filepath": _write_wav(tmp_path / "file-only.wav", samples=29)},
        )

        assert isinstance(stage.process(task), AudioTask)
        inferred_waveform = stage._model.call_args.args[0]
        assert stage._model.call_args.kwargs["sr"] == 16000
        assert inferred_waveform.shape == (1, 29)
        assert "waveform" not in task.data
        assert "sample_rate" not in task.data

    def test_nested_partial_auto_hydration_is_declared_and_conformant(self, tmp_path: Path) -> None:
        stage = _stage(mode="auto", input_residency="auto")
        path = _write_wav(tmp_path / "nested-file.wav", samples=31)

        def fixture() -> AudioTask:
            return AudioTask(
                dataset_name="test",
                data={"segments": [{"audio_filepath": path, "sample_rate": 8000}]},
            )

        assert_agent_ready(stage, fixture, expected_cardinality="1:1", segments_key="segments")
        task = fixture()
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        segment = task.data["segments"][0]
        inferred_waveform = stage._model.call_args.args[0]
        assert stage._model.call_args.kwargs["sr"] == segment["sample_rate"] == 16000
        assert torch.allclose(inferred_waveform, segment["waveform"])

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_process_filters_below_threshold(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=4.0)
        stage._model = _mock_model(2.5)

        result = stage.process(_make_task())

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_annotate_keeps_below_threshold(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=4.0, action="annotate")
        stage._model = _mock_model(2.5)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert abs(result.data["utmos_mos"] - 2.5) < 1e-3
        assert stage.describe().cardinality == "1:1"

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_none_threshold_passes_all(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=None)
        stage._model = _mock_model(1.0)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert abs(result.data["utmos_mos"] - 1.0) < 1e-3

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_prediction_error_skips(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        model = MagicMock(side_effect=RuntimeError("CUDA error"))
        model.parameters = lambda: iter([torch.tensor([0.0])])
        stage._model = model

        result = stage.process(_make_task())

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_no_waveform_no_filepath_skipped(self, mock_ensure: MagicMock) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        stage._model = _mock_model(4.0)

        task = AudioTask(data={"some_key": "value"}, dataset_name="test")
        result = stage.process(task)

        assert result == []

    def test_model_not_loaded(self) -> None:
        stage = UTMOSFilterStage(mos_threshold=3.0)
        stage._model = None

        with patch.object(stage, "_ensure_model"):
            result = stage.process(_make_task())

        assert result == []

    def test_teardown_clears_model(self) -> None:
        stage = UTMOSFilterStage()
        stage._model = MagicMock()
        stage.teardown()
        assert stage._model is None

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_process_nested_segments_filters(self, mock_ensure: MagicMock) -> None:
        """Nested segments: only segments above threshold survive."""
        stage = UTMOSFilterStage(mos_threshold=3.0)
        call_count = {"n": 0}

        def model_side_effect(_waveform: torch.Tensor, sr: int = 16000) -> torch.Tensor:  # noqa: ARG001
            call_count["n"] += 1
            return torch.tensor([4.0 if call_count["n"] % 2 == 1 else 2.0])

        model = MagicMock(side_effect=model_side_effect)
        model.parameters = lambda: iter([torch.tensor([0.0])])
        stage._model = model

        sr = 16000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(4)]
        task = AudioTask(
            data={"segments": segments, "original_file": "test.wav"},
            dataset_name="test",
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert len(result.data["segments"]) == 2
        for seg in result.data["segments"]:
            assert "utmos_mos" in seg

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_process_nested_all_filtered_returns_empty(self, mock_ensure: MagicMock) -> None:
        """Nested segments: when all fail threshold, return []."""
        stage = UTMOSFilterStage(mos_threshold=4.0)
        stage._model = _mock_model(2.0)

        sr = 16000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(3)]
        task = AudioTask(
            data={"segments": segments},
            dataset_name="test",
        )

        result = stage.process(task)

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_annotate_nested_keeps_all_segments(self, mock_ensure: MagicMock) -> None:
        """Nested annotate mode scores every segment without threshold dropping."""
        stage = UTMOSFilterStage(mos_threshold=4.0, action="annotate")
        stage._model = _mock_model(2.0)

        sr = 16000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(3)]
        task = AudioTask(
            data={"segments": segments},
            dataset_name="test",
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert len(result.data["segments"]) == 3
        assert all(seg["utmos_mos"] == 2.0 for seg in result.data["segments"])


class TestSegmentModeFiltering:
    """Filtering per segment keeps the survivors and their scores, not the whole row.

    Lifted from tests/stages/audio/test_agent_simulation_pipelines.py -- it drives only
    UTMOSFilterStage, and was the sole coverage of partial-survivor segment filtering.
    """

    def test_partial_survivors_keep_their_ids_and_scores(self) -> None:
        task = AudioTask(
            dataset_name="t",
            data={
                "agent_segments": [
                    {"id": i, "agent_waveform": torch.randn(1, 3200), "agent_sr": 16000} for i in range(3)
                ]
            },
        )
        stage = UTMOSFilterStage(
            mos_threshold=4.0,
            action="filter",
            mode="segments",
            input_residency="waveform",
            waveform_key="agent_waveform",
            sample_rate_key="agent_sr",
            segments_key="agent_segments",
            score_key="agent_utmos",
        )
        stage._model = _sequential_model([4.2, 2.0, 4.1])

        out = stage.process(task)

        assert isinstance(out, AudioTask)
        assert [seg["id"] for seg in out.data["agent_segments"]] == [0, 2], "only the passing segments survive"
        assert [seg["agent_utmos"] for seg in out.data["agent_segments"]] == pytest.approx([4.2, 4.1])


@pytest.mark.parametrize("mode", ["task", "segments", "auto"])
@pytest.mark.parametrize("scorable", [True, False], ids=["success", "unscorable"])
def test_utmos_agent_conformance_covers_all_scopes(mode: str, scorable: bool) -> None:
    stage = _stage(mode=mode, scorable=scorable, input_residency="waveform")

    def fixture() -> AudioTask:
        audio = {"waveform": torch.zeros(1, 32), "sample_rate": 16000}
        data = {"segments": [audio]} if mode == "segments" else audio
        return AudioTask(dataset_name="test", data=data)

    assert_agent_ready(stage, fixture, expected_cardinality="1:1", segments_key="segments")


@pytest.mark.parametrize("nested", [False, True], ids=["task", "segments"])
@pytest.mark.parametrize("scorable", [True, False], ids=["success", "unscorable"])
def test_utmos_auto_runtime_outputs_match_selected_scope(nested: bool, scorable: bool) -> None:
    stage = _stage(mode="auto", scorable=scorable, input_residency="waveform")

    def fixture() -> AudioTask:
        audio = {"waveform": torch.zeros(1, 32), "sample_rate": 16000}
        return AudioTask(dataset_name="test", data={"segments": [audio]} if nested else audio)

    assert_agent_ready(stage, fixture, expected_cardinality="1:1", segments_key="segments")
    task = fixture()

    result = stage.process(task)

    assert isinstance(result, AudioTask)
    selected = task.data["segments"][0] if nested else task.data
    assert ("utmos_mos" in selected) is scorable
    if nested:
        assert "utmos_mos" not in task.data
    else:
        assert "segments" not in task.data


def test_utmos_consumes_file_and_waveform_residencies(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "utmos.wav")

    assert_residency_consumption(
        lambda residency: _stage(input_residency=residency),
        file_fixture=lambda: AudioTask(dataset_name="test", data={"audio_filepath": path}),
        waveform_fixture=lambda: AudioTask(
            dataset_name="test",
            data={"waveform": torch.zeros(1, 32), "sample_rate": 16000},
        ),
    )
