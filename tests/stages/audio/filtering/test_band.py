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

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch
from nemo_curator.stages.audio._agent._agent_registry import static_contract
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready, assert_residency_consumption
from nemo_curator.stages.audio._agent._planning import validate_pipeline

from nemo_curator.stages.audio.filtering.band import BandFilterStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path


def _make_task(waveform: torch.Tensor | None = None, sample_rate: int = 48000) -> AudioTask:
    if waveform is None:
        waveform = torch.randn(1, sample_rate)
    return AudioTask(
        data={"waveform": waveform, "sample_rate": sample_rate},
        dataset_name="test",
    )


def _write_wav(path: Path, *, samples: int = 32, sample_rate: int = 16000) -> str:
    sf.write(path, np.linspace(-0.25, 0.25, samples, dtype=np.float32), sample_rate)
    return str(path)


def _stage(*, mode: str = "task", scorable: bool = True, input_residency: str = "auto") -> BandFilterStage:
    stage = BandFilterStage(mode=mode, action="annotate", input_residency=input_residency)
    if scorable:
        stage._predictor = MagicMock()
        stage._predictor.predict_audio.return_value = "full_band"
    return stage


class TestBandFilterStage:
    def test_legacy_positional_constructor_order_is_preserved(self) -> None:
        resources = Resources(cpus=2)

        stage = BandFilterStage("model.joblib", "/cache", "narrow_band", "legacy", 8, resources)

        assert stage.model_path == "model.joblib"
        assert stage.cache_dir == "/cache"
        assert stage.band_value == "narrow_band"
        assert stage.name == "legacy"
        assert stage.batch_size == 8
        assert stage.resources is resources

    @pytest.mark.parametrize("residency", ["wavefrom", "", "FILE"])
    def test_invalid_input_residency_is_rejected(self, residency: str) -> None:
        with pytest.raises(ValueError, match="input_residency"):
            BandFilterStage(input_residency=residency)  # type: ignore[arg-type]

    def test_prediction_key_cannot_overwrite_audio_input(self) -> None:
        with (
            patch("nemo_curator.stages.audio.filtering.band.BandPredictor") as predictor,
            patch("nemo_curator.stages.audio.filtering.band.hf_hub_download") as download,
            pytest.raises(ValueError, match="must not collide"),
        ):
            BandFilterStage(prediction_key="waveform")

        predictor.assert_not_called()
        download.assert_not_called()

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param({"segments_key": "audio_filepath"}, id="segments-aliases-filepath"),
            pytest.param({"waveform_key": "sample_rate"}, id="waveform-aliases-rate"),
        ],
    )
    def test_audio_input_role_collisions_are_rejected(self, params: dict[str, str]) -> None:
        with pytest.raises(ValueError, match="Audio input keys must be distinct"):
            BandFilterStage(**params)

    @pytest.mark.parametrize("mode", ["task", "segments"])
    def test_invalid_resident_sample_rate_is_unscorable_without_inference(self, mode: str) -> None:
        stage = _stage(mode=mode, input_residency="waveform")
        audio = {"waveform": torch.zeros(8), "sample_rate": 16000.5}
        task = AudioTask(dataset_name="test", data=audio if mode == "task" else {"segments": [audio]})

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        stage._predictor.predict_audio.assert_not_called()
        assert stage.prediction_key not in audio

    def test_explicit_valid_model_path_returns_without_download(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"model")
        stage = BandFilterStage(model_path=str(model_path))

        with patch("nemo_curator.stages.audio.filtering.band.hf_hub_download") as download:
            assert stage._resolve_model_path() == str(model_path)

        download.assert_not_called()

    def test_explicit_missing_model_path_does_not_download(self, tmp_path: Path) -> None:
        stage = BandFilterStage(model_path=str(tmp_path / "missing.joblib"))

        with (
            patch("nemo_curator.stages.audio.filtering.band.hf_hub_download") as download,
            pytest.raises(FileNotFoundError, match="model_path"),
        ):
            stage._resolve_model_path()

        download.assert_not_called()

    def test_explicit_empty_model_path_does_not_download(self, tmp_path: Path) -> None:
        model_path = tmp_path / "empty.joblib"
        model_path.touch()
        stage = BandFilterStage(model_path=str(model_path))

        with (
            patch("nemo_curator.stages.audio.filtering.band.hf_hub_download") as download,
            pytest.raises(FileNotFoundError, match="empty"),
        ):
            stage._resolve_model_path()

        download.assert_not_called()

    def test_default_model_path_uses_download(self) -> None:
        stage = BandFilterStage(cache_dir="/cache")

        with patch(
            "nemo_curator.stages.audio.filtering.band.hf_hub_download",
            return_value="/cache/model.joblib",
        ) as download:
            assert stage._resolve_model_path() == "/cache/model.joblib"

        download.assert_called_once()

    def test_setup_on_node_prefetches_only_default_model(self, tmp_path: Path) -> None:
        explicit = tmp_path / "model.joblib"
        explicit.write_bytes(b"model")

        with patch("nemo_curator.stages.audio.filtering.band.hf_hub_download") as download:
            BandFilterStage(model_path=str(explicit)).setup_on_node()
            download.assert_not_called()

            BandFilterStage(cache_dir=str(tmp_path)).setup_on_node()
            download.assert_called_once()

    def test_static_contract_conservatively_reports_download_and_row_independence(self) -> None:
        contract = static_contract(BandFilterStage)

        assert contract.gates.requires_internet_first_run is True
        assert contract.gates.per_row_independent is True
        assert BandFilterStage().describe().gates.requires_internet_first_run is True
        assert (
            BandFilterStage(model_path="/configured/model.joblib").describe().gates.requires_internet_first_run
            is False
        )

    def test_resident_waveform_uses_only_file_header_for_missing_sample_rate(self, tmp_path: Path) -> None:
        stage = _stage()
        resident = torch.full((1, 13), 0.75)
        path = _write_wav(tmp_path / "different.wav", samples=31)
        task = AudioTask(dataset_name="test", data={"waveform": resident, "audio_filepath": path})

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        predicted_waveform, predicted_sample_rate = stage._predictor.predict_audio.call_args.args
        assert predicted_waveform.shape == (1, 13)
        assert torch.allclose(predicted_waveform, resident)
        assert predicted_sample_rate == 16000
        assert task.data["sample_rate"] == 16000

    def test_nested_header_completion_preserves_samples_and_conforms(self, tmp_path: Path) -> None:
        stage = _stage(mode="auto")
        resident = torch.full((1, 13), 0.75)
        path = _write_wav(tmp_path / "nested-header.wav", samples=31)

        def fixture() -> AudioTask:
            return AudioTask(
                dataset_name="test",
                data={"segments": [{"waveform": resident.clone(), "audio_filepath": path}]},
            )

        assert_agent_ready(stage, fixture, expected_cardinality="1:1", segments_key="segments")
        task = fixture()
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        segment = task.data["segments"][0]
        predicted_waveform, predicted_sample_rate = stage._predictor.predict_audio.call_args.args
        assert torch.allclose(predicted_waveform, resident)
        assert torch.allclose(segment["waveform"], resident)
        assert predicted_sample_rate == segment["sample_rate"] == 16000

    def test_auto_orphan_sample_rate_is_replaced_with_file_pair(self, tmp_path: Path) -> None:
        stage = _stage()
        path = _write_wav(tmp_path / "different.wav", samples=31, sample_rate=16000)
        task = AudioTask(dataset_name="test", data={"sample_rate": 8000, "audio_filepath": path})

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        predicted_waveform, predicted_sample_rate = stage._predictor.predict_audio.call_args.args
        assert predicted_sample_rate == task.data["sample_rate"] == 16000
        assert predicted_waveform.shape == task.data["waveform"].shape == (1, 31)
        assert torch.allclose(predicted_waveform, task.data["waveform"])

    def test_file_mode_hydrates_pair_for_downstream_auto_consumption(self, tmp_path: Path) -> None:
        path = _write_wav(tmp_path / "file-only.wav", samples=29, sample_rate=16000)
        first = _stage(input_residency="file")
        task = AudioTask(dataset_name="test", data={"audio_filepath": path})

        assert isinstance(first.process(task), AudioTask)
        first_waveform, first_rate = first._predictor.predict_audio.call_args.args
        assert first_rate == task.data["sample_rate"] == 16000
        assert torch.allclose(first_waveform, task.data["waveform"])

        task.data.pop("audio_filepath")
        downstream = _stage(input_residency="auto")
        assert isinstance(downstream.process(task), AudioTask)
        downstream_waveform, downstream_rate = downstream._predictor.predict_audio.call_args.args
        assert downstream_rate == task.data["sample_rate"]
        assert torch.allclose(downstream_waveform, task.data["waveform"])

    def test_nested_file_hydration_is_declared_and_conformant(self, tmp_path: Path) -> None:
        stage = _stage(mode="auto", input_residency="file")
        path = _write_wav(tmp_path / "nested-file.wav", samples=31)

        def fixture() -> AudioTask:
            return AudioTask(dataset_name="test", data={"segments": [{"audio_filepath": path}]})

        assert_agent_ready(stage, fixture, expected_cardinality="1:1", segments_key="segments")
        task = fixture()
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        segment = task.data["segments"][0]
        predicted_waveform, predicted_sample_rate = stage._predictor.predict_audio.call_args.args
        assert torch.allclose(predicted_waveform, segment["waveform"])
        assert predicted_sample_rate == segment["sample_rate"] == 16000

    @pytest.mark.parametrize("action", ["filter", "annotate"])
    def test_unscorable_rerun_clears_stale_prediction(self, action: str) -> None:
        stage = BandFilterStage(mode="task", action=action, input_residency="waveform")
        stage._predictor = MagicMock()
        stage._predictor.predict_audio.return_value = "Error: unavailable"
        task = AudioTask(
            dataset_name="test",
            data={
                "waveform": torch.zeros(1, 32),
                "sample_rate": 16000,
                "band_prediction": "full_band",
            },
        )

        result = stage.process(task)

        assert "band_prediction" not in task.data
        if action == "annotate":
            assert isinstance(result, AudioTask)
        else:
            assert result == []

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_full_band_passes(self, mock_init: MagicMock) -> None:
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "full_band"
        stage._predictor = predictor

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert result.data["band_prediction"] == "full_band"

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_narrow_band_filtered_out(self, mock_init: MagicMock) -> None:
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "narrow_band"
        stage._predictor = predictor

        result = stage.process(_make_task())

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_annotate_keeps_non_target_band(self, mock_init: MagicMock) -> None:
        stage = BandFilterStage(band_value="full_band", action="annotate")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "narrow_band"
        stage._predictor = predictor

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert result.data["band_prediction"] == "narrow_band"
        assert stage.describe().cardinality == "1:1"

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_narrow_band_passes_when_configured(self, mock_init: MagicMock) -> None:
        stage = BandFilterStage(band_value="narrow_band")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "narrow_band"
        stage._predictor = predictor

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert result.data["band_prediction"] == "narrow_band"

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_error_prediction_skipped(self, mock_init: MagicMock) -> None:
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "Error: model failed"
        stage._predictor = predictor

        result = stage.process(_make_task())

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_no_waveform_no_filepath_skipped(self, mock_init: MagicMock) -> None:
        stage = BandFilterStage(band_value="full_band")
        stage._predictor = MagicMock()

        task = AudioTask(data={"some_key": "value"}, dataset_name="test")
        result = stage.process(task)

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_nested_segments_filters(self, mock_init: MagicMock) -> None:
        """Nested segments: only segments passing the band filter survive."""
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        call_count = {"n": 0}

        def predict_side_effect(_waveform: object, _sample_rate: int) -> str:
            call_count["n"] += 1
            return "full_band" if call_count["n"] % 2 == 1 else "narrow_band"

        predictor.predict_audio = predict_side_effect
        stage._predictor = predictor

        sr = 48000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(4)]
        task = AudioTask(
            data={"segments": segments, "original_file": "test.wav"},
            dataset_name="test",
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert len(result.data["segments"]) == 2

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_process_nested_all_filtered_returns_empty(self, mock_init: MagicMock) -> None:
        """Nested segments: when all segments are filtered, return []."""
        stage = BandFilterStage(band_value="full_band")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "narrow_band"
        stage._predictor = predictor

        sr = 48000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(3)]
        task = AudioTask(
            data={"segments": segments},
            dataset_name="test",
        )

        result = stage.process(task)

        assert result == []

    @patch("nemo_curator.stages.audio.filtering.band.BandFilterStage._initialize_predictor")
    def test_annotate_nested_keeps_all_segments(self, mock_init: MagicMock) -> None:
        """Nested annotate mode predicts every segment without target-band dropping."""
        stage = BandFilterStage(band_value="full_band", action="annotate")
        predictor = MagicMock()
        predictor.predict_audio.return_value = "narrow_band"
        stage._predictor = predictor

        sr = 48000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(3)]
        task = AudioTask(
            data={"segments": segments},
            dataset_name="test",
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert len(result.data["segments"]) == 3
        assert all(seg["band_prediction"] == "narrow_band" for seg in result.data["segments"])


@pytest.mark.parametrize("mode", ["task", "segments", "auto"])
@pytest.mark.parametrize("scorable", [True, False], ids=["success", "unscorable"])
def test_band_agent_conformance_covers_all_scopes(mode: str, scorable: bool) -> None:
    stage = _stage(mode=mode, scorable=scorable, input_residency="waveform")

    def fixture() -> AudioTask:
        audio = {"waveform": torch.zeros(1, 32), "sample_rate": 16000}
        data = {"segments": [audio]} if mode == "segments" else audio
        return AudioTask(dataset_name="test", data=data)

    assert_agent_ready(stage, fixture, expected_cardinality="1:1", segments_key="segments")


@pytest.mark.parametrize("nested", [False, True], ids=["task", "segments"])
@pytest.mark.parametrize("scorable", [True, False], ids=["success", "unscorable"])
def test_band_auto_runtime_outputs_match_selected_scope(nested: bool, scorable: bool) -> None:
    stage = _stage(mode="auto", scorable=scorable, input_residency="waveform")

    def fixture() -> AudioTask:
        audio = {"waveform": torch.zeros(1, 32), "sample_rate": 16000}
        return AudioTask(dataset_name="test", data={"segments": [audio]} if nested else audio)

    assert_agent_ready(stage, fixture, expected_cardinality="1:1", segments_key="segments")
    task = fixture()

    result = stage.process(task)

    assert isinstance(result, AudioTask)
    selected = task.data["segments"][0] if nested else task.data
    assert ("band_prediction" in selected) is scorable
    if nested:
        assert "band_prediction" not in task.data
    else:
        assert "segments" not in task.data


def test_band_auto_contract_rejects_parent_audio_for_unhydrated_segments() -> None:
    report = validate_pipeline(
        [BandFilterStage(action="annotate")],
        initial_keys={"waveform", "sample_rate", "segments"},
        initial_roles={"waveform", "sample_rate", "segments"},
        initial_segment_keys={"segment_num"},
    )

    assert not report.ok
    assert any(issue.code == "unsatisfied_reads" for issue in report.issues)


def test_band_consumes_file_and_waveform_residencies(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "band.wav")

    assert_residency_consumption(
        lambda residency: _stage(input_residency=residency),
        file_fixture=lambda: AudioTask(dataset_name="test", data={"audio_filepath": path}),
        waveform_fixture=lambda: AudioTask(
            dataset_name="test",
            data={"waveform": torch.zeros(1, 32), "sample_rate": 16000},
        ),
    )
