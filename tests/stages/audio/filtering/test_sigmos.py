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
from nemo_curator.stages.audio.filtering.sigmos import SIGMOSFilterStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path

_GOOD_SCORES = {
    "MOS_NOISE": 4.5,
    "MOS_OVRL": 4.0,
    "MOS_SIG": 4.2,
    "MOS_COL": 4.1,
    "MOS_DISC": 4.3,
    "MOS_LOUD": 3.8,
    "MOS_REVERB": 4.0,
}

_BAD_SCORES = {
    "MOS_NOISE": 2.0,
    "MOS_OVRL": 2.0,
    "MOS_SIG": 2.0,
    "MOS_COL": 2.0,
    "MOS_DISC": 2.0,
    "MOS_LOUD": 2.0,
    "MOS_REVERB": 2.0,
}


def _make_task(duration_s: float = 1.0, sample_rate: int = 48000) -> AudioTask:
    num_samples = int(duration_s * sample_rate)
    return AudioTask(
        data={"waveform": torch.randn(1, num_samples), "sample_rate": sample_rate},
        dataset_name="test",
    )


def _make_mock_model(scores: dict) -> MagicMock:
    model = MagicMock()
    model.run.return_value = scores
    return model


def _write_wav(path: Path, *, samples: int = 32, sample_rate: int = 16000) -> str:
    sf.write(path, np.linspace(-0.25, 0.25, samples, dtype=np.float32), sample_rate)
    return str(path)


def _stage(*, mode: str = "task", scorable: bool = True, input_residency: str = "auto") -> SIGMOSFilterStage:
    stage = SIGMOSFilterStage(mode=mode, action="annotate", input_residency=input_residency)
    if scorable:
        stage._model = _make_mock_model(_GOOD_SCORES)
    return stage


class TestSIGMOSFilterStage:
    def test_legacy_positional_constructor_order_is_preserved(self) -> None:
        resources = Resources(cpus=2)
        thresholds = (4.1, 3.6, 3.1, 3.2, 3.3, 3.4, 3.5)

        stage = SIGMOSFilterStage("/models", "model.onnx", *thresholds, "legacy", 8, resources)

        assert stage.model_dir == "/models"
        assert stage.model_path == "model.onnx"
        assert (
            stage.noise_threshold,
            stage.ovrl_threshold,
            stage.sig_threshold,
            stage.col_threshold,
            stage.disc_threshold,
            stage.loud_threshold,
            stage.reverb_threshold,
        ) == thresholds
        assert stage.name == "legacy"
        assert stage.batch_size == 8
        assert stage.resources is resources

    @pytest.mark.parametrize("residency", ["wavefrom", "", "FILE"])
    def test_invalid_input_residency_is_rejected(self, residency: str) -> None:
        with pytest.raises(ValueError, match="input_residency"):
            SIGMOSFilterStage(input_residency=residency)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param({"noise_key": "waveform"}, id="output-overwrites-input"),
            pytest.param({"noise_key": "quality", "ovrl_key": "quality"}, id="duplicate-outputs"),
        ],
    )
    def test_output_key_collisions_are_rejected(self, params: dict[str, str]) -> None:
        with (
            patch("nemo_curator.stages.audio.filtering.sigmos.build_sigmos_model") as build_model,
            patch.object(SIGMOSFilterStage, "_download_model") as download,
            pytest.raises(ValueError, match="Output keys"),
        ):
            SIGMOSFilterStage(**params)

        build_model.assert_not_called()
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
            SIGMOSFilterStage(**params)

    @pytest.mark.parametrize("mode", ["task", "segments"])
    def test_invalid_resident_sample_rate_is_unscorable_without_inference(self, mode: str) -> None:
        stage = _stage(mode=mode, input_residency="waveform")
        audio = {"waveform": torch.zeros(8), "sample_rate": 16000.5}
        task = AudioTask(dataset_name="test", data=audio if mode == "task" else {"segments": [audio]})

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        stage._model.run.assert_not_called()
        score_keys = {
            stage.noise_key,
            stage.ovrl_key,
            stage.sig_key,
            stage.col_key,
            stage.disc_key,
            stage.loud_key,
            stage.reverb_key,
        }
        assert not score_keys & audio.keys()

    def test_explicit_valid_model_path_returns_without_download(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"model")
        stage = SIGMOSFilterStage(model_path=str(model_path))

        with patch.object(stage, "_download_model") as download:
            assert stage._resolve_model_path() == str(model_path)

        download.assert_not_called()

    def test_explicit_missing_model_path_does_not_download(self, tmp_path: Path) -> None:
        stage = SIGMOSFilterStage(model_path=str(tmp_path / "missing.onnx"))

        with (
            patch.object(stage, "_download_model") as download,
            pytest.raises(FileNotFoundError, match="model_path"),
        ):
            stage._resolve_model_path()

        download.assert_not_called()

    def test_explicit_empty_model_path_does_not_download(self, tmp_path: Path) -> None:
        model_path = tmp_path / "empty.onnx"
        model_path.touch()
        stage = SIGMOSFilterStage(model_path=str(model_path))

        with (
            patch.object(stage, "_download_model") as download,
            pytest.raises(FileNotFoundError, match="empty"),
        ):
            stage._resolve_model_path()

        download.assert_not_called()

    def test_default_uncached_model_path_uses_download(self, tmp_path: Path) -> None:
        stage = SIGMOSFilterStage(model_dir=str(tmp_path))

        with patch.object(stage, "_download_model", return_value="/cache/model.onnx") as download:
            assert stage._resolve_model_path() == "/cache/model.onnx"

        download.assert_called_once_with(str(tmp_path))

    def test_cached_model_is_reused_without_network(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model-sigmos_1697718653_41d092e8-epo-200.onnx"
        model_path.write_bytes(b"cached")

        with patch("nemo_curator.stages.audio.filtering.sigmos.requests.get") as request:
            assert SIGMOSFilterStage._download_model(str(tmp_path)) == str(model_path)

        request.assert_not_called()

    def test_setup_on_node_prefetches_only_default_model(self, tmp_path: Path) -> None:
        explicit = tmp_path / "explicit.onnx"
        explicit.write_bytes(b"model")

        with patch.object(SIGMOSFilterStage, "_download_model") as download:
            SIGMOSFilterStage(model_path=str(explicit)).setup_on_node()
            download.assert_not_called()

            SIGMOSFilterStage(model_dir=str(tmp_path)).setup_on_node()
            download.assert_called_once_with(str(tmp_path))

    def test_static_contract_conservatively_reports_download_and_row_independence(self) -> None:
        contract = static_contract(SIGMOSFilterStage)

        assert contract.gates.requires_internet_first_run is True
        assert contract.gates.per_row_independent is True
        assert SIGMOSFilterStage().describe().gates.requires_internet_first_run is True
        assert (
            SIGMOSFilterStage(model_path="/configured/model.onnx").describe().gates.requires_internet_first_run
            is False
        )

    def test_native_defaults_remain_filter_auto_and_two_thresholds(self) -> None:
        stage = SIGMOSFilterStage()

        assert stage.action == "filter"
        assert stage.mode == "auto"
        assert stage.noise_threshold == 4.0
        assert stage.ovrl_threshold == 3.5

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_process_passes_good_scores(self, mock_init: MagicMock) -> None:
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)
        stage._model = _make_mock_model(_GOOD_SCORES)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert result.data["sigmos_noise"] == 4.5
        assert result.data["sigmos_ovrl"] == 4.0

    @pytest.mark.parametrize(
        "resident",
        [
            pytest.param(
                torch.tensor([[32767, -32768], [16384, -16384]], dtype=torch.int16),
                id="torch",
            ),
            pytest.param(
                np.array([[32767, -32768], [16384, -16384]], dtype=np.int16),
                id="numpy",
            ),
        ],
    )
    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_stereo_integer_pcm_preserves_legacy_amplitude_and_decision(
        self,
        mock_init: MagicMock,
        resident: torch.Tensor | np.ndarray,
    ) -> None:
        stage = SIGMOSFilterStage(action="annotate", input_residency="waveform")
        stage._model = _make_mock_model(_GOOD_SCORES)
        task = AudioTask(dataset_name="test", data={"waveform": resident, "sample_rate": 16000})

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        audio = stage._model.run.call_args.kwargs["audio"]
        assert audio.dtype == np.float32
        assert audio.shape == (2,)
        assert audio == pytest.approx([24575.5, -24576.0])

    def test_auto_contract_rejects_parent_audio_for_unhydrated_segments(self) -> None:
        report = validate_pipeline(
            [SIGMOSFilterStage(action="annotate")],
            initial_keys={"waveform", "sample_rate", "segments"},
            initial_roles={"waveform", "sample_rate", "segments"},
            initial_segment_keys={"segment_num"},
        )

        assert not report.ok
        assert any(issue.code == "unsatisfied_reads" for issue in report.issues)

    @pytest.mark.parametrize(
        "orphan",
        [
            pytest.param({"waveform": torch.ones(1, 7)}, id="waveform-without-rate"),
            pytest.param({"sample_rate": 8000}, id="rate-without-waveform"),
        ],
    )
    def test_auto_partial_pair_is_replaced_with_file_audio(self, orphan: dict, tmp_path: Path) -> None:
        stage = _stage(input_residency="auto")
        path = _write_wav(tmp_path / "fallback.wav", samples=31, sample_rate=16000)
        task = AudioTask(dataset_name="test", data={**orphan, "audio_filepath": path})

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        inferred = stage._model.run.call_args.kwargs
        assert inferred["sr"] == task.data["sample_rate"] == 16000
        assert inferred["audio"].shape == (31,)
        assert inferred["audio"] == pytest.approx(task.data["waveform"].squeeze(0).numpy())

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
        inferred = stage._model.run.call_args.kwargs
        assert inferred["sr"] == 16000
        assert inferred["audio"].shape == (29,)
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
        inferred = stage._model.run.call_args.kwargs
        assert inferred["sr"] == segment["sample_rate"] == 16000
        assert inferred["audio"] == pytest.approx(segment["waveform"].squeeze(0).numpy())

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_process_rejects_bad_scores(self, mock_init: MagicMock) -> None:
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)
        stage._model = _make_mock_model(_BAD_SCORES)

        result = stage.process(_make_task())

        assert result == []

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_annotate_keeps_bad_scores(self, mock_init: MagicMock) -> None:
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5, action="annotate")
        stage._model = _make_mock_model(_BAD_SCORES)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        assert result.data["sigmos_noise"] == 2.0
        assert result.data["sigmos_ovrl"] == 2.0
        assert stage.describe().cardinality == "1:1"

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_none_thresholds_disable_checks(self, mock_init: MagicMock) -> None:
        stage = SIGMOSFilterStage(
            noise_threshold=None,
            ovrl_threshold=None,
            sig_threshold=None,
            col_threshold=None,
            disc_threshold=None,
            loud_threshold=None,
            reverb_threshold=None,
        )
        stage._model = _make_mock_model(_BAD_SCORES)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_partial_threshold_fail(self, mock_init: MagicMock) -> None:
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=None)
        stage._model = _make_mock_model(
            {
                "MOS_NOISE": 3.0,
                "MOS_OVRL": 5.0,
                "MOS_SIG": 5.0,
                "MOS_COL": 5.0,
                "MOS_DISC": 5.0,
                "MOS_LOUD": 5.0,
                "MOS_REVERB": 5.0,
            }
        )

        result = stage.process(_make_task())

        assert result == []

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_sigmos_output_keys(self, mock_init: MagicMock) -> None:
        stage = SIGMOSFilterStage(noise_threshold=1.0, ovrl_threshold=1.0)
        stage._model = _make_mock_model(_GOOD_SCORES)

        result = stage.process(_make_task())

        assert isinstance(result, AudioTask)
        for key in [
            "sigmos_noise",
            "sigmos_ovrl",
            "sigmos_sig",
            "sigmos_col",
            "sigmos_disc",
            "sigmos_loud",
            "sigmos_reverb",
        ]:
            assert key in result.data

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_no_audio_no_filepath_skipped(self, mock_init: MagicMock) -> None:
        stage = SIGMOSFilterStage()
        stage._model = _make_mock_model(_GOOD_SCORES)

        task = AudioTask(data={"some_key": "value"}, dataset_name="test")
        result = stage.process(task)

        assert result == []

    def test_model_not_available(self) -> None:
        stage = SIGMOSFilterStage()
        stage._model = None

        with patch.object(stage, "_initialize_model"):
            result = stage.process(_make_task())

        assert result == []

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_process_nested_segments_filters(self, mock_init: MagicMock) -> None:
        """Nested segments: only segments passing thresholds survive."""
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)
        call_count = {"n": 0}

        def fake_run(audio: object, sr: int) -> dict:  # noqa: ARG001
            call_count["n"] += 1
            if call_count["n"] % 2 == 1:
                return _GOOD_SCORES
            return _BAD_SCORES

        model = MagicMock()
        model.run.side_effect = fake_run
        stage._model = model

        sr = 48000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(4)]
        task = AudioTask(
            data={"segments": segments, "original_file": "test.wav"},
            dataset_name="test",
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert len(result.data["segments"]) == 2
        for seg in result.data["segments"]:
            assert "sigmos_noise" in seg

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_process_nested_all_filtered_returns_empty(self, mock_init: MagicMock) -> None:
        """Nested segments: when all fail thresholds, return []."""
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)
        stage._model = _make_mock_model(_BAD_SCORES)

        sr = 48000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(3)]
        task = AudioTask(
            data={"segments": segments},
            dataset_name="test",
        )

        result = stage.process(task)

        assert result == []

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_annotate_nested_keeps_all_segments(self, mock_init: MagicMock) -> None:
        """Nested annotate mode scores every segment without threshold dropping."""
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5, action="annotate")
        stage._model = _make_mock_model(_BAD_SCORES)

        sr = 48000
        segments = [{"waveform": torch.randn(1, sr), "sample_rate": sr, "segment_num": i} for i in range(3)]
        task = AudioTask(
            data={"segments": segments},
            dataset_name="test",
        )

        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert len(result.data["segments"]) == 3
        assert all(seg["sigmos_noise"] == 2.0 for seg in result.data["segments"])


@pytest.mark.parametrize("mode", ["task", "segments", "auto"])
@pytest.mark.parametrize("scorable", [True, False], ids=["success", "unscorable"])
def test_sigmos_agent_conformance_covers_all_scopes(mode: str, scorable: bool) -> None:
    stage = _stage(mode=mode, scorable=scorable, input_residency="waveform")

    def fixture() -> AudioTask:
        audio = {"waveform": torch.zeros(1, 32), "sample_rate": 16000}
        data = {"segments": [audio]} if mode == "segments" else audio
        return AudioTask(dataset_name="test", data=data)

    assert_agent_ready(stage, fixture, expected_cardinality="1:1", segments_key="segments")


@pytest.mark.parametrize("nested", [False, True], ids=["task", "segments"])
@pytest.mark.parametrize("scorable", [True, False], ids=["success", "unscorable"])
def test_sigmos_auto_runtime_outputs_match_selected_scope(nested: bool, scorable: bool) -> None:
    stage = _stage(mode="auto", scorable=scorable, input_residency="waveform")

    def fixture() -> AudioTask:
        audio = {"waveform": torch.zeros(1, 32), "sample_rate": 16000}
        return AudioTask(dataset_name="test", data={"segments": [audio]} if nested else audio)

    assert_agent_ready(stage, fixture, expected_cardinality="1:1", segments_key="segments")
    task = fixture()

    result = stage.process(task)

    assert isinstance(result, AudioTask)
    selected = task.data["segments"][0] if nested else task.data
    score_keys = set(stage.outputs()[1])
    assert score_keys.issubset(selected) is scorable
    if nested:
        assert score_keys.isdisjoint(task.data)
    else:
        assert "segments" not in task.data


def test_sigmos_consumes_file_and_waveform_residencies(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "sigmos.wav")

    assert_residency_consumption(
        lambda residency: _stage(input_residency=residency),
        file_fixture=lambda: AudioTask(dataset_name="test", data={"audio_filepath": path}),
        waveform_fixture=lambda: AudioTask(
            dataset_name="test",
            data={"waveform": torch.zeros(1, 32), "sample_rate": 16000},
        ),
    )
