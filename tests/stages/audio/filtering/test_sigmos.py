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

from unittest.mock import MagicMock, patch

from pydub import AudioSegment

from nemo_curator.stages.audio.configs.sigmos import SIGMOSConfig
from nemo_curator.stages.audio.filtering.sigmos import SIGMOSFilterStage
from nemo_curator.tasks import AudioBatch


def _make_audio_segment(duration_ms: int = 1000) -> AudioSegment:
    return AudioSegment.silent(duration=duration_ms, frame_rate=48000)


def _good_scores(path: str) -> dict:
    return {
        path: {
            "MOS_NOISE": 4.5,
            "MOS_OVRL": 4.0,
            "MOS_SIG": 4.2,
            "MOS_COL": 4.1,
            "MOS_DISC": 4.3,
            "MOS_LOUD": 3.8,
            "MOS_REVERB": 4.0,
        }
    }


def _bad_scores(path: str) -> dict:
    return {
        path: {
            "MOS_NOISE": 2.0,
            "MOS_OVRL": 2.0,
            "MOS_SIG": 2.0,
            "MOS_COL": 2.0,
            "MOS_DISC": 2.0,
            "MOS_LOUD": 2.0,
            "MOS_REVERB": 2.0,
        }
    }


class TestSIGMOSConfig:
    """Tests for SIGMOSConfig."""

    def test_defaults(self) -> None:
        cfg = SIGMOSConfig()
        assert cfg.noise_threshold == 4.0
        assert cfg.ovrl_threshold == 3.5
        assert cfg.sig_threshold is None
        assert cfg.col_threshold is None
        assert cfg.disc_threshold is None
        assert cfg.loud_threshold is None
        assert cfg.reverb_threshold is None

    def test_from_dict(self) -> None:
        cfg = SIGMOSConfig.from_dict({"noise_threshold": 3.0, "sig_threshold": 2.5})
        assert cfg.noise_threshold == 3.0
        assert cfg.sig_threshold == 2.5
        assert cfg.ovrl_threshold == 3.5

    def test_from_dict_none(self) -> None:
        cfg = SIGMOSConfig.from_dict(None)
        assert cfg.noise_threshold == 4.0

    def test_from_dict_ignores_unknown(self) -> None:
        cfg = SIGMOSConfig.from_dict({"unknown": 99, "noise_threshold": 1.0})
        assert cfg.noise_threshold == 1.0

    def test_to_dict(self) -> None:
        cfg = SIGMOSConfig(noise_threshold=3.0, ovrl_threshold=None)
        d = cfg.to_dict()
        assert d["noise_threshold"] == 3.0
        assert d["ovrl_threshold"] is None

    def test_roundtrip(self) -> None:
        original = SIGMOSConfig(noise_threshold=3.5, sig_threshold=2.0, reverb_threshold=3.0)
        restored = SIGMOSConfig.from_dict(original.to_dict())
        assert restored.noise_threshold == original.noise_threshold
        assert restored.sig_threshold == original.sig_threshold
        assert restored.reverb_threshold == original.reverb_threshold

    def test_get_active_thresholds(self) -> None:
        cfg = SIGMOSConfig(noise_threshold=4.0, ovrl_threshold=3.5, sig_threshold=None)
        active = cfg.get_active_thresholds()
        assert "noise" in active
        assert "ovrl" in active
        assert "sig" not in active
        assert active["noise"] == 4.0

    def test_get_active_thresholds_all_none(self) -> None:
        cfg = SIGMOSConfig(
            noise_threshold=None, ovrl_threshold=None, sig_threshold=None,
            col_threshold=None, disc_threshold=None, loud_threshold=None,
            reverb_threshold=None,
        )
        assert cfg.get_active_thresholds() == {}

    def test_get(self) -> None:
        cfg = SIGMOSConfig()
        assert cfg.get("noise_threshold") == 4.0
        assert cfg.get("nonexistent", "default") == "default"


class TestSIGMOSFilterStage:
    """Tests for SIGMOSFilterStage."""

    def test_stage_properties(self) -> None:
        stage = SIGMOSFilterStage()
        assert stage.name == "SIGMOSFilter"
        assert stage.inputs() == (["data"], [])
        _, output_keys = stage.outputs()
        for key in ["sigmos_noise", "sigmos_ovrl", "sigmos_sig", "sigmos_col",
                     "sigmos_disc", "sigmos_loud", "sigmos_reverb"]:
            assert key in output_keys

    def test_config_overrides_params(self) -> None:
        cfg = SIGMOSConfig(noise_threshold=2.0, ovrl_threshold=None, sig_threshold=3.0)
        stage = SIGMOSFilterStage(config=cfg)
        assert stage.noise_threshold == 2.0
        assert stage.ovrl_threshold is None
        assert stage.sig_threshold == 3.0

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._initialize_model")
    def test_process_passes_good_scores(self, mock_init) -> None:
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)

        def fake_predict(paths, gpu_id=0, config=None):
            return _good_scores(paths[0])

        stage._predict_function = fake_predict

        audio = _make_audio_segment()
        batch = AudioBatch(
            data=[{"audio": audio}],
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert isinstance(result, AudioBatch)
        assert len(result.data) == 1
        assert result.data[0]["sigmos_noise"] == 4.5
        assert result.data[0]["sigmos_ovrl"] == 4.0

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._initialize_model")
    def test_process_rejects_bad_scores(self, mock_init) -> None:
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)

        def fake_predict(paths, gpu_id=0, config=None):
            return _bad_scores(paths[0])

        stage._predict_function = fake_predict

        audio = _make_audio_segment()
        batch = AudioBatch(
            data=[{"audio": audio}],
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert len(result.data) == 0

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._initialize_model")
    def test_none_thresholds_disable_checks(self, mock_init) -> None:
        stage = SIGMOSFilterStage(
            noise_threshold=None, ovrl_threshold=None,
            sig_threshold=None, col_threshold=None,
            disc_threshold=None, loud_threshold=None,
            reverb_threshold=None,
        )

        def fake_predict(paths, gpu_id=0, config=None):
            return _bad_scores(paths[0])

        stage._predict_function = fake_predict

        audio = _make_audio_segment()
        batch = AudioBatch(
            data=[{"audio": audio}],
            task_id="test",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert len(result.data) == 1

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._initialize_model")
    def test_partial_threshold_fail(self, mock_init) -> None:
        """Item fails if any active threshold is not met."""
        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=None)

        def fake_predict(paths, gpu_id=0, config=None):
            return {paths[0]: {"MOS_NOISE": 3.0, "MOS_OVRL": 5.0,
                               "MOS_SIG": 5.0, "MOS_COL": 5.0,
                               "MOS_DISC": 5.0, "MOS_LOUD": 5.0,
                               "MOS_REVERB": 5.0}}

        stage._predict_function = fake_predict

        audio = _make_audio_segment()
        batch = AudioBatch(data=[{"audio": audio}], task_id="test", dataset_name="test")
        result = stage.process(batch)

        assert len(result.data) == 0

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._initialize_model")
    def test_sigmos_output_keys(self, mock_init) -> None:
        stage = SIGMOSFilterStage(noise_threshold=1.0, ovrl_threshold=1.0)

        def fake_predict(paths, gpu_id=0, config=None):
            return _good_scores(paths[0])

        stage._predict_function = fake_predict

        audio = _make_audio_segment()
        batch = AudioBatch(data=[{"audio": audio}], task_id="test", dataset_name="test")
        result = stage.process(batch)

        item = result.data[0]
        for key in ["sigmos_noise", "sigmos_ovrl", "sigmos_sig", "sigmos_col",
                     "sigmos_disc", "sigmos_loud", "sigmos_reverb"]:
            assert key in item

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._initialize_model")
    def test_empty_batch(self, mock_init) -> None:
        stage = SIGMOSFilterStage()
        stage._predict_function = MagicMock()

        batch = AudioBatch(data=[], task_id="test", dataset_name="test")
        result = stage.process(batch)

        assert isinstance(result, AudioBatch)
        assert len(result.data) == 0

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._initialize_model")
    def test_no_audio_no_filepath_skipped(self, mock_init) -> None:
        stage = SIGMOSFilterStage()
        stage._predict_function = MagicMock()

        batch = AudioBatch(data=[{"some_key": "value"}], task_id="test", dataset_name="test")
        result = stage.process(batch)

        assert len(result.data) == 0

    @patch("nemo_curator.stages.audio.filtering.sigmos.SIGMOSFilterStage._initialize_model")
    def test_preserves_task_metadata(self, mock_init) -> None:
        stage = SIGMOSFilterStage(noise_threshold=1.0, ovrl_threshold=1.0)

        def fake_predict(paths, gpu_id=0, config=None):
            return _good_scores(paths[0])

        stage._predict_function = fake_predict

        audio = _make_audio_segment()
        batch = AudioBatch(data=[{"audio": audio}], task_id="my-task", dataset_name="my-ds")
        result = stage.process(batch)

        assert result.task_id == "my-task"
        assert result.dataset_name == "my-ds"

    def test_predict_function_not_available(self) -> None:
        stage = SIGMOSFilterStage()
        stage._predict_function = None

        with patch.object(stage, "_initialize_model"):
            batch = AudioBatch(data=[{"audio": _make_audio_segment()}], task_id="test", dataset_name="test")
            result = stage.process(batch)

        assert isinstance(result, AudioBatch)
        assert len(result.data) == 0
