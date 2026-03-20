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

"""
Integration tests for SIGMOSFilterStage.

Stage accepts waveform+sample_rate or audio_filepath (WAV); uses predict_audio_mos
in-memory only (no temp files). SIGMOS ONNX inference requires GPU.

Usage:
    pytest tests/stages/audio/ADV/test_sigmos_filter_stages.py -v -s
    pytest tests/stages/audio/ADV/test_sigmos_filter_stages.py -v -s -m "not gpu"
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import soundfile as sf
import torch

from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_DATA_DIR = Path(
    "/lustre/fsw/portfolios/maxine/users/shbhawsar/debug/nemo_curator"
    "/Curator/nemo_curator/stages/audio/datasets/youtube/batch_small"
)
WAV_SMALL = TEST_DATA_DIR / "0dQmTf6K71U.wav"
WAV_LARGE = TEST_DATA_DIR / "-1KFBkWd5xs.wav"

HAS_GPU = torch.cuda.is_available()
DATA_EXISTS = WAV_SMALL.exists()

gpu = pytest.mark.gpu
skip_no_gpu = pytest.mark.skipif(not HAS_GPU, reason="No GPU available")
skip_no_data = pytest.mark.skipif(not DATA_EXISTS, reason=f"Test data not found: {WAV_SMALL}")

GPU_FRAC_RESOURCES = Resources(cpus=1.0, gpus=0.3)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_mono_waveform(wav_path: str):
    """Load WAV to mono waveform (tensor) and sample_rate."""
    data, sr = sf.read(wav_path, dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wav_path() -> str:
    if not DATA_EXISTS:
        pytest.skip(f"Test data not found: {WAV_SMALL}")
    return str(WAV_SMALL)


# ===========================================================================
# Standalone SIGMOS Tests (GPU required)
# ===========================================================================


@skip_no_data
class TestSIGMOSFilterReal:
    """Test SIGMOSFilterStage on real audio files."""

    @skip_no_gpu
    @gpu
    def test_gpu_from_filepath(self, wav_path: str) -> None:
        """Test SIGMOS loading audio directly from filepath."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage

        stage = SIGMOSFilterStage(
            noise_threshold=1.0,
            ovrl_threshold=None,
            resources=GPU_FRAC_RESOURCES,
        )
        stage.setup()
        batch = AudioBatch(
            data=[{"audio_filepath": wav_path}],
            task_id="sigmos-standalone",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert result is not None
        print(f"  SIGMOS from filepath: {len(result.data)} passed")
        if len(result.data) > 0:
            item = result.data[0]
            assert "sigmos_noise" in item
            assert "sigmos_ovrl" in item
            assert "sigmos_sig" in item
            print(f"    NOISE={item['sigmos_noise']:.3f}, OVRL={item['sigmos_ovrl']:.3f}, "
                  f"SIG={item['sigmos_sig']:.3f}")

    @skip_no_gpu
    @gpu
    def test_gpu_low_threshold_passes(self, wav_path: str) -> None:
        """With very low thresholds, most audio should pass."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage

        stage = SIGMOSFilterStage(
            noise_threshold=1.0,
            ovrl_threshold=1.0,
            resources=GPU_FRAC_RESOURCES,
        )
        stage.setup()
        batch = AudioBatch(
            data=[{"audio_filepath": wav_path}],
            task_id="sigmos-low",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert result is not None
        assert len(result.data) == 1
        print(f"  SIGMOS low threshold: PASSED")

    @skip_no_gpu
    @gpu
    def test_gpu_high_threshold_filters(self, wav_path: str) -> None:
        """With very high thresholds, audio should be filtered out."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage

        stage = SIGMOSFilterStage(
            noise_threshold=5.0,
            ovrl_threshold=5.0,
            sig_threshold=5.0,
            resources=GPU_FRAC_RESOURCES,
        )
        stage.setup()
        batch = AudioBatch(
            data=[{"audio_filepath": wav_path}],
            task_id="sigmos-high",
            dataset_name="test",
        )
        result = stage.process(batch)

        assert result is not None
        assert len(result.data) == 0
        print(f"  SIGMOS high threshold: correctly filtered out")

    @skip_no_gpu
    @gpu
    def test_gpu_multi_file(self) -> None:
        """Process multiple wav files."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage

        paths = [p for p in [WAV_SMALL, WAV_LARGE] if p.exists()]
        if len(paths) < 2:
            pytest.skip("Need at least 2 test wav files")

        stage = SIGMOSFilterStage(
            noise_threshold=1.0,
            ovrl_threshold=None,
            resources=GPU_FRAC_RESOURCES,
        )
        stage.setup()
        batch = AudioBatch(
            data=[{"audio_filepath": str(p)} for p in paths],
            task_id="sigmos-multi",
        )
        result = stage.process(batch)

        assert result is not None
        print(f"  SIGMOS multi-file: {len(result.data)}/{len(paths)} passed")

    @skip_no_gpu
    @gpu
    @pytest.mark.parametrize(
        "res",
        [
            pytest.param(Resources(cpus=1.0, gpus=0.3), id="frac-gpu"),
            pytest.param(Resources(cpus=1.0, gpus=1.0), id="full-gpu"),
        ],
    )
    def test_sigmos_resources(self, wav_path: str, res: Resources) -> None:
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage

        stage = SIGMOSFilterStage(
            noise_threshold=1.0,
            ovrl_threshold=None,
            resources=res,
        )
        stage.setup()
        batch = AudioBatch(
            data=[{"audio_filepath": wav_path}],
            task_id="sigmos-res",
        )
        result = stage.process(batch)
        assert result is not None
        print(f"  SIGMOS resources({res}): {len(result.data)} passed")


# ===========================================================================
# Single format: waveform+sample_rate or audio_filepath; no temp file
# ===========================================================================


@skip_no_data
class TestSIGMOSSingleFormatNoTempFile:
    """Stage uses only predict_audio_mos (in-memory); no temp files."""

    def test_stage_does_not_import_tempfile(self) -> None:
        """SIGMOS stage must not use tempfile (single in-memory path)."""
        import inspect
        from nemo_curator.stages.audio.filtering import sigmos as sigmos_module
        source = inspect.getsource(sigmos_module)
        assert "tempfile" not in source, "SIGMOS stage must not use tempfile"

    @skip_no_gpu
    @gpu
    def test_from_waveform_and_sample_rate(self, wav_path: str) -> None:
        """Input with waveform + sample_rate uses in-memory path (no file write)."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage

        waveform, sr = _load_mono_waveform(wav_path)
        stage = SIGMOSFilterStage(
            noise_threshold=1.0,
            ovrl_threshold=1.0,
            resources=GPU_FRAC_RESOURCES,
        )
        stage.setup()
        batch = AudioBatch(
            data=[{"waveform": waveform, "sample_rate": sr}],
            task_id="sigmos-waveform",
            dataset_name="test",
        )
        result = stage.process(batch)
        assert result is not None
        assert len(result.data) <= 1
        if result.data:
            assert "sigmos_noise" in result.data[0]
            assert "sigmos_ovrl" in result.data[0]

    @skip_no_gpu
    @gpu
    def test_from_audio_filepath_wav(self, wav_path: str) -> None:
        """Input with audio_filepath (WAV) loads then uses in-memory predict (no temp file)."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage

        stage = SIGMOSFilterStage(
            noise_threshold=1.0,
            ovrl_threshold=None,
            resources=GPU_FRAC_RESOURCES,
        )
        stage.setup()
        batch = AudioBatch(
            data=[{"audio_filepath": wav_path}],
            task_id="sigmos-path",
            dataset_name="test",
        )
        result = stage.process(batch)
        assert result is not None
        if result.data:
            assert "sigmos_ovrl" in result.data[0]

    @skip_no_gpu
    @gpu
    def test_order_preserved(self, wav_path: str) -> None:
        """Output order matches input order of passing items."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage

        waveform, sr = _load_mono_waveform(wav_path)
        stage = SIGMOSFilterStage(
            noise_threshold=1.0,
            ovrl_threshold=1.0,
            resources=GPU_FRAC_RESOURCES,
        )
        stage.setup()
        batch = AudioBatch(
            data=[
                {"waveform": waveform, "sample_rate": sr, "id": "a"},
                {"audio_filepath": wav_path, "id": "b"},
            ],
            task_id="sigmos-order",
            dataset_name="test",
        )
        result = stage.process(batch)
        assert result is not None
        order = [item.get("id") for item in result.data]
        # Order must match input (a then b); allow subset if one fails
        assert order in (["a", "b"], ["a"], ["b"]), f"Order should match input, got {order}"
        if len(result.data) == 2:
            assert result.data[0].get("id") == "a" and result.data[1].get("id") == "b"

    def test_empty_task(self) -> None:
        """Empty task.data returns AudioBatch with data=[]."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage

        stage = SIGMOSFilterStage(noise_threshold=4.0, ovrl_threshold=3.5)
        batch = AudioBatch(data=[], task_id="empty", dataset_name="test")
        result = stage.process(batch)
        assert result is not None
        assert result.data == []
        assert result.task_id == "empty"

    def test_missing_audio_skipped(self) -> None:
        """Item with no waveform and no valid audio_filepath is skipped (not in result)."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage

        stage = SIGMOSFilterStage(noise_threshold=1.0, ovrl_threshold=1.0)
        batch = AudioBatch(
            data=[{"id": "no-audio"}],
            task_id="skip",
            dataset_name="test",
        )
        result = stage.process(batch)
        assert result is not None
        assert len(result.data) == 0

    def test_config_overrides_thresholds(self) -> None:
        """SIGMOSConfig overrides constructor thresholds."""
        from nemo_curator.stages.audio.configs import SIGMOSConfig
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage

        config = SIGMOSConfig(noise_threshold=2.5, ovrl_threshold=2.0)
        stage = SIGMOSFilterStage(config=config, noise_threshold=99.0, ovrl_threshold=99.0)
        assert stage.noise_threshold == 2.5
        assert stage.ovrl_threshold == 2.0

    @skip_no_gpu
    @gpu
    def test_predict_audio_mos_called_with_numpy(self, wav_path: str) -> None:
        """Stage calls predict_audio_mos with (numpy array, sample_rate); no file path."""
        import numpy as np
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        from nemo_curator.stages.audio.filtering.sigmos_filter_module import sigmos_pipeline

        waveform, sr = _load_mono_waveform(wav_path)
        calls = []

        def record_predict_audio_mos(audio_data, sample_rate, config=None):
            assert isinstance(audio_data, np.ndarray), "Must be numpy array"
            assert isinstance(sample_rate, (int, float)), "Must be sample rate"
            calls.append((audio_data.shape, sample_rate))
            return {
                "MOS_NOISE": 4.0, "MOS_OVRL": 4.0, "MOS_SIG": 4.0, "MOS_COL": 4.0,
                "MOS_DISC": 4.0, "MOS_LOUD": 4.0, "MOS_REVERB": 4.0,
            }

        stage = SIGMOSFilterStage(
            noise_threshold=1.0,
            ovrl_threshold=1.0,
            resources=GPU_FRAC_RESOURCES,
        )
        with patch.object(sigmos_pipeline, "predict_audio_mos", side_effect=record_predict_audio_mos):
            stage.setup()
            batch = AudioBatch(
                data=[{"waveform": waveform, "sample_rate": sr}],
                task_id="mock",
                dataset_name="test",
            )
            result = stage.process(batch)
        assert len(calls) == 1
        assert calls[0][1] == sr
        assert result is not None and len(result.data) == 1


# ===========================================================================
# GPU device handling (no explicit gpu_id, auto-detect via current_device)
# ===========================================================================


class TestSIGMOSGPUDeviceHandling:
    """Verify pipeline auto-detects GPU and stage does not pass gpu_id."""

    def test_predict_audio_mos_has_no_gpu_id_param(self) -> None:
        """predict_audio_mos signature must be (audio_data, sample_rate, config=None)."""
        import inspect
        from nemo_curator.stages.audio.filtering.sigmos_filter_module.sigmos_pipeline import (
            predict_audio_mos,
        )
        sig = inspect.signature(predict_audio_mos)
        assert "gpu_id" not in sig.parameters, \
            f"predict_audio_mos should not have gpu_id param, got: {list(sig.parameters)}"

    def test_stage_does_not_pass_gpu_id(self) -> None:
        """Stage calls predict_audio_mos without gpu_id."""
        import inspect
        from nemo_curator.stages.audio.filtering import sigmos as sigmos_module
        source = inspect.getsource(sigmos_module)
        assert "gpu_id" not in source, \
            "Stage should not reference gpu_id; GPU is auto-detected in the pipeline"

    def test_pipeline_no_gpu_id_constructor(self) -> None:
        """_SIGMOSPipeline constructor should not accept gpu_id."""
        import inspect
        from nemo_curator.stages.audio.filtering.sigmos_filter_module.sigmos_pipeline import (
            _SIGMOSPipeline,
        )
        sig = inspect.signature(_SIGMOSPipeline.__init__)
        assert "gpu_id" not in sig.parameters, \
            f"_SIGMOSPipeline should not have gpu_id param, got: {list(sig.parameters)}"

    @skip_no_gpu
    @gpu
    def test_pipeline_uses_current_device(self) -> None:
        """Pipeline picks up torch.cuda.current_device() automatically."""
        from nemo_curator.stages.audio.filtering.sigmos_filter_module.sigmos_pipeline import (
            _SIGMOSPipeline,
            _MODEL_CACHE,
        )
        expected_id = int(torch.cuda.current_device())
        _MODEL_CACHE.clear()
        try:
            _SIGMOSPipeline(config=None)
        except Exception:
            pytest.skip("SIGMOS model files not available")
        matching = [k for k in _MODEL_CACHE if k.startswith(f"gpu_{expected_id}")]
        assert len(matching) > 0, \
            f"Expected cache key with gpu_{expected_id}, got: {list(_MODEL_CACHE.keys())}"

    def test_cpu_fallback_when_no_gpu(self) -> None:
        """When CUDA is unavailable, pipeline uses CPU without error."""
        from unittest.mock import patch as mock_patch
        from nemo_curator.stages.audio.filtering.sigmos_filter_module.sigmos_pipeline import (
            _SIGMOSPipeline,
            _MODEL_CACHE,
        )
        _MODEL_CACHE.clear()
        with mock_patch("torch.cuda.is_available", return_value=False):
            try:
                _SIGMOSPipeline(config=None)
            except Exception:
                pytest.skip("SIGMOS model files not available")
            matching = [k for k in _MODEL_CACHE if k.startswith("cpu")]
            assert len(matching) > 0, \
                f"Expected cache key starting with 'cpu', got: {list(_MODEL_CACHE.keys())}"


# ===========================================================================
# Corner cases (read-only property, all-None thresholds, from_dict)
# ===========================================================================


class TestSIGMOSCornerCases:
    """Corner-case tests for SIGMOSConfig."""

    def test_config_model_path_is_property(self) -> None:
        """model_path is a read-only property returning a non-empty string."""
        from nemo_curator.stages.audio.configs import SIGMOSConfig

        config = SIGMOSConfig()
        assert isinstance(config.model_path, str)
        assert len(config.model_path) > 0

    def test_all_thresholds_none(self) -> None:
        """Config with all thresholds None yields empty active thresholds."""
        from nemo_curator.stages.audio.configs import SIGMOSConfig

        config = SIGMOSConfig(
            noise_threshold=None,
            ovrl_threshold=None,
            sig_threshold=None,
            col_threshold=None,
            disc_threshold=None,
            loud_threshold=None,
            reverb_threshold=None,
        )
        assert config.get_active_thresholds() == {}

    def test_config_from_dict_ignores_model_path(self) -> None:
        """from_dict ignores model_path (not a dataclass field) and sets valid fields."""
        from nemo_curator.stages.audio.configs import SIGMOSConfig

        config = SIGMOSConfig.from_dict({"model_path": "/fake", "noise_threshold": 3.0})
        assert config.noise_threshold == 3.0
        assert config.model_path != "/fake"
