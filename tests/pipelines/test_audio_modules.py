# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
Tests for Audio Data Filter modules.

This module tests:
1. Standalone mode: Each module works independently with just audio_filepath
2. Pipeline mode: Modules work together passing waveform/audio data
3. Input format tests: Different input formats (filepath, waveform, pydub)
4. Sample rate tests: Different sample rates (16kHz, 48kHz)

Uses real audio data from:
/workdir/Curator/nemo_curator/stages/audio/datasets/youtube/batch_small

Run tests with:
    pytest tests/pipelines/test_audio_modules.py -v
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from pydub import AudioSegment

from nemo_curator.tasks import AudioBatch


# =============================================================================
# Constants and Data Paths
# =============================================================================

DATA_DIR = Path("/workdir/Curator/nemo_curator/stages/audio/datasets/youtube/batch_small")
SAMPLE_AUDIO_FILES = [
    DATA_DIR / "-1KFBkWd5xs.wav",
    DATA_DIR / "0dQmTf6K71U.wav",
]


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def real_audio_48k() -> str:
    """Get path to real 48kHz stereo audio file."""
    audio_path = SAMPLE_AUDIO_FILES[1]  # Shorter file (581 sec)
    if not audio_path.exists():
        pytest.skip(f"Test audio not found: {audio_path}")
    return str(audio_path)


@pytest.fixture
def short_audio_segment(real_audio_48k: str) -> str:
    """Create a 5-minute segment for testing with real speech."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    # Load first 5 minutes (300 seconds = 300000 ms)
    audio = AudioSegment.from_wav(real_audio_48k)[:300000]
    audio.export(temp_path, format="wav")
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def audio_16k(short_audio_segment: str) -> str:
    """Create 16kHz version of audio for sample rate testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    audio = AudioSegment.from_wav(short_audio_segment)
    audio = audio.set_frame_rate(16000)
    audio.export(temp_path, format="wav")
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def audio_24bit(short_audio_segment: str) -> str:
    """Create 24-bit audio for bit depth testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    # Load and save as 24-bit
    data, sr = sf.read(short_audio_segment, dtype='float32')
    sf.write(temp_path, data, sr, subtype='PCM_24')
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mono_audio(short_audio_segment: str) -> str:
    """Create mono version of audio."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    audio = AudioSegment.from_wav(short_audio_segment)
    audio = audio.set_channels(1)
    audio.export(temp_path, format="wav")
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# =============================================================================
# Helper Functions
# =============================================================================

def create_audio_batch_filepath(audio_path: str, task_id: str = "test") -> AudioBatch:
    """Create AudioBatch with only audio_filepath."""
    return AudioBatch(
        task_id=task_id,
        dataset_name="test_dataset",
        data=[{"audio_filepath": audio_path}]
    )


def create_audio_batch_waveform(audio_path: str, task_id: str = "test") -> AudioBatch:
    """Create AudioBatch with waveform tensor loaded (converted to mono)."""
    data, sample_rate = sf.read(audio_path, dtype='float32')
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    
    # Convert to mono if stereo (many modules expect mono)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    return AudioBatch(
        task_id=task_id,
        dataset_name="test_dataset",
        data=[{
            "audio_filepath": audio_path,
            "waveform": waveform,
            "sample_rate": sample_rate
        }]
    )


def create_audio_batch_pydub(audio_path: str, task_id: str = "test") -> AudioBatch:
    """Create AudioBatch with PyDub AudioSegment loaded."""
    audio = AudioSegment.from_wav(audio_path)
    
    return AudioBatch(
        task_id=task_id,
        dataset_name="test_dataset",
        data=[{
            "audio_filepath": audio_path,
            "audio": audio,
            "sample_rate": audio.frame_rate
        }]
    )


def print_result(stage_name: str, result, input_format: str = ""):
    """Print test result details."""
    format_str = f" ({input_format})" if input_format else ""
    print(f"\n{'='*60}")
    print(f"{stage_name}{format_str} Result:")
    print(f"{'='*60}")
    
    if result is None:
        print("  Result: None (filtered out)")
        return
    
    if isinstance(result, list):
        print(f"  Type: List of {len(result)} AudioBatch(es)")
        for i, batch in enumerate(result[:3]):  # Show first 3
            print(f"  Batch {i}: {len(batch.data)} items")
            if batch.data:
                item = batch.data[0]
                print(f"    Keys: {list(item.keys())}")
    else:
        print(f"  Type: AudioBatch with {len(result.data)} items")
        if result.data:
            item = result.data[0]
            print(f"    Keys: {list(item.keys())}")
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: Tensor shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, AudioSegment):
                    print(f"    {key}: AudioSegment duration={len(value)}ms, channels={value.channels}")
                elif isinstance(value, (int, float, str, bool)):
                    print(f"    {key}: {value}")


# =============================================================================
# TEST CATEGORY 1: MonoConversionStage Tests
# =============================================================================

class TestMonoConversionStage:
    """Tests for MonoConversionStage module."""
    
    def test_mono_with_filepath_48k_stereo(self, short_audio_segment: str):
        """Test with audio_filepath input - 48kHz stereo."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=True)
        task = create_audio_batch_filepath(short_audio_segment)
        
        result = stage.process(task)
        print_result("MonoConversion", result, "filepath 48kHz stereo")
        
        assert result is not None
        assert len(result.data) == 1
        item = result.data[0]
        assert "waveform" in item
        assert "sample_rate" in item
        assert item["sample_rate"] == 48000
        assert item["is_mono"] is True
        assert item["waveform"].shape[0] == 1  # 1 channel
    
    def test_mono_with_filepath_16k(self, audio_16k: str):
        """Test with 16kHz audio - should reject with strict mode."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=True)
        task = create_audio_batch_filepath(audio_16k)
        
        result = stage.process(task)
        print_result("MonoConversion", result, "filepath 16kHz (strict=True)")
        
        assert result is None, "Should reject 16kHz with strict_sample_rate=True"
    
    def test_mono_with_filepath_16k_flexible(self, audio_16k: str):
        """Test with 16kHz audio - should accept with non-strict mode."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=False)
        task = create_audio_batch_filepath(audio_16k)
        
        result = stage.process(task)
        print_result("MonoConversion", result, "filepath 16kHz (strict=False)")
        
        assert result is not None
        assert result.data[0]["sample_rate"] == 16000
    
    def test_mono_with_24bit(self, audio_24bit: str):
        """Test with 24-bit audio."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=True)
        task = create_audio_batch_filepath(audio_24bit)
        
        result = stage.process(task)
        print_result("MonoConversion", result, "filepath 24-bit")
        
        assert result is not None
        assert result.data[0]["waveform"].dtype == torch.float32
    
    def test_mono_already_mono(self, mono_audio: str):
        """Test with already mono audio."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=True)
        task = create_audio_batch_filepath(mono_audio)
        
        result = stage.process(task)
        print_result("MonoConversion", result, "filepath mono")
        
        assert result is not None
        assert result.data[0]["is_mono"] is True


# =============================================================================
# TEST CATEGORY 2: BandFilterStage Tests
# =============================================================================

class TestBandFilterStage:
    """Tests for BandFilterStage module."""
    
    @pytest.fixture(autouse=True)
    def check_model(self):
        """Check if band model exists, skip if not."""
        from nemo_curator.stages.audio.filtering import BandFilterStage
        stage = BandFilterStage()
        model_path = stage._resolve_model_path()
        if not os.path.exists(model_path):
            pytest.skip(f"Band model not found: {model_path}")
    
    def test_band_with_filepath(self, short_audio_segment: str):
        """Test with audio_filepath input (auto-load)."""
        from nemo_curator.stages.audio.filtering import BandFilterStage
        
        stage = BandFilterStage(band_value="full_band")
        task = create_audio_batch_filepath(short_audio_segment)
        
        result = stage.process(task)
        print_result("BandFilter", result, "filepath")
        
        # May return None if classified as narrow_band
        if result is not None:
            assert "band_prediction" in result.data[0]
    
    def test_band_with_waveform(self, short_audio_segment: str):
        """Test with waveform tensor input."""
        from nemo_curator.stages.audio.filtering import BandFilterStage
        
        stage = BandFilterStage(band_value="full_band")
        task = create_audio_batch_waveform(short_audio_segment)
        
        result = stage.process(task)
        print_result("BandFilter", result, "waveform")
        
        if result is not None:
            assert "band_prediction" in result.data[0]
            assert "waveform" in result.data[0]  # Preserved
    
    def test_band_with_16k(self, audio_16k: str):
        """Test with 16kHz audio."""
        from nemo_curator.stages.audio.filtering import BandFilterStage
        
        stage = BandFilterStage(band_value="narrow_band")
        task = create_audio_batch_filepath(audio_16k)
        
        result = stage.process(task)
        print_result("BandFilter", result, "filepath 16kHz")
        
        # 16kHz is typically classified as narrow_band


# =============================================================================
# TEST CATEGORY 3: VADSegmentationStage Tests
# =============================================================================

class TestVADSegmentationStage:
    """Tests for VADSegmentationStage module."""
    
    def test_vad_with_filepath(self, short_audio_segment: str):
        """Test with audio_filepath input (auto-load)."""
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        stage = VADSegmentationStage(
            min_duration_sec=0.5,
            max_duration_sec=30.0,
            threshold=0.5
        )
        stage.setup()
        
        task = create_audio_batch_filepath(short_audio_segment)
        
        result = stage.process(task)
        print_result("VAD", result, "filepath")
        
        assert isinstance(result, list)
        print(f"  Detected {len(result)} speech segments")
        
        for i, batch in enumerate(result[:3]):
            item = batch.data[0]
            assert "audio" in item or "waveform" in item
            assert "sample_rate" in item
            assert "start_ms" in item
            assert "end_ms" in item
            print(f"  Segment {i}: {item.get('start_ms', 0)}-{item.get('end_ms', 0)} ms")
    
    def test_vad_with_waveform(self, short_audio_segment: str):
        """Test with waveform tensor input."""
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        stage.setup()
        
        task = create_audio_batch_waveform(short_audio_segment)
        
        result = stage.process(task)
        print_result("VAD", result, "waveform")
        
        assert isinstance(result, list)
    
    def test_vad_with_16k(self, audio_16k: str):
        """Test with 16kHz audio (Silero VAD native rate)."""
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        stage.setup()
        
        task = create_audio_batch_filepath(audio_16k)
        
        result = stage.process(task)
        print_result("VAD", result, "filepath 16kHz")
        
        assert isinstance(result, list)


# =============================================================================
# TEST CATEGORY 4: NISQAFilterStage Tests
# =============================================================================

class TestNISQAFilterStage:
    """Tests for NISQAFilterStage module."""
    
    @pytest.fixture(autouse=True)
    def check_model(self):
        """Check if NISQA model exists, skip if not."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        stage = NISQAFilterStage()
        model_path = stage._resolve_model_path()
        if not os.path.exists(model_path):
            pytest.skip(f"NISQA model not found: {model_path}")
    
    def test_nisqa_with_filepath(self, short_audio_segment: str):
        """Test with audio_filepath input (auto-load)."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        stage = NISQAFilterStage(mos_threshold=None)  # No filtering, just assess
        task = create_audio_batch_filepath(short_audio_segment)
        
        result = stage.process(task)
        print_result("NISQA", result, "filepath")
        
        if result is not None:
            item = result.data[0]
            assert "nisqa_mos" in item
            print(f"  MOS: {item.get('nisqa_mos', 'N/A')}")
            print(f"  NOI: {item.get('nisqa_noi', 'N/A')}")
    
    def test_nisqa_with_pydub(self, short_audio_segment: str):
        """Test with PyDub AudioSegment input."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        stage = NISQAFilterStage(mos_threshold=None)
        task = create_audio_batch_pydub(short_audio_segment)
        
        result = stage.process(task)
        print_result("NISQA", result, "pydub")
        
        if result is not None:
            assert "nisqa_mos" in result.data[0]
    
    def test_nisqa_with_16k(self, audio_16k: str):
        """Test with 16kHz audio."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        stage = NISQAFilterStage(mos_threshold=None)
        task = create_audio_batch_filepath(audio_16k)
        
        result = stage.process(task)
        print_result("NISQA", result, "filepath 16kHz")
    
    def test_nisqa_all_thresholds(self, short_audio_segment: str):
        """Test NISQA with all threshold parameters set to 3."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        # Test with all thresholds set to 3 (should pass for good audio)
        stage = NISQAFilterStage(
            mos_threshold=3.0,
            noi_threshold=3.0,
            col_threshold=3.0,
            dis_threshold=3.0,
            loud_threshold=3.0
        )
        task = create_audio_batch_filepath(short_audio_segment)
        
        result = stage.process(task)
        print_result("NISQA", result, "all thresholds=3")
        
        if result is not None:
            item = result.data[0]
            print(f"  MOS: {item.get('nisqa_mos', 'N/A')} (threshold: 3.0)")
            print(f"  NOI: {item.get('nisqa_noi', 'N/A')} (threshold: 3.0)")
            print(f"  COL: {item.get('nisqa_col', 'N/A')} (threshold: 3.0)")
            print(f"  DIS: {item.get('nisqa_dis', 'N/A')} (threshold: 3.0)")
            print(f"  LOUD: {item.get('nisqa_loud', 'N/A')} (threshold: 3.0)")
            # Verify all metrics are present
            assert "nisqa_mos" in item
            assert "nisqa_noi" in item
    
    def test_nisqa_strict_thresholds(self, short_audio_segment: str):
        """Test NISQA with strict thresholds (should filter out most audio)."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        # Very strict thresholds - most audio should fail
        stage = NISQAFilterStage(
            mos_threshold=4.8,
            noi_threshold=4.8,
            col_threshold=4.8,
            dis_threshold=4.8,
            loud_threshold=4.8
        )
        task = create_audio_batch_filepath(short_audio_segment)
        
        result = stage.process(task)
        print_result("NISQA", result, "strict thresholds=4.8")
        # Result may be None (filtered out) or passed if audio is excellent
    
    def test_nisqa_individual_thresholds(self, short_audio_segment: str):
        """Test NISQA with individual threshold configurations."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        # Test only MOS threshold
        stage_mos = NISQAFilterStage(
            mos_threshold=3.0,
            noi_threshold=None,
            col_threshold=None,
            dis_threshold=None,
            loud_threshold=None
        )
        task = create_audio_batch_filepath(short_audio_segment)
        result = stage_mos.process(task)
        print_result("NISQA", result, "only MOS>=3.0")
        
        # Test only NOI threshold
        stage_noi = NISQAFilterStage(
            mos_threshold=None,
            noi_threshold=3.0,
            col_threshold=None,
            dis_threshold=None,
            loud_threshold=None
        )
        task = create_audio_batch_filepath(short_audio_segment)
        result = stage_noi.process(task)
        print_result("NISQA", result, "only NOI>=3.0")
        
        # Test combined MOS + NOI
        stage_combo = NISQAFilterStage(
            mos_threshold=3.0,
            noi_threshold=3.0,
            col_threshold=None,
            dis_threshold=None,
            loud_threshold=None
        )
        task = create_audio_batch_filepath(short_audio_segment)
        result = stage_combo.process(task)
        print_result("NISQA", result, "MOS>=3.0 + NOI>=3.0")


# =============================================================================
# TEST CATEGORY 5: SIGMOSFilterStage Tests
# =============================================================================

class TestSIGMOSFilterStage:
    """Tests for SIGMOSFilterStage module."""
    
    @pytest.fixture(autouse=True)
    def check_model(self):
        """Check if SIGMOS model exists, skip if not."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        stage = SIGMOSFilterStage()
        model_path = stage._resolve_model_path()
        if not os.path.exists(model_path):
            pytest.skip(f"SIGMOS model not found: {model_path}")
    
    def test_sigmos_with_filepath(self, short_audio_segment: str):
        """Test with audio_filepath input (auto-load)."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        
        stage = SIGMOSFilterStage(ovrl_threshold=None)  # No filtering
        task = create_audio_batch_filepath(short_audio_segment)
        
        result = stage.process(task)
        print_result("SIGMOS", result, "filepath")
        
        if result is not None:
            item = result.data[0]
            print(f"  NOISE: {item.get('sigmos_noise', 'N/A')}")
            print(f"  OVRL: {item.get('sigmos_ovrl', 'N/A')}")
    
    def test_sigmos_with_pydub(self, short_audio_segment: str):
        """Test with PyDub AudioSegment input."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        
        stage = SIGMOSFilterStage(ovrl_threshold=None)
        task = create_audio_batch_pydub(short_audio_segment)
        
        result = stage.process(task)
        print_result("SIGMOS", result, "pydub")
    
    def test_sigmos_with_16k(self, audio_16k: str):
        """Test with 16kHz audio."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        
        stage = SIGMOSFilterStage(ovrl_threshold=None)
        task = create_audio_batch_filepath(audio_16k)
        
        result = stage.process(task)
        print_result("SIGMOS", result, "filepath 16kHz")
    
    def test_sigmos_all_thresholds(self, short_audio_segment: str):
        """Test SIGMOS with all threshold parameters set to 3."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        
        # Test with all thresholds set to 3 (should pass for good audio)
        stage = SIGMOSFilterStage(
            noise_threshold=3.0,
            ovrl_threshold=3.0,
            sig_threshold=3.0,
            col_threshold=3.0,
            disc_threshold=3.0,
            loud_threshold=3.0,
            reverb_threshold=3.0
        )
        task = create_audio_batch_filepath(short_audio_segment)
        
        result = stage.process(task)
        print_result("SIGMOS", result, "all thresholds=3")
        
        if result is not None:
            item = result.data[0]
            print(f"  NOISE: {item.get('sigmos_noise', 'N/A')} (threshold: 3.0)")
            print(f"  OVRL: {item.get('sigmos_ovrl', 'N/A')} (threshold: 3.0)")
            print(f"  SIG: {item.get('sigmos_sig', 'N/A')} (threshold: 3.0)")
            print(f"  COL: {item.get('sigmos_col', 'N/A')} (threshold: 3.0)")
            print(f"  DISC: {item.get('sigmos_disc', 'N/A')} (threshold: 3.0)")
            print(f"  LOUD: {item.get('sigmos_loud', 'N/A')} (threshold: 3.0)")
            print(f"  REVERB: {item.get('sigmos_reverb', 'N/A')} (threshold: 3.0)")
            # Verify metrics are present
            assert "sigmos_ovrl" in item or "sigmos_noise" in item
    
    def test_sigmos_strict_thresholds(self, short_audio_segment: str):
        """Test SIGMOS with strict thresholds (should filter out most audio)."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        
        # Very strict thresholds - most audio should fail
        stage = SIGMOSFilterStage(
            noise_threshold=4.8,
            ovrl_threshold=4.8,
            sig_threshold=4.8,
            col_threshold=4.8,
            disc_threshold=4.8,
            loud_threshold=4.8,
            reverb_threshold=4.8
        )
        task = create_audio_batch_filepath(short_audio_segment)
        
        result = stage.process(task)
        print_result("SIGMOS", result, "strict thresholds=4.8")
        # Result may be None (filtered out) or passed if audio is excellent
    
    def test_sigmos_individual_thresholds(self, short_audio_segment: str):
        """Test SIGMOS with individual threshold configurations."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        
        # Test only NOISE threshold
        stage_noise = SIGMOSFilterStage(
            noise_threshold=3.0,
            ovrl_threshold=None,
            sig_threshold=None,
            col_threshold=None,
            disc_threshold=None,
            loud_threshold=None,
            reverb_threshold=None
        )
        task = create_audio_batch_filepath(short_audio_segment)
        result = stage_noise.process(task)
        print_result("SIGMOS", result, "only NOISE>=3.0")
        
        # Test only OVRL threshold
        stage_ovrl = SIGMOSFilterStage(
            noise_threshold=None,
            ovrl_threshold=3.0,
            sig_threshold=None,
            col_threshold=None,
            disc_threshold=None,
            loud_threshold=None,
            reverb_threshold=None
        )
        task = create_audio_batch_filepath(short_audio_segment)
        result = stage_ovrl.process(task)
        print_result("SIGMOS", result, "only OVRL>=3.0")
        
        # Test combined NOISE + OVRL + REVERB
        stage_combo = SIGMOSFilterStage(
            noise_threshold=3.0,
            ovrl_threshold=3.0,
            sig_threshold=None,
            col_threshold=None,
            disc_threshold=None,
            loud_threshold=None,
            reverb_threshold=3.0
        )
        task = create_audio_batch_filepath(short_audio_segment)
        result = stage_combo.process(task)
        print_result("SIGMOS", result, "NOISE>=3.0 + OVRL>=3.0 + REVERB>=3.0")


# =============================================================================
# TEST CATEGORY 6: SpeakerSeparationStage Tests
# =============================================================================

class TestSpeakerSeparationStage:
    """Tests for SpeakerSeparationStage module."""
    
    @pytest.fixture(autouse=True)
    def check_model(self):
        """Check if speaker separation model exists, skip if not."""
        from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
        stage = SpeakerSeparationStage()
        model_path = stage._resolve_model_path()
        if not os.path.exists(model_path):
            pytest.skip(f"Speaker separation model not found: {model_path}")
    
    def test_speaker_sep_with_filepath(self, short_audio_segment: str):
        """Test with audio_filepath input (auto-load)."""
        from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
        
        stage = SpeakerSeparationStage(min_duration=0.5)
        stage.setup()
        
        task = create_audio_batch_filepath(short_audio_segment)
        
        result = stage.process(task)
        print_result("SpeakerSeparation", result, "filepath")
        
        assert isinstance(result, list)
        print(f"  Detected {len(result)} speaker(s)")
        
        for i, batch in enumerate(result[:3]):
            item = batch.data[0]
            print(f"  Speaker {item.get('speaker_id', i)}: duration={item.get('duration_sec', 0):.2f}s")
    
    def test_speaker_sep_with_pydub(self, short_audio_segment: str):
        """Test with PyDub AudioSegment input."""
        from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
        
        stage = SpeakerSeparationStage(min_duration=0.5)
        stage.setup()
        
        task = create_audio_batch_pydub(short_audio_segment)
        
        result = stage.process(task)
        print_result("SpeakerSeparation", result, "pydub")
        
        assert isinstance(result, list)
    
    def test_speaker_sep_with_waveform(self, short_audio_segment: str):
        """Test with waveform tensor input."""
        from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
        
        stage = SpeakerSeparationStage(min_duration=0.5)
        stage.setup()
        
        task = create_audio_batch_waveform(short_audio_segment)
        
        result = stage.process(task)
        print_result("SpeakerSeparation", result, "waveform")
        
        assert isinstance(result, list)


# =============================================================================
# TEST CATEGORY 7: Pipeline Integration Tests
# =============================================================================

class TestPipelineIntegration:
    """Tests for modules working together in pipelines."""
    
    def test_pipeline_mono_vad(self, short_audio_segment: str):
        """Test MonoConversion -> VAD pipeline."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        mono_stage = MonoConversionStage(output_sample_rate=48000)
        vad_stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        vad_stage.setup()
        
        task = create_audio_batch_filepath(short_audio_segment)
        
        # Stage 1: Mono
        mono_result = mono_stage.process(task)
        assert mono_result is not None
        assert "waveform" in mono_result.data[0]
        print_result("Pipeline: Mono", mono_result)
        
        # Stage 2: VAD
        vad_result = vad_stage.process(mono_result)
        assert isinstance(vad_result, list)
        print_result("Pipeline: VAD", vad_result)
    
    def test_pipeline_mono_vad_band(self, short_audio_segment: str):
        """Test MonoConversion -> VAD -> BandFilter pipeline."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        from nemo_curator.stages.audio.filtering import BandFilterStage
        
        # Check band model
        band_stage = BandFilterStage(band_value="full_band")
        if not os.path.exists(band_stage._resolve_model_path()):
            pytest.skip("Band model not found")
        
        mono_stage = MonoConversionStage(output_sample_rate=48000)
        vad_stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        vad_stage.setup()
        
        task = create_audio_batch_filepath(short_audio_segment)
        
        # Stage 1: Mono
        mono_result = mono_stage.process(task)
        
        # Stage 2: VAD
        vad_results = vad_stage.process(mono_result)
        
        # Stage 3: BandFilter on each segment
        passed = 0
        filtered = 0
        for segment_batch in vad_results[:5]:  # Test first 5 segments
            band_result = band_stage.process(segment_batch)
            if band_result is not None:
                passed += 1
                assert "band_prediction" in band_result.data[0]
            else:
                filtered += 1
        
        print(f"\nPipeline Mono->VAD->Band: {passed} passed, {filtered} filtered")
    
    def test_pipeline_mono_vad_nisqa(self, short_audio_segment: str):
        """Test MonoConversion -> VAD -> NISQA pipeline."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        # Check NISQA model
        nisqa_stage = NISQAFilterStage(mos_threshold=None)
        if not os.path.exists(nisqa_stage._resolve_model_path()):
            pytest.skip("NISQA model not found")
        
        mono_stage = MonoConversionStage(output_sample_rate=48000)
        vad_stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        vad_stage.setup()
        
        task = create_audio_batch_filepath(short_audio_segment)
        
        # Stage 1: Mono
        mono_result = mono_stage.process(task)
        
        # Stage 2: VAD
        vad_results = vad_stage.process(mono_result)
        
        # Stage 3: NISQA on each segment
        for segment_batch in vad_results[:3]:
            nisqa_result = nisqa_stage.process(segment_batch)
            if nisqa_result is not None:
                item = nisqa_result.data[0]
                print(f"  Segment MOS: {item.get('nisqa_mos', 'N/A')}")
    
    def test_data_preservation_through_pipeline(self, short_audio_segment: str):
        """Test that custom metadata is preserved through pipeline."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        from nemo_curator.stages.audio.filtering import BandFilterStage
        
        band_stage = BandFilterStage(band_value="full_band")
        if not os.path.exists(band_stage._resolve_model_path()):
            pytest.skip("Band model not found")
        
        mono_stage = MonoConversionStage(output_sample_rate=48000)
        
        # Create task with custom metadata
        task = AudioBatch(
            task_id="test_preserve",
            dataset_name="test",
            data=[{
                "audio_filepath": short_audio_segment,
                "custom_id": "audio_001",
                "source": "youtube",
                "timestamp": 12345
            }]
        )
        
        mono_result = mono_stage.process(task)
        assert mono_result.data[0]["custom_id"] == "audio_001"
        assert mono_result.data[0]["source"] == "youtube"
        
        band_result = band_stage.process(mono_result)
        if band_result is not None:
            assert band_result.data[0]["custom_id"] == "audio_001"
            assert band_result.data[0]["source"] == "youtube"
            print("  Custom metadata preserved through pipeline!")


# =============================================================================
# TEST CATEGORY 8: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_missing_file(self):
        """Test handling of missing audio file."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000)
        task = create_audio_batch_filepath("/nonexistent/path/audio.wav")
        
        result = stage.process(task)
        assert result is None
    
    def test_empty_batch(self):
        """Test handling of empty batch."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000)
        task = AudioBatch(task_id="empty", dataset_name="test", data=[])
        
        result = stage.process(task)
        assert result is None
    
    def test_multiple_items(self, short_audio_segment: str, mono_audio: str):
        """Test processing multiple items in a batch."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000)
        task = AudioBatch(
            task_id="multi",
            dataset_name="test",
            data=[
                {"audio_filepath": short_audio_segment, "id": 1},
                {"audio_filepath": mono_audio, "id": 2}
            ]
        )
        
        result = stage.process(task)
        assert result is not None
        assert len(result.data) == 2
        assert result.data[0]["id"] == 1
        assert result.data[1]["id"] == 2


# =============================================================================
# TEST CATEGORY 9: Batch Size and process_batch() Tests
# =============================================================================

class TestBatchSizeProcessing:
    """Tests for batch_size parameter and process_batch() method."""
    
    def test_mono_batch_size_config(self, short_audio_segment: str):
        """Test MonoConversionStage batch_size configuration."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        # Default batch_size
        stage_default = MonoConversionStage(output_sample_rate=48000)
        print(f"\n  MonoConversion default batch_size: {stage_default._batch_size}")
        assert stage_default._batch_size == 1
        
        # Custom batch_size via with_()
        stage_batch_3 = MonoConversionStage(output_sample_rate=48000).with_(batch_size=3)
        print(f"  MonoConversion custom batch_size: {stage_batch_3._batch_size}")
        assert stage_batch_3._batch_size == 3
        
        # Verify processing still works
        task = create_audio_batch_filepath(short_audio_segment)
        result = stage_batch_3.process(task)
        assert result is not None
    
    def test_band_batch_size_config(self, short_audio_segment: str):
        """Test BandFilterStage batch_size configuration."""
        from nemo_curator.stages.audio.filtering import BandFilterStage
        
        stage = BandFilterStage(band_value="full_band")
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("Band model not found")
        
        print(f"\n  BandFilter default batch_size: {stage._batch_size}")
        assert stage._batch_size == 1
        
        # Custom batch_size
        stage_batch_5 = BandFilterStage(band_value="full_band").with_(batch_size=5)
        print(f"  BandFilter custom batch_size: {stage_batch_5._batch_size}")
        assert stage_batch_5._batch_size == 5
        
        # Verify processing still works
        task = create_audio_batch_waveform(short_audio_segment)
        result = stage_batch_5.process(task)
        print(f"  Result: {'passed' if result else 'filtered'}")
    
    def test_nisqa_batch_size_config(self, short_audio_segment: str):
        """Test NISQAFilterStage batch_size configuration."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        stage = NISQAFilterStage(mos_threshold=3.0, noi_threshold=3.0)
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("NISQA model not found")
        
        print(f"\n  NISQA default batch_size: {stage._batch_size}")
        
        # Custom batch_size
        stage_batch_4 = NISQAFilterStage(mos_threshold=3.0).with_(batch_size=4)
        print(f"  NISQA custom batch_size: {stage_batch_4._batch_size}")
        assert stage_batch_4._batch_size == 4
        
        # Verify processing still works
        task = create_audio_batch_pydub(short_audio_segment)
        result = stage_batch_4.process(task)
        assert result is not None
    
    def test_sigmos_batch_size_config(self, short_audio_segment: str):
        """Test SIGMOSFilterStage batch_size configuration."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        
        stage = SIGMOSFilterStage(noise_threshold=3.0, ovrl_threshold=3.0)
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("SIGMOS model not found")
        
        print(f"\n  SIGMOS default batch_size: {stage._batch_size}")
        
        # Custom batch_size with lower threshold to ensure pass
        stage_batch_2 = SIGMOSFilterStage(noise_threshold=3.0, ovrl_threshold=3.0).with_(batch_size=2)
        print(f"  SIGMOS custom batch_size: {stage_batch_2._batch_size}")
        assert stage_batch_2._batch_size == 2
        
        # Verify processing still works
        task = create_audio_batch_pydub(short_audio_segment)
        result = stage_batch_2.process(task)
        assert result is not None, "SIGMOS should pass with thresholds=3.0"
    
    def test_vad_batch_size_config(self, short_audio_segment: str):
        """Test VADSegmentationStage batch_size configuration."""
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        print(f"\n  VAD default batch_size: {stage._batch_size}")
        
        # Custom batch_size
        stage_batch_10 = VADSegmentationStage(min_duration_sec=0.5).with_(batch_size=10)
        stage_batch_10.setup()
        print(f"  VAD custom batch_size: {stage_batch_10._batch_size}")
        assert stage_batch_10._batch_size == 10
        
        # Verify processing still works
        task = create_audio_batch_filepath(short_audio_segment)
        result = stage_batch_10.process(task)
        assert isinstance(result, list)
        print(f"  VAD with batch_size=10: {len(result)} segments produced")
    
    def test_speaker_sep_batch_size_config(self, short_audio_segment: str):
        """Test SpeakerSeparationStage batch_size configuration."""
        from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
        
        stage = SpeakerSeparationStage(min_duration=0.5)
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("Speaker separation model not found")
        
        print(f"\n  SpeakerSep default batch_size: {stage._batch_size}")
        
        # Custom batch_size
        stage_batch_2 = SpeakerSeparationStage(min_duration=0.5).with_(batch_size=2)
        stage_batch_2.setup()
        print(f"  SpeakerSep custom batch_size: {stage_batch_2._batch_size}")
        assert stage_batch_2._batch_size == 2
        
        # Verify processing still works
        task = create_audio_batch_filepath(short_audio_segment)
        result = stage_batch_2.process(task)
        assert isinstance(result, list)
        print(f"  SpeakerSep with batch_size=2: {len(result)} speaker segments")
    
    def test_sequential_process_multiple_tasks(self, short_audio_segment: str, mono_audio: str, audio_16k: str):
        """Test processing multiple tasks sequentially (simulating batch processing).
        
        Note: AudioBatch uses list[dict] for data, which doesn't work with the standard
        validate_input() that expects task.data to have attributes. Instead, we test
        sequential processing of multiple tasks which is how the executor would call
        process() for each task in a batch.
        """
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=False)
        
        # Simulate what an executor with batch_size=3 would do
        tasks = [
            create_audio_batch_filepath(short_audio_segment, task_id="task_1"),
            create_audio_batch_filepath(mono_audio, task_id="task_2"),
            create_audio_batch_filepath(audio_16k, task_id="task_3"),
        ]
        
        # Process each task individually (batch_size controls how many tasks executor collects)
        results = []
        for task in tasks:
            result = stage.process(task)
            results.append(result)
        
        print(f"\n  Sequential processing of {len(tasks)} tasks:")
        for i, result in enumerate(results):
            if result is not None:
                print(f"    Task {i}: SR={result.data[0].get('sample_rate')}, mono={result.data[0].get('is_mono')}")
            else:
                print(f"    Task {i}: filtered out")
        
        # All should pass with flexible sample rate
        assert len([r for r in results if r is not None]) == 3
    
    def test_sequential_band_filter_tasks(self, short_audio_segment: str, mono_audio: str):
        """Test sequential BandFilter processing of multiple tasks."""
        from nemo_curator.stages.audio.filtering import BandFilterStage
        
        stage = BandFilterStage(band_value="full_band")
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("Band model not found")
        
        # Simulate batch processing - each task is processed individually
        tasks = [
            create_audio_batch_waveform(short_audio_segment, task_id="band_task_1"),
            create_audio_batch_waveform(mono_audio, task_id="band_task_2"),
        ]
        
        results = []
        for task in tasks:
            result = stage.process(task)
            results.append(result)
        
        print(f"\n  BandFilter sequential processing ({len(tasks)} tasks):")
        for i, result in enumerate(results):
            if result is not None:
                print(f"    Task {i}: band={result.data[0].get('band_prediction')}")
            else:
                print(f"    Task {i}: filtered out")
    
    def test_sequential_nisqa_tasks(self, short_audio_segment: str, mono_audio: str):
        """Test sequential NISQA processing of multiple tasks."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        stage = NISQAFilterStage(
            mos_threshold=3.0,
            noi_threshold=3.0,
            col_threshold=3.0,
            dis_threshold=3.0,
            loud_threshold=3.0
        )
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("NISQA model not found")
        
        tasks = [
            create_audio_batch_pydub(short_audio_segment, task_id="nisqa_task_1"),
            create_audio_batch_pydub(mono_audio, task_id="nisqa_task_2"),
        ]
        
        results = []
        for task in tasks:
            result = stage.process(task)
            results.append(result)
        
        print(f"\n  NISQA sequential processing ({len(tasks)} tasks):")
        for i, result in enumerate(results):
            if result is not None:
                print(f"    Task {i}: MOS={result.data[0].get('nisqa_mos', 0):.2f}")
    
    def test_sequential_vad_tasks(self, short_audio_segment: str, mono_audio: str):
        """Test sequential VAD processing produces multiple segment batches per task."""
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        stage.setup()
        
        tasks = [
            create_audio_batch_filepath(short_audio_segment, task_id="vad_task_1"),
            create_audio_batch_filepath(mono_audio, task_id="vad_task_2"),
        ]
        
        all_segments = []
        for task in tasks:
            result = stage.process(task)
            if isinstance(result, list):
                all_segments.extend(result)
        
        print(f"\n  VAD sequential processing ({len(tasks)} tasks):")
        print(f"    Total segments generated: {len(all_segments)}")
        for i, seg in enumerate(all_segments[:5]):
            print(f"    Segment {i}: {seg.data[0].get('start_ms', 0)}-{seg.data[0].get('end_ms', 0)} ms")
    
    def test_multi_items_per_batch(self, short_audio_segment: str, mono_audio: str, audio_16k: str):
        """Test processing batch with multiple items (audio files)."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=False)
        
        # Single batch with multiple items
        task = AudioBatch(
            task_id="multi_item_batch",
            dataset_name="test",
            data=[
                {"audio_filepath": short_audio_segment, "id": "file_1", "source": "youtube"},
                {"audio_filepath": mono_audio, "id": "file_2", "source": "local"},
                {"audio_filepath": audio_16k, "id": "file_3", "source": "converted"},
            ]
        )
        
        result = stage.process(task)
        assert result is not None
        assert len(result.data) == 3, "All 3 items should be processed"
        
        for i, item in enumerate(result.data):
            print(f"  Item {i}: id={item.get('id')}, SR={item.get('sample_rate')}, "
                  f"mono={item.get('is_mono')}, source={item.get('source')}")
            assert "waveform" in item
            assert "sample_rate" in item
            assert item.get("is_mono") is True
        
        print(f"\nMulti-item batch: {len(result.data)} items processed in single batch")


# =============================================================================
# TEST CATEGORY 10: Resource Configuration Tests
# =============================================================================

class TestResourceConfigurations:
    """Tests for different resource configurations passed to stages."""
    
    def test_mono_cpu_resources(self, short_audio_segment: str):
        """Test MonoConversionStage with various CPU resource settings."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        from nemo_curator.stages.resources import Resources
        
        # Default resources
        stage_default = MonoConversionStage(output_sample_rate=48000)
        print(f"\n  Default resources: cpus={stage_default._resources.cpus}, "
              f"gpus={stage_default._resources.gpus}")
        
        # Custom CPU resources
        stage_cpu_2 = MonoConversionStage(output_sample_rate=48000).with_(
            resources=Resources(cpus=2.0)
        )
        print(f"  Custom resources: cpus={stage_cpu_2._resources.cpus}")
        assert stage_cpu_2._resources.cpus == 2.0
        
        # Process with custom resources
        task = create_audio_batch_filepath(short_audio_segment)
        result = stage_cpu_2.process(task)
        assert result is not None
        print(f"  Stage with cpus=2.0 processed successfully")
    
    def test_band_gpu_resources(self, short_audio_segment: str):
        """Test BandFilterStage with GPU resource settings."""
        from nemo_curator.stages.audio.filtering import BandFilterStage
        from nemo_curator.stages.resources import Resources
        
        stage = BandFilterStage(band_value="full_band")
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("Band model not found")
        
        # Check default GPU resources
        print(f"\n  BandFilter default: cpus={stage._resources.cpus}, "
              f"gpus={stage._resources.gpus}, gpu_memory_gb={stage._resources.gpu_memory_gb}")
        
        # Custom GPU resources using gpus
        stage_gpu = BandFilterStage(band_value="full_band").with_(
            resources=Resources(cpus=2.0, gpus=0.5)
        )
        print(f"  Custom GPU (gpus=0.5): cpus={stage_gpu._resources.cpus}, "
              f"gpus={stage_gpu._resources.gpus}")
        assert stage_gpu._resources.gpus == 0.5
        assert stage_gpu._resources.cpus == 2.0
        
        # Process with custom GPU resources
        task = create_audio_batch_waveform(short_audio_segment)
        result = stage_gpu.process(task)
        print(f"  Stage with gpus=0.5 processed: {'passed' if result else 'filtered'}")
    
    def test_nisqa_gpu_memory_resources(self, short_audio_segment: str):
        """Test NISQAFilterStage with gpu_memory_gb resource setting."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        from nemo_curator.stages.resources import Resources
        
        stage = NISQAFilterStage(mos_threshold=None)
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("NISQA model not found")
        
        # Check default resources
        print(f"\n  NISQA default: cpus={stage._resources.cpus}, "
              f"gpus={stage._resources.gpus}, gpu_memory_gb={stage._resources.gpu_memory_gb}")
        
        # Custom GPU memory resources
        stage_mem = NISQAFilterStage(mos_threshold=None).with_(
            resources=Resources(cpus=4.0, gpu_memory_gb=8.0)
        )
        print(f"  Custom (gpu_memory_gb=8.0): cpus={stage_mem._resources.cpus}, "
              f"gpu_memory_gb={stage_mem._resources.gpu_memory_gb}")
        assert stage_mem._resources.gpu_memory_gb == 8.0
        assert stage_mem._resources.cpus == 4.0
        
        # Process with custom resources
        task = create_audio_batch_pydub(short_audio_segment)
        result = stage_mem.process(task)
        assert result is not None
        print(f"  Stage with gpu_memory_gb=8.0 processed successfully")
    
    def test_sigmos_cpu_only_resources(self, short_audio_segment: str):
        """Test SIGMOSFilterStage with CPU-only resources."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        from nemo_curator.stages.resources import Resources
        
        stage = SIGMOSFilterStage(ovrl_threshold=None)
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("SIGMOS model not found")
        
        # Force CPU-only resources
        stage_cpu = SIGMOSFilterStage(ovrl_threshold=None).with_(
            resources=Resources(cpus=8.0, gpus=0.0)
        )
        print(f"\n  SIGMOS CPU-only: cpus={stage_cpu._resources.cpus}, "
              f"gpus={stage_cpu._resources.gpus}")
        assert stage_cpu._resources.gpus == 0.0
        assert stage_cpu._resources.cpus == 8.0
        
        # Process with CPU-only
        task = create_audio_batch_pydub(short_audio_segment)
        result = stage_cpu.process(task)
        assert result is not None
        print(f"  Stage with cpus=8.0 (CPU-only) processed successfully")
    
    def test_vad_resources(self, short_audio_segment: str):
        """Test VADSegmentationStage with custom resources."""
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        from nemo_curator.stages.resources import Resources
        
        stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        print(f"\n  VAD default: cpus={stage._resources.cpus}, "
              f"gpus={stage._resources.gpus}")
        
        # Custom resources
        stage_custom = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5).with_(
            resources=Resources(cpus=4.0)
        )
        stage_custom.setup()
        print(f"  VAD custom: cpus={stage_custom._resources.cpus}")
        assert stage_custom._resources.cpus == 4.0
        
        # Process with custom resources
        task = create_audio_batch_filepath(short_audio_segment)
        result = stage_custom.process(task)
        assert isinstance(result, list)
        print(f"  VAD with cpus=4.0: {len(result)} segments produced")
    
    def test_speaker_sep_resources(self, short_audio_segment: str):
        """Test SpeakerSeparationStage with different resources."""
        from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
        from nemo_curator.stages.resources import Resources
        
        stage = SpeakerSeparationStage(min_duration=0.5)
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("Speaker separation model not found")
        
        print(f"\n  SpeakerSep default: cpus={stage._resources.cpus}, "
              f"gpus={stage._resources.gpus}")
        
        # Custom GPU resources
        stage_gpu = SpeakerSeparationStage(min_duration=0.5).with_(
            resources=Resources(cpus=2.0, gpus=1.0)
        )
        stage_gpu.setup()
        print(f"  SpeakerSep custom: cpus={stage_gpu._resources.cpus}, "
              f"gpus={stage_gpu._resources.gpus}")
        assert stage_gpu._resources.gpus == 1.0
        
        # Process with custom resources
        task = create_audio_batch_filepath(short_audio_segment)
        result = stage_gpu.process(task)
        assert isinstance(result, list)
        print(f"  SpeakerSep with gpus=1.0: {len(result)} speaker segments")
    
    def test_resources_preserved_through_with(self, short_audio_segment: str):
        """Test that resources are preserved and not lost with chained with_()."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        from nemo_curator.stages.resources import Resources
        
        # Create with resources and check they persist
        stage = NISQAFilterStage(
            mos_threshold=3.0,
            noi_threshold=3.0
        ).with_(resources=Resources(cpus=4.0, gpus=0.5))
        
        print(f"\n  After with_(): cpus={stage._resources.cpus}, gpus={stage._resources.gpus}")
        assert stage._resources.cpus == 4.0
        assert stage._resources.gpus == 0.5
        
        # The stage parameters should still be accessible
        assert stage.mos_threshold == 3.0
        assert stage.noi_threshold == 3.0
        print(f"  Thresholds preserved: mos={stage.mos_threshold}, noi={stage.noi_threshold}")
    
    def test_combined_resources_and_batch_size(self, short_audio_segment: str):
        """Test combining resources and batch_size configuration."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        from nemo_curator.stages.resources import Resources
        
        stage = NISQAFilterStage(mos_threshold=3.0, noi_threshold=3.0)
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("NISQA model not found")
        
        # Check defaults
        print(f"\n  Defaults: cpus={stage._resources.cpus}, gpus={stage._resources.gpus}, "
              f"batch_size={stage._batch_size}")
        
        # Set both resources AND batch_size
        stage_custom = NISQAFilterStage(
            mos_threshold=3.0,
            noi_threshold=3.0
        ).with_(
            resources=Resources(cpus=4.0, gpus=0.5),
            batch_size=8
        )
        
        print(f"  Custom: cpus={stage_custom._resources.cpus}, gpus={stage_custom._resources.gpus}, "
              f"batch_size={stage_custom._batch_size}")
        
        assert stage_custom._resources.cpus == 4.0
        assert stage_custom._resources.gpus == 0.5
        assert stage_custom._batch_size == 8
        assert stage_custom.mos_threshold == 3.0
        assert stage_custom.noi_threshold == 3.0
        
        # Verify processing works
        task = create_audio_batch_pydub(short_audio_segment)
        result = stage_custom.process(task)
        assert result is not None
        print(f"  Combined config processed successfully!")
    
    def test_all_modules_with_combined_config(self, short_audio_segment: str):
        """Test all modules accept combined resources and batch_size."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        from nemo_curator.stages.audio.filtering import BandFilterStage, NISQAFilterStage, SIGMOSFilterStage
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage, SpeakerSeparationStage
        from nemo_curator.stages.resources import Resources
        
        configs_tested = []
        
        # MonoConversion
        mono = MonoConversionStage(output_sample_rate=48000).with_(
            resources=Resources(cpus=2.0), batch_size=4
        )
        assert mono._resources.cpus == 2.0
        assert mono._batch_size == 4
        configs_tested.append(("MonoConversion", mono._resources.cpus, mono._batch_size))
        
        # BandFilter
        band = BandFilterStage(band_value="full_band").with_(
            resources=Resources(cpus=2.0, gpus=0.3), batch_size=3
        )
        assert band._resources.gpus == 0.3
        assert band._batch_size == 3
        configs_tested.append(("BandFilter", band._resources.gpus, band._batch_size))
        
        # NISQA
        nisqa = NISQAFilterStage(mos_threshold=3.0).with_(
            resources=Resources(cpus=4.0, gpus=0.5), batch_size=6
        )
        assert nisqa._resources.gpus == 0.5
        assert nisqa._batch_size == 6
        configs_tested.append(("NISQA", nisqa._resources.gpus, nisqa._batch_size))
        
        # SIGMOS
        sigmos = SIGMOSFilterStage(noise_threshold=3.0).with_(
            resources=Resources(cpus=4.0, gpus=0.4), batch_size=5
        )
        assert sigmos._resources.gpus == 0.4
        assert sigmos._batch_size == 5
        configs_tested.append(("SIGMOS", sigmos._resources.gpus, sigmos._batch_size))
        
        # VAD
        vad = VADSegmentationStage(min_duration_sec=0.5).with_(
            resources=Resources(cpus=4.0), batch_size=10
        )
        assert vad._resources.cpus == 4.0
        assert vad._batch_size == 10
        configs_tested.append(("VAD", vad._resources.cpus, vad._batch_size))
        
        # SpeakerSeparation
        speaker_sep = SpeakerSeparationStage(min_duration=0.5).with_(
            resources=Resources(cpus=2.0, gpus=1.0), batch_size=2
        )
        assert speaker_sep._resources.gpus == 1.0
        assert speaker_sep._batch_size == 2
        configs_tested.append(("SpeakerSep", speaker_sep._resources.gpus, speaker_sep._batch_size))
        
        print("\n  All modules configured with combined resources + batch_size:")
        for name, resource_val, batch_size in configs_tested:
            print(f"    {name}: resource={resource_val}, batch_size={batch_size}")
    
    def test_requires_gpu_property(self, short_audio_segment: str):
        """Test the requires_gpu property on Resources."""
        from nemo_curator.stages.resources import Resources
        
        # CPU only - should not require GPU
        cpu_only = Resources(cpus=4.0)
        assert not cpu_only.requires_gpu
        print(f"\n  CPU-only (cpus=4.0): requires_gpu={cpu_only.requires_gpu}")
        
        # With gpus - should require GPU
        with_gpus = Resources(cpus=2.0, gpus=0.5)
        assert with_gpus.requires_gpu
        print(f"  With gpus (gpus=0.5): requires_gpu={with_gpus.requires_gpu}")
        
        # With gpu_memory_gb - should require GPU
        with_gpu_mem = Resources(cpus=2.0, gpu_memory_gb=8.0)
        assert with_gpu_mem.requires_gpu
        print(f"  With gpu_memory_gb (8.0): requires_gpu={with_gpu_mem.requires_gpu}")
        
        # Zero gpus - should not require GPU
        zero_gpus = Resources(cpus=2.0, gpus=0.0)
        assert not zero_gpus.requires_gpu
        print(f"  Zero gpus (gpus=0.0): requires_gpu={zero_gpus.requires_gpu}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
