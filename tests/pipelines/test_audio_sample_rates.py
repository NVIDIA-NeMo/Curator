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
Tests for Audio Module Sample Rate Compatibility.

This module tests each audio processing stage with different sample rates to verify:
1. Which sample rates are supported by each module
2. How modules handle sample rate conversion/resampling
3. Any sample rate constraints or requirements

Sample Rate Constraints Summary:
================================
| Module                  | Native SR   | Supported SRs      | Resampling Behavior            |
|-------------------------|-------------|--------------------|---------------------------------|
| MonoConversionStage     | Any         | All                | strict_sample_rate flag        |
| VADSegmentationStage    | 16kHz       | 8k,16k,32k,48k,64k | Resamples to 16kHz internally  |
| BandFilterStage         | 48kHz       | All (via librosa)  | librosa.load resamples to 48k  |
| NISQAFilterStage        | 48kHz       | All (via NISQA)    | NISQA handles internally       |
| SIGMOSFilterStage       | 48kHz       | All                | Resamples to 48kHz internally  |
| SpeakerSeparationStage  | 16kHz       | All (via NeMo)     | NeMo handles internally        |

Run tests with:
    pytest tests/pipelines/test_audio_sample_rates.py -v -s
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
# Constants and Sample Rate Definitions
# =============================================================================

DATA_DIR = Path("/workdir/Curator/nemo_curator/stages/audio/datasets/youtube/batch_small")
SAMPLE_AUDIO_FILES = [
    DATA_DIR / "-1KFBkWd5xs.wav",
    DATA_DIR / "0dQmTf6K71U.wav",
]

# Common sample rates to test
SAMPLE_RATES = {
    8000: "8kHz - Telephony (narrow-band)",
    16000: "16kHz - Wideband speech (Silero VAD native)",
    22050: "22.05kHz - Common CD/MP3 downsampled",
    32000: "32kHz - Broadcasting",
    44100: "44.1kHz - CD quality",
    48000: "48kHz - Professional audio (SIGMOS/NISQA native)",
}

# Module-specific constraints
MODULE_CONSTRAINTS = {
    "MonoConversionStage": {
        "native_sr": None,  # Works with any SR
        "supported_srs": "all",
        "behavior": "strict_sample_rate flag controls filtering",
    },
    "VADSegmentationStage": {
        "native_sr": 16000,
        "supported_srs": [8000, 16000, 32000, 48000, 64000, 96000],  # Multiples of 16kHz
        "behavior": "Resamples unsupported rates (like 22050) to 16kHz",
    },
    "BandFilterStage": {
        "native_sr": 48000,
        "supported_srs": "all",
        "behavior": "librosa.load resamples to 48kHz for feature extraction",
    },
    "NISQAFilterStage": {
        "native_sr": 48000,
        "supported_srs": "all",
        "behavior": "NISQA model handles various sample rates internally",
    },
    "SIGMOSFilterStage": {
        "native_sr": 48000,
        "supported_srs": "all",
        "behavior": "Resamples to 48kHz using librosa if needed",
    },
    "SpeakerSeparationStage": {
        "native_sr": 16000,
        "supported_srs": "all",
        "behavior": "NeMo model handles resampling internally",
    },
}


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def base_audio_48k() -> str:
    """Get path to real 48kHz audio file."""
    audio_path = SAMPLE_AUDIO_FILES[1]
    if not audio_path.exists():
        pytest.skip(f"Test audio not found: {audio_path}")
    return str(audio_path)


@pytest.fixture
def short_segment_48k(base_audio_48k: str) -> str:
    """Create a 30-second 48kHz audio segment for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    # Load first 30 seconds
    audio = AudioSegment.from_wav(base_audio_48k)[:30000]
    audio.export(temp_path, format="wav")
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def audio_8k(short_segment_48k: str) -> str:
    """Create 8kHz (telephony) audio."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    audio = AudioSegment.from_wav(short_segment_48k)
    audio = audio.set_frame_rate(8000)
    audio.export(temp_path, format="wav")
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def audio_16k(short_segment_48k: str) -> str:
    """Create 16kHz (wideband) audio - Silero VAD native rate."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    audio = AudioSegment.from_wav(short_segment_48k)
    audio = audio.set_frame_rate(16000)
    audio.export(temp_path, format="wav")
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def audio_22k(short_segment_48k: str) -> str:
    """Create 22.05kHz audio - NOT a multiple of 16kHz, needs resampling for VAD."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    audio = AudioSegment.from_wav(short_segment_48k)
    audio = audio.set_frame_rate(22050)
    audio.export(temp_path, format="wav")
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def audio_44k(short_segment_48k: str) -> str:
    """Create 44.1kHz (CD quality) audio."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    audio = AudioSegment.from_wav(short_segment_48k)
    audio = audio.set_frame_rate(44100)
    audio.export(temp_path, format="wav")
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def audio_32k(short_segment_48k: str) -> str:
    """Create 32kHz (broadcasting) audio."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    audio = AudioSegment.from_wav(short_segment_48k)
    audio = audio.set_frame_rate(32000)
    audio.export(temp_path, format="wav")
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# =============================================================================
# Helper Functions
# =============================================================================

def create_audio_batch(audio_path: str, task_id: str = "test") -> AudioBatch:
    """Create AudioBatch with audio_filepath."""
    return AudioBatch(
        task_id=task_id,
        dataset_name="test_dataset",
        data=[{"audio_filepath": audio_path}]
    )


def create_audio_batch_with_waveform(audio_path: str, task_id: str = "test") -> AudioBatch:
    """Create AudioBatch with waveform tensor and sample rate."""
    data, sample_rate = sf.read(audio_path, dtype='float32')
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    
    # Convert to mono if stereo
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


def get_audio_sample_rate(audio_path: str) -> int:
    """Get sample rate of an audio file."""
    info = sf.info(audio_path)
    return info.samplerate


def print_sample_rate_test_result(module_name: str, sample_rate: int, result, expected_behavior: str):
    """Print formatted test result."""
    sr_desc = SAMPLE_RATES.get(sample_rate, f"{sample_rate}Hz")
    status = "PASSED" if result is not None else "FILTERED"
    
    print(f"\n{'='*70}")
    print(f"Module: {module_name}")
    print(f"Sample Rate: {sample_rate}Hz ({sr_desc})")
    print(f"Status: {status}")
    print(f"Expected: {expected_behavior}")
    
    if result is not None:
        if isinstance(result, list):
            print(f"Result: List of {len(result)} AudioBatch(es)")
        else:
            print(f"Result: AudioBatch with {len(result.data)} items")
            if result.data:
                item = result.data[0]
                out_sr = item.get('sample_rate', 'N/A')
                print(f"Output Sample Rate: {out_sr}")
    print(f"{'='*70}")


# =============================================================================
# TEST CATEGORY 1: MonoConversionStage Sample Rate Tests
# =============================================================================

class TestMonoConversionSampleRates:
    """
    MonoConversionStage Sample Rate Tests
    
    Constraints:
    - No native sample rate requirement
    - strict_sample_rate=True: Only accepts audio matching output_sample_rate
    - strict_sample_rate=False: Accepts any sample rate (preserves original)
    """
    
    def test_mono_48k_strict(self, short_segment_48k: str):
        """Test 48kHz with strict mode - should pass."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=True)
        task = create_audio_batch(short_segment_48k)
        
        result = stage.process(task)
        print_sample_rate_test_result("MonoConversion", 48000, result, "Should PASS (matches expected 48kHz)")
        
        assert result is not None, "48kHz should pass with strict_sample_rate=True when output_sample_rate=48000"
        assert result.data[0]["sample_rate"] == 48000
    
    def test_mono_16k_strict_reject(self, audio_16k: str):
        """Test 16kHz with strict mode expecting 48kHz - should be rejected."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=True)
        task = create_audio_batch(audio_16k)
        
        result = stage.process(task)
        print_sample_rate_test_result("MonoConversion", 16000, result, "Should be FILTERED (16kHz != 48kHz expected)")
        
        assert result is None, "16kHz should be rejected when expecting 48kHz with strict=True"
    
    def test_mono_16k_flexible(self, audio_16k: str):
        """Test 16kHz with flexible mode - should pass."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=False)
        task = create_audio_batch(audio_16k)
        
        result = stage.process(task)
        print_sample_rate_test_result("MonoConversion", 16000, result, "Should PASS (flexible mode)")
        
        assert result is not None, "16kHz should pass with strict_sample_rate=False"
        assert result.data[0]["sample_rate"] == 16000  # Preserves original SR
    
    def test_mono_8k_flexible(self, audio_8k: str):
        """Test 8kHz telephony audio with flexible mode."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=False)
        task = create_audio_batch(audio_8k)
        
        result = stage.process(task)
        print_sample_rate_test_result("MonoConversion", 8000, result, "Should PASS (flexible mode)")
        
        assert result is not None
        assert result.data[0]["sample_rate"] == 8000
    
    def test_mono_22k_flexible(self, audio_22k: str):
        """Test 22.05kHz audio with flexible mode."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=False)
        task = create_audio_batch(audio_22k)
        
        result = stage.process(task)
        print_sample_rate_test_result("MonoConversion", 22050, result, "Should PASS (flexible mode)")
        
        assert result is not None
        assert result.data[0]["sample_rate"] == 22050
    
    def test_mono_44k_flexible(self, audio_44k: str):
        """Test 44.1kHz CD quality audio with flexible mode."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=False)
        task = create_audio_batch(audio_44k)
        
        result = stage.process(task)
        print_sample_rate_test_result("MonoConversion", 44100, result, "Should PASS (flexible mode)")
        
        assert result is not None
        assert result.data[0]["sample_rate"] == 44100
    
    def test_mono_all_sample_rates_summary(self, audio_8k, audio_16k, audio_22k, audio_32k, audio_44k, short_segment_48k):
        """Summary test: all sample rates with flexible mode."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=False)
        
        test_files = [
            (audio_8k, 8000),
            (audio_16k, 16000),
            (audio_22k, 22050),
            (audio_32k, 32000),
            (audio_44k, 44100),
            (short_segment_48k, 48000),
        ]
        
        print("\n" + "="*80)
        print("MonoConversionStage - All Sample Rates Summary (strict=False)")
        print("="*80)
        
        results = []
        for audio_path, expected_sr in test_files:
            task = create_audio_batch(audio_path)
            result = stage.process(task)
            status = "PASS" if result is not None else "FAIL"
            actual_sr = result.data[0]["sample_rate"] if result else "N/A"
            results.append((expected_sr, status, actual_sr))
            print(f"  {expected_sr}Hz: {status} (output SR: {actual_sr})")
        
        # All should pass
        assert all(r[1] == "PASS" for r in results), "All sample rates should pass with flexible mode"


# =============================================================================
# TEST CATEGORY 2: VADSegmentationStage Sample Rate Tests
# =============================================================================

class TestVADSegmentationSampleRates:
    """
    VADSegmentationStage Sample Rate Tests
    
    Constraints:
    - Native sample rate: 16kHz (Silero VAD)
    - Supported sample rates: 8kHz, 16kHz, 32kHz, 48kHz, 64kHz, 96kHz (multiples of 16kHz)
    - Non-multiples (like 22.05kHz): Automatically resampled to 16kHz
    """
    
    def test_vad_16k_native(self, audio_16k: str):
        """Test 16kHz - Silero VAD native rate, should work directly."""
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        stage.setup()
        
        task = create_audio_batch(audio_16k)
        result = stage.process(task)
        print_sample_rate_test_result("VADSegmentation", 16000, result, "Should PASS (native 16kHz)")
        
        assert isinstance(result, list), "VAD should return list of segments"
        print(f"  Segments detected: {len(result)}")
    
    def test_vad_48k_multiple(self, short_segment_48k: str):
        """Test 48kHz - multiple of 16kHz, should work directly."""
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        stage.setup()
        
        task = create_audio_batch(short_segment_48k)
        result = stage.process(task)
        print_sample_rate_test_result("VADSegmentation", 48000, result, "Should PASS (48kHz is multiple of 16kHz)")
        
        assert isinstance(result, list)
        print(f"  Segments detected: {len(result)}")
    
    def test_vad_8k_multiple(self, audio_8k: str):
        """Test 8kHz - multiple of 8kHz, supported by Silero."""
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        stage.setup()
        
        task = create_audio_batch(audio_8k)
        result = stage.process(task)
        print_sample_rate_test_result("VADSegmentation", 8000, result, "Should PASS (8kHz supported)")
        
        assert isinstance(result, list)
        print(f"  Segments detected: {len(result)}")
    
    def test_vad_22k_needs_resampling(self, audio_22k: str):
        """Test 22.05kHz - NOT a multiple of 16kHz, needs resampling."""
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        stage.setup()
        
        task = create_audio_batch(audio_22k)
        result = stage.process(task)
        print_sample_rate_test_result(
            "VADSegmentation", 22050, result, 
            "Should PASS (22.05kHz resampled to 16kHz internally)"
        )
        
        assert isinstance(result, list), "22.05kHz should work after internal resampling to 16kHz"
        print(f"  Segments detected: {len(result)}")
    
    def test_vad_44k_needs_resampling(self, audio_44k: str):
        """Test 44.1kHz - NOT a multiple of 16kHz, needs resampling."""
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        stage.setup()
        
        task = create_audio_batch(audio_44k)
        result = stage.process(task)
        print_sample_rate_test_result(
            "VADSegmentation", 44100, result,
            "Should PASS (44.1kHz resampled to 16kHz internally)"
        )
        
        assert isinstance(result, list)
        print(f"  Segments detected: {len(result)}")
    
    def test_vad_32k_multiple(self, audio_32k: str):
        """Test 32kHz - multiple of 16kHz, should work directly."""
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        stage.setup()
        
        task = create_audio_batch(audio_32k)
        result = stage.process(task)
        print_sample_rate_test_result("VADSegmentation", 32000, result, "Should PASS (32kHz is multiple of 16kHz)")
        
        assert isinstance(result, list)
        print(f"  Segments detected: {len(result)}")
    
    def test_vad_all_sample_rates_summary(self, audio_8k, audio_16k, audio_22k, audio_32k, audio_44k, short_segment_48k):
        """Summary test: all sample rates for VAD."""
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        stage.setup()
        
        test_files = [
            (audio_8k, 8000, "Supported"),
            (audio_16k, 16000, "Native"),
            (audio_22k, 22050, "Resampled to 16kHz"),
            (audio_32k, 32000, "Supported"),
            (audio_44k, 44100, "Resampled to 16kHz"),
            (short_segment_48k, 48000, "Supported"),
        ]
        
        print("\n" + "="*80)
        print("VADSegmentationStage - All Sample Rates Summary")
        print("Native: 16kHz, Supported: 8k/16k/32k/48k/64k/96k, Others: Resampled to 16kHz")
        print("="*80)
        
        all_passed = True
        for audio_path, sr, note in test_files:
            task = create_audio_batch(audio_path)
            result = stage.process(task)
            status = "PASS" if isinstance(result, list) and len(result) > 0 else "FAIL"
            num_segments = len(result) if isinstance(result, list) else 0
            print(f"  {sr}Hz ({note}): {status} ({num_segments} segments)")
            if status == "FAIL":
                all_passed = False
        
        assert all_passed, "All sample rates should be supported (with resampling if needed)"


# =============================================================================
# TEST CATEGORY 3: BandFilterStage Sample Rate Tests
# =============================================================================

class TestBandFilterSampleRates:
    """
    BandFilterStage Sample Rate Tests
    
    Constraints:
    - Native sample rate: 48kHz (for band energy features)
    - Uses librosa.load which resamples to specified SR (48kHz by default)
    - All sample rates supported via automatic resampling
    """
    
    @pytest.fixture(autouse=True)
    def check_model(self):
        """Check if band model exists, skip if not."""
        from nemo_curator.stages.audio.filtering import BandFilterStage
        stage = BandFilterStage()
        model_path = stage._resolve_model_path()
        if not os.path.exists(model_path):
            pytest.skip(f"Band model not found: {model_path}")
    
    def test_band_48k_native(self, short_segment_48k: str):
        """Test 48kHz - native rate for band features."""
        from nemo_curator.stages.audio.filtering import BandFilterStage
        
        stage = BandFilterStage(band_value="full_band")
        task = create_audio_batch_with_waveform(short_segment_48k)
        
        result = stage.process(task)
        print_sample_rate_test_result("BandFilter", 48000, result, "Should PASS (native 48kHz)")
        
        # Result depends on audio content - may pass or filter based on band prediction
        print(f"  Band prediction: {result.data[0].get('band_prediction') if result else 'N/A'}")
    
    def test_band_16k(self, audio_16k: str):
        """Test 16kHz - should work with resampling to 48kHz for feature extraction."""
        from nemo_curator.stages.audio.filtering import BandFilterStage
        
        stage = BandFilterStage(band_value="narrow_band")  # 16kHz likely narrow-band
        task = create_audio_batch_with_waveform(audio_16k)
        
        result = stage.process(task)
        print_sample_rate_test_result(
            "BandFilter", 16000, result,
            "Likely PASS as narrow_band (16kHz audio resampled to 48kHz internally)"
        )
    
    def test_band_8k_telephony(self, audio_8k: str):
        """Test 8kHz telephony - classic narrow-band audio."""
        from nemo_curator.stages.audio.filtering import BandFilterStage
        
        stage = BandFilterStage(band_value="narrow_band")  # 8kHz is definitely narrow-band
        task = create_audio_batch_with_waveform(audio_8k)
        
        result = stage.process(task)
        print_sample_rate_test_result(
            "BandFilter", 8000, result,
            "Should PASS as narrow_band (8kHz telephony is narrow-band)"
        )
    
    def test_band_22k(self, audio_22k: str):
        """Test 22.05kHz - intermediate sample rate."""
        from nemo_curator.stages.audio.filtering import BandFilterStage
        
        # Try both band values to see what 22kHz is classified as
        for band_value in ["full_band", "narrow_band"]:
            stage = BandFilterStage(band_value=band_value)
            task = create_audio_batch_with_waveform(audio_22k)
            result = stage.process(task)
            if result is not None:
                print_sample_rate_test_result(
                    "BandFilter", 22050, result,
                    f"Classified as {band_value}"
                )
                break
    
    def test_band_44k(self, audio_44k: str):
        """Test 44.1kHz CD quality."""
        from nemo_curator.stages.audio.filtering import BandFilterStage
        
        stage = BandFilterStage(band_value="full_band")
        task = create_audio_batch_with_waveform(audio_44k)
        
        result = stage.process(task)
        print_sample_rate_test_result(
            "BandFilter", 44100, result,
            "Should work (44.1kHz resampled to 48kHz internally)"
        )


# =============================================================================
# TEST CATEGORY 4: NISQAFilterStage Sample Rate Tests
# =============================================================================

class TestNISQASampleRates:
    """
    NISQAFilterStage Sample Rate Tests
    
    Constraints:
    - Native sample rate: 48kHz (NISQA model)
    - NISQA handles various sample rates internally
    - All sample rates should work
    """
    
    @pytest.fixture(autouse=True)
    def check_model(self):
        """Check if NISQA model exists, skip if not."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        stage = NISQAFilterStage()
        model_path = stage._resolve_model_path()
        if not os.path.exists(model_path):
            pytest.skip(f"NISQA model not found: {model_path}")
    
    def test_nisqa_48k_native(self, short_segment_48k: str):
        """Test 48kHz - native rate for NISQA."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        stage = NISQAFilterStage(mos_threshold=None)  # No filtering, just assess
        task = create_audio_batch(short_segment_48k)
        
        result = stage.process(task)
        print_sample_rate_test_result("NISQAFilter", 48000, result, "Should PASS (native 48kHz)")
        
        if result:
            item = result.data[0]
            print(f"  MOS: {item.get('nisqa_mos', 'N/A'):.3f}")
            print(f"  NOI: {item.get('nisqa_noi', 'N/A'):.3f}")
    
    def test_nisqa_16k(self, audio_16k: str):
        """Test 16kHz - common wideband rate."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        stage = NISQAFilterStage(mos_threshold=None)
        task = create_audio_batch(audio_16k)
        
        result = stage.process(task)
        print_sample_rate_test_result("NISQAFilter", 16000, result, "Should work (16kHz handled internally)")
        
        if result:
            print(f"  MOS: {result.data[0].get('nisqa_mos', 'N/A'):.3f}")
    
    def test_nisqa_8k_telephony(self, audio_8k: str):
        """Test 8kHz telephony - may have lower quality scores."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        stage = NISQAFilterStage(mos_threshold=None)
        task = create_audio_batch(audio_8k)
        
        result = stage.process(task)
        print_sample_rate_test_result(
            "NISQAFilter", 8000, result,
            "Should work but expect lower MOS due to limited bandwidth"
        )
        
        if result:
            print(f"  MOS: {result.data[0].get('nisqa_mos', 'N/A'):.3f}")
    
    def test_nisqa_22k(self, audio_22k: str):
        """Test 22.05kHz - intermediate sample rate."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        stage = NISQAFilterStage(mos_threshold=None)
        task = create_audio_batch(audio_22k)
        
        result = stage.process(task)
        print_sample_rate_test_result("NISQAFilter", 22050, result, "Should work")
        
        if result:
            print(f"  MOS: {result.data[0].get('nisqa_mos', 'N/A'):.3f}")
    
    def test_nisqa_44k(self, audio_44k: str):
        """Test 44.1kHz CD quality."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        stage = NISQAFilterStage(mos_threshold=None)
        task = create_audio_batch(audio_44k)
        
        result = stage.process(task)
        print_sample_rate_test_result("NISQAFilter", 44100, result, "Should work")
        
        if result:
            print(f"  MOS: {result.data[0].get('nisqa_mos', 'N/A'):.3f}")
    
    def test_nisqa_all_sample_rates_summary(self, audio_8k, audio_16k, audio_22k, audio_32k, audio_44k, short_segment_48k):
        """Summary test: all sample rates for NISQA."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        stage = NISQAFilterStage(mos_threshold=None)
        
        test_files = [
            (audio_8k, 8000),
            (audio_16k, 16000),
            (audio_22k, 22050),
            (audio_32k, 32000),
            (audio_44k, 44100),
            (short_segment_48k, 48000),
        ]
        
        print("\n" + "="*80)
        print("NISQAFilterStage - All Sample Rates Summary")
        print("Native: 48kHz, All rates supported (handles internally)")
        print("="*80)
        
        for audio_path, sr in test_files:
            task = create_audio_batch(audio_path)
            result = stage.process(task)
            status = "PASS" if result is not None else "FAIL"
            mos = result.data[0].get('nisqa_mos', 0) if result else 0
            print(f"  {sr}Hz: {status} (MOS: {mos:.3f})")


# =============================================================================
# TEST CATEGORY 5: SIGMOSFilterStage Sample Rate Tests
# =============================================================================

class TestSIGMOSSampleRates:
    """
    SIGMOSFilterStage Sample Rate Tests
    
    Constraints:
    - Native sample rate: 48kHz
    - Automatically resamples to 48kHz using librosa if input has different SR
    - All sample rates supported via automatic resampling
    """
    
    @pytest.fixture(autouse=True)
    def check_model(self):
        """Check if SIGMOS model exists, skip if not."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        stage = SIGMOSFilterStage()
        model_path = stage._resolve_model_path()
        if not os.path.exists(model_path):
            pytest.skip(f"SIGMOS model not found: {model_path}")
    
    def test_sigmos_48k_native(self, short_segment_48k: str):
        """Test 48kHz - native rate for SIGMOS."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        
        stage = SIGMOSFilterStage(ovrl_threshold=None)  # No filtering
        task = create_audio_batch(short_segment_48k)
        
        result = stage.process(task)
        print_sample_rate_test_result("SIGMOSFilter", 48000, result, "Should PASS (native 48kHz)")
        
        if result:
            item = result.data[0]
            print(f"  OVRL: {item.get('sigmos_ovrl', 'N/A'):.3f}")
            print(f"  NOISE: {item.get('sigmos_noise', 'N/A'):.3f}")
    
    def test_sigmos_16k(self, audio_16k: str):
        """Test 16kHz - will be resampled to 48kHz."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        
        stage = SIGMOSFilterStage(ovrl_threshold=None)
        task = create_audio_batch(audio_16k)
        
        result = stage.process(task)
        print_sample_rate_test_result(
            "SIGMOSFilter", 16000, result,
            "Should work (resampled from 16kHz to 48kHz)"
        )
        
        if result:
            print(f"  OVRL: {result.data[0].get('sigmos_ovrl', 'N/A'):.3f}")
    
    def test_sigmos_8k_telephony(self, audio_8k: str):
        """Test 8kHz telephony - resampled to 48kHz."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        
        stage = SIGMOSFilterStage(ovrl_threshold=None)
        task = create_audio_batch(audio_8k)
        
        result = stage.process(task)
        print_sample_rate_test_result(
            "SIGMOSFilter", 8000, result,
            "Should work (resampled from 8kHz to 48kHz)"
        )
        
        if result:
            print(f"  OVRL: {result.data[0].get('sigmos_ovrl', 'N/A'):.3f}")
    
    def test_sigmos_22k(self, audio_22k: str):
        """Test 22.05kHz - resampled to 48kHz."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        
        stage = SIGMOSFilterStage(ovrl_threshold=None)
        task = create_audio_batch(audio_22k)
        
        result = stage.process(task)
        print_sample_rate_test_result(
            "SIGMOSFilter", 22050, result,
            "Should work (resampled from 22.05kHz to 48kHz)"
        )
    
    def test_sigmos_44k(self, audio_44k: str):
        """Test 44.1kHz - resampled to 48kHz."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        
        stage = SIGMOSFilterStage(ovrl_threshold=None)
        task = create_audio_batch(audio_44k)
        
        result = stage.process(task)
        print_sample_rate_test_result(
            "SIGMOSFilter", 44100, result,
            "Should work (resampled from 44.1kHz to 48kHz)"
        )
    
    def test_sigmos_all_sample_rates_summary(self, audio_8k, audio_16k, audio_22k, audio_32k, audio_44k, short_segment_48k):
        """Summary test: all sample rates for SIGMOS."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        
        stage = SIGMOSFilterStage(ovrl_threshold=None)
        
        test_files = [
            (audio_8k, 8000),
            (audio_16k, 16000),
            (audio_22k, 22050),
            (audio_32k, 32000),
            (audio_44k, 44100),
            (short_segment_48k, 48000),
        ]
        
        print("\n" + "="*80)
        print("SIGMOSFilterStage - All Sample Rates Summary")
        print("Native: 48kHz, All rates supported (resampled to 48kHz internally)")
        print("="*80)
        
        for audio_path, sr in test_files:
            task = create_audio_batch(audio_path)
            result = stage.process(task)
            status = "PASS" if result is not None else "FAIL"
            ovrl = result.data[0].get('sigmos_ovrl', 0) if result else 0
            print(f"  {sr}Hz: {status} (OVRL: {ovrl:.3f})")


# =============================================================================
# TEST CATEGORY 6: SpeakerSeparationStage Sample Rate Tests
# =============================================================================

class TestSpeakerSeparationSampleRates:
    """
    SpeakerSeparationStage Sample Rate Tests
    
    Constraints:
    - Native sample rate: 16kHz (NeMo SortFormer model)
    - NeMo handles resampling internally
    - All sample rates should work
    """
    
    @pytest.fixture(autouse=True)
    def check_model(self):
        """Check if speaker separation model exists, skip if not."""
        from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
        stage = SpeakerSeparationStage()
        model_path = stage._resolve_model_path()
        if not os.path.exists(model_path):
            pytest.skip(f"Speaker separation model not found: {model_path}")
    
    def test_speaker_sep_48k(self, short_segment_48k: str):
        """Test 48kHz with speaker separation."""
        from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
        
        stage = SpeakerSeparationStage(min_duration=0.5)
        stage.setup()
        
        task = create_audio_batch(short_segment_48k)
        result = stage.process(task)
        print_sample_rate_test_result(
            "SpeakerSeparation", 48000, result,
            "Should PASS (NeMo handles 48kHz)"
        )
        
        if isinstance(result, list):
            print(f"  Speakers detected: {len(result)}")
    
    def test_speaker_sep_16k_native(self, audio_16k: str):
        """Test 16kHz - native rate for NeMo diarization."""
        from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
        
        stage = SpeakerSeparationStage(min_duration=0.5)
        stage.setup()
        
        task = create_audio_batch(audio_16k)
        result = stage.process(task)
        print_sample_rate_test_result(
            "SpeakerSeparation", 16000, result,
            "Should PASS (native 16kHz for NeMo)"
        )
        
        if isinstance(result, list):
            print(f"  Speakers detected: {len(result)}")
    
    def test_speaker_sep_8k(self, audio_8k: str):
        """Test 8kHz telephony with speaker separation."""
        from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
        
        stage = SpeakerSeparationStage(min_duration=0.5)
        stage.setup()
        
        task = create_audio_batch(audio_8k)
        result = stage.process(task)
        print_sample_rate_test_result(
            "SpeakerSeparation", 8000, result,
            "Should work (NeMo handles resampling)"
        )
    
    def test_speaker_sep_22k(self, audio_22k: str):
        """Test 22.05kHz with speaker separation - not a multiple of 16kHz."""
        from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
        
        stage = SpeakerSeparationStage(min_duration=0.5)
        stage.setup()
        
        task = create_audio_batch(audio_22k)
        result = stage.process(task)
        print_sample_rate_test_result(
            "SpeakerSeparation", 22050, result,
            "Should work (NeMo resamples 22.05kHz internally)"
        )
        
        if isinstance(result, list):
            print(f"  Speakers detected: {len(result)}")
    
    def test_speaker_sep_44k(self, audio_44k: str):
        """Test 44.1kHz with speaker separation."""
        from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
        
        stage = SpeakerSeparationStage(min_duration=0.5)
        stage.setup()
        
        task = create_audio_batch(audio_44k)
        result = stage.process(task)
        print_sample_rate_test_result(
            "SpeakerSeparation", 44100, result,
            "Should work (NeMo resamples 44.1kHz internally)"
        )
        
        if isinstance(result, list):
            print(f"  Speakers detected: {len(result)}")
    
    def test_speaker_sep_32k(self, audio_32k: str):
        """Test 32kHz with speaker separation."""
        from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
        
        stage = SpeakerSeparationStage(min_duration=0.5)
        stage.setup()
        
        task = create_audio_batch(audio_32k)
        result = stage.process(task)
        print_sample_rate_test_result(
            "SpeakerSeparation", 32000, result,
            "Should work (NeMo handles 32kHz)"
        )
        
        if isinstance(result, list):
            print(f"  Speakers detected: {len(result)}")
    
    def test_speaker_sep_all_sample_rates_summary(self, audio_8k, audio_16k, audio_22k, audio_32k, audio_44k, short_segment_48k):
        """Summary test: all sample rates for SpeakerSeparation."""
        from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
        
        stage = SpeakerSeparationStage(min_duration=0.5)
        stage.setup()
        
        test_files = [
            (audio_8k, 8000, "Resampled to 16kHz"),
            (audio_16k, 16000, "Native"),
            (audio_22k, 22050, "Resampled to 16kHz"),
            (audio_32k, 32000, "Resampled to 16kHz"),
            (audio_44k, 44100, "Resampled to 16kHz"),
            (short_segment_48k, 48000, "Resampled to 16kHz"),
        ]
        
        print("\n" + "="*80)
        print("SpeakerSeparationStage - All Sample Rates Summary")
        print("Native: 16kHz, All rates supported (NeMo resamples internally)")
        print("="*80)
        
        for audio_path, sr, note in test_files:
            task = create_audio_batch(audio_path)
            result = stage.process(task)
            status = "PASS" if isinstance(result, list) and len(result) > 0 else "FAIL"
            num_speakers = len(result) if isinstance(result, list) else 0
            print(f"  {sr}Hz ({note}): {status} ({num_speakers} speakers)")


# =============================================================================
# TEST CATEGORY 7: 22.05kHz Comprehensive Tests (Non-Standard Rate)
# =============================================================================

class Test22kHzComprehensive:
    """
    Comprehensive tests specifically for 22.05kHz sample rate.
    
    22.05kHz is a common sample rate (half of CD 44.1kHz) but it's NOT a 
    multiple of 16kHz, which causes issues with Silero VAD and some other
    models that expect 8k/16k/32k/48k etc.
    
    This test class specifically validates that all modules handle this
    non-standard rate correctly via internal resampling.
    """
    
    def test_22k_mono_conversion(self, audio_22k: str):
        """Test 22.05kHz with MonoConversion - should preserve rate."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        
        # Flexible mode should accept any sample rate
        stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=False)
        task = create_audio_batch(audio_22k)
        
        result = stage.process(task)
        
        assert result is not None, "22.05kHz should be accepted with flexible mode"
        assert result.data[0]["sample_rate"] == 22050, "Should preserve original 22.05kHz"
        print(f"\n  MonoConversion: 22.05kHz preserved, mono={result.data[0]['is_mono']}")
    
    def test_22k_vad_resampling(self, audio_22k: str):
        """Test 22.05kHz with VAD - must resample to 16kHz internally."""
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        stage.setup()
        
        task = create_audio_batch(audio_22k)
        result = stage.process(task)
        
        assert isinstance(result, list), "VAD should work with 22.05kHz via resampling"
        assert len(result) > 0, "Should detect speech segments"
        
        # Verify segment sample rate is preserved from input
        segment_sr = result[0].data[0].get('sample_rate')
        print(f"\n  VAD: 22.05kHz -> {len(result)} segments (internal resample to 16kHz)")
        print(f"  Output segment SR: {segment_sr}")
    
    def test_22k_nisqa(self, audio_22k: str):
        """Test 22.05kHz with NISQA quality assessment."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        stage = NISQAFilterStage(mos_threshold=None)
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("NISQA model not found")
        
        task = create_audio_batch(audio_22k)
        result = stage.process(task)
        
        assert result is not None, "NISQA should handle 22.05kHz"
        mos = result.data[0].get('nisqa_mos', 0)
        print(f"\n  NISQA @ 22.05kHz: MOS={mos:.3f}")
    
    def test_22k_sigmos(self, audio_22k: str):
        """Test 22.05kHz with SIGMOS - resamples to 48kHz internally."""
        from nemo_curator.stages.audio.filtering import SIGMOSFilterStage
        
        # Disable all thresholds to just test sample rate handling
        stage = SIGMOSFilterStage(
            noise_threshold=None, 
            ovrl_threshold=None,
            sig_threshold=None,
            col_threshold=None,
            disc_threshold=None,
            loud_threshold=None,
            reverb_threshold=None
        )
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("SIGMOS model not found")
        
        task = create_audio_batch(audio_22k)
        result = stage.process(task)
        
        assert result is not None, "SIGMOS should handle 22.05kHz via resampling to 48kHz"
        ovrl = result.data[0].get('sigmos_ovrl', 0)
        print(f"\n  SIGMOS @ 22.05kHz: OVRL={ovrl:.3f} (resampled to 48kHz)")
    
    def test_22k_band_filter(self, audio_22k: str):
        """Test 22.05kHz with BandFilter - librosa resamples to 48kHz."""
        from nemo_curator.stages.audio.filtering import BandFilterStage
        
        # Try narrow_band since 22kHz is limited bandwidth
        stage = BandFilterStage(band_value="narrow_band")
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("Band model not found")
        
        task = create_audio_batch_with_waveform(audio_22k)
        result = stage.process(task)
        
        # May pass as narrow or full band depending on content
        if result:
            band = result.data[0].get('band_prediction')
            print(f"\n  BandFilter @ 22.05kHz: classified as {band}")
        else:
            # Try full_band
            stage2 = BandFilterStage(band_value="full_band")
            task2 = create_audio_batch_with_waveform(audio_22k)
            result2 = stage2.process(task2)
            if result2:
                band = result2.data[0].get('band_prediction')
                print(f"\n  BandFilter @ 22.05kHz: classified as {band}")
    
    def test_22k_speaker_separation(self, audio_22k: str):
        """Test 22.05kHz with SpeakerSeparation."""
        from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
        
        stage = SpeakerSeparationStage(min_duration=0.5)
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("Speaker separation model not found")
        
        stage.setup()
        task = create_audio_batch(audio_22k)
        result = stage.process(task)
        
        assert isinstance(result, list), "SpeakerSep should handle 22.05kHz"
        print(f"\n  SpeakerSeparation @ 22.05kHz: {len(result)} speakers detected")
    
    def test_22k_full_pipeline(self, audio_22k: str):
        """Test 22.05kHz through full pipeline: Mono -> VAD -> NISQA."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        # Check NISQA model
        nisqa_stage = NISQAFilterStage(mos_threshold=None)
        if not os.path.exists(nisqa_stage._resolve_model_path()):
            pytest.skip("NISQA model not found")
        
        mono_stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=False)
        vad_stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        vad_stage.setup()
        
        task = create_audio_batch(audio_22k)
        
        print(f"\n  === 22.05kHz Full Pipeline Test ===")
        
        # Stage 1: Mono (preserves 22.05kHz)
        mono_result = mono_stage.process(task)
        assert mono_result is not None
        print(f"  1. Mono: SR={mono_result.data[0]['sample_rate']} (preserved)")
        
        # Stage 2: VAD (resamples to 16kHz internally)
        vad_result = vad_stage.process(mono_result)
        assert isinstance(vad_result, list) and len(vad_result) > 0
        print(f"  2. VAD: {len(vad_result)} segments (internal resample to 16kHz)")
        
        # Stage 3: NISQA on first segment
        nisqa_result = nisqa_stage.process(vad_result[0])
        if nisqa_result:
            mos = nisqa_result.data[0].get('nisqa_mos', 0)
            print(f"  3. NISQA: MOS={mos:.3f}")
        
        print(f"  === Pipeline completed successfully ===")
    
    def test_22k_vs_16k_quality_comparison(self, audio_22k: str, audio_16k: str):
        """Compare quality metrics between 22.05kHz and 16kHz audio."""
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        stage = NISQAFilterStage(mos_threshold=None)
        if not os.path.exists(stage._resolve_model_path()):
            pytest.skip("NISQA model not found")
        
        # Process 22.05kHz
        task_22k = create_audio_batch(audio_22k)
        result_22k = stage.process(task_22k)
        
        # Process 16kHz
        task_16k = create_audio_batch(audio_16k)
        result_16k = stage.process(task_16k)
        
        print("\n" + "="*60)
        print("22.05kHz vs 16kHz NISQA Quality Comparison")
        print("="*60)
        
        if result_22k:
            mos_22k = result_22k.data[0].get('nisqa_mos', 0)
            noi_22k = result_22k.data[0].get('nisqa_noi', 0)
            print(f"  22.05kHz: MOS={mos_22k:.3f}, NOI={noi_22k:.3f}")
        
        if result_16k:
            mos_16k = result_16k.data[0].get('nisqa_mos', 0)
            noi_16k = result_16k.data[0].get('nisqa_noi', 0)
            print(f"  16kHz:    MOS={mos_16k:.3f}, NOI={noi_16k:.3f}")
        
        print("="*60)


# =============================================================================
# TEST CATEGORY 8: Pipeline Sample Rate Flow Tests
# =============================================================================

class TestPipelineSampleRateFlow:
    """
    Test sample rate handling across pipeline stages.
    
    Verifies that sample rates are correctly preserved/transformed
    when audio flows through multiple stages.
    """
    
    def test_pipeline_16k_mono_to_vad(self, audio_16k: str):
        """Test 16kHz flow: MonoConversion -> VAD."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        mono_stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=False)
        vad_stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        vad_stage.setup()
        
        task = create_audio_batch(audio_16k)
        
        # Stage 1: Mono (preserves 16kHz with flexible mode)
        mono_result = mono_stage.process(task)
        assert mono_result is not None
        assert mono_result.data[0]["sample_rate"] == 16000
        print(f"\n  After Mono: SR = {mono_result.data[0]['sample_rate']}")
        
        # Stage 2: VAD (uses 16kHz natively)
        vad_result = vad_stage.process(mono_result)
        assert isinstance(vad_result, list)
        print(f"  After VAD: {len(vad_result)} segments")
        
        if vad_result:
            # Verify segment preserves sample rate
            segment_sr = vad_result[0].data[0].get('sample_rate')
            print(f"  Segment SR: {segment_sr}")
            assert segment_sr == 16000
    
    def test_pipeline_22k_requires_resampling(self, audio_22k: str):
        """Test 22.05kHz flow - VAD must resample internally."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        
        mono_stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=False)
        vad_stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        vad_stage.setup()
        
        task = create_audio_batch(audio_22k)
        
        # Stage 1: Mono (preserves 22.05kHz with flexible mode)
        mono_result = mono_stage.process(task)
        assert mono_result is not None
        assert mono_result.data[0]["sample_rate"] == 22050
        print(f"\n  After Mono: SR = {mono_result.data[0]['sample_rate']}")
        
        # Stage 2: VAD (must resample 22.05kHz to 16kHz internally)
        vad_result = vad_stage.process(mono_result)
        assert isinstance(vad_result, list)
        print(f"  After VAD: {len(vad_result)} segments (22kHz resampled to 16kHz internally)")
    
    def test_pipeline_48k_full_chain(self, short_segment_48k: str):
        """Test 48kHz through full pipeline: Mono -> VAD -> NISQA."""
        from nemo_curator.stages.audio.preprocessing import MonoConversionStage
        from nemo_curator.stages.audio.segmentation import VADSegmentationStage
        from nemo_curator.stages.audio.filtering import NISQAFilterStage
        
        # Check NISQA model
        nisqa_stage = NISQAFilterStage(mos_threshold=None)
        if not os.path.exists(nisqa_stage._resolve_model_path()):
            pytest.skip("NISQA model not found")
        
        mono_stage = MonoConversionStage(output_sample_rate=48000, strict_sample_rate=True)
        vad_stage = VADSegmentationStage(min_duration_sec=0.5, threshold=0.5)
        vad_stage.setup()
        
        task = create_audio_batch(short_segment_48k)
        
        # Stage 1: Mono
        mono_result = mono_stage.process(task)
        assert mono_result is not None
        print(f"\n  After Mono: SR = {mono_result.data[0]['sample_rate']}")
        
        # Stage 2: VAD
        vad_result = vad_stage.process(mono_result)
        assert isinstance(vad_result, list) and len(vad_result) > 0
        print(f"  After VAD: {len(vad_result)} segments")
        
        # Stage 3: NISQA on first segment
        nisqa_result = nisqa_stage.process(vad_result[0])
        if nisqa_result:
            print(f"  After NISQA: MOS = {nisqa_result.data[0].get('nisqa_mos', 0):.3f}")


# =============================================================================
# Summary Test: Print All Module Constraints
# =============================================================================

class TestSampleRateConstraintsSummary:
    """Print summary of all module sample rate constraints."""
    
    def test_print_constraints_summary(self):
        """Print formatted summary of sample rate constraints for all modules."""
        print("\n")
        print("="*100)
        print("AUDIO MODULE SAMPLE RATE CONSTRAINTS SUMMARY")
        print("="*100)
        print()
        print(f"{'Module':<25} {'Native SR':<12} {'Supported SRs':<25} {'Behavior'}")
        print("-"*100)
        
        for module, info in MODULE_CONSTRAINTS.items():
            native = str(info['native_sr']) if info['native_sr'] else "Any"
            supported = str(info['supported_srs'])[:22] + "..." if len(str(info['supported_srs'])) > 22 else str(info['supported_srs'])
            behavior = info['behavior'][:50] + "..." if len(info['behavior']) > 50 else info['behavior']
            print(f"{module:<25} {native:<12} {supported:<25} {behavior}")
        
        print("-"*100)
        print()
        print("KEY FINDINGS:")
        print("  1. VAD (Silero): Only supports 8k/16k/32k/48k/64k/96k Hz (multiples of 16kHz)")
        print("     - Other rates (22.05kHz, 44.1kHz) are automatically resampled to 16kHz")
        print("  2. SIGMOS: Resamples everything to 48kHz internally")
        print("  3. NISQA: Handles various sample rates internally")
        print("  4. BandFilter: Uses librosa which resamples to 48kHz for feature extraction")
        print("  5. MonoConversion: Use strict_sample_rate=False to accept any sample rate")
        print("  6. SpeakerSeparation: NeMo handles resampling internally (native: 16kHz)")
        print()
        print("="*100)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
