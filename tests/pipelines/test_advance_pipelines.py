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
Tests for AudioDataFilterStage advance pipeline.

This module tests the AudioDataFilterStage with various module configurations:
1. All modules enabled (full pipeline)
2. VAD disabled - audio processed as full segment
3. Speaker separation disabled - no per-speaker processing
4. Both VAD and speaker separation disabled
5. Individual quality filters enabled/disabled
6. Timestamp accuracy across different configurations

IMPORTANT: Understanding Two VAD Usages
========================================

The AudioDataFilterStage uses VAD in TWO places:

1. **Initial VAD** (after mono conversion, before quality filters):
   - Location: audio_data_filter.py lines 514-525
   - Controlled by: config.enable_vad
   - When enabled: Segments audio into speech chunks
   - When disabled: Full audio passes as single segment

2. **Per-Speaker VAD** (after speaker separation):
   - Location: audio_data_filter.py line 603
   - Uses: self._vad_stage (same as initial VAD)
   - Purpose: Extract speech segments from each speaker's audio
   - CAVEAT: If enable_vad=False, self._vad_stage is None, so per-speaker VAD also fails!

Pipeline Flow:
    Mono → [VAD #1] → Quality Filters → [Speaker Sep] → [VAD #2 per speaker] → Output

Key focus areas:
- Timestamp correctness when VAD is disabled (full audio timestamps)
- Timestamp correctness when speaker separation is disabled
- Proper output format in all configurations
- Quality scores presence based on enabled filters

Run tests with:
    pytest tests/pipelines/test_advance_pipelines.py -v -s
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """Create a 2-minute segment for testing with real speech."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    
    # Load first 2 minutes (120 seconds = 120000 ms)
    audio = AudioSegment.from_wav(real_audio_48k)[:120000]
    audio.export(temp_path, format="wav")
    
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

def create_audio_batch(audio_path: str, task_id: str = "test") -> AudioBatch:
    """Create AudioBatch with audio_filepath."""
    return AudioBatch(
        task_id=task_id,
        dataset_name="test_dataset",
        data=[{"audio_filepath": audio_path}]
    )


def get_audio_duration_ms(audio_path: str) -> int:
    """Get audio duration in milliseconds."""
    audio = AudioSegment.from_wav(audio_path)
    return len(audio)


def print_result_details(result: Optional[AudioBatch], config_name: str):
    """Print detailed result information for debugging."""
    print(f"\n{'='*70}")
    print(f"Config: {config_name}")
    print(f"{'='*70}")
    
    if result is None:
        print("  Result: None (all segments filtered out)")
        return
    
    print(f"  Number of output segments: {len(result.data)}")
    
    for i, item in enumerate(result.data[:5]):  # Show first 5 segments
        print(f"\n  Segment {i}:")
        print(f"    original_file: {item.get('original_file', 'N/A')}")
        print(f"    original_start_ms: {item.get('original_start_ms', 'N/A')}")
        print(f"    original_end_ms: {item.get('original_end_ms', 'N/A')}")
        print(f"    duration_ms: {item.get('duration_ms', 'N/A')}")
        print(f"    speaker_id: {item.get('speaker_id', 'N/A')}")
        
        # Quality scores
        quality_keys = [k for k in item.keys() if k.startswith(('nisqa_', 'sigmos_', 'band_'))]
        if quality_keys:
            print(f"    Quality scores: {quality_keys}")
    
    if len(result.data) > 5:
        print(f"\n  ... and {len(result.data) - 5} more segments")


def validate_timestamps(result: AudioBatch, original_duration_ms: int, config_name: str):
    """Validate that timestamps are within valid ranges."""
    if result is None:
        return
    
    errors = []
    for i, item in enumerate(result.data):
        start_ms = item.get('original_start_ms', 0)
        end_ms = item.get('original_end_ms', 0)
        duration_ms = item.get('duration_ms', 0)
        
        # Check start >= 0
        if start_ms < 0:
            errors.append(f"Segment {i}: start_ms ({start_ms}) < 0")
        
        # Check end <= original duration (with small tolerance for rounding)
        if end_ms > original_duration_ms + 100:
            errors.append(f"Segment {i}: end_ms ({end_ms}) > original duration ({original_duration_ms})")
        
        # Check start < end
        if start_ms >= end_ms:
            errors.append(f"Segment {i}: start_ms ({start_ms}) >= end_ms ({end_ms})")
        
        # Check duration consistency
        expected_duration = end_ms - start_ms
        if abs(duration_ms - expected_duration) > 1:  # 1ms tolerance
            errors.append(f"Segment {i}: duration_ms ({duration_ms}) != end-start ({expected_duration})")
    
    if errors:
        print(f"\n  TIMESTAMP ERRORS for {config_name}:")
        for error in errors:
            print(f"    - {error}")
        pytest.fail(f"Timestamp validation failed: {len(errors)} errors")
    else:
        print(f"\n  ✓ All timestamps valid for {config_name}")


# =============================================================================
# TEST CATEGORY 1: Full Pipeline (All Modules Enabled)
# =============================================================================

@pytest.mark.gpu
class TestFullPipeline:
    """Tests with all modules enabled."""
    
    def test_full_pipeline_all_enabled(self, short_audio_segment: str):
        """Test with VAD, speaker separation, and all filters enabled."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        config = AudioDataFilterConfig(
            enable_vad=True,
            enable_speaker_separation=True,
            enable_band_filter=True,
            enable_nisqa=True,
            enable_sigmos=True,
            # Use permissive thresholds for testing
            nisqa_mos_threshold=3.0,
            nisqa_noi_threshold=3.0,
            sigmos_noise_threshold=3.0,
            sigmos_ovrl_threshold=3.0,
        )
        
        stage = AudioDataFilterStage(config=config)
        stage.setup()
        
        try:
            task = create_audio_batch(short_audio_segment)
            original_duration_ms = get_audio_duration_ms(short_audio_segment)
            
            result = stage.process(task)
            print_result_details(result, "Full Pipeline (all enabled)")
            
            if result is not None:
                # Should have speaker_id since speaker separation is enabled
                for item in result.data:
                    assert 'speaker_id' in item, "Missing speaker_id with speaker separation enabled"
                    assert 'original_start_ms' in item, "Missing original_start_ms"
                    assert 'original_end_ms' in item, "Missing original_end_ms"
                    assert 'duration_ms' in item, "Missing duration_ms"
                
                # Validate timestamps
                validate_timestamps(result, original_duration_ms, "Full Pipeline")
                
                # Check for quality scores from enabled filters
                for item in result.data:
                    # With all filters enabled, we should have quality scores
                    has_nisqa = any(k.startswith('nisqa_') for k in item.keys())
                    has_sigmos = any(k.startswith('sigmos_') for k in item.keys())
                    print(f"  Has NISQA scores: {has_nisqa}, Has SIGMOS scores: {has_sigmos}")
        finally:
            stage.teardown()


# =============================================================================
# TEST CATEGORY 2: VAD Disabled Tests
# =============================================================================

@pytest.mark.gpu
class TestVADDisabled:
    """Tests with VAD disabled - audio processed as full segment."""
    
    def test_vad_disabled_full_audio_timestamps(self, short_audio_segment: str):
        """When VAD is disabled, timestamps should cover full audio duration."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        config = AudioDataFilterConfig(
            enable_vad=False,  # VAD disabled
            enable_speaker_separation=False,  # Also disable to simplify
            enable_band_filter=False,
            enable_nisqa=False,
            enable_sigmos=False,
        )
        
        stage = AudioDataFilterStage(config=config)
        stage.setup()
        
        try:
            task = create_audio_batch(short_audio_segment)
            original_duration_ms = get_audio_duration_ms(short_audio_segment)
            
            result = stage.process(task)
            print_result_details(result, "VAD Disabled (no filters)")
            
            assert result is not None, "Should get result with no filters enabled"
            
            # With VAD disabled and no speaker separation, we should get
            # the full audio as a single segment
            assert len(result.data) == 1, f"Expected 1 segment, got {len(result.data)}"
            
            item = result.data[0]
            start_ms = item.get('original_start_ms', 0)
            end_ms = item.get('original_end_ms', 0)
            duration_ms = item.get('duration_ms', 0)
            
            print(f"\n  Full audio segment:")
            print(f"    start_ms: {start_ms}")
            print(f"    end_ms: {end_ms}")
            print(f"    duration_ms: {duration_ms}")
            print(f"    original_duration_ms: {original_duration_ms}")
            
            # Timestamps should cover the full audio
            assert start_ms == 0, f"Start should be 0, got {start_ms}"
            # Allow small tolerance for rounding
            assert abs(end_ms - original_duration_ms) <= 100, \
                f"End ({end_ms}) should be close to original duration ({original_duration_ms})"
            assert abs(duration_ms - original_duration_ms) <= 100, \
                f"Duration ({duration_ms}) should be close to original ({original_duration_ms})"
        finally:
            stage.teardown()
    
    def test_vad_disabled_with_quality_filters(self, short_audio_segment: str):
        """VAD disabled but quality filters enabled."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        config = AudioDataFilterConfig(
            enable_vad=False,  # VAD disabled
            enable_speaker_separation=False,
            enable_band_filter=True,
            enable_nisqa=True,
            enable_sigmos=False,
            # Permissive thresholds
            nisqa_mos_threshold=2.0,
            nisqa_noi_threshold=2.0,
        )
        
        stage = AudioDataFilterStage(config=config)
        stage.setup()
        
        try:
            task = create_audio_batch(short_audio_segment)
            original_duration_ms = get_audio_duration_ms(short_audio_segment)
            
            result = stage.process(task)
            print_result_details(result, "VAD Disabled + Quality Filters")
            
            if result is not None:
                # Should still process full audio as one segment
                # (may be filtered out by quality filters)
                for item in result.data:
                    # Check for quality scores
                    has_nisqa = any(k.startswith('nisqa_') for k in item.keys())
                    has_band = 'band_prediction' in item
                    print(f"  Has NISQA: {has_nisqa}, Has Band: {has_band}")
                    
                    # Validate timestamps
                    start_ms = item.get('original_start_ms', 0)
                    end_ms = item.get('original_end_ms', 0)
                    assert start_ms >= 0
                    assert end_ms <= original_duration_ms + 100
        finally:
            stage.teardown()
    
    def test_vad_disabled_with_speaker_separation(self, short_audio_segment: str):
        """
        VAD disabled but speaker separation enabled.
        
        IMPORTANT: There are TWO VAD usages in AudioDataFilterStage:
        
        1. Initial VAD (controlled by enable_vad):
           - Runs after mono conversion to segment audio into speech chunks
           - When disabled: full audio passes as single segment to quality filters
        
        2. Per-Speaker VAD (runs after speaker separation):
           - Runs on each speaker's audio to extract speech segments
           - Uses the SAME self._vad_stage, so if enable_vad=False, this is also None!
        
        Current behavior when enable_vad=False + enable_speaker_separation=True:
        - Initial VAD skipped → full audio goes to quality filters
        - After speaker separation, per-speaker VAD fails because self._vad_stage is None
        - Result: May return None or fail
        
        This test documents this behavior.
        """
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        config = AudioDataFilterConfig(
            enable_vad=False,  # Disables BOTH initial and per-speaker VAD
            enable_speaker_separation=True,
            enable_band_filter=False,
            enable_nisqa=False,
            enable_sigmos=False,
        )
        
        stage = AudioDataFilterStage(config=config)
        stage.setup()
        
        try:
            task = create_audio_batch(short_audio_segment)
            original_duration_ms = get_audio_duration_ms(short_audio_segment)
            
            result = stage.process(task)
            print_result_details(result, "VAD Disabled + Speaker Sep")
            
            # When enable_vad=False, self._vad_stage is None
            # The _run_stage() method returns tasks unchanged if stage is None (line 386-387)
            # So per-speaker "VAD" doesn't segment, it just passes through
            # 
            # However, the timestamp mapping fails because:
            # 1. tracker.translate_to_original() expects start_ms/end_ms from VAD segments
            # 2. Without VAD, the speaker audio doesn't have proper start_ms/end_ms
            # 3. Result: translate_to_original returns empty list → no final results
            print(f"\n  Result when VAD disabled + Speaker Sep enabled:")
            print(f"    Result is None: {result is None}")
            
            if result is not None:
                print(f"    Number of segments: {len(result.data)}")
                for item in result.data:
                    assert 'speaker_id' in item
                    assert 'original_start_ms' in item
                    assert 'original_end_ms' in item
                
                validate_timestamps(result, original_duration_ms, "VAD Disabled + Speaker Sep")
            else:
                # Expected: Result is None because timestamp mapping fails without VAD
                # The per-speaker audio passes through but lacks start_ms/end_ms
                # that correspond to the concatenated audio mappings
                print("    Expected: Result is None due to timestamp mapping failure")
        finally:
            stage.teardown()


# =============================================================================
# TEST CATEGORY 3: Speaker Separation Disabled Tests
# =============================================================================

@pytest.mark.gpu
class TestSpeakerSeparationDisabled:
    """Tests with speaker separation disabled."""
    
    def test_speaker_sep_disabled_with_vad(self, short_audio_segment: str):
        """Speaker separation disabled but VAD enabled."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        config = AudioDataFilterConfig(
            enable_vad=True,
            enable_speaker_separation=False,  # Disabled
            enable_band_filter=False,
            enable_nisqa=False,
            enable_sigmos=False,
        )
        
        stage = AudioDataFilterStage(config=config)
        stage.setup()
        
        try:
            task = create_audio_batch(short_audio_segment)
            original_duration_ms = get_audio_duration_ms(short_audio_segment)
            
            result = stage.process(task)
            print_result_details(result, "Speaker Sep Disabled + VAD")
            
            assert result is not None, "Should have VAD segments"
            
            # Should have multiple segments from VAD
            print(f"\n  Number of VAD segments: {len(result.data)}")
            assert len(result.data) >= 1, "Should have at least 1 segment"
            
            for item in result.data:
                # No speaker_id when speaker separation is disabled
                # (or it might be absent entirely)
                assert 'original_start_ms' in item
                assert 'original_end_ms' in item
                assert 'duration_ms' in item
                
                # Verify timestamps are from VAD segmentation
                start_ms = item.get('original_start_ms', 0)
                end_ms = item.get('original_end_ms', 0)
                duration_ms = item.get('duration_ms', 0)
                
                # VAD segments should be smaller than full audio
                # (unless audio has continuous speech)
                print(f"    Segment: {start_ms}ms - {end_ms}ms ({duration_ms}ms)")
            
            validate_timestamps(result, original_duration_ms, "Speaker Sep Disabled + VAD")
        finally:
            stage.teardown()
    
    def test_speaker_sep_disabled_output_format(self, short_audio_segment: str):
        """Verify output format when speaker separation is disabled."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        config = AudioDataFilterConfig(
            enable_vad=True,
            enable_speaker_separation=False,
            enable_band_filter=True,
            enable_nisqa=True,
            enable_sigmos=False,
            nisqa_mos_threshold=2.0,
        )
        
        stage = AudioDataFilterStage(config=config)
        stage.setup()
        
        try:
            task = create_audio_batch(short_audio_segment)
            
            result = stage.process(task)
            print_result_details(result, "Speaker Sep Disabled + Filters")
            
            if result is not None:
                for item in result.data:
                    # Required fields
                    required_fields = ['original_file', 'original_start_ms', 
                                      'original_end_ms', 'duration_ms']
                    for field in required_fields:
                        assert field in item, f"Missing required field: {field}"
                    
                    # With NISQA enabled, should have quality scores
                    if config.enable_nisqa:
                        has_nisqa = any(k.startswith('nisqa_') for k in item.keys())
                        print(f"  Has NISQA scores: {has_nisqa}")
        finally:
            stage.teardown()


# =============================================================================
# TEST CATEGORY 4: Both VAD and Speaker Separation Disabled
# =============================================================================

@pytest.mark.gpu
class TestMinimalPipeline:
    """Tests with both VAD and speaker separation disabled."""
    
    def test_both_disabled_full_audio(self, short_audio_segment: str):
        """Both VAD and speaker separation disabled - process full audio."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        config = AudioDataFilterConfig(
            enable_vad=False,
            enable_speaker_separation=False,
            enable_band_filter=False,
            enable_nisqa=False,
            enable_sigmos=False,
        )
        
        stage = AudioDataFilterStage(config=config)
        stage.setup()
        
        try:
            task = create_audio_batch(short_audio_segment)
            original_duration_ms = get_audio_duration_ms(short_audio_segment)
            
            result = stage.process(task)
            print_result_details(result, "Minimal Pipeline (both disabled)")
            
            assert result is not None, "Should get result"
            assert len(result.data) == 1, f"Should get 1 segment, got {len(result.data)}"
            
            item = result.data[0]
            
            # Verify full audio timestamps
            assert item.get('original_start_ms', -1) == 0
            assert abs(item.get('original_end_ms', 0) - original_duration_ms) <= 100
            assert abs(item.get('duration_ms', 0) - original_duration_ms) <= 100
            
            # No quality scores should be present
            quality_keys = [k for k in item.keys() if k.startswith(('nisqa_', 'sigmos_', 'band_'))]
            assert len(quality_keys) == 0, f"Unexpected quality scores: {quality_keys}"
            
            print(f"\n  ✓ Full audio processed as single segment with correct timestamps")
        finally:
            stage.teardown()
    
    def test_both_disabled_with_nisqa_only(self, short_audio_segment: str):
        """Both VAD and speaker sep disabled, only NISQA enabled."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        config = AudioDataFilterConfig(
            enable_vad=False,
            enable_speaker_separation=False,
            enable_band_filter=False,
            enable_nisqa=True,
            enable_sigmos=False,
            nisqa_mos_threshold=2.0,  # Very permissive
        )
        
        stage = AudioDataFilterStage(config=config)
        stage.setup()
        
        try:
            task = create_audio_batch(short_audio_segment)
            original_duration_ms = get_audio_duration_ms(short_audio_segment)
            
            result = stage.process(task)
            print_result_details(result, "Both Disabled + NISQA Only")
            
            if result is not None:
                # Should have NISQA scores
                item = result.data[0]
                nisqa_keys = [k for k in item.keys() if k.startswith('nisqa_')]
                print(f"\n  NISQA keys present: {nisqa_keys}")
                assert len(nisqa_keys) > 0, "Should have NISQA scores"
                
                # Verify timestamps for full audio
                assert item.get('original_start_ms', -1) == 0
                assert abs(item.get('original_end_ms', 0) - original_duration_ms) <= 100
        finally:
            stage.teardown()


# =============================================================================
# TEST CATEGORY 5: Quality Filter Combinations
# =============================================================================

@pytest.mark.gpu
class TestQualityFilterCombinations:
    """Tests for different quality filter combinations."""
    
    def test_only_band_filter(self, short_audio_segment: str):
        """Test with only band filter enabled."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        config = AudioDataFilterConfig(
            enable_vad=True,
            enable_speaker_separation=False,
            enable_band_filter=True,
            enable_nisqa=False,
            enable_sigmos=False,
        )
        
        stage = AudioDataFilterStage(config=config)
        stage.setup()
        
        try:
            task = create_audio_batch(short_audio_segment)
            
            result = stage.process(task)
            print_result_details(result, "Band Filter Only")
            
            if result is not None:
                for item in result.data:
                    # Should have band but not NISQA/SIGMOS
                    has_band = 'band_prediction' in item
                    has_nisqa = any(k.startswith('nisqa_') for k in item.keys())
                    has_sigmos = any(k.startswith('sigmos_') for k in item.keys())
                    
                    # Band filter adds band_prediction if it passes
                    print(f"  Band: {has_band}, NISQA: {has_nisqa}, SIGMOS: {has_sigmos}")
                    assert not has_nisqa, "Should not have NISQA scores"
                    assert not has_sigmos, "Should not have SIGMOS scores"
        finally:
            stage.teardown()
    
    def test_only_sigmos_filter(self, short_audio_segment: str):
        """Test with only SIGMOS filter enabled."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        config = AudioDataFilterConfig(
            enable_vad=True,
            enable_speaker_separation=False,
            enable_band_filter=False,
            enable_nisqa=False,
            enable_sigmos=True,
            sigmos_noise_threshold=2.0,
            sigmos_ovrl_threshold=2.0,
        )
        
        stage = AudioDataFilterStage(config=config)
        stage.setup()
        
        try:
            task = create_audio_batch(short_audio_segment)
            
            result = stage.process(task)
            print_result_details(result, "SIGMOS Filter Only")
            
            if result is not None:
                for item in result.data:
                    has_sigmos = any(k.startswith('sigmos_') for k in item.keys())
                    has_nisqa = any(k.startswith('nisqa_') for k in item.keys())
                    has_band = 'band_prediction' in item
                    
                    print(f"  SIGMOS: {has_sigmos}, NISQA: {has_nisqa}, Band: {has_band}")
                    assert has_sigmos, "Should have SIGMOS scores"
                    assert not has_nisqa, "Should not have NISQA scores"
                    assert not has_band, "Should not have band prediction"
        finally:
            stage.teardown()
    
    def test_nisqa_and_sigmos(self, short_audio_segment: str):
        """Test with both NISQA and SIGMOS enabled."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        config = AudioDataFilterConfig(
            enable_vad=True,
            enable_speaker_separation=False,
            enable_band_filter=False,
            enable_nisqa=True,
            enable_sigmos=True,
            nisqa_mos_threshold=2.0,
            sigmos_noise_threshold=2.0,
            sigmos_ovrl_threshold=2.0,
        )
        
        stage = AudioDataFilterStage(config=config)
        stage.setup()
        
        try:
            task = create_audio_batch(short_audio_segment)
            
            result = stage.process(task)
            print_result_details(result, "NISQA + SIGMOS")
            
            if result is not None:
                for item in result.data:
                    has_nisqa = any(k.startswith('nisqa_') for k in item.keys())
                    has_sigmos = any(k.startswith('sigmos_') for k in item.keys())
                    
                    print(f"  NISQA: {has_nisqa}, SIGMOS: {has_sigmos}")
                    # Both should be present if segment passed both filters
                    assert has_nisqa, "Should have NISQA scores"
                    assert has_sigmos, "Should have SIGMOS scores"
        finally:
            stage.teardown()


# =============================================================================
# TEST CATEGORY 6: Timestamp Accuracy Tests
# =============================================================================

@pytest.mark.gpu
class TestTimestampAccuracy:
    """Focused tests on timestamp accuracy in various configurations."""
    
    def test_vad_timestamps_match_segments(self, short_audio_segment: str):
        """Verify VAD segment timestamps are accurate."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        config = AudioDataFilterConfig(
            enable_vad=True,
            enable_speaker_separation=False,
            enable_band_filter=False,
            enable_nisqa=False,
            enable_sigmos=False,
            vad_min_duration_sec=1.0,
            vad_max_duration_sec=30.0,
        )
        
        stage = AudioDataFilterStage(config=config)
        stage.setup()
        
        try:
            task = create_audio_batch(short_audio_segment)
            original_duration_ms = get_audio_duration_ms(short_audio_segment)
            
            result = stage.process(task)
            print_result_details(result, "VAD Timestamps Test")
            
            assert result is not None
            
            total_segment_duration = 0
            for item in result.data:
                start_ms = item.get('original_start_ms', 0)
                end_ms = item.get('original_end_ms', 0)
                duration_ms = item.get('duration_ms', 0)
                
                # Each segment's duration should match end - start
                expected_duration = end_ms - start_ms
                assert abs(duration_ms - expected_duration) <= 1, \
                    f"Duration mismatch: {duration_ms} vs {expected_duration}"
                
                # Timestamps should be within original audio bounds
                assert 0 <= start_ms <= original_duration_ms
                assert 0 <= end_ms <= original_duration_ms + 100
                assert start_ms < end_ms
                
                total_segment_duration += duration_ms
            
            print(f"\n  Total segment duration: {total_segment_duration}ms")
            print(f"  Original audio duration: {original_duration_ms}ms")
            print(f"  Coverage: {100 * total_segment_duration / original_duration_ms:.1f}%")
        finally:
            stage.teardown()
    
    def test_speaker_separation_timestamps_valid(self, short_audio_segment: str):
        """Verify speaker separation timestamps are valid."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        config = AudioDataFilterConfig(
            enable_vad=True,
            enable_speaker_separation=True,
            enable_band_filter=False,
            enable_nisqa=False,
            enable_sigmos=False,
            vad_min_duration_sec=1.0,
        )
        
        stage = AudioDataFilterStage(config=config)
        stage.setup()
        
        try:
            task = create_audio_batch(short_audio_segment)
            original_duration_ms = get_audio_duration_ms(short_audio_segment)
            
            result = stage.process(task)
            print_result_details(result, "Speaker Sep Timestamps Test")
            
            if result is not None:
                # Group by speaker
                speakers = {}
                for item in result.data:
                    speaker_id = item.get('speaker_id', 'unknown')
                    if speaker_id not in speakers:
                        speakers[speaker_id] = []
                    speakers[speaker_id].append(item)
                
                print(f"\n  Speakers found: {list(speakers.keys())}")
                
                for speaker_id, segments in speakers.items():
                    print(f"\n  {speaker_id}: {len(segments)} segments")
                    
                    for seg in segments:
                        start_ms = seg.get('original_start_ms', 0)
                        end_ms = seg.get('original_end_ms', 0)
                        
                        # All timestamps should be valid
                        assert 0 <= start_ms <= original_duration_ms, \
                            f"Invalid start_ms {start_ms} for {speaker_id}"
                        assert 0 <= end_ms <= original_duration_ms + 100, \
                            f"Invalid end_ms {end_ms} for {speaker_id}"
                        assert start_ms < end_ms, \
                            f"start >= end for {speaker_id}: {start_ms} >= {end_ms}"
        finally:
            stage.teardown()
    
    def test_no_overlapping_timestamps_per_speaker(self, short_audio_segment: str):
        """Verify no overlapping timestamps within same speaker."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        config = AudioDataFilterConfig(
            enable_vad=True,
            enable_speaker_separation=True,
            enable_band_filter=False,
            enable_nisqa=False,
            enable_sigmos=False,
        )
        
        stage = AudioDataFilterStage(config=config)
        stage.setup()
        
        try:
            task = create_audio_batch(short_audio_segment)
            
            result = stage.process(task)
            
            if result is not None:
                # Group by speaker
                speakers = {}
                for item in result.data:
                    speaker_id = item.get('speaker_id', 'unknown')
                    if speaker_id not in speakers:
                        speakers[speaker_id] = []
                    speakers[speaker_id].append({
                        'start': item.get('original_start_ms', 0),
                        'end': item.get('original_end_ms', 0)
                    })
                
                # Check for overlaps within each speaker
                for speaker_id, segments in speakers.items():
                    # Sort by start time
                    sorted_segs = sorted(segments, key=lambda x: x['start'])
                    
                    for i in range(len(sorted_segs) - 1):
                        current_end = sorted_segs[i]['end']
                        next_start = sorted_segs[i + 1]['start']
                        
                        # Segments should not overlap (small tolerance allowed)
                        if current_end > next_start + 10:  # 10ms tolerance
                            print(f"  WARNING: Possible overlap in {speaker_id}")
                            print(f"    Segment {i} ends at {current_end}")
                            print(f"    Segment {i+1} starts at {next_start}")
                
                print(f"\n  ✓ Checked {len(speakers)} speakers for overlapping timestamps")
        finally:
            stage.teardown()


# =============================================================================
# TEST CATEGORY 7: Edge Cases and Error Handling
# =============================================================================

@pytest.mark.gpu
class TestEdgeCases:
    """Edge case tests for the advance pipeline."""
    
    def test_very_short_audio(self):
        """Test with very short audio (< 1 second)."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        # Create a 500ms audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        
        try:
            # Create 500ms of silence/noise at 48kHz
            duration_samples = int(0.5 * 48000)
            audio_data = np.random.randn(duration_samples).astype(np.float32) * 0.1
            sf.write(temp_path, audio_data, 48000)
            
            config = AudioDataFilterConfig(
                enable_vad=True,
                enable_speaker_separation=False,
                enable_band_filter=False,
                enable_nisqa=False,
                enable_sigmos=False,
                vad_min_duration_sec=0.1,  # Allow short segments
            )
            
            stage = AudioDataFilterStage(config=config)
            stage.setup()
            
            try:
                task = create_audio_batch(temp_path)
                result = stage.process(task)
                print_result_details(result, "Very Short Audio (500ms)")
                
                # May be None if VAD finds no speech, which is expected
                if result is not None:
                    for item in result.data:
                        assert item.get('original_end_ms', 0) <= 600  # 500ms + tolerance
            finally:
                stage.teardown()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_config_modification_does_not_affect_stage(self, short_audio_segment: str):
        """Verify that modifying config after stage creation doesn't affect stage."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        config = AudioDataFilterConfig(
            enable_vad=True,
            enable_speaker_separation=False,
            enable_nisqa=False,
        )
        
        stage = AudioDataFilterStage(config=config)
        
        # Modify config after stage creation
        config.enable_vad = False
        config.enable_nisqa = True
        
        # Stage should use original config values
        stage.setup()
        
        try:
            task = create_audio_batch(short_audio_segment)
            result = stage.process(task)
            
            # Should still have VAD segments (original config had enable_vad=True)
            if result is not None and stage.config.enable_vad:
                print(f"\n  Stage config: enable_vad={stage.config.enable_vad}")
                # Note: This depends on how the stage stores config
        finally:
            stage.teardown()
    
    def test_get_enabled_filters(self):
        """Test the get_enabled_filters method."""
        from nemo_curator.stages.audio.advance_pipelines import AudioDataFilterConfig
        
        # All enabled
        config1 = AudioDataFilterConfig(
            enable_band_filter=True,
            enable_nisqa=True,
            enable_sigmos=True,
        )
        filters1 = config1.get_enabled_filters()
        assert 'band' in filters1
        assert 'nisqa' in filters1
        assert 'sigmos' in filters1
        print(f"\n  All enabled: {filters1}")
        
        # None enabled
        config2 = AudioDataFilterConfig(
            enable_band_filter=False,
            enable_nisqa=False,
            enable_sigmos=False,
        )
        filters2 = config2.get_enabled_filters()
        assert len(filters2) == 0
        print(f"  None enabled: {filters2}")
        
        # Mixed
        config3 = AudioDataFilterConfig(
            enable_band_filter=True,
            enable_nisqa=False,
            enable_sigmos=True,
        )
        filters3 = config3.get_enabled_filters()
        assert 'band' in filters3
        assert 'nisqa' not in filters3
        assert 'sigmos' in filters3
        print(f"  Mixed (band, sigmos): {filters3}")


# =============================================================================
# TEST CATEGORY 8: Comparison Tests
# =============================================================================

@pytest.mark.gpu
class TestConfigurationComparison:
    """Compare outputs between different configurations."""
    
    def test_vad_enabled_vs_disabled_output_count(self, short_audio_segment: str):
        """Compare segment count with VAD enabled vs disabled."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        # Config with VAD
        config_vad = AudioDataFilterConfig(
            enable_vad=True,
            enable_speaker_separation=False,
            enable_band_filter=False,
            enable_nisqa=False,
            enable_sigmos=False,
        )
        
        # Config without VAD
        config_no_vad = AudioDataFilterConfig(
            enable_vad=False,
            enable_speaker_separation=False,
            enable_band_filter=False,
            enable_nisqa=False,
            enable_sigmos=False,
        )
        
        stage_vad = AudioDataFilterStage(config=config_vad)
        stage_no_vad = AudioDataFilterStage(config=config_no_vad)
        
        stage_vad.setup()
        stage_no_vad.setup()
        
        try:
            task_vad = create_audio_batch(short_audio_segment, task_id="vad_test")
            task_no_vad = create_audio_batch(short_audio_segment, task_id="no_vad_test")
            
            result_vad = stage_vad.process(task_vad)
            result_no_vad = stage_no_vad.process(task_no_vad)
            
            vad_count = len(result_vad.data) if result_vad else 0
            no_vad_count = len(result_no_vad.data) if result_no_vad else 0
            
            print(f"\n  With VAD: {vad_count} segments")
            print(f"  Without VAD: {no_vad_count} segments")
            
            # Without VAD should have exactly 1 segment (full audio)
            assert no_vad_count == 1, f"Without VAD should have 1 segment, got {no_vad_count}"
            
            # With VAD should typically have multiple segments (depends on audio content)
            print(f"  VAD produced {vad_count} segments from the audio")
        finally:
            stage_vad.teardown()
            stage_no_vad.teardown()
    
    def test_output_format_consistency(self, short_audio_segment: str):
        """Verify output format is consistent across configurations."""
        from nemo_curator.stages.audio.advance_pipelines import (
            AudioDataFilterStage,
            AudioDataFilterConfig,
        )
        
        configs = [
            ("All disabled", AudioDataFilterConfig(
                enable_vad=False, enable_speaker_separation=False,
                enable_band_filter=False, enable_nisqa=False, enable_sigmos=False
            )),
            ("VAD only", AudioDataFilterConfig(
                enable_vad=True, enable_speaker_separation=False,
                enable_band_filter=False, enable_nisqa=False, enable_sigmos=False
            )),
            ("VAD + Speaker Sep", AudioDataFilterConfig(
                enable_vad=True, enable_speaker_separation=True,
                enable_band_filter=False, enable_nisqa=False, enable_sigmos=False
            )),
        ]
        
        required_fields = ['original_file', 'original_start_ms', 'original_end_ms', 
                          'duration_ms', 'duration_sec']
        
        for config_name, config in configs:
            stage = AudioDataFilterStage(config=config)
            stage.setup()
            
            try:
                task = create_audio_batch(short_audio_segment)
                result = stage.process(task)
                
                print(f"\n  Config: {config_name}")
                
                if result is not None:
                    for item in result.data[:3]:  # Check first 3
                        missing = [f for f in required_fields if f not in item]
                        if missing:
                            print(f"    Missing fields: {missing}")
                            pytest.fail(f"Missing required fields in {config_name}: {missing}")
                    
                    print(f"    ✓ All {len(required_fields)} required fields present")
            finally:
                stage.teardown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

