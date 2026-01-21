# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
Advanced Audio Processing Pipelines.

This module provides composite pipeline stages that combine multiple
audio processing steps into single, easy-to-use stages.

Stages:
    - AudioDataFilterStage: Complete audio curation pipeline with VAD,
      quality filtering, speaker separation, and timestamp tracking.

Example:
    from advance_pipelines import (
        AudioDataFilterStage,
        AudioDataFilterConfig,
    )
    
    # Create config
    config = AudioDataFilterConfig(
        enable_nisqa=True,
        enable_sigmos=True,
        enable_speaker_separation=True,
        nisqa_mos_threshold=4.5,
    )
    
    # Create stage
    stage = AudioDataFilterStage(config=config)
    
    # Process audio
    results = stage.process(audio_batch)
"""

from .audio_data_filter import AudioDataFilterStage
from .config import AudioDataFilterConfig

__all__ = [
    "AudioDataFilterStage",
    "AudioDataFilterConfig",
]
