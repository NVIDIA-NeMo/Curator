# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
    
    # Add to pipeline (CompositeStage decomposes into independent stages)
    pipeline.add_stage(AudioDataFilterStage(config=config))
"""

from .audio_data_filter import AudioDataFilterStage
from .config import AudioDataFilterConfig

__all__ = [
    "AudioDataFilterStage",
    "AudioDataFilterConfig",
]
