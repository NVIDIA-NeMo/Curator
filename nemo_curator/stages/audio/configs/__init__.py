# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Configuration classes for audio processing stages."""

from .vad import VADConfig
from .nisqa import NISQAConfig
from .sigmos import SIGMOSConfig
from .band import BandFilterConfig
from .speaker import SpeakerSeparationConfig
from .mono_conversion import MonoConversionConfig
from .concatenation import SegmentConcatenationConfig

__all__ = [
    # Preprocessing configs
    "MonoConversionConfig",
    "SegmentConcatenationConfig",
    # Segmentation configs
    "VADConfig",
    "SpeakerSeparationConfig",
    # Filtering configs
    "NISQAConfig",
    "SIGMOSConfig",
    "BandFilterConfig",
]
