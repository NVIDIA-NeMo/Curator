# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Configuration classes for audio processing stages."""

from .mono_conversion import MonoConversionConfig
from .concatenation import SegmentConcatenationConfig

__all__ = [
    "MonoConversionConfig",
    "SegmentConcatenationConfig",
]
