# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

"""
Audio quality filtering stages.

These stages filter audio based on various quality metrics:
- NISQAFilterStage: Speech quality assessment (MOS, noisiness, etc.)
- SIGMOSFilterStage: Signal-based quality metrics
- BandFilterStage: Bandwidth classification (full-band vs narrow-band)

Example:
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.audio.filtering import NISQAFilterStage, SIGMOSFilterStage
    
    pipeline = Pipeline(name="quality_pipeline")
    pipeline.add_stage(NISQAFilterStage(mos_threshold=4.5))
    pipeline.add_stage(SIGMOSFilterStage(noise_threshold=4.0))
"""

from .nisqa import NISQAFilterStage
from .sigmos import SIGMOSFilterStage
from .band import BandFilterStage

__all__ = [
    "NISQAFilterStage",
    "SIGMOSFilterStage",
    "BandFilterStage",
]
