#!/usr/bin/env python3
"""Verify audio modality dependencies are installed."""

import nemo.collections.asr as nemo_asr

from nemo_curator.models.asr.nemo_asr import NeMoASRAdapter
from nemo_curator.stages.audio.inference.asr.stage import ASRStage

_AUDIO_IMPORTS = (nemo_asr, ASRStage, NeMoASRAdapter)
print("✓ Audio adapter imports verified")
