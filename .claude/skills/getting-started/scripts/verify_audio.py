#!/usr/bin/env python3
"""Verify audio modality dependencies are installed."""
import nemo.collections.asr as nemo_asr
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
print("✓ Audio modality imports verified")
