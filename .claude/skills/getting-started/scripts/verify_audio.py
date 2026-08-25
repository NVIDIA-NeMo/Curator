#!/usr/bin/env python3
"""Verify audio modality dependencies are installed."""

import nemo.collections.asr as nemo_asr

from nemo_curator.models.asr.nemo_asr import NeMoASRAdapter
from nemo_curator.stages.audio.inference.asr.stage import ASRStage

adapter = NeMoASRAdapter(model_id="nvidia/stt_en_fastconformer_ctc_large")
stage = ASRStage(
    adapter_target="nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
    model_id=adapter.model_id,
    audio_filepath_key="audio_filepath",
)

print(f"✓ Audio modality imports verified ({nemo_asr.__name__}, {type(stage).__name__}, {type(adapter).__name__})")
