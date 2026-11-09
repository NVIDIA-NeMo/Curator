#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Verify audio modality dependencies are installed."""

import nemo.collections.asr as nemo_asr

from nemo_curator.models.asr.nemo_asr import NeMoASRAdapter
from nemo_curator.stages.audio.inference.asr.stage import ASRStage

adapter = NeMoASRAdapter(model_id="nvidia/stt_en_fastconformer_ctc_large")
stage = ASRStage(
    adapter_target="nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
    model_id=adapter.model_id,
    max_audio_sec_per_actor=2400.0,
    audio_filepath_key="audio_filepath",
)

print(f"✓ Audio modality imports verified ({nemo_asr.__name__}, {type(stage).__name__}, {type(adapter).__name__})")
