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

"""Stages for cutting long-form diarized audio into snippets for ALM pretraining.

The output JSONL produced by this pipeline is a per-snippet manifest where
each row preserves the original audio's metadata (with ``alignment``
removed), references the cut snippet WAV/FLAC file, and carries a list of
diarization+transcription segments with timestamps relative to the snippet.
That shape can be reused to construct interleaved audio/text continuation
data, ASR training data, TTS training data, and speaker-diarization
training data without re-cutting the source audio.
"""

from nemo_curator.stages.audio.alm.pretrain.extraction import (
    SnippetExtractionStage,
)
from nemo_curator.stages.audio.alm.pretrain.finalize import (
    finalize_audio_pretrain_outputs,
    prepare_audio_pretrain_outputs,
)
from nemo_curator.stages.audio.alm.pretrain.io import (
    PretrainMetricsAggregatorStage,
    ReadLongFormManifestStage,
    SnippetManifestWriterStage,
)
from nemo_curator.stages.audio.alm.pretrain.pipeline import (
    build_audio_pretrain_pipeline,
)
from nemo_curator.stages.audio.alm.pretrain.planning import (
    OverlapFilterStage,
    SnippetCutPlannerStage,
    SnippetRepetitionFilterStage,
)

__all__ = [
    "OverlapFilterStage",
    "PretrainMetricsAggregatorStage",
    "ReadLongFormManifestStage",
    "SnippetCutPlannerStage",
    "SnippetExtractionStage",
    "SnippetManifestWriterStage",
    "SnippetRepetitionFilterStage",
    "build_audio_pretrain_pipeline",
    "finalize_audio_pretrain_outputs",
    "prepare_audio_pretrain_outputs",
]
