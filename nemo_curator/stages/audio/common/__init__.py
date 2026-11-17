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

from nemo_curator.stages.audio.common.duration import GetAudioDurationStage as GetAudioDurationStage
from nemo_curator.stages.audio.common.duration import get_audio_duration as get_audio_duration
from nemo_curator.stages.audio.common.folder_source import (
    CreateInitialManifestAudioFolderStage as CreateInitialManifestAudioFolderStage,
)
from nemo_curator.stages.audio.common.folder_source import _AudioFolderTask as _AudioFolderTask
from nemo_curator.stages.audio.common.helpers import ensure_mono as ensure_mono
from nemo_curator.stages.audio.common.helpers import ensure_waveform_2d as ensure_waveform_2d
from nemo_curator.stages.audio.common.helpers import load_audio_file as load_audio_file
from nemo_curator.stages.audio.common.helpers import resolve_model_path as resolve_model_path
from nemo_curator.stages.audio.common.helpers import resolve_waveform_from_item as resolve_waveform_from_item
from nemo_curator.stages.audio.common.manifest_checkpoint import ManifestCheckpointStage as ManifestCheckpointStage
from nemo_curator.stages.audio.common.manifest_reader import ManifestReader as ManifestReader
from nemo_curator.stages.audio.common.manifest_reader import ManifestReaderStage as ManifestReaderStage
from nemo_curator.stages.audio.common.manifest_writer import ManifestWriterStage as ManifestWriterStage
from nemo_curator.stages.audio.common.value_filters import (
    PreserveByValueConditionsStage as PreserveByValueConditionsStage,
)
from nemo_curator.stages.audio.common.value_filters import PreserveByValueStage as PreserveByValueStage
from nemo_curator.stages.audio.common.value_filters import _ValueCondition as _ValueCondition

__all__ = [
    "CreateInitialManifestAudioFolderStage",
    "GetAudioDurationStage",
    "ManifestCheckpointStage",
    "ManifestReader",
    "ManifestReaderStage",
    "ManifestWriterStage",
    "PreserveByValueConditionsStage",
    "PreserveByValueStage",
    "ensure_mono",
    "ensure_waveform_2d",
    "get_audio_duration",
    "load_audio_file",
    "resolve_model_path",
    "resolve_waveform_from_item",
]

# Preserve pipeline fingerprints and existing pickle paths across the module split.
for _export in (
    get_audio_duration,
    GetAudioDurationStage,
    PreserveByValueStage,
    PreserveByValueConditionsStage,
    ManifestReaderStage,
    ManifestReader,
    CreateInitialManifestAudioFolderStage,
    ManifestWriterStage,
    ManifestCheckpointStage,
    load_audio_file,
    ensure_waveform_2d,
    ensure_mono,
    resolve_waveform_from_item,
    resolve_model_path,
    _AudioFolderTask,
    _ValueCondition,
):
    _export.__module__ = __name__
del _export
