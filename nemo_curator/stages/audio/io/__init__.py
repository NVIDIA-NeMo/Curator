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

"""Audio input/output stages with lazy optional-dependency imports."""

from importlib import import_module

_LAZY = {
    "NeMoSpeechAudioReader": "nemo_curator.stages.audio.io.nemo_speech_reader",
    "NeMoSpeechDiscoveryStage": "nemo_curator.stages.audio.io.nemo_speech_reader",
    "NeMoSpeechReaderStage": "nemo_curator.stages.audio.io.nemo_speech_reader",
    "NeMoSpeechWriterStage": "nemo_curator.stages.audio.io.nemo_speech_writer",
    "derive_manifest_shard_key": "nemo_curator.stages.audio.io.shard_key",
    "finalize_nemo_speech_output": "nemo_curator.stages.audio.io.nemo_speech_writer",
}

__all__ = [
    "NeMoSpeechAudioReader",
    "NeMoSpeechDiscoveryStage",
    "NeMoSpeechReaderStage",
    "NeMoSpeechWriterStage",
    "derive_manifest_shard_key",
    "finalize_nemo_speech_output",
]


def __getattr__(name: str) -> object:
    target = _LAZY.get(name)
    if target is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(import_module(target), name)


def __dir__() -> list[str]:
    return __all__
