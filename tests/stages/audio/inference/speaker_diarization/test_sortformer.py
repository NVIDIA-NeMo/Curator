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

"""Compatibility-import tests for the historic Sortformer stage module."""

from unittest.mock import MagicMock

from nemo_curator.models.audio.speaker_diarization.sortformer import parse_sortformer_segments
from nemo_curator.stages.audio.inference.speaker_diarization.sortformer import (
    InferenceSortformerStage,
    _parse_sortformer_segments,
)
from nemo_curator.stages.audio.inference.speaker_diarization.stage import (
    InferenceSortformerStage as CanonicalStage,
)
from nemo_curator.tasks import AudioTask


def test_compatibility_module_reexports_canonical_stage() -> None:
    assert InferenceSortformerStage is CanonicalStage


def test_compatibility_parser_aliases_model_adapter_helper() -> None:
    assert _parse_sortformer_segments is parse_sortformer_segments


def test_legacy_preloaded_model_supports_direct_process_without_setup() -> None:
    model = MagicMock()
    model.sortformer_modules = MagicMock()
    model.parameters.return_value = iter([])
    model.diarize.return_value = [["0.00 1.00 speaker_0"]]
    stage = InferenceSortformerStage(
        diar_model=model,
        adapter_kwargs={"bounded_stft": False, "max_positional_encoding_length": None},
    )

    task = AudioTask(data={"audio_filepath": "/provider/virtual.wav"})
    result = stage.process(task)

    assert result.data["diar_segments"] == [{"start": 0.0, "end": 1.0, "speaker": "speaker_0"}]
    assert result is not task
    assert result.filepath_key == "audio_filepath"
    assert "diar_segments" not in task.data
    model.diarize.assert_called_once_with(audio=["/provider/virtual.wav"], batch_size=1)


def test_legacy_preloaded_model_supports_direct_diarize_without_setup() -> None:
    model = MagicMock()
    model.diarize.return_value = [["0.00 1.00 speaker_0"]]
    stage = InferenceSortformerStage(diar_model=model)

    result = stage.diarize(["/provider/virtual.wav"])

    assert result == [[{"start": 0.0, "end": 1.0, "speaker": "speaker_0"}]]
    model.diarize.assert_called_once_with(audio=["/provider/virtual.wav"], batch_size=1)
