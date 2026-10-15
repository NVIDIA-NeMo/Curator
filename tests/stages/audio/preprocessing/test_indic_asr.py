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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio.preprocessing import PrepareIndicASRInputStage
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def indic_task(tmp_path: Path) -> AudioTask:
    path = tmp_path / "clip.wav"
    sf.write(path, np.full(1600, 0.25, dtype=np.float32), 16000, subtype="FLOAT")
    return AudioTask(
        task_id="clip",
        dataset_name="hindi",
        data={
            "audio_filepath": str(path),
            "audio_item_id": "clip",
            "duration": 0.1,
            "source_lang": "hi",
            "text": "  यह एक परीक्षण है।  ",
        },
    )


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("stereo", [False, True])
def test_prepares_audio_without_changing_reference_or_metadata(
    indic_task: AudioTask, tmp_path: Path, relative: bool, stereo: bool
) -> None:
    reference = indic_task.data["text"]
    path = indic_task.data["audio_filepath"]
    if stereo:
        sf.write(path, np.tile([0.25, 0.75], (1600, 1)), 16000, subtype="FLOAT")
    if relative:
        indic_task.data["audio_filepath"] = "clip.wav"
    indic_task.data.update(
        additional_notes={"dataset": "public", "primary_model": "old"},
        _skipme="old reason",
        custom_metadata="preserved",
    )

    result = PrepareIndicASRInputStage(audio_root=str(tmp_path) if relative else None).process(indic_task)

    assert result is indic_task
    assert result.task_id == "clip"
    assert result.dataset_name == "hindi"
    assert result.data["audio_filepath"] == path
    assert result.data["audio_item_id"] == "clip"
    assert result.data["source_lang"] == "hi"
    assert result.data["duration"] == 0.1
    assert result.data["custom_metadata"] == "preserved"
    assert "text" not in result.data
    assert result.data["reference_text"] == result.data["granary_v1_prediction"] == reference
    assert result.data["additional_notes"] == {
        "dataset": "public",
        "primary_model": "indic_canary",
        "recovery_model": "parakeet_riva",
    }
    assert result.data["_skipme"] == ""
    assert result.data["sampling_rate"] == 16000
    assert result.data["waveform"].dtype == torch.float32
    torch.testing.assert_close(result.data["waveform"], torch.full((1, 1600), 0.5 if stereo else 0.25))


@pytest.mark.parametrize("notes", [None, "not a mapping"])
def test_initializes_notes(indic_task: AudioTask, notes: object) -> None:
    if notes is not None:
        indic_task.data["additional_notes"] = notes
    result = PrepareIndicASRInputStage().process(indic_task)
    assert result.data["additional_notes"] == {
        "primary_model": "indic_canary",
        "recovery_model": "parakeet_riva",
    }


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("audio_item_id", "", "audio_item_id must be a nonempty string"),
        ("audio_item_id", 123, "audio_item_id must be a nonempty string"),
        ("text", "", "Reference text is empty"),
        ("text", " \t ", "Reference text is empty"),
        ("text", None, "Reference text is empty"),
        ("source_lang", "en", "Expected source_lang='hi'"),
        ("duration", 0.102, "Manifest duration mismatch"),
    ],
)
def test_rejects_invalid_manifest_fields(indic_task: AudioTask, key: str, value: object, message: str) -> None:
    indic_task.data[key] = value
    with pytest.raises(RuntimeError, match=message):
        PrepareIndicASRInputStage().process(indic_task)


def test_relative_path_requires_audio_root(indic_task: AudioTask) -> None:
    indic_task.data["audio_filepath"] = "clip.wav"
    with pytest.raises(RuntimeError, match="Relative audio path requires audio_root"):
        PrepareIndicASRInputStage().process(indic_task)


def test_rejects_missing_file(indic_task: AudioTask, tmp_path: Path) -> None:
    indic_task.data["audio_filepath"] = str(tmp_path / "missing.wav")
    with pytest.raises(FileNotFoundError, match="Input audio is missing"):
        PrepareIndicASRInputStage().process(indic_task)


def test_rejects_wrong_sample_rate_without_resampling(indic_task: AudioTask) -> None:
    sf.write(indic_task.data["audio_filepath"], np.zeros(800, dtype=np.float32), 8000)
    with pytest.raises(RuntimeError, match="Expected mono 16000 Hz audio"):
        PrepareIndicASRInputStage().process(indic_task)


def test_preserves_manifest_duration_within_tolerance(indic_task: AudioTask) -> None:
    indic_task.data["duration"] = 0.1005
    result = PrepareIndicASRInputStage().process(indic_task)
    assert result.data["duration"] == 0.1005


@pytest.mark.parametrize("remap", [False, True])
def test_declared_contract_matches_processing(indic_task: AudioTask, remap: bool) -> None:
    stage = PrepareIndicASRInputStage()
    if remap:
        stage = PrepareIndicASRInputStage(
            text_key="transcript", reference_text_key="reference", waveform_key="samples"
        )
        indic_task.data["transcript"] = indic_task.data.pop("text")
    reference = indic_task.data[stage.text_key]

    assert_agent_ready(
        stage,
        fixture_factory=lambda: indic_task,
        expected_cardinality="1:1",
        available_keys=indic_task.data.keys(),
    )

    assert indic_task.data[stage.reference_text_key] == reference
    assert indic_task.data[stage.waveform_key].shape == (1, 1600)
    assert stage.text_key not in indic_task.data
