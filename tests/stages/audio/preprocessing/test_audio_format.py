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

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo_curator.stages.audio._agent._agent_registry import build_contract
from nemo_curator.stages.audio.preprocessing import AudioFormatStage
from nemo_curator.stages.audio.preprocessing import audio_format as af
from nemo_curator.tasks import AudioTask


def _wav(tmp_path: Path, channels: int = 2, rate: int = 16000, seconds: float = 0.5) -> str:
    path = tmp_path / f"a_{channels}ch_{rate}.wav"
    frames = int(rate * seconds)
    data = np.stack([np.full(frames, 0.25 * (c + 1), dtype=np.float32) for c in range(channels)], axis=1)
    sf.write(path, data, rate, subtype="FLOAT")
    return str(path)


def _task(**data: object) -> AudioTask:
    return AudioTask(dataset_name="d", data=dict(data))


def test_annotates_from_header_without_decoding(tmp_path: Path) -> None:
    path = _wav(tmp_path, channels=2, rate=22050)
    with mock.patch.object(af, "load_audio_file", side_effect=AssertionError("decoded")):
        out = AudioFormatStage().process(_task(audio_filepath=path))
    assert (out.data["sample_rate"], out.data["num_channels"]) == (22050, 2)
    assert "waveform" not in out.data


@pytest.mark.parametrize(
    ("options", "kept"),
    [
        ({"allowed_sample_rates": [16000]}, True),
        ({"allowed_sample_rates": [48000]}, False),
        ({"min_sample_rate": 22050}, False),
        ({"max_channels": 1}, False),
        ({"allowed_channels": [2], "max_sample_rate": 16000}, True),
    ],
)
def test_selects_on_rate_and_channels(tmp_path: Path, options: dict, kept: bool) -> None:
    out = AudioFormatStage(**options).process(_task(audio_filepath=_wav(tmp_path)))
    assert (out != []) is kept


def test_resident_waveform_is_observed_without_reading_disk() -> None:
    task = _task(waveform=torch.zeros(3, 10), sample_rate=8000, audio_filepath="/missing.wav")
    out = AudioFormatStage(allowed_channels=[3]).process(task)
    assert (out.data["sample_rate"], out.data["num_channels"]) == (8000, 3)


def test_downmixes_to_mono_and_resamples(tmp_path: Path) -> None:
    out = AudioFormatStage(target_channels=1, target_sample_rate=8000).process(_task(audio_filepath=_wav(tmp_path)))
    assert out.data["waveform"].shape == (1, 4000)
    # Away from the resampler's edge ringing, the downmix of 0.25 and 0.5 is their mean.
    assert torch.allclose(out.data["waveform"][:, 500:3500], torch.full((1, 3000), 0.375), atol=1e-2)
    assert (out.data["sample_rate"], out.data["num_channels"], out.data["duration"]) == (8000, 1, 0.5)


def test_duplicates_mono_and_refuses_layout_downmix() -> None:
    stage = AudioFormatStage(target_channels=2)
    assert stage.process(_task(waveform=torch.ones(1, 4), sample_rate=8)).data["waveform"].shape == (2, 4)
    assert stage.process(_task(waveform=torch.ones(6, 4), sample_rate=8)) == []


def test_writes_to_disk_and_repoints_path(tmp_path: Path) -> None:
    source = _wav(tmp_path)
    stage = AudioFormatStage(
        target_sample_rate=8000,
        keep_waveform_in_task=False,
        write_to_disk=True,
        update_audio_filepath=True,
        output_dir=str(tmp_path / "out"),
    )
    out = stage.process(_task(audio_filepath=source))
    written = out.data["audio_filepath"]
    assert out.data["original_audio_filepath"] == source
    assert out.data["converted_audio_filepath"] == written
    assert sf.info(written).samplerate == 8000
    assert "waveform" not in out.data


def test_unreadable_file_drops_row(tmp_path: Path) -> None:
    assert AudioFormatStage().process(_task(audio_filepath=str(tmp_path / "missing.wav"))) == []


@pytest.mark.parametrize(
    "options",
    [
        {"allowed_sample_rates": []},
        {"min_channels": 0},
        {"min_sample_rate": 48000, "max_sample_rate": 16000},
        {"write_to_disk": True},
        {"target_channels": 1, "keep_waveform_in_task": False},
    ],
)
def test_rejects_invalid_configuration(options: dict) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        AudioFormatStage(**options)


def test_contract_names_filtering_and_conversion() -> None:
    assert build_contract(AudioFormatStage()).cardinality == "1:1"
    assert build_contract(AudioFormatStage(allowed_channels=[1])).cardinality == "filter"
    contract = build_contract(AudioFormatStage(target_channels=1))
    assert contract.cardinality == "1:1"
    assert {"waveform", "duration"} <= set(contract.writes.data_keys)
