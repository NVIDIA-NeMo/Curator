# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from nemo_curator.stages.audio._agent._agent_registry import build_contract
from nemo_curator.stages.audio._agent._residency import (
    resolve_audio,
)
from nemo_curator.stages.audio.common import (
    ensure_waveform_2d,
)
from nemo_curator.stages.audio.preprocessing import (
    ChannelCountStage,
    MonoConversionStage,
    SampleRateFilterStage,
)
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path


def _stereo_task(tmp_path: Path, sample_rate: int = 16000) -> tuple[AudioTask, str]:
    """A row carrying BOTH a resident stereo waveform and the file it came from."""
    path = str(tmp_path / "stereo.wav")
    waveform = torch.stack([torch.zeros(sample_rate), torch.ones(sample_rate) * 0.5])
    sf.write(path, waveform.T.numpy(), sample_rate)
    task = AudioTask(
        dataset_name="resident",
        data={"audio_filepath": path, "waveform": waveform, "sample_rate": sample_rate},
    )
    return task, path


@pytest.mark.parametrize(
    ("factory", "channels_key"),
    [
        (
            lambda out: MonoConversionStage(
                output_sample_rate=16000,
                input_residency="waveform",
                keep_waveform_in_task=False,
                write_to_disk=True,
                update_audio_filepath=True,
                output_dir=out,
            ),
            "is_mono",
        ),
        (
            lambda out: ChannelCountStage(
                action="convert",
                target_channels=1,
                input_residency="waveform",
                keep_waveform_in_task=False,
                write_to_disk=True,
                update_audio_filepath=True,
                output_dir=out,
            ),
            "num_channels",
        ),
    ],
    ids=["mono_conversion", "channel_count"],
)
def test_disk_only_conversion_does_not_leave_the_pre_conversion_waveform(
    tmp_path: Path,
    factory,  # noqa: ANN001
    channels_key: str,
) -> None:
    """Resident input -> disk-only conversion -> auto-residency consumer must not read stale audio."""
    task, original = _stereo_task(tmp_path)
    stage = factory(str(tmp_path / "out"))

    result = stage.process(task)
    assert result is not None
    assert not isinstance(result, list)

    # The converted metadata and the audio a downstream stage can reach have to agree.
    assert result.data[channels_key] in (True, 1)
    assert "waveform" not in result.data
    assert "sample_rate" not in result.data

    consumed = resolve_audio(result.data, residency="auto")
    assert consumed is not None
    assert ensure_waveform_2d(consumed[0]).shape[0] == 1
    assert result.data["audio_filepath"] != original

    # And validation knows, so a downstream waveform reader is caught before the run.
    assert set(build_contract(stage).removes_keys) == {"waveform", "sample_rate"}


@pytest.mark.parametrize(
    "cls",
    [MonoConversionStage, ChannelCountStage],
    ids=["mono_conversion", "channel_count"],
)
def test_conversion_without_an_output_sink_is_rejected(cls) -> None:  # noqa: ANN001
    """Converting into neither the task nor disk keeps the original audio under converted metadata."""
    extra = {"action": "convert", "target_channels": 1} if cls is ChannelCountStage else {}
    with pytest.raises(ValueError, match="keep_waveform_in_task or write_to_disk"):
        cls(keep_waveform_in_task=False, write_to_disk=False, **extra)
    with pytest.raises(ValueError, match="update_audio_filepath"):
        cls(write_to_disk=False, update_audio_filepath=True, **extra)


def test_a_null_waveform_does_not_authenticate_a_stale_sample_rate(tmp_path: Path) -> None:
    """Residency is about the VALUE; a present-but-empty column must not vouch for metadata."""
    path = tmp_path / "a.wav"
    sf.write(path, torch.zeros(48000).numpy(), 48000)  # really 48 kHz
    stage = SampleRateFilterStage(allowed_sample_rates=[16000])

    for data in (
        {"audio_filepath": str(path), "sample_rate": 16000},  # no waveform column
        {"audio_filepath": str(path), "sample_rate": 16000, "waveform": None},  # column, no value
    ):
        task = AudioTask(dataset_name="d", data=dict(data))
        assert stage._observed_rate(task) == 48000, "the file header must win over stale metadata"
        assert not stage.process(task), "a 48 kHz file must not pass a 16 kHz-only filter"

    # A genuinely resident waveform still authenticates its own rate without a header read.
    resident = AudioTask(
        dataset_name="d",
        data={"audio_filepath": str(path), "sample_rate": 16000, "waveform": torch.zeros(1, 16000)},
    )
    assert stage._observed_rate(resident) == 16000


@pytest.mark.parametrize(
    "waveform",
    [
        torch.tensor([[32767, -32768], [0, 16384]], dtype=torch.int16),
        np.array([[2147483647, -2147483648], [0, 1073741824]], dtype=np.int32),
    ],
    ids=["torch_pcm16", "numpy_pcm32"],
)
@pytest.mark.parametrize("stage_kind", ["mono", "channel_count"])
def test_resident_pcm_stereo_is_normalized_before_downmix(waveform: object, stage_kind: str) -> None:
    if stage_kind == "mono":
        stage = MonoConversionStage(
            output_sample_rate=16000,
            input_residency="waveform",
            strict_sample_rate=True,
        )
    else:
        stage = ChannelCountStage(
            action="convert",
            target_channels=1,
            input_residency="waveform",
        )
    task = AudioTask(dataset_name="d", data={"waveform": waveform, "sample_rate": 16000})

    result = stage.process(task)

    assert isinstance(result, AudioTask)
    assert result.data["waveform"].dtype == torch.float32
    assert result.data["waveform"].shape == (1, 2)
    assert torch.isfinite(result.data["waveform"]).all()


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("allowed_sample_rates", "16000", TypeError),
        ("allowed_sample_rates", [16000, True], ValueError),
        ("allowed_sample_rates", [16000.5], ValueError),
        ("allowed_sample_rates", [0], ValueError),
        ("min_sample_rate", True, ValueError),
        ("min_sample_rate", -1, ValueError),
        ("max_sample_rate", 48000.0, ValueError),
    ],
)
def test_sample_rate_filter_rejects_invalid_config(
    field_name: str, value: object, error_type: type[Exception]
) -> None:
    with pytest.raises(error_type, match=field_name):
        SampleRateFilterStage(**{field_name: value})
