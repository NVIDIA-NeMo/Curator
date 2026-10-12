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
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio.tts.studio_voice import (
    StudioVoiceEnhancementStage,
    StudioVoiceTritonClient,
    is_fatal_runtime_error,
)
from nemo_curator.tasks import AudioTask
from tests.stages.audio.tts.test_audio_io import make_tar_segments


class _FakeInput:
    def __init__(self, name: str, shape: tuple, dtype: str) -> None:
        self.name, self.shape, self.dtype = name, shape, dtype
        self.data: np.ndarray | None = None

    def set_data_from_numpy(self, data: np.ndarray) -> None:
        self.data = data


class _FakeServer:
    """Fake ``tritonclient.grpc.InferenceServerClient`` that halves every chunk."""

    def __init__(self, url: str, verbose: bool = False) -> None:
        self.url, self.verbose = url, verbose
        self.calls: list[dict] = []

    def is_server_ready(self) -> bool:
        return True

    def is_model_ready(self, model_name: str) -> bool:
        return model_name == "studio_voice_low_latency-48k"

    def infer(self, **kwargs: object) -> SimpleNamespace:
        data = kwargs["inputs"][0].data
        self.calls.append({**kwargs, "shape": data.shape})
        return SimpleNamespace(as_numpy=lambda _name: data * 0.5)


def _fake_grpc_module() -> SimpleNamespace:
    servers: list[_FakeServer] = []

    def make_server(url: str, verbose: bool = False) -> _FakeServer:
        servers.append(_FakeServer(url, verbose))
        return servers[-1]

    return SimpleNamespace(
        InferenceServerClient=make_server,
        InferInput=_FakeInput,
        InferRequestedOutput=lambda name: name,
        servers=servers,
    )


def test_client_streams_480_sample_chunks_and_trims() -> None:
    grpc = _fake_grpc_module()
    client = StudioVoiceTritonClient(target="host:1", grpcclient=grpc, wait_ready_sec=0)
    audio = np.linspace(-1, 1, 1000, dtype=np.float32)
    enhanced = client.enhance(audio)

    calls = grpc.servers[0].calls
    assert len(calls) == 3  # 1000 samples -> padded to 1440 = 3 x 480
    assert all(c["shape"] == (1, 480) for c in calls)
    assert [c["sequence_start"] for c in calls] == [True, False, False]
    assert all(c["sequence_end"] is False for c in calls)
    assert len({c["sequence_id"] for c in calls}) == 1
    np.testing.assert_allclose(enhanced, audio * 0.5)


def test_client_rejects_empty_and_wrong_rate() -> None:
    grpc = _fake_grpc_module()
    with pytest.raises(ValueError, match="48000"):
        StudioVoiceTritonClient(target="h:1", grpcclient=grpc, sample_rate=16000, wait_ready_sec=0)
    client = StudioVoiceTritonClient(target="h:1", grpcclient=grpc, wait_ready_sec=0)
    with pytest.raises(ValueError, match="empty"):
        client.enhance(np.zeros(0, dtype=np.float32))


def test_client_not_ready_raises() -> None:
    grpc = _fake_grpc_module()
    with pytest.raises(RuntimeError, match="did not become ready"):
        StudioVoiceTritonClient(target="h:1", grpcclient=grpc, model_name="other", wait_ready_sec=0)


def test_fatal_error_classification() -> None:
    assert is_fatal_runtime_error(RuntimeError("[StatusCode.UNAVAILABLE] failed to connect"))
    assert is_fatal_runtime_error(RuntimeError("Request for unknown model: x"))
    assert not is_fatal_runtime_error(ValueError("bad audio"))


class _ScaleClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.inputs: list[int] = []

    def enhance(self, samples: np.ndarray) -> np.ndarray:
        self.inputs.append(len(samples))
        if self.error is not None:
            raise self.error
        return samples * 0.5


def test_stage_writes_48k_wavs_and_annotations(tmp_path: Path) -> None:
    segments = make_tar_segments(tmp_path, [("vid_0.wav", 0.5, 16000), ("vid_1.wav", 0.25, 22050)])
    segments.append({"duration": 1.0})  # no audio address
    client = _ScaleClient()
    out_dir = tmp_path / "enh"
    stage = StudioVoiceEnhancementStage(output_dir=str(out_dir), client=client)
    stage.setup()
    task = stage.process(AudioTask(data={"orig_sample_id": "vid", "segments": segments}))

    first, second, third = (s["enhanced_audio"] for s in task.data["segments"])
    assert first["error"] is None
    assert first["sampling_rate"] == 48000
    assert first["duration"] == pytest.approx(0.5, abs=1e-3)
    assert Path(first["audio_filepath"]) == out_dir / "vid" / "vid_0.wav"
    info = sf.info(first["audio_filepath"])
    assert info.samplerate == 48000
    assert info.subtype == "PCM_16"
    assert second["duration"] == pytest.approx(0.25, abs=1e-3)
    assert third == {
        "audio_filepath": None,
        "duration": None,
        "sampling_rate": None,
        "method": "studio_voice",
        "error": "missing_tar_member",
    }
    assert client.inputs[0] == 24000
    assert abs(client.inputs[1] - 12000) <= 1  # 5512 samples @ 22.05 kHz -> 48 kHz
    calls_before = list(client.inputs)

    # resume: existing outputs are reused without calling the server
    stage.process(AudioTask(data={"orig_sample_id": "vid", "segments": segments[:2]}))
    assert client.inputs == calls_before


def test_stage_records_nonfatal_and_raises_fatal(tmp_path: Path) -> None:
    segments = make_tar_segments(tmp_path, [("a.wav", 0.1, 16000)])
    stage = StudioVoiceEnhancementStage(output_dir=str(tmp_path / "o"), client=_ScaleClient(ValueError("boom")))
    task = stage.process(AudioTask(data={"orig_sample_id": "s", "segments": segments}))
    annotation = task.data["segments"][0]["enhanced_audio"]
    assert annotation["error"] == "studio_voice_failed"
    assert "boom" in annotation["error_detail"]

    fatal = StudioVoiceEnhancementStage(
        output_dir=str(tmp_path / "o2"), client=_ScaleClient(RuntimeError("StatusCode.UNAVAILABLE"))
    )
    with pytest.raises(RuntimeError, match="UNAVAILABLE"):
        fatal.process(AudioTask(data={"orig_sample_id": "s", "segments": segments}))
