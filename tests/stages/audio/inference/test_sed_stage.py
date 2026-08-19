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

"""Stage-level tests for SEDInferenceStage (real stage, mocked CNN14 model).

Complements ``test_sed.py`` (which unit-tests the vendored model utilities in
isolation) by exercising the actual ``SEDInferenceStage.process_batch`` code
path: waveform extraction, mono-mixing, batching, valid-frame accounting, and
the task-data keys written for the downstream postprocessing stage.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from nemo_curator.stages.audio.inference.sed import SEDInferenceStage
from nemo_curator.tasks import AudioTask

CLASSES_NUM = 527
SAMPLE_RATE = 16000
HOP_SIZE = 320


class _MockCnn14(nn.Module):
    """Tiny stand-in returning random framewise output of the correct shape."""

    def __init__(self, classes_num: int = CLASSES_NUM, hop_size: int = HOP_SIZE) -> None:
        super().__init__()
        self.classes_num = classes_num
        self.hop_size = hop_size

    def forward(self, x: torch.Tensor, _mixup_lambda: object = None) -> dict[str, torch.Tensor]:
        batch, samples = x.shape
        frames = samples // self.hop_size + 1
        fw = torch.rand(batch, frames, self.classes_num)
        return {"framewise_output": fw, "clipwise_output": fw.mean(dim=1)}


def _make_stage(**kwargs: object) -> SEDInferenceStage:
    """Build a stage with the model injected (skips checkpoint loading in setup())."""
    stage = SEDInferenceStage(checkpoint_path="unused-in-tests", **kwargs)
    stage._model = _MockCnn14()
    stage._device = torch.device("cpu")
    return stage


def _audio_task(task_id: str, data: dict) -> AudioTask:
    return AudioTask(task_id=task_id, dataset_name="test", data=data)


class TestSEDInferenceStage:
    def test_writes_framewise_keys(self) -> None:
        stage = _make_stage()
        wav = np.random.default_rng(0).standard_normal(SAMPLE_RATE).astype(np.float32)
        task = _audio_task("t1", {"waveform": wav, "sample_rate": SAMPLE_RATE, "audio_filepath": "a.wav"})

        (out,) = stage.process_batch([task])

        assert out.data["_sed_framewise"].shape[1] == CLASSES_NUM
        assert out.data["_sed_framewise"].dtype == np.float16
        assert out.data["sed_fps"] == float(SAMPLE_RATE) / HOP_SIZE
        assert out.data["sed_valid_frames"] == 50  # ceil(16000 / 320)

    def test_float32_dtype_option(self) -> None:
        stage = _make_stage(framewise_dtype="float32")
        wav = np.zeros(SAMPLE_RATE, dtype=np.float32)
        task = _audio_task("t1", {"waveform": wav, "sample_rate": SAMPLE_RATE})

        (out,) = stage.process_batch([task])

        assert out.data["_sed_framewise"].dtype == np.float32

    def test_batches_multiple_tasks_of_different_length(self) -> None:
        stage = _make_stage()
        tasks = [
            _audio_task("t1", {"waveform": np.zeros(SAMPLE_RATE, dtype=np.float32), "sample_rate": SAMPLE_RATE}),
            _audio_task("t2", {"waveform": np.zeros(SAMPLE_RATE * 2, dtype=np.float32), "sample_rate": SAMPLE_RATE}),
        ]

        out = stage.process_batch(tasks)

        assert len(out) == 2
        assert out[0].data["sed_valid_frames"] == 50
        assert out[1].data["sed_valid_frames"] == 100

    def test_stereo_is_mono_mixed(self) -> None:
        stage = _make_stage()
        stereo = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
        task = _audio_task("t1", {"waveform": stereo, "sample_rate": SAMPLE_RATE})

        (out,) = stage.process_batch([task])

        assert out.data["_sed_framewise"].ndim == 2

    def test_skipme_flag_is_passed_through_untouched(self) -> None:
        stage = _make_stage()
        wav = np.zeros(SAMPLE_RATE, dtype=np.float32)
        task = _audio_task("t1", {"waveform": wav, "sample_rate": SAMPLE_RATE, "_skipme": True})

        (out,) = stage.process_batch([task])

        assert "_sed_framewise" not in out.data

    def test_missing_waveform_is_skipped(self) -> None:
        stage = _make_stage()
        task = _audio_task("t1", {"audio_filepath": "no-waveform.wav"})

        (out,) = stage.process_batch([task])

        assert "_sed_framewise" not in out.data

    def test_save_npz_writes_sidecar(self, tmp_path: object) -> None:
        stage = _make_stage(save_npz=True, output_dir=str(tmp_path))
        wav = np.zeros(SAMPLE_RATE, dtype=np.float32)
        task = _audio_task("t1", {"waveform": wav, "sample_rate": SAMPLE_RATE, "audio_filepath": "a.wav"})

        (out,) = stage.process_batch([task])

        npz_path = out.data["npz_filepath"]
        with np.load(npz_path) as d:
            assert d["framewise"].shape[1] == CLASSES_NUM
            assert d["framewise"].dtype == np.float16
            assert int(d["valid_frames"]) == 50
            # real sidecar schema (no phantom keys)
            assert set(d.files) == {"framewise", "fps", "audio_filepath", "original_num_samples", "valid_frames"}

    def test_short_audio_is_padded_and_processed(self) -> None:
        # 80 samples (0.005s) is far below min_input=max(1024, 320*32); the real
        # pad_short_segments branch must still yield valid framewise output.
        stage = _make_stage()
        wav = np.zeros(80, dtype=np.float32)
        task = _audio_task("t1", {"waveform": wav, "sample_rate": SAMPLE_RATE})

        (out,) = stage.process_batch([task])

        assert out.data["_sed_framewise"].shape[1] == CLASSES_NUM
        assert out.data["sed_valid_frames"] == 1  # ceil(80 / 320), clamped to true length

    def test_num_workers_is_configurable(self) -> None:
        assert SEDInferenceStage(checkpoint_path="x").num_workers() is None
        assert SEDInferenceStage(checkpoint_path="x", xenna_num_workers=4).num_workers() == 4
