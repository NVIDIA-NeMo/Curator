# MIT License
#
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

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

import os
from enum import Enum

import librosa
import numpy as np
import onnxruntime as ort
import scipy
from loguru import logger

__all__ = ["SigMOS", "Version"]


class Version(Enum):
    V1 = "v1"  # 15.10.2023


class SigMOS:
    """
    MOS Estimator for the P.804 standard.
    See https://arxiv.org/pdf/2309.07385.pdf
    """

    def __init__(
        self,
        model_dir: str,
        model_version: Version = Version.V1,
        force_cpu: bool = False,
        device_id: int = 0,
        model_path: str | None = None,
    ):
        if model_version not in Version:
            msg = f"model_version must be a Version enum member, got {model_version!r}"
            raise ValueError(msg)

        if model_path:
            model_file_path = model_path
        else:
            model_path_history = {Version.V1: os.path.join(model_dir, "model-sigmos_1697718653_41d092e8-epo-200.onnx")}
            model_file_path = model_path_history[model_version]

        self.sampling_rate = 48_000
        self.resample_type = "fft"
        self.model_version = model_version
        self.device_id = device_id

        # STFT params
        self.dft_size = 960
        self.frame_size = 480
        self.window_length = 960
        self.window = np.sqrt(np.hanning(int(self.window_length) + 1)[:-1]).astype(np.float32)

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1

        use_gpu = not force_cpu

        if use_gpu:
            logger.info("SIGMOS ort inference on cuda device_id %s", self.device_id)
            ort_provider = ("CUDAExecutionProvider", {"device_id": str(self.device_id)})
            self.session = ort.InferenceSession(model_file_path, options, providers=[ort_provider])
            provider_options = self.session.get_provider_options()
            if "CUDAExecutionProvider" not in provider_options:
                logger.warning(
                    "CUDAExecutionProvider requested but not available at runtime "
                    "(missing cuDNN or CUDA libraries). Falling back to CPUExecutionProvider."
                )
                self.session = ort.InferenceSession(model_file_path, options, providers=["CPUExecutionProvider"])
        else:
            self.session = ort.InferenceSession(model_file_path, options, providers=["CPUExecutionProvider"])

    def stft(self, signal: np.ndarray) -> np.ndarray:
        last_frame = len(signal) % self.frame_size
        if last_frame == 0:
            last_frame = self.frame_size

        padded_signal = np.pad(signal, ((self.window_length - self.frame_size, self.window_length - last_frame),))
        frames = librosa.util.frame(padded_signal, frame_length=len(self.window), hop_length=self.frame_size, axis=0)
        spec = scipy.fft.rfft(frames * self.window, n=self.dft_size)
        return spec.astype(np.complex64)

    @staticmethod
    def compressed_mag_complex(x: np.ndarray, compress_factor: float = 0.3) -> np.ndarray:
        x = x.view(np.float32).reshape((*x.shape, 2)).swapaxes(-1, -2)
        x2 = np.maximum((x * x).sum(axis=-2, keepdims=True), 1e-12)
        if compress_factor == 1:
            mag = np.sqrt(x2)
        else:
            x = np.power(x2, (compress_factor - 1) / 2) * x
            mag = np.power(x2, compress_factor / 2)

        features = np.concatenate((mag, x), axis=-2)
        features = np.transpose(features, (1, 0, 2))
        return np.expand_dims(features, 0)

    def run(self, audio: np.ndarray, sr: int | None = None) -> dict[str, float]:
        if sr is not None and sr != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate, res_type=self.resample_type)

        features = self.stft(audio)
        features = self.compressed_mag_complex(features)

        onnx_inputs = {inp.name: features for inp in self.session.get_inputs()}
        output = self.session.run(None, onnx_inputs)[0][0]

        return {
            "MOS_COL": float(output[0]),
            "MOS_DISC": float(output[1]),
            "MOS_LOUD": float(output[2]),
            "MOS_NOISE": float(output[3]),
            "MOS_REVERB": float(output[4]),
            "MOS_SIG": float(output[5]),
            "MOS_OVRL": float(output[6]),
        }


def build_sigmos_model(force_cpu: bool = False, device_id: int = 0, model_path: str | None = None) -> "SigMOS":
    model_dir = os.path.dirname(os.path.abspath(__file__))
    return SigMOS(model_dir=model_dir, force_cpu=force_cpu, device_id=device_id, model_path=model_path)
