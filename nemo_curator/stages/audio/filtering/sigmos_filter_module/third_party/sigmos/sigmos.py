import os
import scipy
import librosa
import torch
import numpy as np
import onnxruntime as ort
from enum import Enum


__all__ = ["SigMOS", "Version"]


class Version(Enum):
    V1 = "v1"  # 15.10.2023


def is_gpu_support_available():
    if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
        return True
    else:
        return False


class SigMOS:
    '''
    MOS Estimator for the P.804 standard.
    See https://arxiv.org/pdf/2309.07385.pdf
    '''
    def __init__(self, model_dir, model_version=Version.V1, force_cpu=False, device_id=0, model_path=None):
        assert model_version in [v for v in Version]

        if model_path:
            model_file_path = model_path
        else:
            model_path_history = {
                Version.V1: os.path.join(model_dir, 'model-sigmos_1697718653_41d092e8-epo-200.onnx')
            }
            model_file_path = model_path_history[model_version]

        self.sampling_rate = 48_000
        self.resample_type = 'fft'
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

        use_gpu = True if is_gpu_support_available() and not force_cpu else False

        if not use_gpu:
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1

        if use_gpu:
            print(f"ort inference on cuda device_id {self.device_id}")
            ort_provider = ("CUDAExecutionProvider", {"device_id": str(self.device_id)})
            self.session = ort.InferenceSession(model_file_path, options,
                                                providers=[ort_provider])
            # Verify CUDA provider was actually loaded; fall back to CPU if not
            provider_options = self.session.get_provider_options()
            if "CUDAExecutionProvider" not in provider_options:
                print(
                    f"WARNING: CUDAExecutionProvider requested but not available at runtime "
                    f"(missing cuDNN or CUDA libraries). Falling back to CPUExecutionProvider."
                )
                self.session = ort.InferenceSession(model_file_path, options,
                                                    providers=["CPUExecutionProvider"])
        else:
            self.session = ort.InferenceSession(model_file_path, options,
                                                providers=["CPUExecutionProvider"])

    def stft(self, signal):
        last_frame = len(signal) % self.frame_size
        if last_frame == 0:
            last_frame = self.frame_size

        padded_signal = np.pad(signal, ((self.window_length - self.frame_size, self.window_length - last_frame),))
        frames = librosa.util.frame(padded_signal, frame_length=len(self.window), hop_length=self.frame_size, axis=0)
        spec = scipy.fft.rfft(frames * self.window, n=self.dft_size)
        return spec.astype(np.complex64)

    @staticmethod
    def compressed_mag_complex(x: np.ndarray, compress_factor=0.3):
        x = x.view(np.float32).reshape(x.shape + (2,)).swapaxes(-1, -2)
        x2 = np.maximum((x * x).sum(axis=-2, keepdims=True), 1e-12)
        if compress_factor == 1:
            mag = np.sqrt(x2)
        else:
            x = np.power(x2, (compress_factor - 1) / 2) * x
            mag = np.power(x2, compress_factor / 2)

        features = np.concatenate((mag, x), axis=-2)
        features = np.transpose(features, (1, 0, 2))
        return np.expand_dims(features, 0)

    def run(self, audio: np.ndarray, sr=None):
        if sr is not None and sr != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate, res_type=self.resample_type)
            # print(f"Audio file resampled from {sr} to {self.sampling_rate}!")

        features = self.stft(audio)
        features = self.compressed_mag_complex(features)

        onnx_inputs = {inp.name: features for inp in self.session.get_inputs()}
        output = self.session.run(None, onnx_inputs)[0][0]

        result = {
            'MOS_COL': float(output[0]), 'MOS_DISC': float(output[1]), 'MOS_LOUD': float(output[2]),
            'MOS_NOISE': float(output[3]), 'MOS_REVERB': float(output[4]), 'MOS_SIG': float(output[5]),
            'MOS_OVRL': float(output[6])
        }
        return result


def build_sigmos_model(force_cpu=False, device_id=0, model_path=None):
    model_dir = os.path.dirname(os.path.abspath(__file__))
    return SigMOS(model_dir=model_dir, force_cpu=force_cpu, device_id=device_id, model_path=model_path)


if __name__ == '__main__':
    ''' 
        Sample code to run the SigMOS estimator. 
        V1 (current model) is an alpha version and should be used in accordance.
    '''
    model_dir = r"."
    sigmos_estimator = SigMOS(model_dir=model_dir)

    # input data must have sr=48kHz, otherwise please specify the sr (it will be resampled to 48kHz internally)
    sampling_rate = 48_000
    dummy_data = np.random.rand(5 * sampling_rate)
    dummy_result = sigmos_estimator.run(dummy_data, sr=sampling_rate)
    print(dummy_result)
