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

"""Internal helper to install and probe audio dependencies without reinstalling Curator.

Invoked by install_audio.sh after FFmpeg preparation. Requirements come from
the installer's own Curator checkout, including when used as the documented
fallback for older images. Existing inference-stack versions are constraints,
not permission to upgrade the Curator-under-test.
"""

import argparse
import contextlib
import shutil
import subprocess
import sys
import tempfile
import tomllib
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

# This file is a standalone installer helper, not a Python package.
# ruff: noqa: INP001

# https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions
TORCHCODEC_SERIES = {(2, 8): "0.7", (2, 9): "0.9", (2, 10): "0.10", (2, 11): "0.11"}


def audio_requirements() -> list[Requirement]:
    """Read this checkout's platform-applicable audio requirements, excluding Curator itself."""
    # Selecting CPU/CUDA requirements needs the installed Torch, not host autodetection.
    import torch

    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]
    requirements = []
    for extra in ("audio_common", "audio_cuda12" if torch.version.cuda else "audio_cpu"):
        for text in extras[extra]:
            requirement = Requirement(text)
            if requirement.name.replace("_", "-").lower() == "nemo-curator":
                continue
            if requirement.marker is None or requirement.marker.evaluate():
                requirements.append(requirement)
    return requirements


def check_audio() -> None:
    """Check real imports and local decoding without downloads or GPU inference."""
    if shutil.which("ffmpeg") is None:
        msg = "ffmpeg executable not found on PATH (required by ResampleAudioStage)."
        raise RuntimeError(msg)

    # Optional runtime imports must run only inside the environment being checked.
    import numpy as np
    import onnx
    import onnxruntime as ort
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import ASRModel  # noqa: F401
    from nemo_text_processing.text_normalization.normalize import Normalizer
    from pyannote.audio.core.io import Audio
    from torchcodec.decoders import AudioDecoder
    from whisperx.vads.pyannote import load_vad_model  # noqa: F401

    from nemo_curator.stages.audio.datasets.readspeech import CreateInitialManifestReadSpeechStage  # noqa: F401
    from nemo_curator.stages.audio.metrics.wer import ComputeWERStage  # noqa: F401

    if Normalizer(input_case="cased", lang="en").normalize("2", verbose=False) != "two":
        msg = "NeMo text normalization failed its number-rewrite check."
        raise RuntimeError(msg)

    for requirement in audio_requirements():
        installed = version(requirement.name)
        if not requirement.specifier.contains(installed, prereleases=True):
            msg = f"{requirement.name} {installed} does not satisfy {requirement.specifier}"
            raise RuntimeError(msg)

    with tempfile.TemporaryDirectory() as directory:
        sample_rate_hz = 16000
        samples = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
        for suffix, subtype in (("flac", "PCM_16"), ("wav", "PCM_16"), ("wav", "FLOAT")):
            path = Path(directory) / f"{subtype}.{suffix}"
            sf.write(path, samples, sample_rate_hz, subtype=subtype)
            expected, _ = sf.read(path, dtype="float32", always_2d=True)
            decoded = AudioDecoder(str(path)).get_all_samples()
            waveform, sample_rate = Audio(sample_rate=sample_rate_hz)(str(path))
            if decoded.sample_rate != sample_rate_hz or sample_rate != sample_rate_hz:
                msg = f"Unexpected sample rate decoding {path.name}"
                raise RuntimeError(msg)
            if waveform.shape != (1, len(samples)) or decoded.data.shape != waveform.shape:
                msg = f"Unexpected sample shape decoding {path.name}"
                raise RuntimeError(msg)
            torch.testing.assert_close(waveform, decoded.data)
            torch.testing.assert_close(decoded.data, torch.from_numpy(expected.T), rtol=0, atol=1e-6)

    # SIGMOS uses ONNX Runtime; a CPU session also validates the ARM fallback.
    tensor = onnx.helper.make_tensor_value_info("audio", onnx.TensorProto.FLOAT, [1])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["audio"], ["output"])], "probe", [tensor], [output]
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=10)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(session.run(None, {"audio": np.ones(1, dtype=np.float32)})[0], [1])
    print("Audio imports, text normalization, ONNX Runtime CPU inference, and FLAC/PCM WAV decoding passed.")


def installed_constraints(configured_overrides: list[str]) -> tuple[list[str], list[str]]:
    """Protect the existing inference stack and retain its applicable uv overrides."""
    constraints = []
    overrides = []
    for name in (
        "nemo-curator",
        "torch",
        "torchaudio",
        "torchvision",
        "numba",
        "fsspec",
        "huggingface-hub",
        "transformers",
    ):
        with contextlib.suppress(PackageNotFoundError):
            constraints.append(f"{name}=={version(name)}")
            if name in {"torch", "torchaudio", "torchvision"}:
                overrides.append(f"{name}=={version(name)}")
    # Preserve exceptions already used to build this Curator, without adopting
    # a newer checkout's overrides when they would change the installed stack.
    for text in configured_overrides:
        requirement = Requirement(text)
        if requirement.name in {"torch", "torchaudio", "torchvision", "torchcodec"}:
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        with contextlib.suppress(PackageNotFoundError):
            installed = version(requirement.name)
            if requirement.specifier.contains(installed, prereleases=True):
                overrides.append(f"{requirement.name}=={installed}")
    with contextlib.suppress(PackageNotFoundError):
        # Honor the installed Numba's NumPy bounds rather than pinning one image's NumPy.
        for text in distribution("numba").requires or []:
            requirement = Requirement(text)
            if requirement.name == "numpy" and (requirement.marker is None or requirement.marker.evaluate()):
                constraints.append(str(requirement))
    return constraints, overrides


def install_audio() -> None:
    """Resolve audio extras against the installed Torch/CUDA and NumPy constraints."""
    # Import Torch here so command parsing does not initialize an optional runtime.
    import torch

    torch_version = Version(torch.__version__)
    series = TORCHCODEC_SERIES.get(torch_version.release[:2])
    if series is None:
        msg = f"No verified TorchCodec series for Torch {torch.__version__}; consult the upstream compatibility table."
        raise RuntimeError(msg)
    codec = f"torchcodec~={series}.0"
    flavor = "cu" + torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"
    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    requirements = [str(requirement) for requirement in audio_requirements() if requirement.name != "torchcodec"]

    constraints, overrides = installed_constraints(
        project.get("tool", {}).get("uv", {}).get("override-dependencies", [])
    )
    constraints.append(codec)
    overrides.append(codec)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "constraints.txt"
        path.write_text("\n".join(constraints) + "\n")
        override_path = Path(directory) / "overrides.txt"
        override_path.write_text("\n".join(overrides) + "\n")
        uv = shutil.which("uv")
        if uv is None:
            msg = "uv is required to preserve Curator's existing dependency overrides during audio installation."
            raise RuntimeError(msg)
        # An installed wrong-CUDA wheel can satisfy the series; replace it from the matching index.
        subprocess.run(  # noqa: S603
            [
                uv,
                "--no-config",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                "--index-url",
                f"https://download.pytorch.org/whl/{flavor}",
                "--reinstall-package",
                "torchcodec",
                codec,
            ],
            check=True,
        )
        subprocess.run(  # noqa: S603
            [
                uv,
                "--no-config",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--constraint",
                str(path),
                "--override",
                str(override_path),
                *requirements,
            ],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        if not args.check:
            install_audio()
        check_audio()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: audio environment: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
