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

"""Importing a stage module must not load the heavy model stack it only needs at setup time."""

import subprocess
import sys

import pytest

HEAVY = ("whisperx", "pyannote.audio", "nemo.collections.asr", "nemo_text_processing", "onnxruntime")


@pytest.mark.parametrize(
    "module",
    [
        "nemo_curator.stages.audio.filtering",
        "nemo_curator.stages.audio.filtering.sigmos",
        "nemo_curator.stages.audio.inference.vad.whisperx_vad",
        "nemo_curator.stages.audio.inference.speaker_diarization.pyannote",
        "nemo_curator.stages.audio.metrics.wer",
        "nemo_curator.stages.audio.tagging.inference.nemo_asr_align",
        "nemo_curator.stages.audio.tagging.split",
    ],
)
def test_stage_module_import_is_light(module: str) -> None:
    code = f"import sys, {module}; print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603
    assert loaded == ""


_BLOCK = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == {blocked!r} or name.startswith({blocked!r} + "."):
            raise ImportError(f"No module named {{name!r}}")
        return None
sys.meta_path.insert(0, Block())
"""
_SIGMOS = "nemo_curator.stages.audio.filtering.sigmos_filter_module.third_party.sigmos.sigmos"


@pytest.mark.parametrize(
    ("blocked", "call", "extra"),
    [
        (
            "whisperx",
            "from nemo_curator.stages.audio.inference.vad.whisperx_vad import WhisperXVADModel; WhisperXVADModel(device='cpu')",
            "--extra audio_common",
        ),
        (
            "pyannote.audio",
            "from nemo_curator.stages.audio.inference.speaker_diarization.pyannote import _load_pyannote_pipeline; _load_pyannote_pipeline('m', None)",
            "--extra audio_common",
        ),
        (
            "nemo",
            "from nemo_curator.stages.audio.metrics.wer import word_error_rate_detail; word_error_rate_detail(['a'], ['a'])",
            "--extra audio_common",
        ),
        (
            "nemo_text_processing",
            "from nemo_curator.stages.audio.metrics.wer import ComputeWERStage; ComputeWERStage().setup()",
            "--extra audio_common",
        ),
        (
            "nemo",
            "from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import NeMoASRAlignerStage; NeMoASRAlignerStage().load_model()",
            "--extra audio_common",
        ),
        ("onnxruntime", f"from {_SIGMOS} import _onnxruntime_module; _onnxruntime_module()", "--extra audio_cpu"),
        ("librosa", f"from {_SIGMOS} import _librosa_module; _librosa_module()", "--extra audio_cpu"),
    ],
)
def test_missing_model_package_names_its_extra(blocked: str, call: str, extra: str) -> None:
    code = _BLOCK.format(blocked=blocked) + call
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert proc.returncode != 0
    assert proc.stderr.splitlines()[-1].startswith("ImportError:"), proc.stderr[-2000:]
    assert extra in proc.stderr.splitlines()[-1]
