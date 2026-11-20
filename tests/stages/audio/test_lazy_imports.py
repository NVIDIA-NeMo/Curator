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
