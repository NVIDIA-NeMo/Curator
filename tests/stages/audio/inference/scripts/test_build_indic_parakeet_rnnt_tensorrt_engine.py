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

import subprocess
import sys
from pathlib import Path

from nemo_curator.stages.audio.inference.scripts import build_indic_parakeet_rnnt_tensorrt_engine as builder
from nemo_curator.stages.audio.inference.scripts import tensorrt_encoder_utils


def test_builder_imports_packaged_encoder_utility() -> None:
    assert builder.build_encoder_bundle is tensorrt_encoder_utils.build_encoder_bundle


def test_builder_remains_directly_executable_from_a_source_checkout() -> None:
    completed = subprocess.run(  # noqa: S603 - executable and script are trusted test-environment paths
        [sys.executable, str(Path(builder.__file__).resolve()), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Build an FP16 TensorRT encoder bundle" in completed.stdout
