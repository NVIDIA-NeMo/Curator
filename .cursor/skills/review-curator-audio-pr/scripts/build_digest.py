#!/usr/bin/env python3
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
"""Audio wrapper around the generic review-curator-pr digest builder."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

AUDIO_PATH_REGEX = (
    r"^(nemo_curator/stages/audio/|nemo_curator/tasks/audio_task\.py|"
    r"tutorials/audio/|tests/stages/audio/|tests/tasks/test_audio|"
    r"benchmarking/.*([Aa]udio|ALM|alm))"
)

generic = Path(__file__).resolve().parents[2] / "review-curator-pr" / "scripts" / "build_digest.py"
sys.argv[1:1] = ["--path-regex", AUDIO_PATH_REGEX, "--modality-label", "audio"]
runpy.run_path(str(generic), run_name="__main__")
