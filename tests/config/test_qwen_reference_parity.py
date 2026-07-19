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

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PROMPT_DIR = _ROOT / "examples/audio/qwen_omni_inprocess/prompts"
_REFERENCE_PROMPT_SHA256 = {
    "en_qwen3_omni_disfluency_asr.md": (
        "124c9c3fedd45ab23002ccdf47c1253848ff1b629bd0e6d22078ab4a54ac9a77"  # pragma: allowlist secret
    ),
    "en_qwen3_omni_reference_improvement.md": (
        "62514fcf26630556fd85b0851563338069c5b24acfa0862ddd91979f10d571dc"  # pragma: allowlist secret
    ),
    "ml_qwen3_omni_disfluency_asr.md": (
        "80748ef9c62170131ccf9884282dce59c2d0f447b85bb0ac855a7a987ef49b6c"  # pragma: allowlist secret
    ),
    "ml_qwen3_omni_reference_improvement.md": (
        "7dfc9128fdf3a8b4b5d3f509621885ca71f54b78335ddb613dc08f9a9025de9c"  # pragma: allowlist secret
    ),
}


def test_reference_prompt_assets_are_byte_exact() -> None:
    observed = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(_PROMPT_DIR.glob("*.md"))}
    assert observed == _REFERENCE_PROMPT_SHA256
