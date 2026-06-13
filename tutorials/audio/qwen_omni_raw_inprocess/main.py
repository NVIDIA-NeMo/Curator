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

"""Raw-manifest entry point for the Qwen-Omni in-process pipeline.

The generic runner lives in ``qwen_omni_inprocess``. This sibling tutorial
exists so launchers can route raw JSONL audio manifests through the normal
audio download/preprocess path without changing the tarred-data tutorial.
"""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runner = Path(__file__).resolve().parents[1] / "qwen_omni_inprocess" / "main.py"
    runpy.run_path(str(runner), run_name="__main__")
