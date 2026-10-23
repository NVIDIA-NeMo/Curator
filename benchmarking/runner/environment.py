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

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_SUBPROCESS_ENVIRONMENT = {
    # Enable Ray Data scheduler diagnostics for all benchmark subprocesses by
    # default so entry logs include scheduler events useful for post-run analysis.
    "NEMO_CURATOR_RAY_DATA_DIAGNOSTICS": "1",
}


def merge_subprocess_environment(configured_environment: Mapping[str, str]) -> dict[str, str]:
    """Return subprocess env with benchmark defaults, parent env, then config overrides."""
    return {**DEFAULT_SUBPROCESS_ENVIRONMENT, **os.environ, **configured_environment}
