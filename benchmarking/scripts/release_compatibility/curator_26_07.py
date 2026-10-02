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

"""26.07 benchmark adapters; no changes to the installed Curator package."""

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def create_minhash_stage(stage_class: Callable[..., T], *, normalize_text: bool, **kwargs) -> T:
    # 26.07 lacks this keyword. Reject requested normalization rather than
    # silently changing the workload being compared with newer releases.
    if normalize_text:
        message = "The 26.07 compatibility profile requires normalize_text=False"
        raise ValueError(message)
    return stage_class(**kwargs)
