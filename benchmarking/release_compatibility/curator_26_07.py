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

import logging
import os
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


def create_minhash_stage(stage_class: Callable[..., T], *, normalize_text: bool, **kwargs) -> T:
    # 26.07 lacks this keyword. Reject requested normalization rather than
    # silently changing the workload being compared with newer releases.
    if normalize_text:
        message = "The 26.07 compatibility profile requires normalize_text=False"
        raise ValueError(message)
    return stage_class(**kwargs)


def create_dedup_workflow(
    workflow_class: Callable[..., T], *, normalize_text: bool, use_async_memory: bool, **kwargs
) -> T:
    """Keep the workload unchanged while using the release's allocator implementation."""
    if normalize_text:
        message = "The 26.07 compatibility profile requires normalize_text=False"
        raise ValueError(message)
    logger.warning(
        "26.07 dedup uses the release's allocator; use_async_memory=%s is unavailable. "
        "Allocator differences must be considered when comparing performance.",
        use_async_memory,
    )
    return workflow_class(**kwargs)


def create_diarization_stage(stage_class: Callable[..., T], **kwargs) -> T:
    """Supply the old required auth argument without changing models or logging credentials."""
    return stage_class(hf_token=os.environ.get("HF_TOKEN"), **kwargs)
