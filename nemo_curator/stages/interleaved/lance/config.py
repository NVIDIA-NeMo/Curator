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

"""Shared Lance table configuration for interleaved stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LanceTableConfig:
    """Configuration for opening a pinned Lance table."""

    uri: str
    version: int | None = None
    storage_options: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.uri:
            msg = "Lance table URI must not be empty"
            raise ValueError(msg)
