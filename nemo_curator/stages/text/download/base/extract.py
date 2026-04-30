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

from abc import ABC, abstractmethod
from typing import Any

from nemo_curator.stages.resources import Resources


class DocumentExtractor(ABC):
    """Abstract base class for document extractors.

    Takes a record dict and returns one or more processed record dicts, or None to skip.
    Can transform any fields in the input dict.
    """

    resources = Resources(cpus=1.0)

    @abstractmethod
    def extract(self, record: dict[str, str]) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Extract/transform a record dict into one or more final record dicts."""
        ...

    @abstractmethod
    def input_columns(self) -> list[str]:
        """Define input columns - produces DocumentBatch with records."""
        ...

    @abstractmethod
    def output_columns(self) -> list[str]:
        """Define output columns - produces DocumentBatch with records."""
        ...

    def setup_on_node(self, *_args, **_kwargs) -> None:
        """Optional setup hook executed once per node."""
        del _args, _kwargs

    def setup(self, *_args, **_kwargs) -> None:
        """Optional setup hook executed once per worker."""
        del _args, _kwargs

    def teardown(self) -> None:
        """Optional teardown hook."""
        _teardown_completed = True
        del _teardown_completed

    def ray_stage_spec(self) -> dict[str, Any]:
        """Optional Ray configuration for iterate-extract stages."""
        return {}
