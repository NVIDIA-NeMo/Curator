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

from dataclasses import dataclass, field
from typing import TypeAlias

from nemo_curator.utils.hash_utils import get_deterministic_hash

from .tasks import Task

FragmentIds: TypeAlias = list[int]


@dataclass
class LanceReadTask(Task[FragmentIds]):
    """Task containing Lance fragment ids assigned to one read partition.

    Args:
        path: Path or URI of the Lance dataset to read.
        version: Lance dataset version to read.
        data: Lance fragment ids to read.
    """

    path: str = field(kw_only=True)
    version: int = field(kw_only=True)
    data: FragmentIds = field(default_factory=list)

    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return bool(self.data)

    def get_deterministic_id(self) -> str:
        parts = [
            self.path,
            str(self.version),
            *(str(fragment_id) for fragment_id in self.data),
        ]
        return get_deterministic_hash(parts)
