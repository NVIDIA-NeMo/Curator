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
"""Payload-less marker tasks.

``EmptyTask`` seeds a pipeline (the implicit root id ``"0"``). All markers
share the :class:`SentinelTask` base and carry no payload (``data is None``).
Construct one with ``EmptyTask()``.
"""

from dataclasses import dataclass, field

from nemo_curator.tasks.tasks import Task


@dataclass
class SentinelTask(Task[None]):
    """Base for payload-less marker tasks. Always carries no data; ``task_id``
    is framework-assigned like any other task."""

    data: None = None

    def __post_init__(self) -> None:
        assert self.data is None, "SentinelTask carries no data"  # noqa: S101
        super().__post_init__()

    @property
    def num_items(self) -> int:
        return 0

    def validate(self) -> bool:
        return True


@dataclass
class EmptyTask(SentinelTask):
    """Payload-less task that seeds a pipeline. Its ``task_id`` is fixed to
    ``"0"`` — the implicit root every task in a run descends from, so all
    ``task_id``s share the ``"0"`` prefix (source partitions become
    ``"0_<id>"``, user-provided initial tasks become ``"0_0"``, ``"0_1"``, …).
    """

    dataset_name: str = "empty"
    task_id: str = field(init=False, default="0")
