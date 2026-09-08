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

``EmptyTask`` seeds a pipeline (the implicit root id ``"0"``). The resumability
layer adds two more markers on the same :class:`SentinelTask` base:

- ``NoneTask`` — this slot was intentionally filtered. The resumability counter
  treats it as a consumed branch (decrements). The adapter auto-wraps a
  returned ``None`` as a ``NoneTask``.
- ``FailedTask`` — this slot failed and should be retried on resume. The counter
  is NOT decremented, so its source stays pending and reruns.

All carry no payload (``data is None``) and get their ``task_id`` assigned by
the executor adapter; sentinels are stripped before the next stage. Construct
with ``EmptyTask()`` / ``NoneTask()`` / ``FailedTask()``.
"""

from dataclasses import dataclass, field

from nemo_curator.tasks.tasks import Task


@dataclass
class SentinelTask(Task[None]):
    """Base for payload-less marker tasks: no data, framework-assigned ``task_id``."""

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
    """Seeds a pipeline with ``task_id="0"`` — the implicit root every task
    descends from (so all ids share the ``"0"`` prefix)."""

    dataset_name: str = "empty"
    task_id: str = field(init=False, default="0")


@dataclass
class NoneTask(SentinelTask):
    """Marks a slot as intentionally filtered (resumability counter decrements)."""

    dataset_name: str = "none"


@dataclass
class FailedTask(SentinelTask):
    """Marks a slot as failed → retried on resume (counter does NOT decrement)."""

    dataset_name: str = "failed"
