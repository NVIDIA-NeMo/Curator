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
"""Unit tests for the payload-less sentinel tasks: the ``SentinelTask`` base,
``EmptyTask`` (pipeline seed, ``task_id="0"``), and the ``NoneTask`` /
``FailedTask`` resumability markers (framework-assigned ``task_id``).
"""

from __future__ import annotations

import pytest

from nemo_curator.tasks import EmptyTask, FailedTask, NoneTask, SentinelTask, Task


class TestSentinelBase:
    def test_subclasses_are_tasks(self) -> None:
        for obj in (SentinelTask(dataset_name="s"), EmptyTask(), NoneTask(), FailedTask()):
            assert isinstance(obj, Task)
            assert isinstance(obj, SentinelTask)

    def test_carry_no_data(self) -> None:
        for obj in (SentinelTask(dataset_name="s"), EmptyTask(), NoneTask(), FailedTask()):
            assert obj.data is None

    def test_num_items_is_zero(self) -> None:
        for obj in (SentinelTask(dataset_name="s"), EmptyTask(), NoneTask(), FailedTask()):
            assert obj.num_items == 0

    def test_validate_is_true(self) -> None:
        for obj in (SentinelTask(dataset_name="s"), EmptyTask(), NoneTask(), FailedTask()):
            assert obj.validate() is True

    def test_rejects_payload(self) -> None:
        # The base asserts ``data is None`` so a sentinel can never carry data.
        with pytest.raises(AssertionError):
            SentinelTask(dataset_name="s", data="oops")


class TestEmptyTask:
    def test_is_rooted_at_zero(self) -> None:
        # EmptyTask is the implicit root every task descends from.
        assert EmptyTask().task_id == "0"
        assert EmptyTask().dataset_name == "empty"

    def test_task_id_is_not_user_settable(self) -> None:
        # ``task_id`` is init=False, so it cannot be passed positionally/kw.
        with pytest.raises(TypeError):
            EmptyTask(task_id="5")  # type: ignore[call-arg]


class TestResumabilityMarkers:
    def test_dataset_names(self) -> None:
        assert NoneTask().dataset_name == "none"
        assert FailedTask().dataset_name == "failed"

    def test_task_id_unset_until_assigned(self) -> None:
        # Unlike EmptyTask, these get their id from the adapter; default empty.
        assert NoneTask().task_id == ""
        assert FailedTask().task_id == ""

    def test_none_and_failed_are_distinct(self) -> None:
        assert not isinstance(NoneTask(), FailedTask)
        assert not isinstance(FailedTask(), NoneTask)
