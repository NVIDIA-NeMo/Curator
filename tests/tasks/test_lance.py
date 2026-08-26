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

from dataclasses import replace

from nemo_curator.tasks import LanceReadTask


def test_lance_read_task_deterministic_id() -> None:
    task = LanceReadTask(dataset_name="docs", path="s3://bucket/docs.lance", version=1, data=[1, 2])
    task_id = task.get_deterministic_id()

    assert task_id == replace(task).get_deterministic_id()
    assert task_id != replace(task, path="s3://bucket/other.lance").get_deterministic_id()
    assert task_id != replace(task, version=2).get_deterministic_id()
    assert task_id != replace(task, data=[1, 3]).get_deterministic_id()
