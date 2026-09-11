# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemo_curator.pipeline.payload_lifecycle import cleanup_stage_run_resources


class _CleanableStage:
    def __init__(self, name: str, calls: list[str], *, fails: bool = False) -> None:
        self.name = name
        self._calls = calls
        self._fails = fails

    def cleanup_run_resources(self) -> None:
        self._calls.append(self.name)
        if self._fails:
            msg = "cleanup exploded"
            raise RuntimeError(msg)


def test_executor_cleans_up_stages_in_reverse_and_survives_failures() -> None:
    calls: list[str] = []
    stages = [
        _CleanableStage("materialize", calls),
        _CleanableStage("consumer", calls, fails=True),
        object(),
    ]

    cleanup_stage_run_resources(stages)

    assert calls == ["consumer", "materialize"]
