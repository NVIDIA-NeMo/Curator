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
"""Unit tests for source/sink role assignment in ``Pipeline.build``."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import Task


@dataclass
class _NoopStage(ProcessingStage[Task, Task]):
    name: str = "noop"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: Task) -> Task:
        return task


class TestSourceSinkRoleAssignment:
    def test_defaults_first_stage_to_source(self) -> None:
        s0 = _NoopStage(name="s0")
        s1 = _NoopStage(name="s1")
        s2 = _NoopStage(name="s2")
        Pipeline(name="t", stages=[s0, s1, s2]).build()
        assert s0.is_source_stage is True
        assert s1.is_source_stage is False
        assert s2.is_source_stage is False

    def test_defaults_last_stage_to_sink(self) -> None:
        s0 = _NoopStage(name="s0")
        s1 = _NoopStage(name="s1")
        s2 = _NoopStage(name="s2")
        Pipeline(name="t", stages=[s0, s1, s2]).build()
        assert s2.is_sink_stage is True
        assert s0.is_sink_stage is False
        assert s1.is_sink_stage is False

    def test_explicit_source_overrides_default(self) -> None:
        s0 = _NoopStage(name="s0")
        s1 = _NoopStage(name="s1")
        s1.is_source_stage = True
        Pipeline(name="t", stages=[s0, s1]).build()
        assert s0.is_source_stage is False
        assert s1.is_source_stage is True

    def test_explicit_sink_overrides_default(self) -> None:
        s0 = _NoopStage(name="s0")
        s1 = _NoopStage(name="s1")
        s2 = _NoopStage(name="s2")
        s1.is_sink_stage = True
        Pipeline(name="t", stages=[s0, s1, s2]).build()
        assert s1.is_sink_stage is True
        assert s2.is_sink_stage is False

    def test_multiple_explicit_source_stages_raises(self) -> None:
        s0 = _NoopStage(name="s0")
        s1 = _NoopStage(name="s1")
        s0.is_source_stage = True
        s1.is_source_stage = True
        with pytest.raises(ValueError, match="multiple source stages marked"):
            Pipeline(name="t", stages=[s0, s1]).build()

    def test_multiple_explicit_sink_stages_raises(self) -> None:
        s0 = _NoopStage(name="s0")
        s1 = _NoopStage(name="s1")
        s0.is_sink_stage = True
        s1.is_sink_stage = True
        with pytest.raises(ValueError, match="multiple sink stages marked"):
            Pipeline(name="t", stages=[s0, s1]).build()

    def test_single_stage_is_both_source_and_sink(self) -> None:
        s0 = _NoopStage(name="s0")
        Pipeline(name="t", stages=[s0]).build()
        assert s0.is_source_stage is True
        assert s0.is_sink_stage is True
