# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

from nemo_curator.stages.audio._agent._composite import expand_composites
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.audio.common import (
    ManifestReader,
)


def test_nested_composite_is_reported_as_unrunnable(monkeypatch) -> None:  # noqa: ANN001
    """A shape rejected by Pipeline must not be downgraded to opaque."""
    stage = ManifestReader("manifest.jsonl")
    nested_children = [ManifestReader("one.jsonl"), ManifestReader("two.jsonl")]
    monkeypatch.setattr(stage, "decompose_and_apply_with", lambda: nested_children)

    expansion = expand_composites([stage])
    assert expansion.stages == []
    assert 0 not in expansion.opaque
    assert "nested composition" in expansion.unrunnable[0]

    report = validate_pipeline([stage])
    assert not report.ok
    assert any(issue.code == "composite_unrunnable" for issue in report.issues)
