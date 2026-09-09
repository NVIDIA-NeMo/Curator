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

"""Checkpoint guards for stateful audio IO stages."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.io.convert import DocumentBatchJsonlWriterStage
from nemo_curator.stages.audio.io.extract_segments import SegmentExtractionStage
from nemo_curator.stages.audio.io.group_export import ManifestGroupExportStage

if TYPE_CHECKING:
    from pathlib import Path


def test_unsafe_io_stages_reject_checkpointing_before_touching_outputs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    cases = [
        (
            DocumentBatchJsonlWriterStage(output_path=str(tmp_path / "documents.jsonl")),
            tmp_path / "documents.jsonl",
        ),
        (
            ManifestGroupExportStage(output_dir=str(tmp_path / "groups")),
            tmp_path / "groups",
        ),
        (
            SegmentExtractionStage(output_dir=str(tmp_path / "segments")),
            tmp_path / "segments",
        ),
    ]

    for stage, output in cases:
        assert stage.is_resumable is False
        with pytest.raises(ValueError, match=stage.name):
            Pipeline(name="checkpoint_guard", stages=[stage]).run(checkpoint_path=checkpoint)
        assert not output.exists()

    assert not checkpoint.exists()
