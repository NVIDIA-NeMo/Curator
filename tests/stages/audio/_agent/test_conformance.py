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

from typing import TYPE_CHECKING

import pandas as pd
import pytest
import soundfile as sf
import torch

from nemo_curator.stages.audio._agent._agent_ready import (
    AgentReady,
    IOSpec,
    StageContract,
)
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio.common import (
    ManifestReaderStage,
    PreserveByValueStage,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, DocumentBatch, FileGroupTask

if TYPE_CHECKING:
    from pathlib import Path


def test_fanout_conformance_checks_every_emitted_result(tmp_path: Path) -> None:
    """A later fan-out row cannot omit a write that only the first row carries."""
    manifest = tmp_path / "mixed.jsonl"
    manifest.write_text(
        '{"recording_path": "/tmp/a.wav"}\n{"text": "missing the declared recording_path"}\n',
        encoding="utf-8",
    )
    reader = ManifestReaderStage(include_files_key="recording_path")

    with pytest.raises(AssertionError, match="missing from result 1"):
        assert_agent_ready(
            reader,
            lambda: FileGroupTask(dataset_name="d", data=[str(manifest)]),
            expected_cardinality="1:N fan-out",
            available_keys=set(),
        )


class _DataFrameFanInStage(AgentReady, ProcessingStage[AudioTask, DocumentBatch]):
    """Small stand-in for the full agent branch's AudioToDocumentStage."""

    BATCH_ONLY = True
    name = "dataframe_fan_in"

    def describe(self) -> StageContract:
        return StageContract(
            writes=IOSpec(data_keys=["text"]),
            cardinality="N:1",
        )

    def process(self, _task: AudioTask) -> DocumentBatch:
        raise NotImplementedError

    def process_batch(self, tasks: list[AudioTask]) -> list[DocumentBatch]:
        return [
            DocumentBatch(
                dataset_name=tasks[0].dataset_name,
                data=pd.DataFrame([{"text": task.data["text"]} for task in tasks]),
            )
        ]


def test_n_to_one_conformance_reads_document_batch_columns() -> None:
    """Declared N:1 writes are DataFrame columns in a DocumentBatch, not dict keys."""
    assert_agent_ready(
        _DataFrameFanInStage(),
        lambda: [
            AudioTask(dataset_name="d", data={"text": "one"}),
            AudioTask(dataset_name="d", data={"text": "two"}),
        ],
        expected_cardinality="N:1",
        available_keys={"text"},
    )


def test_conformance_requires_exact_literal_key_for_unknown_role_read(tmp_path: Path) -> None:
    """A custom (unknown-role) read must be satisfied by its exact key, not waved through."""
    wav = tmp_path / "a.wav"
    sf.write(wav, torch.zeros(48000).numpy(), 48000)
    selector = PreserveByValueStage("mos", 3.0, "ge")  # input_value_key='mos' -> unknown role

    # 'mos' is not among the available keys, so the unknown-role read is unsatisfied.
    with pytest.raises(AssertionError, match="not satisfied"):
        assert_agent_ready(selector, available_keys={"audio_filepath"}, run=False)

    # Present exactly, it passes.
    assert_agent_ready(selector, available_keys={"mos"}, run=False)
