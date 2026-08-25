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

"""Tests for the opt-in DocumentBatch JSONL sink."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from fsspec.core import url_to_fs

from nemo_curator.stages.audio._agent._agent_registry import build_contract, static_contract
from nemo_curator.stages.audio.io.convert import DocumentBatchJsonlWriterStage
from nemo_curator.tasks import DocumentBatch


def _batch(rows: list[dict], *, dataset_name: str = "audio") -> DocumentBatch:
    return DocumentBatch(
        dataset_name=dataset_name,
        data=pd.DataFrame(rows),
        _metadata={"trace_id": "kept"},
        _stage_perf=["seed"],
    )


def test_writes_one_jsonl_line_per_document_row_and_preserves_task(tmp_path) -> None:  # noqa: ANN001
    output = tmp_path / "curated.jsonl"
    stage = DocumentBatchJsonlWriterStage(output_path=str(output))
    first = _batch([{"id": 1, "text": "héllo"}, {"id": 2, "text": "world"}])
    second = _batch([{"id": 3, "text": "again"}])

    stage.setup()
    returned = stage.process(first)
    stage.process(second)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"id": 1, "text": "héllo"},
        {"id": 2, "text": "world"},
        {"id": 3, "text": "again"},
    ]
    assert returned is first
    assert returned.dataset_name == "audio"
    assert returned._metadata == {"trace_id": "kept"}
    assert returned._stage_perf == ["seed"]


def test_setup_truncates_only_between_runs(tmp_path) -> None:  # noqa: ANN001
    output = tmp_path / "curated.jsonl"
    stage = DocumentBatchJsonlWriterStage(output_path=str(output))
    batch = _batch([{"id": 1}])

    stage.setup()
    stage.process(batch)
    stage.process(batch)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2

    stage.setup()
    stage.process(batch)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_supports_fsspec_cloud_paths(tmp_path) -> None:  # noqa: ANN001
    output = f"memory://document-writer/{tmp_path.parent.name}/{tmp_path.name}/curated.jsonl"
    stage = DocumentBatchJsonlWriterStage(output_path=output)

    stage.setup()
    stage.process(_batch([{"id": 1, "text": "café"}, {"id": 2, "text": "茶"}]))

    filesystem, path = url_to_fs(output)
    rows = [json.loads(line) for line in filesystem.cat(path).decode("utf-8").splitlines()]
    assert rows == [{"id": 1, "text": "café"}, {"id": 2, "text": "茶"}]


def test_contract_is_document_batch_only_and_declares_disk_lifecycle(tmp_path) -> None:  # noqa: ANN001
    stage = DocumentBatchJsonlWriterStage(output_path=str(tmp_path / "out.jsonl"))
    dynamic = build_contract(stage)
    static = static_contract(DocumentBatchJsonlWriterStage)

    for contract in (dynamic, static):
        assert contract.accepts_task_type == "DocumentBatch"
        assert contract.produces_task_type == "DocumentBatch"
        assert contract.gates.writes_to_disk is True
        assert contract.gates.lifecycle_side_effects is True
        assert contract.gates.requires_serializable_input is True
    assert stage.num_workers() == 1


def test_rejects_an_empty_output_path() -> None:
    with pytest.raises(ValueError, match="output_path is required"):
        DocumentBatchJsonlWriterStage(output_path="")
