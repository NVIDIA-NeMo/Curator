# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking/scripts"))
from text_metrics import MeasureText, collect_text_metrics, count_registry_documents

from nemo_curator.tasks import DocumentBatch, FileGroupTask


@pytest.fixture(scope="session")
def shared_ray_cluster() -> None:
    """These measurement and registry checks do not use a scheduler."""


@pytest.mark.parametrize("arrow", [False, True])
def test_fully_filtered_batch_retains_input_volume_after_serialization(arrow: bool) -> None:
    data = {"body": ["é", "🙂", ""]}
    task = DocumentBatch(dataset_name="test", data=pa.table(data) if arrow else pd.DataFrame(data))
    MeasureText("input", "body").process(task)
    task.data = task.data.slice(0, 0) if arrow else task.data.iloc[:0]
    MeasureText("output", "body").process(task)
    result = FileGroupTask(dataset_name="output", data=["empty.parquet"], _metadata=task._metadata)
    result = pickle.loads(pickle.dumps(result))  # noqa: S301
    assert collect_text_metrics([result]) == {
        "num_input_documents": 3,
        "num_output_documents": 0,
        "input_text_bytes": 6,
        "output_text_bytes": 0,
    }
    with pytest.raises(ValueError, match="changed the document count"):
        collect_text_metrics([result], preserve_document_count=True)


def test_missing_measurement_is_not_zero() -> None:
    with pytest.raises(ValueError, match="No completed"):
        collect_text_metrics([])
    with pytest.raises(KeyError):
        collect_text_metrics([FileGroupTask(dataset_name="output", data=["unknown.parquet"])])


def test_document_denominator_uses_inclusive_registry_ranges(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"next_id": 109, "batch_registry": {"file1": [100, 102], "file2": [103, 108]}}))
    assert count_registry_documents(str(registry)) == 9
    assert count_registry_documents(None) is None


@pytest.mark.parametrize("ranges", [{"a": [0, 2], "b": [2, 4]}, {"a": [4, 1]}, {"a": ["0", 2]}])
def test_corrupt_registry_cannot_produce_document_throughput(tmp_path: Path, ranges: dict) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"next_id": 5, "batch_registry": ranges}))
    with pytest.raises((TypeError, ValueError)):
        count_registry_documents(str(registry))
