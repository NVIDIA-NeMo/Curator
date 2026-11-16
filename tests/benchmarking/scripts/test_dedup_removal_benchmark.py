# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking/scripts"))
from dedup_removal_benchmark import MeasuredRemovalWorkflow
from text_metrics import collect_text_metrics

from tests.benchmarking.scripts.test_modifier_benchmark import SerialExecutor


@pytest.fixture(scope="session")
def shared_ray_cluster() -> None:
    """Explicit IDs allow a real removal workflow without a Ray ID registry."""


def test_removal_accounts_for_a_fully_removed_batch(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    pq.write_table(pa.table({"id": [1, 2], "text": ["é", "keep"]}), source / "a.parquet")
    pq.write_table(pa.table({"id": [3], "text": ["bad"]}), source / "b.parquet")
    ids = tmp_path / "ids.parquet"
    pq.write_table(pa.table({"id": [1, 3]}), ids)
    workflow = MeasuredRemovalWorkflow(
        input_path=str(source),
        ids_to_remove_path=str(ids),
        output_path=str(tmp_path / "output"),
        id_field="id",
        input_filetype="parquet",
        input_files_per_partition=1,
    )
    results = workflow.run(SerialExecutor())
    tasks = results.pipeline_tasks["removal"]
    assert collect_text_metrics(tasks) == {
        "num_input_documents": 3,
        "num_output_documents": 1,
        "input_text_bytes": 9,
        "output_text_bytes": 4,
    }
    assert sum(pq.read_table(task.data[0]).num_rows for task in tasks) == 1
