# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking/scripts"))
from dedup_removal_benchmark import MeasuredRemovalWorkflow, run_removal_benchmark
from text_metrics import collect_text_metrics

from nemo_curator.pipeline.workflow import WorkflowRunResult
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.performance_utils import StagePerfStats
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


@pytest.mark.parametrize("measurement_seconds", [0.0, 1000.0])
def test_io_percentage_preserves_original_stage_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, measurement_seconds: float
) -> None:
    from nemo_curator.backends import xenna

    task = DocumentBatch(dataset_name="fixture", data=pa.table({"text": ["kept"]}))
    task._metadata["text_volume"] = {
        "num_input_documents": 3,
        "num_output_documents": 1,
        "input_text_bytes": 9,
        "output_text_bytes": 4,
    }
    task._stage_perf = [
        StagePerfStats(stage_name=name, process_time=seconds)
        for name, seconds in [
            ("jsonl_reader", 3.0),
            ("measure_input_text", measurement_seconds),
            ("DuplicatesRemovalStage", 2.0),
            ("measure_output_text", measurement_seconds),
            ("parquet_writer", 5.0),
        ]
    ]
    result = WorkflowRunResult(
        workflow_name="removal", pipeline_tasks={"removal": [task]}, metadata={"num_duplicates_removed": 2}
    )
    monkeypatch.setattr(xenna, "XennaExecutor", SerialExecutor)
    monkeypatch.setattr(MeasuredRemovalWorkflow, "run", lambda *_, **__: result)
    measured = run_removal_benchmark("input", "ids", str(tmp_path), "xenna", blocksize="1GiB")
    assert measured["metrics"]["io_percentage"] == 80.0
    assert measured["metrics"]["num_removed_documents"] == 2
