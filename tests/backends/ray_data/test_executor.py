# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import pytest
import ray

from nemo_curator.backends.ray_data import executor as executor_module
from nemo_curator.backends.ray_data.executor import RayDataExecutor
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.reassembly import PDFDocumentMergeStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.schema import PAGE_OUTPUT_SCHEMA
from nemo_curator.stages.interleaved.pdf.nemotron_parse.validation import PDFPageBudgetValidationStage
from nemo_curator.tasks import EmptyTask, InterleavedBatch, Task
from tests.stages.interleaved.pdf.nemotron_parse._page_budget_helpers import TEST_OUTPUT_VERSION, partition, project


def test_driver_validation_observes_empty_distributed_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest, _ = partition(tmp_path, [{"file_name": "missing.pdf", "num_pages": 1}])
    stage = PDFPageBudgetValidationStage(manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION)
    runner = RayDataExecutor()
    monkeypatch.setattr(executor_module, "install_ray_data_diagnostics", lambda: None)
    monkeypatch.setattr(executor_module, "register_loguru_serializer", lambda: None)
    monkeypatch.setattr(executor_module, "execute_setup_on_node", lambda *_a, **_k: None)
    monkeypatch.setattr(ray, "init", lambda **_kwargs: None)
    monkeypatch.setattr(ray, "shutdown", lambda: None)
    monkeypatch.setattr(runner, "_tasks_to_dataset", lambda _tasks: None)
    monkeypatch.setattr(runner, "_dataset_to_tasks", lambda _dataset: [])
    with pytest.raises(ValueError, match="Missing whole PDFs"):
        runner.execute([stage])


def test_driver_validation_must_follow_distributed_stages(tmp_path: Path):
    manifest, _ = partition(tmp_path, [{"file_name": "a.pdf", "num_pages": 1}])
    with pytest.raises(ValueError, match="must follow"):
        RayDataExecutor().execute(
            [
                PDFPageBudgetValidationStage(manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION),
                PDFDocumentMergeStage(),
            ]
        )


def test_page_merge_cannot_execute_without_final_validation():
    with pytest.raises(ValueError, match="Required final driver validation"):
        RayDataExecutor().execute([PDFDocumentMergeStage()])


@dataclass
class _PassThroughStage(ProcessingStage[Task, Task]):
    name: str = "pass_through"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: Task) -> Task:
        return task


@pytest.mark.parametrize("driver_last", [True, False])
def test_driver_stages_use_identity_when_equal_stages_have_different_placement(
    monkeypatch: pytest.MonkeyPatch, driver_last: bool
):
    distributed = _PassThroughStage()
    driver = _PassThroughStage().with_(ray_stage_spec={RayStageSpecKeys.RUN_ON_DRIVER: True})
    assert distributed == driver
    assert distributed is not driver
    stages = [distributed, driver] if driver_last else [driver, distributed]
    runner = RayDataExecutor()
    if not driver_last:
        with pytest.raises(ValueError, match="must follow"):
            runner.execute(stages)
        return

    monkeypatch.setattr(executor_module, "install_ray_data_diagnostics", lambda: None)
    monkeypatch.setattr(executor_module, "register_loguru_serializer", lambda: None)
    monkeypatch.setattr(executor_module, "execute_setup_on_node", lambda *_a, **_k: None)
    monkeypatch.setattr(ray, "init", lambda **_kwargs: None)
    monkeypatch.setattr(ray, "shutdown", lambda: None)
    monkeypatch.setattr(runner, "_tasks_to_dataset", lambda _tasks: None)
    monkeypatch.setattr(runner, "_dataset_to_tasks", lambda _dataset: [EmptyTask()])
    distributed_adapter, driver_adapter = Mock(), Mock()
    driver_adapter.return_value.process_batch.return_value = [EmptyTask()]
    monkeypatch.setattr(executor_module, "RayDataStageAdapter", distributed_adapter)
    monkeypatch.setattr(executor_module, "BaseStageAdapter", driver_adapter)

    runner.execute(stages)

    assert distributed_adapter.call_args.args[0] is distributed
    assert driver_adapter.call_args.args[0] is driver
    distributed_adapter.return_value.process_dataset.assert_called_once()
    driver_adapter.return_value.process_batch.assert_called_once()


@pytest.mark.usefixtures("shared_ray_client")
def test_real_ray_fanin_and_validation_for_large_document(tmp_path: Path):
    manifest, tasks = partition(
        tmp_path,
        [
            {"file_name": "largest.pdf", "num_pages": 2227},
            {"file_name": "tail.pdf", "num_pages": 3},
            {"file_name": "failed.pdf", "num_pages": 1},
            {"file_name": "empty.pdf", "num_pages": 2, "selected_page_indices": [1]},
        ],
    )
    chunks = [
        chunk for task in tasks for chunk in project(task, failed={0}, no_elements={1}, text_only=set(range(2, 64)))
    ]
    results = RayDataExecutor().execute(
        [
            PDFDocumentMergeStage(),
            InterleavedParquetWriterStage(
                path=str(tmp_path / "out"),
                materialize_on_write=False,
                on_materialize_error="warn",
                schema=PAGE_OUTPUT_SCHEMA,
                reject_unknown_columns=True,
                atomic_local_write=True,
                use_output_file_id=True,
                mode="error",
            ),
            PDFPageBudgetValidationStage(manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION),
        ],
        initial_tasks=list(reversed(chunks)),
    )
    assert len(results) == 4
    assert all(task._metadata["pdf_page_budget_validation"]["pages"] == 2232 for task in results)


@dataclass
class _LostOutputStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    name: str = "simulate_lost_distributed_output"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, _task: InterleavedBatch) -> None:
        return None


@pytest.mark.usefixtures("shared_ray_client")
def test_real_ray_empty_upstream_still_fails_final_inventory(tmp_path: Path):
    manifest, tasks = partition(tmp_path, [{"file_name": "lost.pdf", "num_pages": 2}])
    with pytest.raises(ValueError, match="Missing whole PDFs"):
        RayDataExecutor().execute(
            [
                _LostOutputStage(),
                PDFPageBudgetValidationStage(manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION),
            ],
            initial_tasks=project(tasks[0]),
        )
