# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import ray

from nemo_curator.backends.ray_data import executor as executor_module
from nemo_curator.backends.ray_data.executor import RayDataExecutor
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.reassembly import PDFDocumentMergeStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.schema import PAGE_OUTPUT_SCHEMA
from nemo_curator.stages.interleaved.pdf.nemotron_parse.validation import PDFPageBudgetValidationStage
from nemo_curator.tasks import InterleavedBatch
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


def test_real_ray_fanin_and_validation_for_large_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if os.environ.get("PAGE_BUDGET_RUN_RAY_INTEGRATION") != "1":
        pytest.skip("Run explicitly in an isolated CPU Ray cluster after the active benchmark releases")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("RAY_ADDRESS", "local")
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
    ray_temp = tempfile.mkdtemp(prefix="pb-ray-", dir="/tmp")
    print(f"Retained Ray CPU qualification logs: {ray_temp}")
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=256 * 1024**2,
        _temp_dir=ray_temp,
    )
    try:
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
    finally:
        ray.shutdown()


@dataclass
class _LostOutputStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    name: str = "simulate_lost_distributed_output"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, _task: InterleavedBatch) -> None:
        return None


def test_real_ray_empty_upstream_still_fails_final_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if os.environ.get("PAGE_BUDGET_RUN_RAY_INTEGRATION") != "1":
        pytest.skip("Run explicitly in an isolated CPU Ray cluster after the active benchmark releases")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("RAY_ADDRESS", "local")
    manifest, tasks = partition(tmp_path, [{"file_name": "lost.pdf", "num_pages": 2}])
    ray_temp = tempfile.mkdtemp(prefix="pb-ray-", dir="/tmp")
    print(f"Retained Ray CPU qualification logs: {ray_temp}")
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=256 * 1024**2,
        _temp_dir=ray_temp,
    )
    try:
        with pytest.raises(ValueError, match="Missing whole PDFs"):
            RayDataExecutor().execute(
                [
                    _LostOutputStage(),
                    PDFPageBudgetValidationStage(
                        manifest_path=str(manifest), expected_output_version=TEST_OUTPUT_VERSION
                    ),
                ],
                initial_tasks=project(tasks[0]),
            )
    finally:
        ray.shutdown()
