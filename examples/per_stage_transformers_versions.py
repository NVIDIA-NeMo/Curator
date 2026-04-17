"""Example: run two pipeline stages with incompatible transformers versions.

This demonstrates the primary use case for per-stage runtime environments:
stages that require different, potentially incompatible library versions can
coexist in the same pipeline without conflict.

Usage (inside the NeMo Curator container):
    python3 examples/per_stage_transformers_versions.py
"""

from typing import ClassVar

import pandas as pd

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch


class TransformersStage1(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage that requires transformers==4.40.0."""

    name = "transformers_stage_1"
    resources = Resources(cpus=0.5)
    batch_size = 1
    runtime_env: ClassVar[dict] = {"pip": ["transformers==4.40.0"]}

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["stage1_transformers_version"]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        import transformers

        batch = task.to_pandas().copy()
        batch["stage1_transformers_version"] = transformers.__version__
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=batch,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


class TransformersStage2(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage that requires transformers==4.45.0."""

    name = "transformers_stage_2"
    resources = Resources(cpus=0.5)
    batch_size = 1
    runtime_env: ClassVar[dict] = {"pip": ["transformers==4.45.0"]}

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["stage1_transformers_version"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["stage2_transformers_version"]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        import transformers

        batch = task.to_pandas().copy()
        batch["stage2_transformers_version"] = transformers.__version__
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=batch,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


if __name__ == "__main__":
    initial = DocumentBatch(
        task_id="example",
        dataset_name="example",
        data=pd.DataFrame({"text": ["hello world"]}),
    )

    pipeline = Pipeline(
        name="per_stage_transformers_example",
        stages=[TransformersStage1(), TransformersStage2()],
    )

    print("Running pipeline (first run installs virtualenvs, subsequent runs use cache)...")
    results = pipeline.run(executor=RayDataExecutor(), initial_tasks=[initial])

    result_batch = results[0].to_pandas()
    v1 = result_batch["stage1_transformers_version"].iloc[0]
    v2 = result_batch["stage2_transformers_version"].iloc[0]
    print(f"Stage 1 transformers version: {v1}")
    print(f"Stage 2 transformers version: {v2}")
    if v1 != "4.40.0":
        msg = f"Expected stage 1 transformers==4.40.0, got {v1}"
        raise RuntimeError(msg)
    if v2 != "4.45.0":
        msg = f"Expected stage 2 transformers==4.45.0, got {v2}"
        raise RuntimeError(msg)
    print("OK — both stages ran with their own isolated transformers version.")
