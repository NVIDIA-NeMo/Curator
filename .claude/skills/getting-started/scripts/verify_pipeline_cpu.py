#!/usr/bin/env python3
"""Verify CPU-only pipeline execution works (no GPU required)."""
import pandas as pd
from dataclasses import field

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import Task, _EmptyTask


class SampleTask(Task[pd.DataFrame]):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def num_items(self):
        return len(self.data)

    def validate(self):
        return True


class TaskCreationStage(ProcessingStage[_EmptyTask, SampleTask]):
    name = "TaskCreationStage"

    def inputs(self):
        return [], []

    def outputs(self):
        return ["data"], ["text"]

    def process(self, _):
        return [
            SampleTask(
                data=pd.DataFrame({"text": ["Hello world", "Test sentence"]}),
                task_id=1,
                dataset_name="test",
            )
        ]


class WordCountStage(ProcessingStage[SampleTask, SampleTask]):
    name = "WordCountStage"
    resources = Resources(cpus=1.0)
    batch_size = 1

    def inputs(self):
        return ["data"], ["text"]

    def outputs(self):
        return ["data"], ["text", "word_count"]

    def process(self, task):
        task.data["word_count"] = task.data["text"].str.split().str.len()
        return task


if __name__ == "__main__":
    rc = RayClient()
    rc.start()

    pipeline = Pipeline(name="verify_cpu")
    pipeline.add_stage(TaskCreationStage())
    pipeline.add_stage(WordCountStage())

    results = pipeline.run(XennaExecutor())

    print("✓ CPU pipeline executed successfully")
    print(f"  Output: {len(results)} task(s) processed")

    rc.stop()
