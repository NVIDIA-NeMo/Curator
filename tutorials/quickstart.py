# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Quickstart example for NeMo Curator

This example shows 3 stages:
1. TaskCreationStage: None -> List[Task] : This creates tasks with sample sentences
2. WordCountStage: List[Task] -> List[Task] : This adds a new column with the word count of the sentences
3. SentimentStage: List[Task] -> List[Task] : This is a GPU stage that adds a new column with the sentiment of the sentences
"""

import random
from dataclasses import field

import huggingface_hub
import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import Task, _EmptyTask

SAMPLE_SENTENCES = [
    "I love this product",
    "I hate this product",
    "I'm neutral about this product",
]


class SampleTask(Task[pd.DataFrame]):
    """
    A sample task that contains a dataframe with a single column "sentence"
    """

    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return True


class TaskCreationStage(ProcessingStage[_EmptyTask, SampleTask]):
    _name: str = "TaskCreationStage"

    def __init__(self, num_sentences_per_task: int, num_tasks: int):
        self.num_sentences_per_task = num_sentences_per_task
        self.num_tasks = num_tasks

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sentence"]

    def process(self, _: _EmptyTask) -> SampleTask:
        """
        Process the input task and return a new task with the processed data
        """
        # randomly sample the sentences
        tasks = []
        for _ in range(self.num_tasks):
            sampled_sentences = random.sample(SAMPLE_SENTENCES, self.num_sentences_per_task)
            tasks.append(
                SampleTask(
                    data=pd.DataFrame({"sentence": sampled_sentences}),
                    task_id=random.randint(0, 1000000),  # noqa: S311
                    dataset_name="SampleDataset",
                )
            )
        return tasks


class WordCountStage(ProcessingStage[SampleTask, SampleTask]):
    _name: str = "WordCountStage"
    _resources: Resources = Resources(cpus=1.0)
    _batch_size: int = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sentence"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sentence", "word_count"]

    def process(self, task: SampleTask) -> SampleTask:
        """
        Process the input task and return a new task with the processed data
        """
        task.data["word_count"] = task.data["sentence"].str.split().str.len()
        return task


class SentimentStage(ProcessingStage[SampleTask, SampleTask]):
    _name: str = "SentimentStage"
    _resources: Resources = Resources(cpus=1.0, gpu_memory_gb=10.0)
    _batch_size: int = 1

    def __init__(self, model_name: str, batch_size: int):
        """
        Args:
            model_name: The name of the model to use
            batch_size: The batch size (in terms of number of tasks) to use for the model
        """
        self.model_name = model_name
        self._batch_size = batch_size
        self.model = None
        self.tokenizer = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sentence", "word_count"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["sentence", "word_count", "sentiment"]

    def setup_on_node(self, node_info: NodeInfo, _: WorkerMetadata) -> None:
        """Cache this model on the node. You can assume that this only gets called once per node."""
        logger.info(f"Ensuring model {self.model_name} artifacts are cached on node {node_info.node_id}...")
        # Use snapshot_download to download all files without loading the model into memory.
        huggingface_hub.snapshot_download(
            repo_id=self.model_name,
            local_files_only=False,  # Download if not cached
            resume_download=True,  # Resume interrupted downloads
        )
        logger.info(f"Model {self.model_name} artifacts are cached or downloading on node {node_info.node_id}.")

    def setup(self, _: WorkerMetadata) -> None:
        """Load the Hugging Face model and tokenizer from the cache."""
        logger.info(f"Loading model {self.model_name} from cache on worker...")
        # Load sentiment model from cache only.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            local_files_only=True,  # Fail if not cached
        )
        self.model.to("cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True,  # Fail if not cached
        )
        logger.info(f"Model {self.model_name} and tokenizer loaded successfully from cache.")

    def process_batch(self, tasks: list[SampleTask]) -> list[SampleTask]:
        """Process a batch of tasks using the sentiment model."""
        if not self.model or not self.tokenizer:
            error_message = "Model and tokenizer not loaded. Setup must be called first."
            logger.error(error_message)
            raise RuntimeError(error_message)

        # Collect all sentences from all tasks
        all_sentences = []
        task_sentence_counts = []

        for task in tasks:
            sentences = task.data["sentence"].tolist()
            all_sentences.extend(sentences)
            task_sentence_counts.append(len(sentences))

        # Tokenize all sentences at once
        inputs = self.tokenizer(all_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
            "cuda"
        )

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Get the sentiment labels (assuming 0=negative, 1=neutral, 2=positive)
            sentiment_scores = predictions.cpu().numpy()
            sentiment_labels = [
                "negative" if s[0] > 0.5 else "positive" if s[2] > 0.5 else "neutral"  # noqa: PLR2004
                for s in sentiment_scores
            ]

        # Distribute results back to tasks
        result_tasks = []
        sentence_idx = 0

        for i, task in enumerate(tasks):
            num_sentences = task_sentence_counts[i]
            task_sentiments = sentiment_labels[sentence_idx : sentence_idx + num_sentences]

            # Create new task with sentiment column
            new_data = task.data.copy()
            new_data["sentiment"] = task_sentiments

            result_task = SampleTask(data=new_data, task_id=task.task_id, dataset_name=task.dataset_name)
            result_tasks.append(result_task)

            sentence_idx += num_sentences

        return result_tasks

    def process(self, task: SampleTask) -> SampleTask:
        """Single task processing - fallback method."""
        return self.process_batch([task])[0]


def main() -> None:
    """Main function to run the pipeline."""
    # Create pipeline
    ray_client = RayClient()
    ray_client.start()
    pipeline = Pipeline(name="sentiment_analysis", description="Analyze sentiment of sample sentences")

    # Add stages
    pipeline.add_stage(TaskCreationStage(num_sentences_per_task=3, num_tasks=200))
    pipeline.add_stage(WordCountStage())
    pipeline.add_stage(SentimentStage(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", batch_size=2))

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Create executor
    executor = XennaExecutor()

    # Execute pipeline
    print("Starting pipeline execution...")
    results = pipeline.run(executor)

    # Print results
    print("\nPipeline completed!")
    print(f"Total output tasks: {len(results) if results else 0}")

    if results:
        for i, task in enumerate(results):
            print(f"\nTask {i}:")
            print(task.data)

    ray_client.stop()


if __name__ == "__main__":
    main()
