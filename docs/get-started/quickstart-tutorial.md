---
description: "Comprehensive getting started guide for NeMo Curator with core concepts, architecture overview, and a complete end-to-end pipeline example"
categories: ["getting-started"]
tags: ["quickstart", "tutorial", "pipeline", "installation", "beginner", "python-api"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "universal"
---

(quickstart-tutorial)=

# Getting Started with NeMo Curator

Welcome to NeMo Curator! This guide introduces you to the core concepts and architecture of NeMo Curator and walks you through building your first data curation pipeline.

## What You'll Learn

By the end of this tutorial, you'll understand:

- The pipeline-based architecture of NeMo Curator
- Core concepts: Pipelines, Stages, Tasks, and Executors
- How to install and set up your environment
- How to build and run a complete end-to-end pipeline
- Where to go next for modality-specific workflows

## What You'll Build

You'll create a three-stage sentiment analysis pipeline that:

1. **Creates sample tasks** with sentences
2. **Counts words** in each sentence using CPU
3. **Analyzes sentiment** using a GPU-accelerated transformer model

This example demonstrates the key patterns you'll use across all NeMo Curator workflows, regardless of modality (text, image, video, or audio).

:::{tip}
**Already familiar with the basics?** Jump directly to modality-specific quickstarts:

- [Text Curation Quickstart](gs-text)
- [Image Curation Quickstart](gs-image)
- [Video Curation Quickstart](gs-video)
- [Audio Curation Quickstart](gs-audio)
:::

---

## Core Concepts

NeMo Curator uses a **pipeline-based architecture** with modular, reusable components. Understanding these four core concepts is essential:

### Pipeline

A `Pipeline` is a container that orchestrates a sequence of processing stages. It manages the flow of data from one stage to the next and coordinates execution across distributed resources.

**Key characteristics**:

- Named and described for clarity
- Composed of one or more stages
- Automatically validates stage compatibility
- Provides detailed execution plans

```python
from nemo_curator.pipeline import Pipeline

pipeline = Pipeline(
    name="sentiment_analysis",
    description="Analyze sentiment of sample sentences"
)
```

### Stage

A `ProcessingStage` is a single, self-contained unit of work in your pipeline. Each stage:

- Defines its **resource requirements** (CPUs, GPUs, memory)
- Declares its **inputs** (what data it expects)
- Declares its **outputs** (what data it produces)
- Implements a `process()` method to transform data

**Example stage types**:

- CPU stages (for example, word counting, text cleaning)
- GPU-accelerated stages (for example, embedding generation, classification)
- Data loading stages
- Export stages

```python
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources

class WordCountStage(ProcessingStage):
    _name = "WordCountStage"
    _resources = Resources(cpus=1.0)  # CPU-only stage
    _batch_size = 1

    def process(self, task):
        task.data["word_count"] = task.data["sentence"].str.split().str.len()
        return task
```

### Task

A `Task` is a unit of data that flows through your pipeline. Each task:

- Contains the actual data for processing (for example, DataFrame, file path)
- Tracks metadata (for example, task ID, dataset name)
- Validates correctness

Think of tasks as "work items" that stages process and transform.

```python
from nemo_curator.tasks import Task
import pandas as pd

task = SampleTask(
    data=pd.DataFrame({"sentence": ["I love this product"]}),
    task_id=1,
    dataset_name="SampleDataset"
)
```

### Executor

An `Executor` manages how and where your pipeline runs. NeMo Curator uses:

- **RayClient**: Manages the Ray cluster (single-node or multi-node)
- **XennaExecutor**: Schedules and executes stages across available resources

The executor handles:

- Resource allocation and scheduling
- Distributed execution across nodes
- GPU and CPU management
- Task batching and parallel processing

```python
from nemo_curator.core.client import RayClient
from nemo_curator.backends.xenna import XennaExecutor

ray_client = RayClient()
ray_client.start()

executor = XennaExecutor()
results = pipeline.run(executor)
```

:::{seealso}
For deeper architectural details, refer to [Core Concepts](about-concepts).
:::

---

## Prerequisites

Before you start, ensure your system meets these requirements:

**System Requirements**:

- **OS**: Ubuntu 24.04, 22.04, or 20.04 (Linux required)
- **Python**: 3.10, 3.11, or 3.12
- **Memory**: 16GB+ RAM
- **GPU** (optional): NVIDIA GPU with CUDA 12+ and 10GB+ VRAM

:::{note}
**CPU mode**: You can run NeMo Curator without a GPU, but GPU acceleration improves performance for operations like classification and embedding generation.
:::

---

## Installation

### Step 1: Install uv

NeMo Curator uses `uv` for fast, reliable package management:

```bash
curl -LsSf https://astral.sh/uv/0.8.22/install.sh | sh
source $HOME/.local/bin/env
```

### Step 2: Create a Virtual Environment

```bash
uv venv
source .venv/bin/activate
```

### Step 3: Install NeMo Curator

For this quickstart, install the base package with text support:

```bash
uv pip install torch wheel_stub psutil setuptools setuptools_scm
echo "transformers==4.55.2" > override.txt
uv pip install https://pypi.nvidia.com --no-build-isolation "nemo-curator[text_cuda12]" --override override.txt
```

:::{tip}
**For other modalities**:

- Image: `"nemo-curator[image_cuda12]"`
- Video: `"nemo-curator[video_cuda12]"`
- Audio: `"nemo-curator[audio_cuda12]"`
- All modalities: `"nemo-curator[all]"`

Refer to the [Installation Guide](admin-installation) for detailed instructions.
:::

### Step 4: Verify Installation

```bash
python -c "import nemo_curator; print(nemo_curator.__version__)"
```

If you see the version number, you're ready to build your first pipeline! 🎉

---

## Your First Pipeline

Let's build a complete sentiment analysis pipeline step by step. This example follows the official quickstart in [`tutorials/quickstart.py`](https://github.com/NVIDIA-NeMo/Curator/blob/main/tutorials/quickstart.py).

### Pipeline Overview

Our pipeline has three stages:

```text
TaskCreationStage -> WordCountStage -> SentimentStage -> Results
```

**Stage 1**: Creates sample tasks with random sentences

**Stage 2**: Counts words in each sentence (CPU)

**Stage 3**: Analyzes sentiment using a Hugging Face model (GPU)

### Complete Example Code

Create a file called `my_first_pipeline.py`:

```python
import random
from dataclasses import field

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from loguru import logger

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
    """Task containing a DataFrame with sentences."""

    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def num_items(self) -> int:
        return len(self.data)

    def validate(self) -> bool:
        return True


class TaskCreationStage(ProcessingStage[_EmptyTask, SampleTask]):
    """Stage 1: Create sample tasks with random sentences."""

    _name = "TaskCreationStage"

    def __init__(self, num_sentences_per_task: int, num_tasks: int):
        self.num_sentences_per_task = num_sentences_per_task
        self.num_tasks = num_tasks

    def inputs(self):
        return [], []

    def outputs(self):
        return ["data"], ["sentence"]

    def process(self, _: _EmptyTask) -> list[SampleTask]:
        tasks = []
        for _ in range(self.num_tasks):
            sampled_sentences = random.sample(
                SAMPLE_SENTENCES, self.num_sentences_per_task
            )
            tasks.append(
                SampleTask(
                    data=pd.DataFrame({"sentence": sampled_sentences}),
                    task_id=random.randint(0, 1000000),
                    dataset_name="SampleDataset",
                )
            )
        return tasks


class WordCountStage(ProcessingStage[SampleTask, SampleTask]):
    """Stage 2: Count words in each sentence (CPU-only)."""

    _name = "WordCountStage"
    _resources = Resources(cpus=1.0)
    _batch_size = 1

    def inputs(self):
        return ["data"], ["sentence"]

    def outputs(self):
        return ["data"], ["sentence", "word_count"]

    def process(self, task: SampleTask) -> SampleTask:
        task.data["word_count"] = task.data["sentence"].str.split().str.len()
        return task


class SentimentStage(ProcessingStage[SampleTask, SampleTask]):
    """Stage 3: Analyze sentiment using GPU-accelerated model."""

    _name = "SentimentStage"
    _resources = Resources(cpus=1.0, gpu_memory_gb=10.0)
    _batch_size = 1

    def __init__(self, model_name: str, batch_size: int):
        self.model_name = model_name
        self._batch_size = batch_size
        self.model = None
        self.tokenizer = None

    def inputs(self):
        return ["data"], ["sentence", "word_count"]

    def outputs(self):
        return ["data"], ["sentence", "word_count", "sentiment"]

    def setup(self, _: WorkerMetadata) -> None:
        """Load the model and tokenizer."""
        logger.info(f"Loading model {self.model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.model.to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def process(self, task: SampleTask) -> SampleTask:
        """Process a single task."""
        sentences = task.data["sentence"].tolist()

        # Tokenize and run inference
        inputs = self.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment_scores = predictions.cpu().numpy()
            sentiment_labels = [
                "negative" if s[0] > 0.5 else "positive" if s[2] > 0.5 else "neutral"
                for s in sentiment_scores
            ]

        # Add sentiment column
        task.data["sentiment"] = sentiment_labels
        return task


def main():
    """Build and run the pipeline."""
    # Step 1: Initialize Ray cluster
    ray_client = RayClient()
    ray_client.start()

    # Step 2: Create pipeline
    pipeline = Pipeline(
        name="sentiment_analysis",
        description="Analyze sentiment of sample sentences"
    )

    # Step 3: Add stages
    pipeline.add_stage(TaskCreationStage(num_sentences_per_task=3, num_tasks=10))
    pipeline.add_stage(WordCountStage())
    pipeline.add_stage(
        SentimentStage(
            model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
            batch_size=2
        )
    )

    # Step 4: Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Step 5: Execute pipeline
    executor = XennaExecutor()
    print("Starting pipeline execution...")
    results = pipeline.run(executor)

    # Step 6: Display results
    print("\nPipeline completed!")
    print(f"Total output tasks: {len(results) if results else 0}")

    if results:
        for i, task in enumerate(results[:3]):  # Show first 3 tasks
            print(f"\nTask {i}:")
            print(task.data)

    ray_client.stop()


if __name__ == "__main__":
    main()
```

### Run Your Pipeline

```bash
python my_first_pipeline.py
```

---

## Understanding the Output

When you run your pipeline, you'll see three types of output:

### 1. Pipeline Description

Before execution, the pipeline prints its structure:

```text
Pipeline: sentiment_analysis
Description: Analyze sentiment of sample sentences
Stages: 3

Stage 1: TaskCreationStage
  Resources: 1.0 CPUs
  Batch size: 1
  Inputs:
    Required attributes: []
    Required columns: []
  Outputs:
    Output attributes: data
    Output columns: sentence

Stage 2: WordCountStage
  Resources: 1.0 CPUs
  Batch size: 1
  Inputs:
    Required attributes: data
    Required columns: sentence
  Outputs:
    Output attributes: data
    Output columns: sentence, word_count

Stage 3: SentimentStage
  Resources: 1.0 CPUs
    GPU Memory: 10.0 GB (1 GPUs)
  Batch size: 1
  Inputs:
    Required attributes: data
    Required columns: sentence, word_count
  Outputs:
    Output attributes: data
    Output columns: sentence, word_count, sentiment
```

This shows:

- Each stage's resource requirements
- Input and output schemas
- Data flow through the pipeline

### 2. Execution Logs

During execution, you'll see logs from Ray and the stages:

```text
Starting pipeline execution...
[RayClient] Starting Ray cluster...
[TaskCreationStage] Creating 10 tasks with 3 sentences each...
[WordCountStage] Processing task 1/10...
[SentimentStage] Loading model cardiffnlp/twitter-roberta-base-sentiment-latest...
[SentimentStage] Processing task 1/10...
```

### 3. Results

After completion, the pipeline outputs the processed tasks:

```text
Pipeline completed!
Total output tasks: 10

Task 0:
                         sentence  word_count sentiment
0           I love this product           4  positive
1           I hate this product           4  negative
2  I'm neutral about this product           5   neutral
```

Each task now has:

- Original `sentence` column
- Computed `word_count` column (from Stage 2)
- Predicted `sentiment` column (from Stage 3)

---

## Next Steps

Congratulations! You've built your first NeMo Curator pipeline. 🎉

### Explore Modality-Specific Workflows

Now that you understand the core concepts, explore how to curate data for specific modalities:

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} {octicon}`typography;1.5em;sd-mr-1` Text Curation
:link: gs-text
:link-type: ref
Curate text datasets with deduplication, quality filtering, and classification.
:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Image Curation
:link: gs-image
:link-type: ref
Curate image-text datasets with embeddings, aesthetic scoring, and NSFW detection.
:::

:::{grid-item-card} {octicon}`video;1.5em;sd-mr-1` Video Curation
:link: gs-video
:link-type: ref
Process videos with clipping, encoding, embeddings, and deduplication.
:::

:::{grid-item-card} {octicon}`unmute;1.5em;sd-mr-1` Audio Curation
:link: gs-audio
:link-type: ref
Curate speech data with ASR transcription and quality assessment.
:::

::::

### Learn Advanced Techniques

- **Distributed Processing**: [Infrastructure Guide](reference-infrastructure-main)
- **Custom Stages**: Explore [Text Tutorials](text-overview) for real-world examples
- **Performance Optimization**: [Benchmarks and Best Practices](about-key-features)
- **Production Deployment**: [Deployment Options](admin-overview)

### Get Help and Contribute

- **API Reference**: [Python API Documentation](apidocs-main)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/NVIDIA-NeMo/Curator/discussions)
- **Report Issues**: [Bug reports and feature requests](https://github.com/NVIDIA-NeMo/Curator/issues)
- **Contributing**: [Contribution guidelines](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md)

---

## Summary

In this tutorial, you learned:

- ✅ NeMo Curator's pipeline-based architecture
- ✅ Core concepts: Pipeline, Stage, Task, Executor
- ✅ How to install and configure your environment
- ✅ How to build and run a complete end-to-end pipeline
- ✅ How to interpret pipeline output and results

You're now ready to tackle real-world data curation challenges across text, image, video, and audio modalities!
