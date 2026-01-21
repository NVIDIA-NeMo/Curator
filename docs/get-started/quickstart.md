---
description: "Your first NeMo Curator pipeline in 10 minutes—from installation to running a complete data curation workflow"
categories: ["getting-started"]
tags: ["quickstart", "installation", "tutorial", "pipeline", "text-curation"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "text-only"
---

(gs-quickstart)=

# Quickstart: Your First Pipeline

Build and run a complete data curation pipeline in 10 minutes. This tutorial introduces NeMo Curator's core concepts through a hands-on example that downloads, filters, and exports a text dataset.

**What you'll learn:**

- Install NeMo Curator
- Understand the Pipeline → Stages → Tasks architecture
- Build a pipeline that filters low-quality documents
- Run the pipeline and inspect results

**Time required:** ~10 minutes

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.10, 3.11, or 3.12**
- **8GB+ RAM** (16GB recommended for larger datasets)
- **Linux** (Ubuntu 22.04/24.04 recommended) or macOS
- **NVIDIA GPU** (optional, for accelerated processing)

---

## Step 1: Install NeMo Curator

Install NeMo Curator with text processing support using `uv`:

```bash
# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create and activate a virtual environment
uv venv nemo-curator-env
source nemo-curator-env/bin/activate

# Install NeMo Curator with text support
uv pip install "nemo-curator[text_cpu]"
```

:::{tip}
For GPU-accelerated processing, use `nemo-curator[text_cuda12]` instead. See [Installation Guide](../admin/installation.md) for all options.
:::

**Verify installation:**

```bash
python -c "import nemo_curator; print(f'NeMo Curator {nemo_curator.__version__}')"
```

:::{tip}
**Want to skip ahead?** Run the pre-built quickstart tutorial directly:
```bash
python tutorials/quickstart.py
```
Then return here to understand what it does.
:::

---

## Step 2: Understand the Architecture

NeMo Curator uses a **Pipeline → Stages → Tasks** architecture built on [Ray](https://docs.ray.io/), a distributed computing framework:

```
┌─────────────────────────────────────────────────────────────┐
│                        Pipeline                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  Reader  │ → │  Filter  │ → │ Modifier │ → │  Writer  │ │
│  │  Stage   │   │  Stage   │   │  Stage   │   │  Stage   │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       ↓              ↓              ↓              ↓        │
│     Tasks          Tasks          Tasks          Tasks      │
│  (batches of      (filtered     (modified      (written    │
│   documents)      documents)    documents)     to disk)    │
└─────────────────────────────────────────────────────────────┘
```

| Component | Purpose | Example |
|-----------|---------|---------|
| **Pipeline** | Orchestrates the workflow | `Pipeline(name="my_pipeline", stages=[...])` |
| **Stage** | A single processing step | `JsonlReader`, `ScoreFilter`, `JsonlWriter` |
| **Task** | A batch of data flowing through stages | `DocumentBatch` containing rows of text |

Each stage processes tasks (batches of documents) and passes them to the next stage. The executor handles parallelization and resource management automatically using Ray workers.

:::{note}
**Tasks vs Documents**: A `Task` (specifically `DocumentBatch` for text) is a container holding multiple documents. Stages process entire batches at once for efficiency.
:::

---

## Step 3: Create Sample Data

Create a sample dataset in [JSONL format](https://jsonlines.org/) (one JSON object per line):

```{note}
JSONL is the standard format for NeMo Curator text data. Each line is a self-contained JSON object with at least a `text` field.
```

```python
import json
import os

# Create a data directory
os.makedirs("data/input", exist_ok=True)
os.makedirs("data/output", exist_ok=True)

# Sample documents (mix of good and low-quality)
documents = [
    {
        "id": "doc_001",
        "text": "Machine learning models require large amounts of high-quality training data. "
                "Data curation involves filtering, cleaning, and deduplicating datasets to "
                "improve model performance. NeMo Curator provides scalable tools for this task."
    },
    {
        "id": "doc_002",
        "text": "Short."  # Too short - will be filtered
    },
    {
        "id": "doc_003",
        "text": "Natural language processing has evolved significantly with the advent of "
                "transformer architectures. These models learn contextual representations "
                "that capture semantic meaning across long sequences of text."
    },
    {
        "id": "doc_004",
        "text": "!!!! #### @@@@ %%%% &&&& ****"  # Low quality - mostly symbols
    },
    {
        "id": "doc_005",
        "text": "GPU acceleration enables processing of massive datasets in hours rather "
                "than days. NVIDIA RAPIDS libraries like cuDF and cuML provide GPU-native "
                "implementations of common data processing operations."
    },
]

# Write to JSONL format
with open("data/input/sample.jsonl", "w") as f:
    for doc in documents:
        f.write(json.dumps(doc) + "\n")

print(f"Created {len(documents)} sample documents in data/input/sample.jsonl")
```

---

## Step 4: Build Your First Pipeline

Create a pipeline that reads documents, filters out low-quality ones, and writes the results:

```python
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules.score_filter import ScoreFilter
from nemo_curator.stages.text.filters.heuristic_filter import (
    WordCountFilter,
    NonAlphaNumericFilter,
)

# Start the Ray cluster (handles distributed execution)
# This manages worker processes that execute pipeline stages in parallel
ray_client = RayClient()
ray_client.start()

# Define the pipeline stages
pipeline = Pipeline(
    name="quality_filter_pipeline",
    description="Filter documents by word count and content quality",
    stages=[
        # Stage 1: Read JSONL files into DocumentBatch tasks
        JsonlReader(
            file_paths="data/input/",
            files_per_partition=1,
            fields=["id", "text"],
        ),
        
        # Stage 2: Keep only documents with 10+ words
        ScoreFilter(
            filter_obj=WordCountFilter(min_words=10, max_words=100000),
            text_field="text",
            score_field="word_count",  # Optionally save the score
        ),
        
        # Stage 3: Remove documents with >25% non-alphanumeric characters
        ScoreFilter(
            filter_obj=NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
            text_field="text",
            score_field="alpha_ratio",
        ),
        
        # Stage 4: Write filtered results to JSONL
        JsonlWriter(path="data/output/"),
    ],
)

# Print pipeline description
print(pipeline.describe())
```

**Pipeline output:**

```
Pipeline: quality_filter_pipeline
Description: Filter documents by word count and content quality
Stages: 5

Stage 1: file_partitioning
  Resources: 1.0 CPUs
  Batch size: 1

Stage 2: jsonl_reader
  Resources: 1.0 CPUs
  Batch size: 1

Stage 3: word_count
  Resources: 1.0 CPUs
  Batch size: 1

Stage 4: alpha_numeric
  Resources: 1.0 CPUs
  Batch size: 1

Stage 5: jsonl_writer
  Resources: 1.0 CPUs
  Batch size: 1
```

:::{note}
**Why 5 stages from 4?** `JsonlReader` is a *composite stage* that automatically decomposes into two execution stages: `file_partitioning` (discovers files) and `jsonl_reader` (reads file contents). This separation enables better parallelization.
:::

---

## Step 5: Run the Pipeline

Execute the pipeline and inspect the results:

```python
# Run the pipeline
print("Running pipeline...")
results = pipeline.run()

# Check results
print(f"\nPipeline completed!")
print(f"Output tasks: {len(results) if results else 0}")

# Examine the output files
import glob
output_files = glob.glob("data/output/*.jsonl")
print(f"Output files: {output_files}")

# Read and display filtered documents
if output_files:
    with open(output_files[0], "r") as f:
        for line in f:
            doc = json.loads(line)
            print(f"\n✓ Kept: {doc['id']}")
            print(f"  Word count: {doc.get('word_count', 'N/A')}")
            print(f"  Preview: {doc['text'][:80]}...")

# Clean up
ray_client.stop()
```

**Expected output:**

```
Running pipeline...

Pipeline completed!
Output tasks: 1
Output files: ['data/output/sample_0.jsonl']

✓ Kept: doc_001
  Word count: 32
  Preview: Machine learning models require large amounts of high-quality training data....

✓ Kept: doc_003
  Word count: 28
  Preview: Natural language processing has evolved significantly with the advent of trans...

✓ Kept: doc_005
  Word count: 26
  Preview: GPU acceleration enables processing of massive datasets in hours rather than d...
```

Documents `doc_002` (too short) and `doc_004` (too many symbols) were filtered out.

---

## Complete Script

Here's the complete working example:

```python
#!/usr/bin/env python3
"""NeMo Curator Quickstart: Your First Pipeline"""

import json
import os
import glob

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules.score_filter import ScoreFilter
from nemo_curator.stages.text.filters.heuristic_filter import (
    WordCountFilter,
    NonAlphaNumericFilter,
)


def create_sample_data():
    """Create sample documents for the tutorial."""
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)

    documents = [
        {
            "id": "doc_001",
            "text": "Machine learning models require large amounts of high-quality training data. "
                    "Data curation involves filtering, cleaning, and deduplicating datasets to "
                    "improve model performance. NeMo Curator provides scalable tools for this task."
        },
        {"id": "doc_002", "text": "Short."},
        {
            "id": "doc_003",
            "text": "Natural language processing has evolved significantly with the advent of "
                    "transformer architectures. These models learn contextual representations "
                    "that capture semantic meaning across long sequences of text."
        },
        {"id": "doc_004", "text": "!!!! #### @@@@ %%%% &&&& ****"},
        {
            "id": "doc_005",
            "text": "GPU acceleration enables processing of massive datasets in hours rather "
                    "than days. NVIDIA RAPIDS libraries like cuDF and cuML provide GPU-native "
                    "implementations of common data processing operations."
        },
    ]

    with open("data/input/sample.jsonl", "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")

    return len(documents)


def main():
    # Create sample data
    num_docs = create_sample_data()
    print(f"Created {num_docs} sample documents\n")

    # Start Ray cluster
    ray_client = RayClient()
    ray_client.start()

    try:
        # Build the pipeline
        pipeline = Pipeline(
            name="quality_filter_pipeline",
            description="Filter documents by word count and content quality",
            stages=[
                JsonlReader(
                    file_paths="data/input/",
                    files_per_partition=1,
                    fields=["id", "text"],
                ),
                ScoreFilter(
                    filter_obj=WordCountFilter(min_words=10, max_words=100000),
                    text_field="text",
                    score_field="word_count",
                ),
                ScoreFilter(
                    filter_obj=NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
                    text_field="text",
                    score_field="alpha_ratio",
                ),
                JsonlWriter(path="data/output/"),
            ],
        )

        # Run the pipeline
        print("Running pipeline...")
        print(pipeline.describe())
        results = pipeline.run()

        # Display results
        print(f"\n{'='*50}")
        print("Pipeline completed!")
        print(f"Input documents: {num_docs}")

        output_files = glob.glob("data/output/*.jsonl")
        if output_files:
            kept_count = 0
            with open(output_files[0], "r") as f:
                for line in f:
                    doc = json.loads(line)
                    kept_count += 1
                    print(f"\n✓ Kept: {doc['id']} (word_count={doc.get('word_count', 'N/A')})")
                    print(f"  {doc['text'][:60]}...")

            print(f"\nFiltered: {num_docs - kept_count} documents removed")
            print(f"Output: {kept_count} documents saved to {output_files[0]}")

    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
```

---

## Key Concepts Recap

| Concept | What It Does | Code Example |
|---------|--------------|--------------|
| **Pipeline** | Connects stages into a workflow | `Pipeline(stages=[...])` |
| **JsonlReader** | Reads JSONL files into DocumentBatch tasks | `JsonlReader(file_paths="data/")` |
| **ScoreFilter** | Computes a score and filters documents | `ScoreFilter(filter_obj=WordCountFilter(...))` |
| **JsonlWriter** | Writes DocumentBatch tasks to JSONL files | `JsonlWriter(path="output/")` |
| **RayClient** | Manages the distributed execution cluster | `RayClient().start()` |

---

## Next Steps

Now that you've built your first pipeline, explore these resources:

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} 📖 Core Concepts
:link: ../about/concepts/index
:link-type: doc

Deep dive into Pipeline, Stages, Tasks, and Executors.
:::

:::{grid-item-card} 📚 Text Curation Guide
:link: ../curate-text/index
:link-type: doc

Complete guide to text processing: filters, classifiers, deduplication.
:::

:::{grid-item-card} 🖼️ Image Curation
:link: gs-image
:link-type: ref

Curate image-text datasets for vision language models.
:::

:::{grid-item-card} 🎬 Video Curation
:link: gs-video
:link-type: ref

Process video corpora with scene detection and embeddings.
:::

::::

**Common next steps:**

1. **Add more filters** – See [Quality Filtering](../curate-text/process-data/quality-assessment/heuristic.md) for 30+ built-in filters
2. **Scale to larger datasets** – Adjust `files_per_partition` and add GPU resources
3. **Use GPU acceleration** – Install `nemo-curator[text_cuda12]` for RAPIDS-powered deduplication
4. **Build custom stages** – See [Creating Custom Stages](../about/concepts/stages.md) to implement your own processing logic

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Ray is already running` message | This is normal if you've run pipelines before. NeMo Curator reuses existing Ray clusters. |
| `ModuleNotFoundError` | Ensure you activated the virtual environment: `source nemo-curator-env/bin/activate` |
| Pipeline hangs | Check available memory. Try reducing `files_per_partition` for large datasets. |
| No output files created | Verify input path exists and contains `.jsonl` files. Check the pipeline logs for errors. |

For additional help, see the [Installation Guide](../admin/installation.md) or open an issue on [GitHub](https://github.com/NVIDIA-NeMo/Curator/issues).
