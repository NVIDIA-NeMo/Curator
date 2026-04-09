---
description: "Advanced synthetic data generation using Nemotron-CC pipelines for text transformation and knowledge extraction"
categories: ["workflows"]
tags: ["nemotron-cc", "paraphrasing", "knowledge-extraction", "distillation"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "workflow"
modality: "text-only"
---

(nemotron-cc-overview)=
# Nemotron-CC Pipelines

Nemotron-CC provides advanced synthetic data generation workflows for transforming and extracting knowledge from existing text documents. Unlike simple generation, these pipelines use sophisticated preprocessing, LLM-based transformation, and postprocessing to create high-quality training data.

## The Composable Pipeline Pattern

Nemotron-CC stages follow a composable pattern with three distinct phases:

1. **Preprocessing**: Segment documents, filter by length, and prepare inputs for the LLM
2. **Generation**: Apply task-specific prompts to transform text using the LLM
3. **Postprocessing**: Clean outputs, remove formatting artifacts, and filter low-quality results

This separation enables fine-grained control over each phase while providing reusable helper functions for common patterns.

## Pipeline Architecture

```{mermaid}
flowchart TB
    subgraph "Preprocessing"
        A[Input Documents] --> B[Token Count Filter]
        B --> C[Document Splitter]
        C --> D[Segment Filter]
        D --> E[Document Joiner]
    end

    subgraph "LLM Generation"
        E --> F[Task-Specific Stage<br/>WikiPara/DiverseQA/Distill/etc.]
    end

    subgraph "Postprocessing"
        F --> G[Token Count Filter]
        G --> H[Markdown Remover]
        H --> I[Task-Specific Cleanup]
        I --> J[Quality Filter]
    end

    J --> K[Output Dataset]
```

## Input Data Requirements

Before running a Nemotron-CC pipeline, prepare your input data as Parquet files with the required schema.

### Required Schema

```{list-table} Required Parquet Columns
:header-rows: 1
:widths: 20 15 65

* - Column
  - Type
  - Description
* - `id`
  - `int64`
  - Unique document identifier. Required by the preprocessing pipeline to reassemble document segments after splitting.
* - `text`
  - `string`
  - Document content to transform. This is the primary input field for all Nemotron-CC stages.
* - `bucketed_results`
  - `int64`
  - Quality score used to route documents to appropriate pipelines. Values typically range from 0-20, where higher scores indicate higher quality content.
```

### Quality Score Field

The `bucketed_results` field contains quality scores that determine which pipeline processes each document:

- **High-quality documents** (`bucketed_results > 11`): Process with DiverseQA, Distill, ExtractKnowledge, or KnowledgeList tasks
- **Low-quality documents** (`bucketed_results <= 11`): Process with WikipediaParaphrasing to improve text quality

### Generating Quality Scores

Use NeMo Curator's quality assessment tools to generate quality scores before running SDG pipelines:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.stages.text.classifiers import FineWebEduClassifier
from nemo_curator.stages.text.modules import AddId

# Create pipeline to score documents
pipeline = Pipeline(name="quality_scoring")

# Read raw documents
pipeline.add_stage(JsonlReader(file_paths="raw_data/*.jsonl", fields=["text"]))

# Add unique document IDs
pipeline.add_stage(AddId(id_field="id"))

# Score document quality (outputs int score 0-5)
pipeline.add_stage(
    FineWebEduClassifier(
        int_score_field="bucketed_results",  # Use this as quality score
    )
)

# Save as Parquet for SDG pipeline
pipeline.add_stage(ParquetWriter(path="scored_data/"))

results = pipeline.run()
```

:::{tip}
The example above uses `FineWebEduClassifier` which outputs scores 0-5. For the Nemotron-CC threshold of 11, you can either:
- Scale the scores (e.g., multiply by 4)
- Adjust the filter threshold in your SDG pipeline
- Use a different classifier that outputs scores in the 0-20 range
:::

```{seealso}
For detailed information on quality scoring options, see {ref}`Quality Assessment & Filtering <text-process-data-filter>`.
```

### Example Data

An example Parquet file with the correct schema is available in the tutorials directory:

```bash
tutorials/synthetic/nemotron_cc/example_data/data.parquet
```

You can inspect its structure:

```python
import pandas as pd

df = pd.read_parquet("tutorials/synthetic/nemotron_cc/example_data/data.parquet")
print(df.columns.tolist())  # ['id', 'text', 'bucketed_results']
print(df.head(2))
```

---

## Available Tasks

Nemotron-CC provides five specialized generation tasks, each designed for specific data transformation needs:

```{list-table} Nemotron-CC Task Types
:header-rows: 1
:widths: 20 25 30 25

* - Task
  - Stage Class
  - Purpose
  - Use Case
* - Wikipedia Paraphrasing
  - `WikipediaParaphrasingStage`
  - Rewrite text as Wikipedia-style prose
  - Improving noisy web data
* - Diverse QA
  - `DiverseQAStage`
  - Generate diverse Q&A pairs
  - Reading comprehension training
* - Distill
  - `DistillStage`
  - Create condensed, informative paraphrases
  - Knowledge distillation
* - Extract Knowledge
  - `ExtractKnowledgeStage`
  - Extract factual content as passages
  - Knowledge base creation
* - Knowledge List
  - `KnowledgeListStage`
  - Extract structured fact lists
  - Fact extraction
```

## Quality-Based Processing Strategy

Nemotron-CC pipelines are designed to process data based on quality scores. The typical approach:

### High-Quality Data Pipeline

For documents with high quality scores, use tasks that leverage the existing quality:
- **DiverseQA**: Generate Q&A pairs from well-structured content
- **Distill**: Create condensed versions preserving key information
- **ExtractKnowledge**: Extract factual passages
- **KnowledgeList**: Extract structured facts

```python
from nemo_curator.stages.text.filters import Filter

# Filter for high-quality documents (score > 11)
pipeline.add_stage(
    Filter(
        filter_fn=lambda x: int(x) > 11,
        filter_field="bucketed_results",
    ),
)
```

### Low-Quality Data Pipeline

For documents with lower quality scores, use Wikipedia Paraphrasing to improve text quality:

```python
# Filter for low-quality documents (score <= 11)
pipeline.add_stage(
    Filter(
        filter_fn=lambda x: int(x) <= 11,
        filter_field="bucketed_results",
    ),
)
```

## Using Helper Functions

The recommended approach is to use the helper functions in `nemotron_cc_pipelines.py`:

:::{note}
The `nemotron_cc_pipelines` helper functions are provided in the [tutorials directory](https://github.com/NVIDIA-NeMo/Curator/blob/main/tutorials/synthetic/nemotron_cc/nemotron_cc_pipelines.py), not as part of the installed package. Copy the `nemotron_cc_pipelines.py` file to your project or reference the patterns when building custom pipelines.
:::

```python
from nemotron_cc_pipelines import (
    add_preprocessing_pipeline,
    add_diverse_qa_postprocessing_pipeline,
)
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import DiverseQAStage

pipeline = Pipeline(name="diverse_qa_pipeline")

# Add preprocessing
pipeline = add_preprocessing_pipeline(
    pipeline=pipeline,
    text_field="text",
    system_prompt=SYSTEM_PROMPT,
    user_prompt_template=PROMPT_TEMPLATE,
    min_document_tokens=30,
    min_segment_tokens=30,
    max_input_tokens=1000,
    args=args,  # Contains tokenizer config
)

# Add generation stage
pipeline.add_stage(
    DiverseQAStage(
        client=llm_client,
        model_name="meta/llama-3.3-70b-instruct",
        generation_config=generation_config,
        input_field="text",
        output_field="diverse_qa",
    )
)

# Add postprocessing
pipeline = add_diverse_qa_postprocessing_pipeline(
    pipeline=pipeline,
    llm_response_field="diverse_qa",
    args=args,
)
```

## Task Configuration

Each task has specific token count and preprocessing requirements:

```{list-table} Recommended Task Configuration
:header-rows: 1
:widths: 25 15 15 20 25

* - Task
  - Min Doc Tokens
  - Min Segment Tokens
  - Max Input Tokens
  - Max Output Tokens
* - Diverse QA
  - 30
  - 30
  - 1000
  - 600
* - Distill
  - 30
  - 10
  - 2000
  - 1600
* - Extract Knowledge
  - 30
  - 30
  - 1400
  - 1400
* - Knowledge List
  - 30
  - 30
  - 1000
  - 600
* - Wikipedia Paraphrasing
  - 5
  - 5
  - 512
  - 512
```

## Quick Example

```python
import os
from transformers import AutoTokenizer
from nemo_curator.core.client import RayClient
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.models.client import AsyncOpenAIClient
from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import DiverseQAStage
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter

# Initialize
client = RayClient(include_dashboard=False)
client.start()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

# Create LLM client
llm_client = AsyncOpenAIClient(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1",
    max_concurrent_requests=5,
)

# Build pipeline (see "Using Helper Functions" section for preprocessing/postprocessing)
pipeline = Pipeline(name="nemotron_cc_diverse_qa")
pipeline.add_stage(ParquetReader(file_paths=["./input_data/*.parquet"]))

# Add preprocessing stages using helper function:
# pipeline = add_preprocessing_pipeline(pipeline, text_field="text", ...)

# Add generation stage
pipeline.add_stage(
    DiverseQAStage(
        client=llm_client,
        model_name="meta/llama-3.3-70b-instruct",
        generation_config=GenerationConfig(temperature=0.5, top_p=0.9),
        input_field="text",
        output_field="diverse_qa",
    )
)

# Add postprocessing stages using helper function:
# pipeline = add_diverse_qa_postprocessing_pipeline(pipeline, llm_response_field="diverse_qa", ...)

pipeline.add_stage(ParquetWriter(path="./output/"))

# Execute
executor = XennaExecutor()
results = pipeline.run(executor)

client.stop()
```

---

## Detailed Reference

::::{grid} 1
:gutter: 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Task Reference
:link: tasks
:link-type: doc
Detailed reference for each Nemotron-CC stage, prompts, and post-processing
+++
{bdg-secondary}`reference`
{bdg-secondary}`api`
:::

::::

```{toctree}
:hidden:

tasks
```

