---
description: "Load text data from various sources including Common Crawl, Wikipedia, and custom datasets using NeMo Curator's ray-curator framework"
categories: ["workflows"]
tags: ["data-loading", "common-crawl", "wikipedia", "custom-data", "distributed", "ray"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "text-only"
---

(text-load-data)=

# Text Data Loading

Load text data from a variety of data sources using NeMo Curator's ray-curator framework.

NeMo Curator provides a task-centric pipeline framework for downloading and processing large-scale public text datasets. The framework uses Ray as the distributed backend and converts raw data formats like Common Crawl's `.warc.gz` to processing-friendly formats like `.jsonl`.

## How it Works

Ray-curator's data loading framework uses a **4-step pipeline pattern** where data flows through stages as tasks:

1. **URL Generation**: Generate URLs from configuration (`URLGenerationStage`)
2. **Download**: Retrieve files from URLs to local storage (`DocumentDownloadStage`)
3. **Iteration**: Parse downloaded files to extract raw records (`DocumentIterateStage`)
4. **Extraction**: Extract and clean structured content from raw records (`DocumentExtractStage`)

Each step uses a `ProcessingStage` that transforms tasks. The pipeline flow is:

```text
_EmptyTask → FileGroupTask(URLs) → FileGroupTask(Files) → DocumentBatch → DocumentBatch
```

Data sources provide composite stages that combine these steps into complete download-extract pipelines, producing `DocumentBatch` tasks for further processing.

::::{tab-set}

:::{tab-item} Python

```python
from ray_curator.pipeline import Pipeline
from ray_curator.backends.xenna import XennaExecutor
from ray_curator.stages.download.text.common_crawl import CommonCrawlDownloadExtractStage
from ray_curator.stages.io.writer import JsonlWriter
from ray_curator.tasks import EmptyTask

# Create a pipeline for downloading Common Crawl data
pipeline = Pipeline(
    name="common_crawl_download",
    description="Download and process Common Crawl web archives"
)

# Add data loading stage
cc_stage = CommonCrawlDownloadExtractStage(
    start_snapshot="2020-50",
    end_snapshot="2020-50", 
    download_dir="/tmp/cc_downloads",
    crawl_type="main",
    url_limit=10  # Limit for testing
)
pipeline.add_stage(cc_stage)

# Add writer stage to save as JSONL
writer = JsonlWriter(output_dir="/output/folder")
pipeline.add_stage(writer)

# Build and execute pipeline
pipeline.build()
executor = XennaExecutor()

# Start with an empty task to trigger URL generation
results = pipeline.run(executor, initial_tasks=[EmptyTask])
```

:::

:::{tab-item} Reading Custom Data

```python
from ray_curator.pipeline import Pipeline
from ray_curator.stages.io.reader import JsonlReader
from ray_curator.stages.modules import ScoreFilter
from ray_curator.stages.filters import WordCountFilter

# Create pipeline for processing existing JSONL files
pipeline = Pipeline(name="custom_data_processing")

# Read JSONL files
reader = JsonlReader(
    file_paths="/path/to/data/*.jsonl",
    files_per_partition=4,
    columns=["text", "url"]  # Only read specific columns
)
pipeline.add_stage(reader)

# Add filtering stage
word_filter = ScoreFilter(
    filter_obj=WordCountFilter(min_words=50, max_words=1000),
    text_field="text"
)
pipeline.add_stage(word_filter)

# Execute pipeline
executor = XennaExecutor()
results = pipeline.run(executor)
```

:::

::::

---

## Data Sources & File Formats

Load data from public datasets and custom data sources using ray-curator stages.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Common Crawl
:link: text-load-data-common-crawl
:link-type: ref
Download and process web archive data from Common Crawl
+++
{bdg-secondary}`web-data`
{bdg-secondary}`warc`
{bdg-secondary}`html-extraction`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Wikipedia
:link: text-load-data-wikipedia  
:link-type: ref
Download and extract Wikipedia articles from Wikipedia dumps
+++
{bdg-secondary}`articles`
{bdg-secondary}`multilingual`
{bdg-secondary}`xml-dumps`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Custom Data
:link: text-load-data-custom
:link-type: ref
Read and process your own text datasets in standard formats
+++
{bdg-secondary}`jsonl`
{bdg-secondary}`parquet`
{bdg-secondary}`file-partitioning`
:::

::::

## Key Components

### Tasks

Ray-curator operates on **Tasks** - batches of data that flow through the pipeline:

- **`_EmptyTask`**: Starting point for pipelines that generate data
- **`FileGroupTask`**: Contains file paths (URLs or local files)  
- **`DocumentBatch`**: Contains text documents as pandas DataFrame or PyArrow Table

### Stages

**ProcessingStages** transform tasks through the pipeline:

- **Composite Stages**: High-level stages like `CommonCrawlDownloadExtractStage` that decompose into several steps
- **Atomic Stages**: Individual processing steps like `DocumentDownloadStage`, `JsonlReaderStage`
- **I/O Stages**: File readers (`JsonlReader`) and writers (`JsonlWriter`, `ParquetWriter`)

### Executors

**Executors** run pipelines on different backends:

- **`XennaExecutor`**: Production-ready executor using Cosmos framework
- **`RayDataExecutor`**: Experimental executor using Ray Data

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

arxiv
common-crawl
wikipedia
Custom Data <custom.md>
```
