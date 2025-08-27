---
description: "Load text data from Common Crawl, Wikipedia, and custom datasets using Curator."
categories: ["workflows"]
tags: ["data-loading", "arxiv", "common-crawl", "wikipedia", "custom-data", "distributed", "ray"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "text-only"
---

(text-load-data)=

# Text Data Loading

Load text data from ArXiv, Common Crawl, Wikipedia, and custom sources using Curator.

Curator provides a task-centric pipeline for downloading and processing large-scale public text datasets. It runs on Ray and converts raw formats like Common Crawl's `.warc.gz` into JSONL.

## How it Works

Curator pipelines use a **4-step pattern** where data flows through stages as tasks:

1. **URL Generation**: Generate URLs from configuration (`URLGenerationStage`)
2. **Download**: Retrieve files from URLs to local storage (`DocumentDownloadStage`)
3. **Iteration**: Parse downloaded files to extract raw records (`DocumentIterateStage`)
4. **Extraction**: Extract and clean structured content from raw records (`DocumentExtractStage`)

Each step uses a `ProcessingStage` that transforms tasks. The pipeline flow is:

```text
Start → FileGroupTask(URLs) → FileGroupTask(Files) → DocumentBatch → DocumentBatch
```

Data sources provide composite stages that combine these steps into complete download-extract pipelines, producing `DocumentBatch` tasks for further processing.

::::{tab-set}

:::{tab-item} Python

```python
from ray_curator.pipeline import Pipeline
from ray_curator.backends.xenna.executor import XennaExecutor
from ray_curator.stages.text.download import CommonCrawlDownloadExtractStage
from ray_curator.stages.io.writer import JsonlWriter

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
writer = JsonlWriter(path="/output/folder")
pipeline.add_stage(writer)

# Build and execute pipeline
pipeline.build()
executor = XennaExecutor()
results = pipeline.run(executor)
```

:::

::::

---

## Data Sources & File Formats

Load data from public datasets and custom data sources using Curator stages.

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

:::{grid-item-card} {octicon}`file;1.5em;sd-mr-1` Read Existing Data (JSONL)
:link: text-load-data-read-existing
:link-type: ref
Read existing JSONL datasets using Curator's reader stage
+++
{bdg-secondary}`jsonl`
:::

::::

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

arxiv
common-crawl
wikipedia
Custom Data <custom.md>
Read Existing Data (JSONL) <read-existing>
```
