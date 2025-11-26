---
description: "Core concepts for acquiring text data from remote sources including DocumentDownloader, DocumentIterator, and DocumentExtractor components"
categories: ["concepts-architecture"]
tags: ["data-acquisition", "remote-sources", "download", "extract", "distributed"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "text-only"
---

(about-concepts-text-data-acquisition)=

# Data Acquisition Concepts

This guide covers the core concepts for acquiring and processing text data from remote sources in NeMo Curator. Data acquisition focuses on downloading, extracting, and converting remote data sources into the `DocumentBatch` format for further processing.

## Overview

Data acquisition in NeMo Curator follows a four-stage architecture:

1. **Generate URLs**: Discover and generate download URLs from minimal input
2. **Download**: Retrieve raw data files from remote sources
3. **Iterate**: Extract individual records from downloaded containers
4. **Extract**: Convert raw content to clean, structured text

This process transforms diverse remote data sources into a standardized `DocumentBatch` that can be used throughout the text curation pipeline.

## Core Components

The data acquisition framework consists of four abstract base classes that define the acquisition workflow:

### URLGenerator

Generates URLs for downloading from minimal input configuration. You need to override `generate_urls` which generates a bunch of URLs that user wants to download.

**Example Implementation**:

```python
from dataclasses import dataclass
from nemo_curator.stages.text.download import URLGenerator

@dataclass
class CustomURLGenerator(URLGenerator):
    def generate_urls(self):
        # Custom URL generation logic
        urls = []
        ...
        return urls
```

### DocumentDownloader

Connects to and downloads data from remote repositories. You must override `_get_output_filename` and `_download_to_path` which are called by an underlying function called `download` which tries to be idempotent.

**Example Implementation**:

```python
from nemo_curator.stages.text.download import DocumentDownloader

class CustomDownloader(DocumentDownloader):
    def __init__(self, download_dir: str):
        super().__init__(download_dir=download_dir)
    
    def _get_output_filename(self, url: str) -> str:
        # Custom logic to extract filename from URL
        return url.split("/")[-1]
    
    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        # Custom download logic
        # Return (success_bool, error_message)
        try:
            # ... download implementation ...
            return True, None
        except Exception as e:
            return False, str(e)
```

### DocumentIterator

Extracts individual records from downloaded containers. You should only override `iterate` and `output_columns` where `iterate` must have logic to load the local file path and return bunch of documents. The `list[dict]` is finally considered to a Pandas DataFrame which is passed to Extractor.

**Example Implementation**:

```python
from collections.abc import Iterator
from typing import Any
from nemo_curator.stages.text.download import DocumentIterator

class CustomIterator(DocumentIterator):
    def __init__(self, log_frequency: int = 1000):
        super().__init__()
        self._log_frequency = log_frequency
    
    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        # Custom iteration logic to load local file and return documents
        for record in load_local_file_fn(file_path):
            yield {"content": record_content, "metadata": record_metadata}
    
    def output_columns(self) -> list[str]:
        return ["content", "metadata"]
```

### DocumentExtractor (Optional)

DocumentExtractor works on a Pandas DataFrame and is optional.

**Example Implementation**:

```python
from typing import Any
from nemo_curator.stages.text.download import DocumentExtractor

class CustomExtractor(DocumentExtractor):
    def __init__(self):
        super().__init__()
    
    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        # Custom extraction logic
        cleaned_text = clean_content(record["content"])
        detected_lang = detect_language(cleaned_text)
        return {"text": cleaned_text, "language": detected_lang}
    
    def input_columns(self):
        return ["content", "metadata"]
    
    def output_columns(self):
        return ["text", "language"]
```

## Supported Data Sources

NeMo Curator provides built-in support for major public text datasets:

::::{grid} 2 2 2 3
:gutter: 2

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Common Crawl
:link: text-load-data-common-crawl
:link-type: ref

Download and extract web archive data from Common Crawl
+++
{bdg-secondary}`web-scale` {bdg-secondary}`multilingual`
:::

:::{grid-item-card} {octicon}`typography;1.5em;sd-mr-1` ArXiv
:link: text-load-data-arxiv
:link-type: ref

Download and extract scientific papers from arXiv
+++
{bdg-secondary}`academic` {bdg-secondary}`scientific`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Wikipedia
:link: text-load-data-wikipedia
:link-type: ref

Download and extract Wikipedia articles from Wikipedia dumps
+++
{bdg-secondary}`encyclopedic` {bdg-secondary}`structured`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Custom Data Sources
:link: text-load-data-custom
:link-type: ref

Implement a download and extract pipeline for a custom data source
+++
{bdg-secondary}`extensible` {bdg-secondary}`specialized`
:::

::::

## Integration with Pipeline Architecture

The data acquisition process seamlessly integrates with NeMo Curator's pipeline-based architecture. The `DocumentDownloadExtractStage` handles parallel processing through the distributed computing framework.

### Acquisition Workflow

```python
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.download import DocumentDownloadExtractStage
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.base import ProcessingStage

# Create composite stage
class CustomDownloadExtractStage(DocumentDownloadExtractStage):
    def __init__(
        self,
        download_dir: str = "./custom_downloads",
        url_limit: int | None = None,
        record_limit: int | None = None,
        add_filename_column: bool | str = True,
    ):
        # Create the URL generator
        self.url_generator = CustomURLGenerator()

        # Create the downloader
        self.downloader = CustomDownloader(download_dir=download_dir)

        # Create the iterator
        self.iterator = CustomIterator()

        # Create the extractor
        self.extractor = CustomExtractor()

        # Initialize the parent composite stage
        super().__init__(
            url_generator=self.url_generator,
            downloader=self.downloader,
            iterator=self.iterator,
            extractor=self.extractor,
            url_limit=url_limit,
            record_limit=record_limit,
            add_filename_column=add_filename_column,
        )
        self.name = "custom_pipeline"

    def decompose(self) -> list[ProcessingStage]:
        """Decompose this composite stage into its constituent stages."""
        return self.stages

    def get_description(self) -> str:
        """Get a description of this composite stage."""
        return "Custom pipeline"

# Initialize Ray client
ray_client = RayClient()
ray_client.start()

# Define acquisition pipeline
pipeline = Pipeline(name="data_acquisition")

# Create download and extract stage with custom components
custom_download_extract_stage = CustomDownloadExtractStage(...)
pipeline.add_stage(custom_download_extract_stage)

# Write the results
pipeline.add_stage(JsonlWriter(...))

# Execute acquisition pipeline
results = pipeline.run()

# Stop Ray client
ray_client.stop()
```

## Performance Optimization

### Parallel Processing

Data acquisition leverages distributed computing frameworks for scalable processing:

- **Parallel Downloads**: Each URL in the generated list downloads through separate workers
- **Concurrent Extraction**: Files process in parallel across workers
- **Memory Management**: Streaming processing for large files

## Integration with Data Loading

Data acquisition produces a standardized output that integrates seamlessly with Curator's {ref}`Data Loading Concepts <about-concepts-text-data-loading>`:

```{note}
Data acquisition includes basic content-level deduplication during extraction (such as removing duplicate HTML content within individual web pages). This is separate from the main deduplication pipeline stages (exact, fuzzy, and semantic deduplication) that operate on the full dataset after acquisition.
```

```python
from nemo_curator.stages.text.io.writer import ParquetWriter

# Create acquisition pipeline with all stages including writer
acquisition_pipeline = Pipeline(name="data_acquisition")
# ... add acquisition stages ...

# Add writer to save results directly
writer = ParquetWriter(path="acquired_data/")
acquisition_pipeline.add_stage(writer)

# Run pipeline to acquire and save data in one execution
results = acquisition_pipeline.run()

# Later: Load using pipeline-based data loading
from nemo_curator.stages.text.io.reader import ParquetReader

load_pipeline = Pipeline(name="load_acquired_data")
reader = ParquetReader(file_paths="acquired_data/")
load_pipeline.add_stage(reader)
```

This enables you to:

- **Separate acquisition from processing** for better workflow management
- **Cache acquired data** to avoid re-downloading
- **Mix acquired and local data** in the same processing pipeline
- **Use standard loading patterns** regardless of data origin
