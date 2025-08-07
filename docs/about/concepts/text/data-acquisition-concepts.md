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

This guide covers the core concepts for acquiring and processing text data from remote sources in NeMo Curator. Data acquisition focuses on downloading, extracting, and converting remote data sources into {ref}`DocumentDataset <documentdataset>` format for further processing.

## Overview

Data acquisition in NeMo Curator follows a three-stage architecture:

1. **Download**: Retrieve raw data files from remote sources
2. **Iterate**: Extract individual records from downloaded containers
3. **Extract**: Convert raw content to clean, structured text

This process transforms diverse remote data sources into a standardized `DocumentDataset` that can be used throughout the text curation pipeline.

## Core Components

The data acquisition framework consists of three abstract base classes that define the acquisition workflow:

### DocumentDownloader

Connects to and downloads data from remote repositories.

```{list-table}
:header-rows: 1

* - Feature
  - Description
* - Remote Connectivity
  - - Handles authentication and connection management
    - Manages download retries and resume capability
    - Supports various protocols (HTTP, S3, FTP)
    - Optimizes bandwidth usage and parallel downloads
* - State Management
  - - Tracks download progress and completion
    - Handles connection timeouts and failures
    - Manages temporary file storage
    - Provides download resumption capabilities
* - Extensibility
  - - Abstract base class for custom implementations
    - Plugin architecture for new data sources
    - Configurable download parameters
    - Integration with existing authentication systems
```

**Example Implementation**:
```python
class CustomDownloader(DocumentDownloader):
    def __init__(self, download_dir):
        super().__init__()
        self._download_dir = download_dir
    
    def download(self, url):
        # Custom download logic
        output_file = os.path.join(self._download_dir, filename)
        # ... download implementation ...
        return output_file
```

### DocumentIterator

Extracts individual records from downloaded containers.

```{list-table}
:header-rows: 1

* - Feature
  - Description
* - Container Parsing
  - - Handles various archive formats (tar, zip, WARC)
    - Parses XML, JSON, and custom file structures
    - Manages memory-efficient streaming
    - Supports nested container structures
* - Record Extraction
  - - Identifies individual documents within containers
    - Provides document metadata and identifiers
    - Handles malformed or corrupted entries
    - Maintains processing order and lineage
* - Streaming Support
  - - Processes large files without loading entirely into memory
    - Enables real-time processing of incoming data
    - Supports parallel iteration across multiple workers
    - Optimizes I/O operations for performance
```

**Example Implementation**:
```python
class CustomIterator(DocumentIterator):
    def __init__(self, log_frequency=1000):
        super().__init__()
        self._log_frequency = log_frequency
    
    def iterate(self, file_path):
        # Custom iteration logic
        for record in parse_container(file_path):
            yield record_metadata, record_content
```

### DocumentExtractor

Converts raw content formats to clean, structured text.

```{list-table}
:header-rows: 1

* - Feature
  - Description
* - Format Conversion
  - - Converts HTML, PDF, LaTeX to clean text
    - Handles character encoding normalization
    - Preserves important structural information
    - Removes boilerplate and navigation content
* - Content Processing
  - - Performs language detection via pycld2
    - Extracts metadata from document headers
    - Handles multi-language documents
    - Applies content-specific cleaning rules
* - Quality Control
  - - Filters malformed or empty documents
    - Validates text encoding and structure
    - Applies basic quality heuristics
    - Preserves extraction lineage information
```

**Example Implementation**:
```python
class CustomExtractor(DocumentExtractor):
    def __init__(self):
        super().__init__()
    
    def extract(self, content):
        # Custom extraction logic
        cleaned_text = clean_content(content)
        return {"text": cleaned_text, "language": detected_lang}
```

## Supported Data Sources

NeMo Curator provides built-in support for major public text datasets:

::::{grid} 2 2 2 3
:gutter: 2

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Common Crawl
:link: text-load-data-common-crawl
:link-type: ref

Web crawl data in WARC format with content-level deduplication during extraction and quality filtering.
+++
{bdg-secondary}`web-scale` {bdg-secondary}`multilingual`
:::

:::{grid-item-card} {octicon}`typography;1.5em;sd-mr-1` ArXiv
:link: text-load-data-arxiv
:link-type: ref

Academic papers in LaTeX/PDF with automatic metadata extraction and language detection.
+++
{bdg-secondary}`academic` {bdg-secondary}`scientific`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Wikipedia
:link: text-load-data-wikipedia
:link-type: ref

Multi-language Wikipedia dumps with structure preservation and content cleaning.
+++
{bdg-secondary}`encyclopedic` {bdg-secondary}`structured`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Custom Sources
:link: text-load-data-custom
:link-type: ref

Extensible framework for implementing custom data loaders through abstract base classes.
+++
{bdg-secondary}`extensible` {bdg-secondary}`specialized`
:::

::::

## Integration with DocumentDataset

The data acquisition process seamlessly integrates with NeMo Curator's core data structures:

### Acquisition Workflow

```python
from nemo_curator.download import download_and_extract

# Define acquisition components
downloader = CustomDownloader(download_dir)
iterator = CustomIterator()
extractor = CustomExtractor()

# Acquire data and create DocumentDataset
dataset = download_and_extract(
    urls=data_urls,
    output_paths=output_paths,
    downloader=downloader,
    iterator=iterator,
    extractor=extractor,
    output_format={"text": str, "language": str, "url": str}
)

# Result is a standard DocumentDataset ready for processing
print(f"Acquired {len(dataset)} documents")
```

### Batch Processing

For large-scale data acquisition, use the batch processing capabilities:

```python
from nemo_curator.download import batch_download

# Download multiple sources in parallel
downloaded_files = batch_download(urls, downloader)

# Process downloaded files through iteration and extraction
# ... (custom processing logic)
```

## Configuration and Customization

### Configuration Files

Data acquisition components can be configured through YAML files:

```yaml
# downloader_config.yaml
download_module: "my_package.CustomDownloader"
download_params:
  download_dir: "/data/downloads"
  parallel_downloads: 4

iterator_module: "my_package.CustomIterator"
iterator_params:
  log_frequency: 1000

extract_module: "my_package.CustomExtractor"  
extract_params:
  language_detection: true

format:
  text: str
  language: str
  url: str
```

### Dynamic Loading

Load acquisition components dynamically:

```python
from nemo_curator.utils.config_utils import build_downloader

downloader, iterator, extractor, format = build_downloader(
    "downloader_config.yaml"
)
```

## Performance Optimization

### Parallel Processing

Data acquisition leverages `Dask` for distributed processing:

- **Parallel Downloads**: Multiple URLs downloaded simultaneously
- **Concurrent Extraction**: Multiple files processed in parallel
- **Memory Management**: Streaming processing for large files
- **Fault Tolerance**: Automatic retry and recovery mechanisms

### Scaling Strategies

**Single Node**: 
- Use multiple worker processes for CPU-bound extraction
- Optimize I/O operations for local storage
- Balance download and processing throughput

**Multi-Node**:
- Distribute download tasks across cluster nodes
- Use shared storage for intermediate files
- Coordinate processing through `Dask` distributed scheduler

**Cloud Deployment**:
- Leverage cloud storage for source data
- Use auto-scaling for variable workloads  
- Optimize network bandwidth and storage costs

## Integration with Data Loading

Data acquisition produces standardized output that integrates seamlessly with {ref}`Data Loading Concepts <about-concepts-text-data-loading>`:

```{note}
Data acquisition includes basic content-level deduplication during extraction (e.g., removing duplicate HTML content within individual web pages). This is separate from the main deduplication pipeline stages (exact, fuzzy, and semantic deduplication) that operate on the full dataset after acquisition.
```

```python
# Acquisition produces DocumentDataset
acquired_dataset = download_and_extract(...)

# Save in standard format for loading
acquired_dataset.to_parquet("acquired_data/")

# Later: Load using standard data loading
from nemo_curator.datasets import DocumentDataset
dataset = DocumentDataset.read_parquet("acquired_data/")
```

This enables you to:

- **Separate acquisition from processing** for better workflow management
- **Cache acquired data** to avoid re-downloading
- **Mix acquired and local data** in the same processing pipeline
- **Use standard loading patterns** regardless of data origin