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

This guide covers the core concepts for acquiring and processing text data from remote sources in NeMo Curator. Data acquisition focuses on downloading, extracting, and converting remote data sources into `DocumentBatch` format for further processing.

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
class CustomURLGenerator(URLGenerator):
    def __init__(self, base_urls):
        super().__init__()
        self._base_urls = base_urls
    
    def generate_urls(self):
        # Custom URL generation logic
        urls = []
        for base_url in self._base_urls:
            # Discover or construct actual download URLs
            urls.extend(discover_files_at_url(base_url))
        return urls
```

### DocumentDownloader

Connects to and downloads data from remote repositories. You must override `_get_output_filename` and `_download_to_path` which are called by function called `download` which tries to be idempotent.

**Example Implementation**:

```python
class CustomDownloader(DocumentDownloader):
    def __init__(self, download_dir):
        super().__init__()
        self._download_dir = download_dir
    
    def _get_output_filename(self, url):
        # Extract filename from URL
        return url.split('/')[-1]
    
    def _download_to_path(self, url, path):
        # Custom download logic
        # Return (success_bool, error_message)
        try:
            # ... download implementation ...
            return True, None
        except Exception as e:
            return False, str(e)
```

### DocumentIterator

Extracts individual records from downloaded containers. You should only override `iterate` and `output_columns` where `iterate` must have logic to load the local file path and return bunch of documents. The list[dict] is finally considered to a pd.DataFrame which is passed to Extractor.

**Example Implementation**:

```python
class CustomIterator(DocumentIterator):
    def __init__(self, log_frequency=1000):
        super().__init__()
        self._log_frequency = log_frequency
    
    def iterate(self, file_path):
        # Custom iteration logic to load local file and return documents
        for record in load_local_file(file_path):
            yield {"content": record_content, "metadata": record_metadata}
    
    def output_columns(self):
        return ["content", "metadata"]
```

### DocumentExtractor (Optional)

DocumentExtractor works on the pd.DataFrame and is optional.

**Example Implementation**:

```python
class CustomExtractor(DocumentExtractor):
    def __init__(self):
        super().__init__()
    
    def extract(self, record):
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

## Integration with Pipeline Architecture

The data acquisition process seamlessly integrates with NeMo Curator's pipeline-based architecture:

### Acquisition Workflow

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.download import DocumentDownloadExtractStage, URLGenerator

# Define acquisition pipeline
pipeline = Pipeline(name="data_acquisition")

# Create download and extract stage with custom components
download_extract_stage = DocumentDownloadExtractStage(
    url_generator=CustomURLGenerator(data_urls),
    downloader=CustomDownloader(download_dir),
    iterator=CustomIterator(),
    extractor=CustomExtractor()
)
pipeline.add_stage(download_extract_stage)

# Execute acquisition pipeline
results = pipeline.run()

```

### Scaling and Parallel Processing

The `DocumentDownloadExtractStage` handles parallel processing through the distributed computing framework:

```python
# Simple pipeline - parallelism handled automatically
pipeline = Pipeline(name="scalable_acquisition")

# The composite stage handles URL generation and parallel downloads internally
download_extract_stage = DocumentDownloadExtractStage(
    url_generator=CustomURLGenerator(base_urls),  # Generates URLs from config
    downloader=CustomDownloader(download_dir),
    iterator=CustomIterator(),
    extractor=CustomExtractor(),
    url_limit=1000  # Limit number of URLs to process
)
pipeline.add_stage(download_extract_stage)

# Execute with distributed executor for automatic scaling
results = pipeline.run(executor)
```

## Performance Optimization

### Parallel Processing

Data acquisition leverages distributed computing frameworks for scalable processing:

- **Parallel Downloads**: Each URL in the generated list downloads through separate workers
- **Concurrent Extraction**: Files process in parallel across workers
- **Memory Management**: Streaming processing for large files

## Integration with Data Loading

Data acquisition produces standardized output that integrates seamlessly with {ref}`Data Loading Concepts <about-concepts-text-data-loading>`:

```{note}
Data acquisition includes basic content-level deduplication during extraction (e.g., removing duplicate HTML content within individual web pages). This is separate from the main deduplication pipeline stages (exact, fuzzy, and semantic deduplication) that operate on the full dataset after acquisition.
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
results = acquisition_pipeline.run(executor)

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