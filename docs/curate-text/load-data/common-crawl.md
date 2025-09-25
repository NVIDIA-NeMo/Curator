---
description: "Download and extract text from Common Crawl web archives using Curator."
categories: ["how-to-guides"]
tags: ["common-crawl", "web-data", "warc", "language-detection", "distributed", "html-extraction", "pipeline"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-load-data-common-crawl)=

# Common Crawl

Download and extract text from Common Crawl snapshots using Curator.

Common Crawl provides petabytes of web data collected over years of web crawling. The data uses a compressed web archive format (`.warc.gz`), which requires processing to extract useful text for language model training.

## How it Works

Curator's Common Crawl processing pipeline consists of four sequential stages:

1. **URL Generation**: Generates WARC file URLs from Common Crawl's index for the specified snapshot range
2. **Download**: Downloads the compressed WARC files from Common Crawl's servers (optionally using S3 for faster downloads)
3. **Iteration**: Extracts individual records from WARC files and decodes HTML content
4. **Extraction**: Performs language detection and extracts clean text using configurable HTML extraction algorithms

The pipeline outputs structured data that you can write to JSONL or Parquet files for further processing.

## Before You Start

Choose your download method and ensure you have the prerequisites:

- HTTPS downloads (default): No AWS account required.
- S3 downloads (set `use_aws_to_download=True`):
  - An AWS account with credentials configured (profile, environment, or instance role).
  - Common Crawl's S3 access uses Requester Pays; you incur charges for requests and data transfer.
  - `s5cmd` installed for fast S3 listing and copy operations:

```bash
# Install s5cmd for faster S3 downloads
pip install s5cmd
```

---

## Usage

Here's how to create and run a Common Crawl processing pipeline:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.download import CommonCrawlDownloadExtractStage
from nemo_curator.stages.text.io.writer import JsonlWriter

def main():
    # Create pipeline
    pipeline = Pipeline(
        name="common_crawl_pipeline",
        description="Download and process Common Crawl data"
    )

    # Add Common Crawl processing stage
    cc_stage = CommonCrawlDownloadExtractStage(
        start_snapshot="2020-50",  # YYYY-WW format for CC-MAIN
        end_snapshot="2020-50",
        download_dir="./cc_downloads",
        crawl_type="main",  # or "news"
        use_aws_to_download=True,  # Faster S3 downloads (requires s5cmd)
        url_limit=10,  # Limit number of WARC files for testing
        record_limit=1000,  # Limit records per WARC file
    )
    pipeline.add_stage(cc_stage)

    # Add output writer stage
    writer = JsonlWriter(output_dir="./cc_output")
    pipeline.add_stage(writer)

    # Run pipeline
    results = pipeline.run()

    print(f"Pipeline completed. Results: {len(results) if results else 0} output files")

if __name__ == "__main__":
    main()
```

For how pipelines execute across backends (`XennaExecutor`, `RayDataExecutor`), refer to {ref}`reference-execution-backends`.

### Writing to Parquet

To write Parquet instead of JSONL, use `ParquetWriter`:

```python
from nemo_curator.stages.text.io.writer import ParquetWriter

# Replace the JSONL writer with ParquetWriter
writer = ParquetWriter(output_dir="./cc_output_parquet")
pipeline.add_stage(writer)
```

### Parameters

```{list-table} CommonCrawlDownloadExtractStage Parameters
:header-rows: 1
:widths: 25 20 35 20

* - Parameter
  - Type
  - Description
  - Default
* - `start_snapshot`
  - str
  - First snapshot to include (format: "YYYY-WW" for main, "YYYY-MM" for news). Not every year and week has a snapshot; refer to the official list at `https://data.commoncrawl.org/`.
  - Required
* - `end_snapshot`
  - str
  - Last snapshot to include (same format as start_snapshot). Ensure your range includes at least one valid snapshot.
  - Required
* - `download_dir`
  - str
  - Directory to store downloaded WARC files
  - Required
* - `crawl_type`
  - Literal["main", "news"]
  - Whether to use CC-MAIN or CC-NEWS dataset
  - "main"
* - `html_extraction`
  - HTMLExtractorAlgorithm | str | None
  - Text extraction algorithm to use. Defaults to `JusTextExtractor()` if not specified.
  - JusTextExtractor() if not specified
* - `html_extraction_kwargs`
  - dict | None
  - Additional arguments for the HTML extractor. Ignored when `html_extraction` is a concrete extractor object (for example, `JusTextExtractor()`); pass kwargs to the extractor constructor instead. When `html_extraction` is a string ("justext", "resiliparse", or "trafilatura"), kwargs are forwarded.
  - None
* - `stop_lists`
  - dict[str, frozenset[str]] | None
  - Language-specific stop words for text quality assessment. If not provided, Curator uses jusText defaults with additional support for Thai, Chinese, and Japanese languages.
  - None
* - `use_aws_to_download`
  - bool
  - Use S3 downloads via s5cmd instead of HTTPS (requires s5cmd installation)
  - False
* - `verbose`
  - bool
  - Enable verbose logging for download operations
  - False
* - `url_limit`
  - int | None
  - Maximum number of WARC files to download (useful for testing)
  - None
* - `record_limit`
  - int | None
  - Maximum number of records to extract per WARC file
  - None
* - `add_filename_column`
  - bool | str
  - Whether to add source filename column to output; if str, uses it as the column name (default name: "file_name")
  - True
```

## Output Format

The pipeline processes Common Crawl data through several stages, ultimately producing structured documents. The extracted text includes the following fields:

```json
{
  "url": "http://example.com/page.html",
  "warc_id": "a515a7b6-b6ec-4bed-998b-8be2f86f8eac", 
  "source_id": "CC-MAIN-20201123153826-20201123183826-00000.warc.gz",
  "language": "ENGLISH",
  "text": "Extracted web page content..."
}
```

```{list-table} Output Fields
:header-rows: 1
:widths: 20 80

* - Field
  - Description
* - `url`
  - Original URL of the web page
* - `warc_id`
  - Unique identifier for the WARC record
* - `source_id`
  - Name of the source WARC file
* - `language`
  - Detected language of the content (e.g., "ENGLISH", "SPANISH")
* - `text`
  - Extracted and cleaned text content
```

If you enable `add_filename_column`, the output includes an extra field `file_name` (or your custom column name).

## Customization Options

### HTML Text Extraction Algorithms

Curator supports several HTML text extraction algorithms:

```{list-table} Available HTML Extractors
:header-rows: 1
:widths: 30 70

* - Extractor
  - Library
* - `JusTextExtractor`
  - [jusText](https://github.com/miso-belica/jusText)
* - `ResiliparseExtractor`
  - [Resiliparse](https://github.com/chatnoir-eu/chatnoir-resiliparse)
* - `TrafilaturaExtractor`
  - [Trafilatura](https://trafilatura.readthedocs.io/)
```

#### Configuring HTML Extractors

```python
from nemo_curator.stages.text.download.html_extractors import ResiliparseExtractor
from nemo_curator.stages.text.download.html_extractors import TrafilaturaExtractor

# Use Resiliparse for extraction
cc_stage = CommonCrawlDownloadExtractStage(
    start_snapshot="2020-50",
    end_snapshot="2020-50",
    download_dir="./downloads",
    html_extraction=ResiliparseExtractor(
        required_stopword_density=0.25,
        main_content=True
    )
)

# Or use Trafilatura with custom parameters
cc_stage = CommonCrawlDownloadExtractStage(
    start_snapshot="2020-50", 
    end_snapshot="2020-50",
    download_dir="./downloads",
    html_extraction=TrafilaturaExtractor(
        min_extracted_size=200,
        max_repetitions=3
    )
)
```

### Language Processing

You can customize language detection and extraction by providing stop words for different languages:

```python
# Define custom stop words for specific languages
stop_lists = {
    "ENGLISH": frozenset(["the", "and", "is", "in", "for", "where", "when", "to", "at"]),
    "SPANISH": frozenset(["el", "la", "de", "que", "y", "en", "un", "es", "se", "no"])
}

cc_stage = CommonCrawlDownloadExtractStage(
    start_snapshot="2020-50",
    end_snapshot="2020-50", 
    download_dir="./downloads",
    stop_lists=stop_lists
)
```

## Advanced Usage

### Using Different Executors

Curator supports several execution backends:

```python
from nemo_curator.backends.xenna import XennaExecutor
executor = XennaExecutor()

# Experimental Ray Data executor (optional)
from nemo_curator.backends.experimental.ray_data.executor import RayDataExecutor
executor = RayDataExecutor()
```

### Processing CC-NEWS Data

For Common Crawl News data, use the `news` crawl type with month-based snapshots:

```python
cc_stage = CommonCrawlDownloadExtractStage(
    start_snapshot="2020-08",  # YYYY-MM format for CC-NEWS
    end_snapshot="2020-10",
    download_dir="./news_downloads",
    crawl_type="news"  # Use CC-NEWS instead of CC-MAIN
)
```

### Large-Scale Processing

For production workloads, consider these optimizations:

```python
cc_stage = CommonCrawlDownloadExtractStage(
    start_snapshot="2020-50",
    end_snapshot="2020-52", 
    download_dir="/fast_storage/cc_downloads",
    use_aws_to_download=True,  # Faster S3 downloads
    verbose=False,  # Reduce logging overhead
    # Remove limits for full processing
    # url_limit=None,
    # record_limit=None
)
```
