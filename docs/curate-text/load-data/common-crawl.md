---
description: "Download and extract text from Common Crawl web archives with language detection and multiple text extraction algorithms"
categories: ["how-to-guides"]
tags: ["common-crawl", "web-data", "warc", "language-detection", "distributed", "html-extraction"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-load-data-common-crawl)=
# Common Crawl

Download and extract text from Common Crawl snapshots using NeMo Curator utilities.

Common Crawl provides petabytes of web data collected over years of web crawling. The data is stored in a compressed web archive format (`.warc.gz`), which needs to be processed to extract useful text for language model training.

## How it Works

NeMo Curator's Common Crawl extraction process:

1. Downloads the compressed WARC files from Common Crawl's servers (optionally using S3 for faster downloads)
2. Decodes the HTML within each record from binary to text
3. Performs language detection using [pyCLD2](https://github.com/aboSamoor/pycld2)
4. Extracts the relevant text using one of several text extraction algorithms
5. Outputs the extracted text as `.jsonl` files for further processing

---

## Usage

Here's how to download and extract Common Crawl data:

::::{tab-set}

:::{tab-item} Python
```python
import os
from nemo_curator import get_client
from nemo_curator.download import download_common_crawl
from nemo_curator.datasets import DocumentDataset

def main():
    # Initialize a Dask client
    client = get_client(cluster_type="cpu")

    # Set parameters for downloading
    output_path = "/extracted/output/folder"
    start_snapshot = "2020-50"
    end_snapshot = "2021-04"
    output_type = "jsonl"
    os.makedirs(output_path, exist_ok=True)

    # Download and extract Common Crawl data
    common_crawl_dataset = download_common_crawl(
        output_path, start_snapshot, end_snapshot, output_type=output_type
    )

    # Write the dataset to disk
    common_crawl_dataset.to_json(output_path=output_path, write_to_filename=True)
    print("Extracted dataset saved to:", output_path)

if __name__ == "__main__":
    main()
```
:::

:::{tab-item} CLI
First, generate a list of URLs:

```bash
get_common_crawl_urls \
  --starting-snapshot="2020-50" \
  --ending-snapshot="2020-50" \
  --output-warc-url-file=./url_data/warc_urls_cc_2020_50.txt
```

Then download and extract:

```bash
download_and_extract \
  --input-url-file=./url_data/warc_urls_cc_2020_50.txt \
  --builder-config-file=./config/cc_warc_builder.yaml \
  --output-json-dir=/datasets/CC-MAIN-2020-50/json
```

The config file should look like:

```yaml
download_module: nemo_curator.download.commoncrawl.CommonCrawlWARCDownloader
download_params:
  aws: True  # Optional: Set to True to use S3 for faster downloads
iterator_module: nemo_curator.download.commoncrawl.CommonCrawlWARCIterator
iterator_params: {}
extract_module: nemo_curator.download.commoncrawl.CommonCrawlWARCExtractor
extract_params: {}
```

```{note}
The `download_params` section can include optional parameters like `aws: True` for S3 downloads or `verbose: True` for detailed logging. If no custom parameters are needed, use `download_params: {}`.
```

:::

::::

### Parameters

```{list-table} Common Crawl Download Parameters
:header-rows: 1
:widths: 20 20 40 20

* - Parameter
  - Type
  - Description
  - Default
* - `output_path`
  - str
  - Path where the extracted files will be placed
  - Required
* - `start_snapshot`
  - str
  - First Common Crawl snapshot to include (format: "YYYY-WW" for CC-MAIN, "YYYY-MM" for CC-NEWS)
  - Required
* - `end_snapshot`
  - str
  - Last Common Crawl snapshot to include
  - Required
* - `output_type`
  - Literal["jsonl", "parquet"]
  - File format for storing data
  - "jsonl"
* - `algorithm`
  - HTMLExtractorAlgorithm
  - Text extraction algorithm to use (JusTextExtractor, ResiliparseExtractor, or TrafilaturaExtractor)
  - JusTextExtractor()
* - `stop_lists`
  - Optional[Dict[str, frozenset]]
  - Dictionary of language-specific stop words
  - None
* - `news`
  - bool
  - Whether to use CC-NEWS dataset instead of CC-MAIN
  - False
* - `aws`
  - bool
  - Whether to download from S3 using s5cmd instead of HTTPS (requires s5cmd to be installed)
  - False
* - `raw_download_dir`
  - Optional[str]
  - Directory to store raw WARC files
  - None
* - `keep_raw_download`
  - bool
  - Whether to keep the raw downloaded files
  - False
* - `force_download`
  - bool
  - Whether to force re-download even if files exist
  - False
* - `url_limit`
  - Optional[int]
  - Maximum number of WARC files to download
  - None
* - `record_limit`
  - Optional[int]
  - Maximum number of records to extract per file
  - None
```

```{admonition} Snapshot Availability
:class: note

Not every year and week has a snapshot. Ensure your range includes at least one valid Common Crawl snapshot. See [the official site](https://data.commoncrawl.org/) for a list of valid snapshots.
```

## Output Format

The extracted text is stored in `.jsonl` files with the following format:

```json
{
  "text": "Extracted web page content...",
  "warc_id": "a515a7b6-b6ec-4bed-998b-8be2f86f8eac",
  "source_id": "CC-MAIN-20201123153826-20201123183826-00000.warc.gz",
  "url": "http://example.com/page.html",
  "language": "ENGLISH",
  "file_name": "CC-MAIN-20201123153826-20201123183826-00000.warc.gz.jsonl"
}
```

## Customization Options

### Text Extraction

NeMo Curator supports multiple HTML text extraction algorithms:

1. **JusTextExtractor** (default): Uses [jusText](https://github.com/miso-belica/jusText) to extract main content
2. **ResiliparseExtractor**: Uses [Resiliparse](https://github.com/chatnoir-eu/chatnoir-resiliparse) for extraction
3. **TrafilaturaExtractor**: Uses [Trafilatura](https://trafilatura.readthedocs.io/en/latest/) for extraction

You can select a different extractor as follows:

```python
from nemo_curator.download import (
    ResiliparseExtractor,
    TrafilaturaExtractor,
    download_common_crawl
)

# Use Resiliparse for extraction
extraction_algorithm = ResiliparseExtractor()

common_crawl_dataset = download_common_crawl(
    output_path,
    start_snapshot,
    end_snapshot,
    output_type=output_type,
    algorithm=extraction_algorithm,
)
```

Each extractor has unique parameters -- check their docstrings for details.

### Language Processing

You can customize language detection and extraction by providing [stop words](text-process-data-languages-stop-words) for different languages:

```python
from nemo_curator.download import download_common_crawl

# Define custom stop words for specific languages
stop_lists = {"ENGLISH": frozenset(["the", "and", "is", "in", "for", "where", "when", "to", "at"])}

common_crawl = download_common_crawl(
    "/extracted/output/folder",
    "2020-50",
    "2021-04",
    output_type="jsonl",
    stop_lists=stop_lists,
)
```

```{note}
If no custom stop lists are provided, NeMo Curator uses jusText's default stop lists with additional support for Thai, Chinese, and Japanese languages.
```
