---
description: "Download and extract text from arXiv academic papers using NeMo Curator with LaTeX processing and automatic metadata extraction"
categories: ["how-to-guides"]
tags: ["arxiv", "academic-papers", "latex", "pdf", "data-loading", "scientific-data"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-load-data-arxiv)=
# ArXiv

Download and extract text from ArXiv papers using NeMo Curator utilities.

ArXiv is a free distribution service and open-access archive for scholarly articles, primarily in fields like physics, mathematics, computer science, and more. ArXiv contains millions of scholarly papers, most of them available in LaTeX source format.

## How it Works

NeMo Curator simplifies the process of:

- Downloading ArXiv papers from S3
- Extracting text from LaTeX source files
- Converting the content to a standardized format for further processing

## Before You Start

ArXiv papers are hosted on Amazon S3, so you'll need to have:

1. Properly configured AWS credentials in `~/.aws/config`
2. [s5cmd](https://github.com/peak/s5cmd) installed (pre-installed in the NVIDIA NeMo Framework Container)

---

## Usage

Here's how to download and extract ArXiv data using NeMo Curator:

::::{tab-set}

:::{tab-item} Python

```python
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.download import download_arxiv

# Initialize a Dask client
client = get_client(cluster_type="cpu")

# Download and extract ArXiv papers
arxiv_dataset = download_arxiv(output_path="/extracted/output/folder")

# Write the dataset to disk
arxiv_dataset.to_json(output_path="/extracted/output/folder", write_to_filename=True)
```

:::

:::{tab-item} CLI

```bash
download_and_extract \
  --input-url-file=./arxiv_urls.txt \
  --builder-config-file=./config/arxiv_builder.yaml \
  --output-json-dir=/datasets/arxiv/json
```

The config file should look like:

```yaml
download_module: nemo_curator.download.arxiv.ArxivDownloader
download_params: {}
iterator_module: nemo_curator.download.arxiv.ArxivIterator
iterator_params: {}
extract_module: nemo_curator.download.arxiv.ArxivExtractor
extract_params: {}
```

:::

::::

If you've already downloaded and extracted ArXiv data to the specified output folder, NeMo Curator will read from those files instead of downloading them again.

```{admonition} Text Processing with Stop Words
:class: tip

When processing academic papers from ArXiv, you may want to customize text extraction and analysis using stop words. Stop words can help identify section boundaries, distinguish main content from references, and support language-specific processing. For a comprehensive guide to stop words in NeMo Curator, see {ref}`Stop Words in Text Processing <text-process-data-languages-stop-words>`.
```

### Parameters

```{list-table} ArXiv Download Parameters
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
* - `output_type`
  - Literal["jsonl", "parquet"]
  - File format for storing data
  - "jsonl"
* - `raw_download_dir`
  - Optional[str]
  - Directory to specify where to download the raw ArXiv files
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
  - Limit the number of papers downloaded (useful for testing)
  - None
* - `record_limit`
  - Optional[int]
  - Limit the number of records processed
  - None
```

## Output Format

NeMo Curator extracts and processes the main text content from LaTeX source files. The extractor focuses on the body text of papers, automatically removing:

- Comments and LaTeX markup
- Content before the first section header
- Bibliography and appendix sections
- LaTeX macro definitions (while expanding their usage)

```{admonition} Limited Metadata Extraction
:class: note

The current ArXiv implementation focuses on text extraction and does not parse document metadata like titles, authors, or categories from the LaTeX source. Only the processed text content and basic file identifiers are returned.
```

```{list-table} ArXiv Output Fields
:header-rows: 1
:widths: 20 20 60

* - Field
  - Type
  - Description
* - `text`
  - str
  - The main text content extracted from LaTeX files (cleaned and processed)
* - `id`
  - str
  - A unique identifier for the paper (formatted ArXiv ID)
* - `source_id`
  - str
  - The source tar file name where the paper was found
* - `file_name`
  - str
  - The filename used for the output file
```