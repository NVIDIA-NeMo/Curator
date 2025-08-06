---
description: "Download and extract text from Wikipedia dumps with support for multiple languages and automatic content processing"
categories: ["how-to-guides"]
tags: ["wikipedia", "dumps", "multilingual", "articles", "data-loading"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-load-data-wikipedia)=
# Wikipedia

Download and extract text from [Wikipedia Dumps](https://dumps.wikimedia.org/backup-index.html) using NeMo Curator utilities.

Wikipedia regularly releases dumps of all its content, which include articles, talk pages, user pages, and more. These dumps are available in various formats, including XML and SQL.

## How it Works

NeMo Curator simplifies the process of:

- Downloading the latest Wikipedia dump
- Extracting the article content
- Converting the content to a usable format for language model training

## Before You Start

NeMo Curator uses `wget` to download Wikipedia dumps. You must have `wget` installed on your system:

- **On macOS**:  `brew install wget`
- **On Ubuntu/Debian**: `sudo apt-get install wget`
- **On CentOS/RHEL**:  `sudo yum install wget`

---

## Usage

Here's how to download and extract Wikipedia data using NeMo Curator:

::::{tab-set}

:::{tab-item} Python

```python
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.download import download_wikipedia

# Initialize a Dask client
client = get_client(cluster_type="cpu")

# Download and extract Wikipedia
wikipedia_dataset = download_wikipedia(
    output_path="/extracted/output/folder",
    dump_date="20240401"  # Optional: specific dump date
)

# The dataset is now available as a DocumentDataset object
print(f"Downloaded {len(wikipedia_dataset)} articles")
print(wikipedia_dataset.head())

# Write the dataset to disk as JSONL files
wikipedia_dataset.to_json(output_path="/path/to/output/files")
```

:::

:::{tab-item} CLI
NeMo Curator provides a CLI for downloading Wikipedia data. 

**Step 1: Generate Wikipedia URLs**

First, generate a list of Wikipedia dump URLs for the desired language:

```bash
get_wikipedia_urls \
  --language=en \
  --output-url-file=./wikipedia_urls.txt
```

**Step 2: Create Configuration File**

Create a configuration file (`wikipedia_builder.yaml`):

```yaml
download_module: nemo_curator.download.wikipedia.WikipediaDownloader
download_params: {}
iterator_module: nemo_curator.download.wikipedia.WikipediaIterator
iterator_params:
  language: 'en'
extract_module: nemo_curator.download.wikipedia.WikipediaExtractor
extract_params:
  language: 'en'
format:
  text: str
  title: str
  id: str
  url: str
  language: str
  source_id: str
```

**Step 3: Run Download and Extraction**

```bash
download_and_extract \
  --input-url-file=./wikipedia_urls.txt \
  --builder-config-file=./wikipedia_builder.yaml \
  --output-json-dir=/datasets/wikipedia/json
```
:::

::::

### Parameters

```{list-table}
:header-rows: 1
:widths: 20 20 20 40

* - Parameter
  - Type
  - Default
  - Description
* - `output_path`
  - str
  - Required
  - Path where the extracted files will be placed
* - `dump_date`
  - Optional[str]
  - None
  - Parameter to specify a particular Wikipedia dump date. The format must be "YYYYMMDD" (for example, "20250401" for April 1, 2025). Wikipedia creates new dumps approximately twice a month (around the 1st and 20th). You can find available dump dates by visiting https://dumps.wikimedia.org/enwiki/. If not specified, NeMo Curator will automatically use the latest available dump.
* - `language`
  - str
  - "en"
  - Language code to download (for example, "en" for English)
* - `url_limit`
  - Optional[int]
  - None
  - Parameter to limit the number of URLs downloaded (useful for testing)
```

If no `dump_date` is specified, NeMo Curator will download the latest available dump.

## Output Format

The extracted Wikipedia articles are stored in `.jsonl` files, with each line containing a JSON object with fields:

- `text`: The main text content of the article
- `id`: A unique identifier for the article
- `title`: The title of the Wikipedia article
- `url`: The URL of the Wikipedia article
- `language`: The language code of the article
- `source_id`: The source file identifier
- `file_name`: The output file name (when using `write_to_filename=True`)

