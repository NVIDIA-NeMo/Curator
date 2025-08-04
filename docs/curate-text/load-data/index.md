---
description: "Load text data from various sources including Common Crawl, arXiv, Wikipedia, and custom datasets using NeMo Curator"
categories: ["workflows"]
tags: ["data-loading", "common-crawl", "arxiv", "wikipedia", "custom-data", "distributed"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "text-only"
---

(text-load-data)=
# Text Data Loading

Load text data from a variety of data sources using NeMo Curator.

NeMo Curator provides tools for downloading and processing large-scale public text datasets. Common data formats like Common Crawl's `.warc.gz` are automatically converted to more processing-friendly formats like `.jsonl`.

## How it Works

NeMo Curator's data loading framework consists of three main components:

1. **Downloaders**: Responsible for retrieving data from source locations (`DocumentDownloader`)
2. **Iterators**: Parse through downloaded data to identify individual documents (`DocumentIterator`)
3. **Extractors**: Extract and clean text from raw document formats (`DocumentExtractor`)

Each supported data source has specific implementations of these components optimized for that data type. The result is a standardized [`DocumentDataset`](documentdataset) that can be used for further curation steps.

::::{tab-set}

:::{tab-item} Python

```python
from nemo_curator import get_client
from nemo_curator.download import download_common_crawl, download_wikipedia, download_arxiv

# Initialize a Dask client
client = get_client(cluster_type="cpu")

# Download and extract data using correct parameter names
dataset = download_common_crawl(
    output_path="/output/folder", 
    start_snapshot="2020-50", 
    end_snapshot="2021-04"
)

# Write to disk in the desired format
dataset.to_json(output_path="/output/folder", write_to_filename=True)
```

:::

:::{tab-item} CLI

```bash
# Generic download and extract utility
# Requires a YAML configuration file specifying downloader, iterator, and extractor implementations
# Example config files: config/cc_warc_builder.yaml, config/arxiv_builder.yaml, config/wikipedia_builder.yaml
download_and_extract \
  --input-url-file=<Path to URL list> \
  --builder-config-file=<Path to YAML config file> \
  --output-json-dir=<Output directory>

# Alternative: Extract from pre-downloaded files (extraction-only mode)
download_and_extract \
  --input-data-dir=<Path to downloaded files> \
  --builder-config-file=<Path to YAML config file> \
  --output-json-dir=<Output directory>

# Common Crawl URL retrieval utility
# Generates a list of WARC file URLs for specified snapshot range
get_common_crawl_urls \
  --starting-snapshot="2020-50" \
  --ending-snapshot="2020-50" \
  --output-warc-url-file=./warc_urls.txt
```

:::

::::


---

## Data Sources & File Formats

Load data from public, local, and custom data sources.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` arXiv
:link: text-load-data-arxiv
:link-type: ref
Extract and process scientific papers from arXiv
+++
{bdg-secondary}`academic`
{bdg-secondary}`pdf`
{bdg-secondary}`latex`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Common Crawl
:link: text-load-data-common-crawl
:link-type: ref
Load and preprocess text data from Common Crawl web archives
+++
{bdg-secondary}`web-data`
{bdg-secondary}`warc`
{bdg-secondary}`distributed`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Custom Data
:link: text-load-data-custom
:link-type: ref
Load your own text datasets in various formats
+++
{bdg-secondary}`jsonl`
{bdg-secondary}`parquet`
{bdg-secondary}`custom-formats`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Wikipedia
:link: text-load-data-wikipedia
:link-type: ref
Import and process Wikipedia articles for training datasets
+++
{bdg-secondary}`articles`
{bdg-secondary}`multilingual`
{bdg-secondary}`dumps`
:::

::::

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

common-crawl
arxiv
wikipedia
Custom Data <custom.md>
```
