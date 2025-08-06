---
description: "Load and process custom datasets using NeMo Curator's extensible framework with custom downloaders, iterators, and extractors"
categories: ["how-to-guides"]
tags: ["custom-data", "extensible", "downloaders", "iterators", "extractors", "framework"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "how-to"
modality: "text-only"
---

(text-load-data-custom)=
# Custom Data Loading

Load and process your own custom datasets using NeMo Curator's extensible framework. This guide explains how to implement custom data loaders that integrate with NeMo Curator's distributed processing capabilities.

## How it Works

NeMo Curator's custom data loading process:

1. Downloads data from your source using a custom `DocumentDownloader`
2. Iterates through the downloaded data using a custom `DocumentIterator`
3. Extracts text using a custom `DocumentExtractor`
4. Outputs the processed data in JSONL or Parquet format

---

## Usage

Here's how to implement and use custom data loaders:

::::{tab-set}

:::{tab-item} Python
```python
from nemo_curator import get_client
from nemo_curator.download import download_and_extract
from my_custom_module import MyCustomDownloader, MyCustomIterator, MyCustomExtractor

def main():
    # Initialize a Dask client
    client = get_client(cluster_type="cpu")

    # Create instances of your custom components
    downloader = MyCustomDownloader()
    iterator = MyCustomIterator()
    extractor = MyCustomExtractor()

    # Use them with NeMo Curator's framework
    dataset = download_and_extract(
        urls=[url1, url2, url3],
        output_paths=[output_path1, output_path2, output_path3],
        downloader=downloader,
        iterator=iterator,
        extractor=extractor,
        output_format={"text": str, "id": str},
        output_type="jsonl",
        keep_raw_download=False,
        force_download=False,
        filename_col="file_name",
        record_limit=None
    )

    # Process the dataset
    dataset.to_json(output_path="/output/folder", write_to_filename=True)

if __name__ == "__main__":
    main()
```
:::

:::{tab-item} CLI
Create a configuration YAML file:

```yaml
# custom_config.yaml
download_module: my_custom_module.MyCustomDownloader
download_params:
  param1: value1
  param2: value2
iterator_module: my_custom_module.MyCustomIterator
iterator_params:
  param3: value3
extract_module: my_custom_module.MyCustomExtractor
extract_params:
  param4: value4
```

Then run the command-line tool:

```bash
# Note: Use the actual script name from nemo_curator/scripts/
python -m nemo_curator.scripts.download_and_extract \
  --input-url-file=./my_urls.txt \
  --builder-config-file=./custom_config.yaml \
  --output-json-dir=/output/folder
```
:::

::::

### Parameters

```{list-table} Custom Data Loading Parameters
:header-rows: 1
:widths: 20 20 40 20

* - Parameter
  - Type
  - Description
  - Default
* - `urls`
  - List[str]
  - List of URLs or paths to download from
  - Required
* - `output_paths`
  - List[str]
  - List of paths where downloaded files will be stored
  - Required
* - `downloader`
  - DocumentDownloader
  - Custom downloader implementation
  - Required
* - `iterator`
  - DocumentIterator
  - Custom iterator implementation
  - Required
* - `extractor`
  - DocumentExtractor
  - Custom extractor implementation
  - Required
* - `output_format`
  - Dict[str, type]
  - Schema for output data
  - Required
* - `output_type`
  - Literal["jsonl", "parquet"]
  - Output file format
  - "jsonl"
* - `keep_raw_download`
  - bool
  - Whether to retain raw downloaded files after extraction
  - False
* - `force_download`
  - bool
  - Whether to re-download and re-extract existing files
  - False
* - `filename_col`
  - str
  - Name of the column for storing filenames in the dataset
  - "file_name"
* - `record_limit`
  - int | None
  - Maximum number of records to extract from each file
  - None
```

## Output Format

The processed data can be stored in either JSONL or Parquet format:

### JSONL Format

```json
{
    "text": "This is a sample text document",
    "id": "unique-id-123",
    "metadata": {
        "source": "example",
        "timestamp": "2024-03-21"
    }
}
```

### Parquet Format

Parquet files maintain the same schema as JSONL files but provide:

- Efficient compression
- Fast query performance
- Column-based operations
- Reduced storage costs

## Implementation Guide

### 1. Create Custom Downloader

```python
from nemo_curator.download.doc_builder import DocumentDownloader

class MyCustomDownloader(DocumentDownloader):
    def download(self, url):
        """Download data from url and return the path to the downloaded file"""
        # Implement download logic
        return "/path/to/downloaded/file"
```

### 2. Create Custom Iterator

```python
from nemo_curator.download.doc_builder import DocumentIterator

class MyCustomIterator(DocumentIterator):
    def iterate(self, file_path):
        """Iterate through documents in the downloaded file"""
        for doc in my_iterator_logic(file_path):
            metadata = {"url": doc.get("url", "")}
            content = doc.get("content", "")
            yield metadata, content
```

### 3. Create Custom Extractor

```python
from nemo_curator.download.doc_builder import DocumentExtractor

class MyCustomExtractor(DocumentExtractor):
    def extract(self, content):
        """Extract text from content and return a dictionary"""
        # Your extraction logic here
        extracted_text = process_content(content)
        unique_id = generate_unique_id(content)
        
        return {
            'text': extracted_text,
            'id': unique_id,
            # Add any other fields as needed
        }
```

```{admonition} Enhancing Custom Extraction
:class: tip

When implementing custom extractors, consider adding robust error handling and metadata extraction to improve the quality of your processed data. You can also implement content filtering and validation logic within your extractor.
```

## Best Practices

1. **Error Handling**: Implement robust error handling for corrupt files and network issues
2. **Logging**: Use Python's logging module for process visibility and debugging
3. **Metadata**: Include useful metadata in extracted documents for downstream processing
4. **Chunking**: Consider chunking large files for efficient distributed processing
5. **Caching**: Implement caching to avoid re-downloading or re-processing data
6. **Parameter Validation**: Validate input parameters in your custom classes
7. **Memory Management**: Be mindful of memory usage when processing large files
8. **Type Annotations**: Use proper type hints to improve code clarity and IDE support
