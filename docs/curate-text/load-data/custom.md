---
description: "Create custom data loading pipelines using Curator."
categories: ["how-to-guides"]
tags: ["custom-data", "stages", "pipelines", "data-loading"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "how-to"
modality: "text-only"
---

(text-load-data-custom)=

# Custom Data Loading

Create custom data loading pipelines using Curator. This guide shows how to build modular stages that run on Curator's distributed processing.

## How It Works

Curator uses the same **4-step pipeline pattern** described in {ref}`Data Acquisition Concepts <about-concepts-text-data-acquisition>` for custom data loading. Each step uses an abstract base class with corresponding processing stages that compose into pipelines.

---

## Architecture Overview

For detailed information about the core components and data flow, see {ref}`Data Acquisition Concepts <about-concepts-text-data-acquisition>` and {ref}`Data Loading Concepts <about-concepts-text-data-loading>`.

---

## Implementation Guide

### 1. Create Directory Structure

```text
your_data_source/
├── __init__.py
├── stage.py           # Main composite stage
├── url_generation.py  # URL generation logic
├── download.py        # Download implementation
├── iterator.py        # File iteration logic
└── extract.py         # Data extraction logic (optional)
```

### 2. Build Core Components

#### URL Generator (`url_generation.py`)

```python
from nemo_curator.stages.text.download.base.url_generation import URLGenerator

class CustomURLGenerator(URLGenerator):
    def __init__(self, config_param: str):
        self.config_param = config_param

    def generate_urls(self) -> list[str]:
        """Generate list of URLs to download."""
        # Your URL generation logic here
        return [
            "https://example.com/dataset1.zip",
            "https://example.com/dataset2.zip",
        ]
```

#### Document Download Handler (`download.py`)

```python
import requests
from nemo_curator.stages.text.download.base.download import DocumentDownloader

class CustomDownloader(DocumentDownloader):
    def __init__(self, download_dir: str, verbose: bool = False):
        super().__init__(download_dir, verbose)

    def _get_output_filename(self, url: str) -> str:
        """Extract filename from URL."""
        return url.split('/')[-1]

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        """Download file from URL to local path."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True, None
        except Exception as e:
            return False, str(e)
```

#### Document Iterator (`iterator.py`)

```python
import json
from collections.abc import Iterator
from typing import Any
from nemo_curator.stages.text.download.base.iterator import DocumentIterator

class CustomIterator(DocumentIterator):
    def __init__(self, record_format: str = "jsonl"):
        self.record_format = record_format

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        """Iterate over records in a file."""
        if self.record_format == "jsonl":
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        # Add other format handlers as needed

    def output_columns(self) -> list[str]:
        """Define output columns."""
        return ["content", "metadata", "id"]
```

#### Document Extractor (`extract.py`)

```python
from typing import Any
from nemo_curator.stages.text.download.base.extract import DocumentExtractor

class CustomExtractor(DocumentExtractor):
    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        """Transform raw record to final format."""
        # Skip invalid records
        if not record.get("content"):
            return None

        # Extract and clean text
        cleaned_text = self._clean_text(record["content"])

        # Generate unique ID if not present
        doc_id = record.get("id", self._generate_id(cleaned_text))

        return {
            "text": cleaned_text,
            "id": doc_id,
            "source": record.get("metadata", {}).get("source", "unknown")
        }

    def input_columns(self) -> list[str]:
        return ["content", "metadata", "id"]

    def output_columns(self) -> list[str]:
        return ["text", "id", "source"]

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Your text cleaning logic here
        return text.strip()

    def _generate_id(self, text: str) -> str:
        """Generate unique ID for text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:16]
```

### 3. Create Composite Stage (`stage.py`)

```python
from nemo_curator.stages.text.download.base.stage import DocumentDownloadExtractStage
from .url_generation import CustomURLGenerator
from .download import CustomDownloader
from .iterator import CustomIterator
from .extract import CustomExtractor

class CustomDataStage(DocumentDownloadExtractStage):
    """Custom data loading stage combining all components."""

    def __init__(
        self,
        config_param: str,
        download_dir: str,
        record_format: str = "jsonl",
        url_limit: int | None = None,
        record_limit: int | None = None,
        **kwargs
    ):
        super().__init__(
            url_generator=CustomURLGenerator(config_param),
            downloader=CustomDownloader(download_dir),
            iterator=CustomIterator(record_format),
            extractor=CustomExtractor(),  # Optional - remove if not needed
            url_limit=url_limit,
            record_limit=record_limit,
            **kwargs
        )
```

---

## Usage Examples

### Basic Pipeline

```python
from nemo_curator.pipeline import Pipeline
from your_data_source.stage import CustomDataStage

def main():
    # Create custom data loading stage
    data_stage = CustomDataStage(
        config_param="production",
        download_dir="/tmp/downloads",
        record_limit=1000  # Limit for testing
    )

    # Create pipeline
    pipeline = Pipeline(
        name="custom_data_pipeline",
        description="Load and process custom dataset"
    )
    pipeline.add_stage(data_stage)

    # Run pipeline
    print("Starting pipeline...")
    results = pipeline.run()

    # Process results
    if results:
        for task in results:
            print(f"Processed {task.num_items} documents")
            # Access data as Pandas DataFrame
            df = task.to_pandas()
            print(df.head())

if __name__ == "__main__":
    main()
```

For executor options and configuration, refer to {ref}`reference-execution-backends`.

<!-- move the following to concepts / general docs on pipelines in separate PR -->
<!-- ### Adding Processing Stages

```python
from nemo_curator.stages.modules import ScoreFilter
from nemo_curator.stages.filters import WordCountFilter
from nemo_curator.stages.text.io.writer import JsonlWriter

def create_full_pipeline():
    pipeline = Pipeline(name="full_processing")

    # Data loading
    pipeline.add_stage(CustomDataStage(
        config_param="production",
        download_dir="/tmp/downloads"
    ))

    # Text filtering
    pipeline.add_stage(ScoreFilter(
        filter_obj=WordCountFilter(min_words=10, max_words=1000),
        text_field="text"
    ))

    # Output
    pipeline.add_stage(JsonlWriter(path="/output/processed"))

    return pipeline
``` -->


---

## Parameters Reference

```{list-table} Custom Data Loading Parameters
:header-rows: 1
:widths: 20 20 40 20

* - Parameter
  - Type
  - Description
  - Default
* - `url_generator`
  - URLGenerator
  - Custom URL generation implementation
  - Required
* - `downloader`
  - DocumentDownloader
  - Custom download implementation
  - Required
* - `iterator`
  - DocumentIterator
  - Custom file iteration implementation
  - Required
* - `extractor`
  - DocumentExtractor | None
  - Optional extraction/transformation step
  - None
* - `url_limit`
  - int | None
  - Maximum number of URLs to process
  - None
* - `record_limit`
  - int | None
  - Maximum records per file
  - None
* - `add_filename_column`
  - bool | str
  - Add filename column to output; if str, uses it as the column name (default name: "file_name")
  - True
```

---

## Output Format

Processed data flows through the pipeline as `DocumentBatch` tasks containing Pandas DataFrames or PyArrow Tables:

### Example Output Schema

```python
{
    "text": "This is the processed document text",
    "id": "unique-document-id",
    "source": "example.com",
    "file_name": "dataset1.jsonl"  # If add_filename_column=True (default column name)
}
```
