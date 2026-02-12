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

Curator uses the same **3-step pipeline pattern** described in {ref}`Data Acquisition Concepts <about-concepts-text-data-acquisition>` for custom data loading. Each step uses an abstract base class with corresponding processing stages that compose into pipelines.

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
from dataclasses import dataclass
from nemo_curator.stages.text.download import URLGenerator

@dataclass
class CustomURLGenerator(URLGenerator):
    def generate_urls(self) -> list[str]:
        """Generate list of URLs to download."""
        # Your URL generation logic here
        return [
            "https://example.com/dataset1.zip",
            "https://example.com/dataset2.zip",
        ]
```

#### Document Downloader (`download.py`)

```python
from nemo_curator.stages.text.download import DocumentDownloader

class CustomDownloader(DocumentDownloader):
    def __init__(self, download_dir: str):
        super().__init__(download_dir=download_dir)

    def _get_output_filename(self, url: str) -> str:
        """Extract filename from URL."""
        return url.split("/")[-1]

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        """Download file from URL to local path."""
        # Custom download logic
        # Return (success_bool, error_message)
        try:
            # ... download implementation ...
            return True, None
        except Exception as e:
            return False, str(e)
```

#### Document Iterator (`iterator.py`)

```python
import json
from collections.abc import Iterator
from typing import Any
from nemo_curator.stages.text.download import DocumentIterator

class CustomIterator(DocumentIterator):
    def __init__(self, log_frequency: int = 1000):
        super().__init__()
        self._log_frequency = log_frequency

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        """Iterate over records in a file."""
        # Custom iteration logic to load local file and return documents
        for record in load_local_file_fn(file_path):
            yield {"content": record_content, "metadata": record_metadata, "id": record_id}

    def output_columns(self) -> list[str]:
        """Define output columns."""
        return ["content", "metadata", "id"]
```

#### Document Extractor (`extract.py`)

```python
from typing import Any
from nemo_curator.stages.text.download import DocumentExtractor

class CustomExtractor(DocumentExtractor):
    def __init__(self):
        super().__init__()

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
from nemo_curator.stages.text.download import DocumentDownloadExtractStage
from nemo_curator.stages.base import ProcessingStage
from .url_generation import CustomURLGenerator
from .download import CustomDownloader
from .iterator import CustomIterator
from .extract import CustomExtractor

class CustomDataStage(DocumentDownloadExtractStage):
    """Custom data loading stage combining all components."""

    def __init__(
        self,
        download_dir: str = "./custom_downloads",
        url_limit: int | None = None,
        record_limit: int | None = None,
        add_filename_column: bool | str = True,
    ):
        self.url_generator = CustomURLGenerator()
        self.downloader = CustomDownloader(download_dir=download_dir)
        self.iterator = CustomIterator()
        self.extractor = CustomExtractor()

        # Initialize the parent composite stage
        super().__init__(
            url_generator=self.url_generator,
            downloader=self.downloader,
            iterator=self.iterator,
            extractor=self.extractor,  # Optional - remove if not needed
            url_limit=url_limit,
            record_limit=record_limit,
            add_filename_column=add_filename_column,
        )
        self.name = "custom_data"

    def decompose(self) -> list[ProcessingStage]:
        """Decompose this composite stage into its constituent stages."""
        return self.stages

    def get_description(self) -> str:
        """Get a description of this composite stage."""
        return "Custom data"
```

---

## Usage Examples

### Basic Pipeline

```python
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from your_data_source.stage import CustomDataStage
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter

def main():
    # Initialize Ray client
    ray_client = RayClient()
    ray_client.start()

    # Create pipeline
    pipeline = Pipeline(
        name="custom_data_pipeline",
        description="Load and process custom dataset"
    )

    # Create custom data loading stage
    data_stage = CustomDataStage(...)

    pipeline.add_stage(data_stage)

    # Save the results to JSONL
    pipeline.add_stage(JsonlWriter(...))

    # Run pipeline
    print("Starting pipeline...")
    results = pipeline.run()

    # Stop Ray client
    ray_client.stop()

if __name__ == "__main__":
    main()
```

For executor options and configuration, refer to {ref}`reference-execution-backends`.

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
