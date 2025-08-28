# Adding Custom Download Pipelines

## 📁 Structure Overview

The framework follows a **4-step pipeline pattern** where each step is implemented as an abstract base class with corresponding stages:

```
1. URLGenerator → URLGenerationStage    (URLs from config/input)
2. DocumentDownloader → DocumentDownloadStage    (local files from URLs)
3. DocumentIterator → DocumentIterateStage    (raw records from files)
4. DocumentExtractor → DocumentExtractStage    (structured data from records)
```

## 🛠️ Implementation Steps

### 1. Create Your Directory Structure
```
your_data_source/
├── __init__.py
├── stage.py           # Main composite stage
├── url_generation.py  # URL generation logic
├── download.py        # Download implementation
├── iterator.py        # File iteration logic
└── extract.py         # Data extraction logic
```

### 2. Implement Core Components

**URL Generator** (`url_generation.py`):
```python
from nemo_curator.stages.text.download import URLGenerator

class YourURLGenerator(URLGenerator):
    def generate_urls(self) -> list[str]:
        # Return list of URLs to download
        return ["https://example.com/file1.zip", ...]
```

**Downloader** (`download.py`):
```python
from nemo_curator.stages.text.download import DocumentDownloader

class YourDownloader(DocumentDownloader):
    def _get_output_filename(self, url: str) -> str:
        # Extract filename from URL
        return url.split('/')[-1]

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        # Download logic - return (success_bool, error_message)
        # Use subprocess.run, requests, etc.
```

**Iterator** (`iterator.py`):
```python
from nemo_curator.stages.text.download import DocumentIterator

class YourIterator(DocumentIterator):
    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        # Parse file and yield record dicts
        yield {"content": "raw_text", "metadata": "value"}

    def output_columns(self) -> list[str]:
        return ["content", "metadata"]
```

**Extractor** (`extract.py`):
```python
from nemo_curator.stages.text.download import DocumentExtractor

class YourExtractor(DocumentExtractor):
    def extract(self, record: dict[str, str]) -> dict[str, Any] | None:
        # Transform raw record to final format
        return {"text": processed_text, "lang": language}

    def input_columns(self) -> list[str]:
        return ["content", "metadata"]

    def output_columns(self) -> list[str]:
        return ["text", "lang"]
```

### 3. Create Composite Stage (`stage.py`)

```python
from nemo_curator.stages.text.download import DocumentDownloadExtractStage

class YourDownloadExtractStage(DocumentDownloadExtractStage):
    def __init__(self, config_param1, config_param2, download_dir: str, **kwargs):
        super().__init__(
            url_generator=YourURLGenerator(config_param1),
            downloader=YourDownloader(download_dir),
            iterator=YourIterator(config_param2),
            extractor=YourExtractor(),  # Optional
            **kwargs
        )
```

## 🚀 Usage

```python
# Create and run your pipeline
stage = YourDownloadExtractStage(
    config_param1="value1",
    download_dir="/tmp/downloads"
)

pipeline = Pipeline("your_pipeline")
pipeline.add_stage(stage)
results = pipeline.run(executor)
```

## 💡 Key Notes

- **Extractor is optional** - omit if iteration produces final format
- **Each component is independent** - mix and match with existing ones
- **Follow the data flow**: `_EmptyTask → FileGroupTask → DocumentBatch`
- **Use existing patterns** - see `common_crawl/` for complete example

## 📖 Examples

See the `common_crawl/` directory for a complete implementation that demonstrates:
- Complex URL generation with date ranges and API calls
- WARC file downloading with multiple backends (wget/s5cmd)
- WARC record iteration and parsing
- HTML content extraction with multiple algorithms
