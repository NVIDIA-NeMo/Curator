---
description: "Implementation guide for downloading and extracting text from arXiv academic papers using ray-curator's modular pipeline architecture"
categories: ["how-to-guides"]
tags: ["arxiv", "academic-papers", "latex", "pdf", "data-loading", "scientific-data", "ray-curator"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "how-to"
modality: "text-only"
---

(text-load-data-arxiv)=

# ArXiv

Implementation guide for downloading and extracting text from ArXiv papers using ray-curator's pipeline architecture.

ArXiv is a free distribution service and open-access archive for scholarly articles, primarily in fields like physics, mathematics, computer science, and more. ArXiv contains millions of scholarly papers, most of them available in LaTeX source format.

```{admonition} Implementation Required
:class: warning

ArXiv functionality is not currently implemented in ray-curator. This guide explains how to create ArXiv support using ray-curator's modular pipeline architecture. For reference implementations, see {ref}`Common Crawl <text-load-data-common-crawl>` and {ref}`Wikipedia <text-load-data-wikipedia>`.
```

## Architecture Overview

Ray-curator uses a 4-step pipeline pattern for document downloading and processing:

1. **URL Generation**: Generate ArXiv tar file URLs from S3 bucket listings
2. **Download**: Download tar files from S3 using s5cmd
3. **Iteration**: Extract and iterate through LaTeX files within tar archives
4. **Extraction**: Process LaTeX content and extract clean text

## Implementation Guide

To create ArXiv support in ray-curator, you'll need to create the following components:

### 1. URL Generator

Create an ArXiv URL generator that lists S3 bucket contents:

```python
from ray_curator.stages.download.text import URLGenerator

class ArxivURLGenerator(URLGenerator):
    def __init__(self, limit: int | None = None):
        self.limit = limit
    
    def generate_urls(self) -> list[str]:
        """Generate ArXiv tar file URLs from S3 bucket listing."""
        # Implementation would use boto3 or s5cmd to list s3://arxiv/src/
        # and return list of tar file URLs
        pass
```

### 2. Document Download Component

Create the S3 download component using s5cmd:

```python
from ray_curator.stages.download.text import DocumentDownloader

class ArxivDownloader(DocumentDownloader):
    def _get_output_filename(self, url: str) -> str:
        return url.split('/')[-1]  # Extract filename from S3 URL
    
    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        """Download using s5cmd with requester-pays."""
        cmd = ["s5cmd", "--request-payer=requester", "cp", url, path]
        # Implementation details...
        pass
    
    def num_workers_per_node(self) -> int | None:
        return 4  # Limit concurrent downloads
```

### 3. Document Iterator

Extract and iterate through LaTeX files in tar archives:

```python
from ray_curator.stages.download.text import DocumentIterator

class ArxivIterator(DocumentIterator):
    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        """Extract LaTeX files from tar archive and yield records."""
        with tarfile.open(file_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith('.tex'):
                    # Extract and yield LaTeX content
                    pass
    
    def output_columns(self) -> list[str]:
        return ["raw_content", "file_name", "tar_source"]
```

### 4. Document Extractor

Process LaTeX content to extract clean text:

```python
from ray_curator.stages.download.text import DocumentExtractor

class ArxivExtractor(DocumentExtractor):
    def extract(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Clean LaTeX and extract main text content."""
        raw_content = record["raw_content"]
        # Process LaTeX: remove comments, expand macros, extract sections
        cleaned_text = self._clean_latex(raw_content)
        
        return {
            "text": cleaned_text,
            "id": self._extract_arxiv_id(record["file_name"]),
            "source_id": record["tar_source"],
            "file_name": record["file_name"]
        }
    
    def input_columns(self) -> list[str]:
        return ["raw_content", "file_name", "tar_source"]
    
    def output_columns(self) -> list[str]:
        return ["text", "id", "source_id", "file_name"]
```

### 5. Composite Stage

Combine all components into a single stage:

```python
from ray_curator.stages.download.text import DocumentDownloadExtractStage

class ArxivDownloadExtractStage(DocumentDownloadExtractStage):
    def __init__(
        self,
        download_dir: str,
        url_limit: int | None = None,
        record_limit: int | None = None,
        **kwargs
    ):
        super().__init__(
            url_generator=ArxivURLGenerator(limit=url_limit),
            downloader=ArxivDownloader(download_dir),
            iterator=ArxivIterator(),
            extractor=ArxivExtractor(),
            url_limit=url_limit,
            record_limit=record_limit,
            **kwargs
        )
```

## Usage Pattern

Once implemented, ArXiv processing would follow the standard ray-curator pattern:

```python
from ray_curator.pipeline import Pipeline
from ray_curator.backends.experimental.ray_data import RayDataExecutor

# Create pipeline
pipeline = Pipeline(
    name="arxiv_processing",
    stages=[
        ArxivDownloadExtractStage(
            download_dir="/tmp/arxiv_downloads",
            url_limit=100,  # For testing
            record_limit=1000
        )
    ]
)

# Execute
executor = RayDataExecutor()
results = pipeline.run(executor)
```

## Prerequisites

For ArXiv implementation, you would need:

1. **AWS Configuration**: Properly configured credentials in `~/.aws/config`
2. **s5cmd**: Installed for efficient S3 transfers (included in NVIDIA NeMo Framework Container)
3. **LaTeX Processing**: Libraries for LaTeX parsing and macro expansion

```{admonition} Text Processing with Stop Words
:class: tip

When processing academic papers from ArXiv, you may want to customize text extraction and analysis using stop words. Stop words can help identify section boundaries, distinguish main content from references, and support language-specific processing. For a comprehensive guide to stop words in NeMo Curator, see {ref}`Stop Words in Text Processing <text-process-data-languages-stop-words>`.
```

## Expected Output Format

The implemented ArXiv extractor would produce DocumentBatch objects with the following schema:

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
  - The original LaTeX filename
```

## Reference Implementations

For guidance on implementing these components, examine the existing implementations:

- **Common Crawl**: `ray_curator/stages/download/text/common_crawl/`
- **Wikipedia**: `ray_curator/stages/download/text/wikipedia/`

Both follow the same 4-step pipeline pattern and provide concrete examples of each component.