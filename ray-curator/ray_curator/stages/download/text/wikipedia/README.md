# Wikipedia Download and Extract Stage

This module provides a complete pipeline for downloading and processing Wikipedia dump files using the ray-curator framework.

## Features

- **URL Generation**: Automatically discovers and generates URLs for Wikipedia dump files
- **Parallel Download**: Downloads multiple .bz2 dump files in parallel
- **Article Extraction**: Parses Wikipedia XML dumps to extract individual articles
- **Content Cleaning**: Removes Wikipedia markup, references, tables, and media links
- **Multi-language Support**: Supports all Wikipedia languages with proper alias handling

## Components

### 1. WikipediaUrlGenerator
- Generates URLs for Wikipedia dump files
- Supports specific dump dates or latest dumps
- Handles multiple languages

### 2. WikipediaDownloader
- Downloads .bz2 dump files using wget
- Supports resume and verification
- Handles multiple files in parallel

### 3. WikipediaIterator
- Parses Wikipedia XML dumps
- Extracts article metadata (title, ID, URL, etc.)
- Filters out redirects and non-main namespace pages

### 4. WikipediaExtractor
- Cleans Wikipedia markup using mwparserfromhell
- Removes references, tables, and media links
- Handles category links and magic words
- Supports language-specific aliases

## Usage

### Basic Usage

```python
from ray_curator.stages.download.text.wikipedia import WikipediaDownloadExtractStage
from ray_curator.pipeline import Pipeline
from ray_curator.executors import XennaExecutor
from ray_curator.tasks import _EmptyTask

# Create the Wikipedia stage
wikipedia_stage = WikipediaDownloadExtractStage(
    language="en",
    download_dir="./wikipedia_downloads",
    verbose=True,
    url_limit=2,        # Download only 2 files for testing
    record_limit=1000,  # Extract only 1000 articles per file
)

# Create and run pipeline
pipeline = Pipeline("wikipedia_pipeline", stages=[wikipedia_stage])
executor = XennaExecutor()
initial_task = _EmptyTask(task_id="wiki", dataset_name="wikipedia")

results = pipeline.run(executor=executor, initial_task=initial_task)
```

### Advanced Configuration

```python
# Configure for specific dump date and language
wikipedia_stage = WikipediaDownloadExtractStage(
    language="es",                    # Spanish Wikipedia
    download_dir="./downloads/es",
    dump_date="20231201",            # Specific dump date
    verbose=True,
    url_limit=None,                  # Download all available files
    record_limit=None,               # Extract all articles
    log_frequency=5000,              # Log every 5000 articles
)
```

## Parameters

- **language** (str): Wikipedia language code (default: "en")
- **download_dir** (str): Directory for downloaded .bz2 files
- **dump_date** (str, optional): Specific dump date in "YYYYMMDD" format
- **wikidumps_index_prefix** (str): Base URL for Wikipedia dumps
- **verbose** (bool): Enable verbose logging
- **url_limit** (int, optional): Maximum number of dump files to process
- **record_limit** (int, optional): Maximum articles to extract per file
- **add_filename_column** (bool): Add source filename to output
- **log_frequency** (int): Progress logging frequency

## Output Format

Each processed article includes:

```json
{
    "text": "Cleaned article text...",
    "title": "Article Title",
    "id": "12345",
    "url": "https://en.wikipedia.org/wiki/Article_Title",
    "language": "en",
    "source_id": "enwiki-20231201-pages-articles-multistream1.xml-p1p41242.bz2",
    "file_name": "enwiki-20231201-pages-articles-multistream1.xml-p1p41242.bz2"
}
```

## Dependencies

Required packages:
- `mwparserfromhell`: For parsing Wikipedia markup
- `beautifulsoup4`: For parsing HTML index pages
- `lxml`: For XML processing
- `requests`: For HTTP requests

Install with:
```bash
pip install mwparserfromhell beautifulsoup4 lxml requests
```

## Supported Languages

All Wikipedia languages are supported. Common examples:
- `en`: English
- `es`: Spanish
- `fr`: French
- `de`: German
- `ja`: Japanese
- `zh`: Chinese
- `ru`: Russian
- `pt`: Portuguese
- `it`: Italian
- `ar`: Arabic
