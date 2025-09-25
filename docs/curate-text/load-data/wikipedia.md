---
description: "Download and extract text from Wikipedia dumps using Curator."
categories: ["how-to-guides"]
tags: ["wikipedia", "dumps", "multilingual", "articles", "data-loading"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-load-data-wikipedia)=

# Wikipedia

Download and extract text from [Wikipedia Dumps](https://dumps.wikimedia.org/backup-index.html) using Curator.

Wikipedia releases compressed dumps of all its content in XML format twice per month. Curator provides a complete pipeline to automatically download, parse, and extract clean text from these dumps.

## How it Works

The Wikipedia pipeline in Curator consists of four stages:

1. **URL Generation**: Automatically discovers Wikipedia dump URLs for the specified language and date
2. **Download**: Downloads compressed .bz2 dump files using `wget`
3. **Iteration**: Parses XML content and extracts individual articles
4. **Extraction**: Cleans Wikipedia markup and converts to plain text

## Before You Start

Wikipedia publishes new dumps around the **first** and **twentieth** of each month. Refer to the English Wikipedia dumps index at `https://dumps.wikimedia.org/enwiki/` for available dates.

Curator uses `wget` to download Wikipedia dumps. You must have `wget` installed on your system:

- **On macOS**:  `brew install wget`
- **On Ubuntu/Debian**: `sudo apt-get install wget`
- **On CentOS/RHEL**:  `sudo yum install wget`

---

## Usage

Here's how to download and extract Wikipedia data using Curator:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.download import WikipediaDownloadExtractStage
from nemo_curator.stages.text.io.writer import JsonlWriter

# Create the Wikipedia processing stage
wikipedia_stage = WikipediaDownloadExtractStage(
    language="en",
    download_dir="./wikipedia_downloads",
    dump_date="20240401",  # Optional: specific dump date (YYYYMMDD format)
    url_limit=5,           # Optional: limit number of dump files (useful for testing)
    record_limit=1000,     # Optional: limit articles per dump file
    verbose=True
)

# Create writer stage to save results
writer_stage = JsonlWriter(
    path="./wikipedia_output"
)

# Create and configure pipeline
pipeline = Pipeline(
    name="wikipedia_pipeline",
    description="Download and process Wikipedia dumps"
)
pipeline.add_stage(wikipedia_stage)
pipeline.add_stage(writer_stage)

# Execute the pipeline
results = pipeline.run()
print(f"Pipeline completed with {len(results) if results else 0} output files")
```

For executor options and configuration, refer to {ref}`reference-execution-backends`.

### Multi-Language Processing

You can process several languages by creating separate pipelines:

```python
languages = ["en", "es", "fr", "de"]

for lang in languages:
    # Create language-specific pipeline
    wikipedia_stage = WikipediaDownloadExtractStage(
        language=lang,
        download_dir=f"./downloads/{lang}",
        dump_date="20240401"
    )

    writer_stage = JsonlWriter(
        path=f"./output/{lang}"
    )

    pipeline = Pipeline(name=f"wikipedia_{lang}")
    pipeline.add_stage(wikipedia_stage)
    pipeline.add_stage(writer_stage)

    # Execute
    results = pipeline.run()
```

### Parameters

```{list-table} WikipediaDownloadExtractStage Parameters
:header-rows: 1
:widths: 20 20 20 40

* - Parameter
  - Type
  - Default
  - Description
* - `language`
  - str
  - "en"
  - Language code for Wikipedia dump (for example, `en`, `es`, `fr`). Most follow ISO 639‑1, with project-specific exceptions such as `simple`. Refer to Meta‑Wiki [List of Wikipedia language editions](https://meta.wikimedia.org/wiki/List_of_Wikipedias) for supported edition codes and [List of ISO 639 language codes](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) for general codes.
* - `download_dir`
  - str
  - "./wikipedia_downloads"
  - Directory to store downloaded .bz2 files
* - `dump_date`
  - Optional[str]
  - None
  - Specific dump date in "YYYYMMDD" format (for example, "20240401"). Dumps are published around the first and twentieth of each month. If None, uses the latest available dump
* - `wikidumps_index_prefix`
  - str
  - "https://dumps.wikimedia.org"
  - Base URL for Wikipedia dumps index
* - `verbose`
  - bool
  - False
  - Enable verbose logging during download
* - `url_limit`
  - Optional[int]
  - None
  - Maximum number of dump URLs to process (useful for testing)
* - `record_limit`
  - Optional[int]
  - None
  - Maximum number of articles to extract per dump file
* - `add_filename_column`
  - bool | str
  - True
  - Whether to add source filename column to output; if str, uses it as the column name (default name: "file_name")
* - `log_frequency`
  - int
  - 1000
  - How often to log progress during article processing
```

### Known Limitations

Parsing relies on `mwparserfromhell`. Complex templates might not be fully rendered, so template-heavy pages can yield incomplete text. Customize the extractor if you need different behavior.

## Output Format

The processed Wikipedia articles become JSONL files, with each line containing a JSON object with these fields:

- `text`: The cleaned main text content of the article
- `title`: The title of the Wikipedia article
- `id`: Wikipedia's unique identifier for the article
- `url`: The constructed Wikipedia URL for the article
- `language`: The language code of the article
- `source_id`: Identifier of the source dump file

If you enable `add_filename_column`, the output includes an extra field `file_name` (or your custom column name).

### Example Output Record

```json
{
  "text": "Python is a high-level, general-purpose programming language...",
  "title": "Python (programming language)",
  "id": "23862",
  "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
  "language": "en",
  "source_id": "enwiki-20240401-pages-articles-multistream1.xml"
}
```
