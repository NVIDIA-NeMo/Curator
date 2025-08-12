---
description: "Download and extract text from arXiv using ray-curator's pipeline framework"
categories: ["how-to-guides"]
tags: ["arxiv", "academic-papers", "latex", "data-loading", "scientific-data", "ray-curator"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

# ArXiv

(text-load-data-arxiv)=

Download and extract text from ArXiv LaTeX source bundles using ray-curator's pipeline framework.

ArXiv hosts millions of scholarly papers, typically distributed as LaTeX source inside `.tar` archives under the `s3://arxiv/src/` requester-pays bucket.

## How it Works

The ArXiv pipeline in ray-curator consists of four stages:

1. URL Generation: Lists available ArXiv source tar files from the S3 bucket
2. Download: Downloads `.tar` archives via s5cmd (requester-pays)
3. Iteration: Extracts LaTeX projects and yields per-paper records
4. Extraction: Cleans LaTeX and produces plain text

Internals (implemented):

- URL generation: `ray_curator/stages/download/text/arxiv/url_generation.py`
- Download: `ray_curator/stages/download/text/arxiv/download.py`
- Iterator: `ray_curator/stages/download/text/arxiv/iterator.py`
- Extractor: `ray_curator/stages/download/text/arxiv/extract.py`
- Composite stage: `ray_curator/stages/download/text/arxiv/stage.py`

## Before You Start

You must have s5cmd installed and AWS credentials configured for requester-pays.

- Install s5cmd: see `https://github.com/peak/s5cmd`
- Configure AWS credentials in your environment (or `~/.aws/credentials`) with access to requester-pays buckets

```{admonition} S3 Requester Pays
:class: tip

The ArXiv `s3://arxiv/src/` bucket is requester-pays. All listing and copy operations set requester-pays via s5cmd.
```

---

## Usage

Create and run an ArXiv processing pipeline and write outputs to JSONL:

```python
from ray_curator.pipeline import Pipeline
from ray_curator.backends.experimental.ray_data import RayDataExecutor
from ray_curator.stages.download.text.arxiv import ArxivDownloadExtractStage
from ray_curator.stages.io.writer import JsonlWriter
from ray_curator.tasks import EmptyTask

def main():
    pipeline = Pipeline(
        name="arxiv_pipeline",
        description="Download and process ArXiv LaTeX sources"
    )

    # Add ArXiv stage
    arxiv_stage = ArxivDownloadExtractStage(
        download_dir="./arxiv_downloads",
        url_limit=5,        # optional: number of tar files to process
        record_limit=1000,  # optional: max papers per tar
        add_filename_column=True,
        verbose=True,
    )
    pipeline.add_stage(arxiv_stage)

    # Add writer stage
    writer = JsonlWriter(output_dir="./arxiv_output")
    pipeline.add_stage(writer)

    # Execute
    executor = RayDataExecutor()
    results = pipeline.run(executor, initial_tasks=[EmptyTask])
    print(f"Completed with {len(results) if results else 0} output files")

if __name__ == "__main__":
    main()
```

### Parameters

```{list-table} ArxivDownloadExtractStage Parameters
:header-rows: 1
:widths: 25 20 35 20

* - Parameter
  - Type
  - Description
  - Default
* - `download_dir`
  - str
  - Directory to store downloaded `.tar` files
  - "./arxiv_downloads"
* - `url_limit`
  - int | None
  - Maximum number of ArXiv tar files to download (useful for testing)
  - None
* - `record_limit`
  - int | None
  - Maximum number of papers to extract per tar file
  - None
* - `add_filename_column`
  - bool | str
  - Whether to add a source filename column to output; if str, use it as the column name
  - True (column name defaults to `file_name`)
* - `log_frequency`
  - int
  - How often to log progress while iterating papers
  - 1000
* - `verbose`
  - bool
  - Enable verbose logging during download
  - False
```

```{note}
URL generation and download use s5cmd with requester-pays to list and copy from `s3://arxiv/src/`.
```

## Output Format

The extractor returns per-paper text; the filename column is optionally added by the pipeline:

```json
{
  "text": "Main body text extracted from LaTeX after cleaning...",
  "file_name": "arXiv-2401.01234.tar"  
}
```

```{list-table} Output Fields
:header-rows: 1
:widths: 20 80

* - Field
  - Description
* - `text`
  - Extracted and cleaned paper text (LaTeX macros inlined where supported, comments and references removed)
* - `file_name`
  - Optional. Name of the source tar file (enabled by `add_filename_column`)
```

```{admonition} Intermediate Fields
:class: note

During iteration the pipeline yields `id` (ArXiv identifier), `source_id` (tar basename), and `content` (list of LaTeX files). The final extractor stage emits only `text` plus the optional filename column.
```

## Advanced Notes

- The pipeline validates paths and extracts tar files with path traversal protection.
- The iterator and extractor adapt RedPajama preprocessing with safety and robustness improvements.
- Macro expansion handles non-argument macros; macros with arguments are not expanded.

See also: {ref}`Common Crawl <text-load-data-common-crawl>`, {ref}`Wikipedia <text-load-data-wikipedia>`
