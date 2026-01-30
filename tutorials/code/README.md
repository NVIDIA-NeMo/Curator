# Code Annotation and Filtering Tutorials

Tutorials demonstrating NeMo Curator's code annotation and filtering capabilities
with distributed Ray pipelines.

## Prerequisites

Install NeMo Curator with the code curation dependencies using [uv](https://docs.astral.sh/uv/):

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install NeMo Curator with code curation dependencies
uv pip install nemo-curator[code]
```

## Dataset

These tutorials use the [bigcode/the-stack-smol-xs](https://huggingface.co/datasets/bigcode/the-stack-smol-xs)
dataset from Hugging Face - a small subset of The Stack with 100 samples per language
across 87 programming languages (8,700 total samples, ~32MB).

Dataset columns: `content`, `lang`, `size`, `ext`, `max_stars_count`, `avg_line_length`, `max_line_length`, `alphanum_fraction`

## Quick Start

```bash
# Step 1: Download the dataset
./tutorials/code/00_download_data.sh

# Step 2: Run the tutorial stages
python tutorials/code/01_language_detection.py
python tutorials/code/02_code_annotation.py
python tutorials/code/03_code_filtering.py
python tutorials/code/04_license_detection.py
```

## Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 0 | `00_download_data.sh` | Download dataset from Hugging Face |
| 1 | `01_language_detection.py` | Detect programming languages |
| 2 | `02_code_annotation.py` | Compute quality metrics |
| 3 | `03_code_filtering.py` | Filter by quality signals |
| 4 | `04_license_detection.py` | Detect software licenses |

## Example Pipeline

```python
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.code import CodeAnnotation, CodeQualityFilter
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

ray_client = RayClient()
ray_client.start()

pipeline = Pipeline(
    name="Code Curation",
    stages=[
        JsonlReader(file_paths=["data/python/data.json", "data/rust/data.json"]),
        CodeAnnotation(content_column="content", filename_column="ext"),
        CodeQualityFilter(min_alpha_percent=0.25),
        JsonlWriter(path="output/"),
    ],
)

pipeline.run()
ray_client.stop()
```

## Available Stages

- `CodeLanguageDetection` - Detect programming language from content and filename
- `CodeAnnotation` - Compute annotations (language, stats, metrics, tokens)
- `CodeQualityFilter` - Filter based on quality metrics
- `CodeLicenseDetection` - Detect SPDX licenses in code

## Output Structure

```
tutorials/code/output/
├── input_data/               # Downloaded dataset from Hugging Face
│   └── data/                 # One subdirectory per language
│       ├── python/
│       │   └── data.json     # JSONL format (one JSON per line)
│       ├── javascript/
│       │   └── data.json
│       └── ...
├── 01_lang_detected/         # Language detection results (JSONL)
├── 02_annotated/             # Code annotation results (JSONL)
├── 03_filtered/              # Filtered results (JSONL)
└── 04_license_detected/      # License detection results (JSONL)
```
