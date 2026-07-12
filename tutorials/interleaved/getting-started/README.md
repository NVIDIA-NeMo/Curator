# Getting Started with Interleaved Multimodal Data Curation

The files in this directory show how to load, explore, filter, and save **interleaved multimodal data** (text and images in reading order) using NeMo Curator.

**API Reference**: [nemo_curator.tasks.interleaved](https://docs.nvidia.com/nemo/curator/latest/nemo-curator/nemo_curator/tasks/interleaved)

- **`interleaved_data_quickstart.ipynb`** — interactive Jupyter notebook that walks through reading a MINT-1T PDF shard, visualising the data, applying image and text quality filters, writing to Parquet, and doing a round-trip verification. A great starting point.
- **`interleaved_pipeline.py`** — a self-contained command-line pipeline for production use: reads WebDataset tar shards, applies configurable filters, and writes to Parquet or WebDataset tar format.

## Setup

```bash
git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
pip install uv

# CPU-only (notebook + pipeline script work without a GPU)
uv sync --extra interleaved_cpu

# GPU-accelerated (required for large-scale runs)
uv sync --extra interleaved_cuda12
```

## Sample Data

Both the notebook and pipeline script use a MINT-1T PDF shard as sample data. Download one (~79 MB) from HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli download mlfoundations/MINT-1T-PDF-CC-2024-18 \
    CC-MAIN-2024-18-shard-0/CC-MAIN-20240412101354-20240412131354-00000.tar \
    --repo-type dataset \
    --local-dir ./sample_data
export MINT1T_TAR_PATH=./sample_data/CC-MAIN-2024-18-shard-0/CC-MAIN-20240412101354-20240412131354-00000.tar
```

## Quick Start — Notebook

Open `tutorials/interleaved/getting-started/interleaved_data_quickstart.ipynb` in JupyterLab or VS Code.

The notebook auto-downloads the shard on first run, or uses `$MINT1T_TAR_PATH` if set (see above).

## Quick Start — Pipeline Script

**Basic run** — read one shard, apply default aspect-ratio filter, write to Parquet:

```bash
python tutorials/interleaved/getting-started/interleaved_pipeline.py \
    --input-path $MINT1T_TAR_PATH \
    --output-path ./output/ \
    --on-materialize-error drop_row \
    --mode overwrite
```

**Add text and image filters**:

```bash
python tutorials/interleaved/getting-started/interleaved_pipeline.py \
    --input-path $MINT1T_TAR_PATH \
    --output-path ./output/ \
    --min-aspect-ratio 0.5 \
    --max-aspect-ratio 2.0 \
    --min-text-chars 50 \
    --on-materialize-error drop_row \
    --mode overwrite
```

**Write to WebDataset tar format**:

```bash
python tutorials/interleaved/getting-started/interleaved_pipeline.py \
    --input-path $MINT1T_TAR_PATH \
    --output-path ./output_tars/ \
    --writer-format webdataset \
    --on-materialize-error drop_row \
    --mode overwrite
```

**Pass extra columns through** (e.g. custom metadata fields):

```bash
python tutorials/interleaved/getting-started/interleaved_pipeline.py \
    --input-path $MINT1T_TAR_PATH \
    --output-path ./output/ \
    --schema-overrides '{"url": "large_string", "pdf_name": "large_string"}' \
    --mode overwrite
```

## Common Use Cases

| Goal | Key flags |
|------|-----------|
| Filter narrow/wide images | `--min-aspect-ratio 0.5 --max-aspect-ratio 2.0` |
| Drop very short text rows | `--min-text-chars 50` |
| Write to WebDataset tar | `--writer-format webdataset` |
| Keep image bytes lazy (faster reads) | `--no-materialize-on-read` |
| Drop rows where image fetch fails | `--on-materialize-error drop_row` |
| Preserve custom JSON fields | `--schema-overrides '{"url": "large_string"}'` |
| Process multiple shards per task | `--files-per-partition 4` |

## Output

**Parquet output** (default):

```
output/
└── <shard_hash>.parquet   # one file per input shard
```

Each Parquet file uses the standard interleaved schema — one row per content element:

| Column | Description |
|--------|-------------|
| `sample_id` | Unique document identifier |
| `position` | Position within the document (images and text share the same sequence) |
| `modality` | `"text"`, `"image"`, or `"metadata"` |
| `content_type` | MIME type (e.g. `"text/plain"`, `"image/tiff"`) |
| `text_content` | Text for text rows |
| `binary_content` | Image bytes for image rows |
| `source_ref` | JSON pointer for lazy image fetch |

**Read the output:**

```python
import pandas as pd

df = pd.read_parquet("output/<shard_hash>.parquet")

# All text
text = df[df["modality"] == "text"][["sample_id", "position", "text_content"]]

# All images (binary_content holds raw image bytes)
images = df[df["modality"] == "image"][["sample_id", "position", "binary_content"]]
```

**WebDataset output** (`--writer-format webdataset`):

```
output_tars/
└── <shard_hash>.tar   # one tar per input shard, one JSON + images per sample
```

## Next Steps

- **Starting from PDFs?** See the [PDF Extraction Pipeline (Nemotron-Parse)](../nemotron_parse_pdf/) to convert PDFs into the same interleaved Parquet format used here.
