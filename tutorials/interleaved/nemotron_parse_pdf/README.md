# Nemotron-Parse PDF Tutorial

Convert PDFs into structured, interleaved parquet — text blocks, tables, images, and captions in reading order — using **Nemotron-Parse v1.2**.

## Setup

```bash
git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
pip install uv
uv sync --extra interleaved_cuda12
```

## Quickstart

**Step 1 — Create a manifest listing your PDFs:**

```bash
# One JSON line per PDF
for f in /path/to/pdfs/*.pdf; do
    echo "{\"file_name\": \"$(basename $f)\"}" >> manifest.jsonl
done
```

**Step 2 — Run the pipeline:**

```bash
python tutorials/interleaved/nemotron_parse_pdf/main.py \
    --manifest manifest.jsonl \
    --pdf-dir /path/to/pdfs \
    --output-dir /path/to/output \
    --backend vllm \
    --enforce-eager
```

## Input formats

The pipeline supports three input formats selected by a mutually exclusive flag:

| Flag | Description |
|------|-------------|
| `--pdf-dir PATH` | Flat directory of `.pdf` files |
| `--zip-base-dir PATH` | CC-MAIN-style numbered zip archives |
| `--jsonl-base-dir PATH` | GitHub-style JSONL with base64-encoded PDFs |

## Output schema

Each row in the output parquet is one **document element** in reading order:

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | PDF filename without extension |
| `position` | int | Element index within document |
| `modality` | string | `text`, `image`, `table`, or `metadata` |
| `content_type` | string | `text/markdown`, `image/png`, or `application/json` |
| `text_content` | string | Extracted text (markdown for text/tables) |
| `binary_content` | bytes | PNG bytes for image elements |
| `page_number` | int | Source page (0-indexed) |
| `url` | string | Source URL from manifest |

**Read the output:**

```python
import pandas as pd

df = pd.read_parquet("output/my_doc.parquet")
print(df[["modality", "content_type", "text_content"]].head(10))

# All text
text_blocks = df[df["modality"] == "text"]["text_content"].tolist()

# All images
from PIL import Image
import io
images = [Image.open(io.BytesIO(b)) for b in df[df["modality"] == "image"]["binary_content"]]
```

## Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `vllm` | Inference backend (`vllm` or `hf`) |
| `--enforce-eager` | off | Skip vLLM CUDA graph capture (~35 min savings on first run) |
| `--max-num-seqs` | 64 | Max concurrent sequences for vLLM |
| `--pdfs-per-task` | 10 | PDFs batched per processing task |
| `--max-pdfs` | — | Cap total PDFs (for testing) |
| `--dpi` | 300 | PDF rendering resolution |
| `--max-pages` | 50 | Max pages per PDF |
| `--text-in-pic` | off | Predict text inside images (v1.2+ feature) |
