# Tutorial: PDF Parsing with Nemotron-Parse

Process PDFs through **Nemotron-Parse v1.2** to produce structured, interleaved parquet output — text blocks, tables, images, and captions in reading order.

## Pipeline Overview

```
manifest.jsonl
     │
     ▼
PDFPartitioningStage  [CPU]  — reads manifest, groups PDFs into batches
     │
     ▼
PDFPreprocessStage    [CPU]  — extracts PDF bytes, renders pages → PNG images
     │
     ▼
NemotronParseInferenceStage [GPU]  — vLLM inference (Nemotron-Parse v1.2)
     │
     ▼
NemotronParsePostprocessStage [CPU] — parses model output, aligns images/captions
     │
     ▼
InterleavedParquetWriterStage [CPU] — writes interleaved parquet output
```

**Model**: `nvidia/NVIDIA-Nemotron-Parse-v1.2`  
**Backend**: vLLM (recommended) or HuggingFace Transformers  
**Hardware**: 1 GPU per vLLM instance (TP1); scales linearly across GPUs/nodes

---

## Input Formats

The pipeline supports **3 input formats** via mutually exclusive flags.

### 1. PDF Directory (`--pdf-dir`)

A flat directory of `.pdf` files. Simplest format for local use.

**Directory layout:**
```
pdfs/
├── doc001.pdf
├── doc002.pdf
└── ...
manifest.jsonl       # one JSON per line
```

**Manifest format** (`manifest.jsonl`):
```json
{"file_name": "doc001.pdf"}
{"file_name": "doc002.pdf"}
```

**Generate manifest from a directory:**
```bash
for f in /path/to/pdfs/*.pdf; do
    echo "{\"file_name\": \"$(basename $f)\"}" >> manifest.jsonl
done
```

**Run command:**
```bash
python tutorials/interleaved/nemotron_parse_pdf/main.py \
    --manifest /path/to/manifest.jsonl \
    --pdf-dir /path/to/pdfs \
    --output-dir /path/to/output \
    --backend vllm \
    --enforce-eager
```

---

### 2. CC-MAIN Zip Archives (`--zip-base-dir`)

CC-MAIN-2021-31-PDF-UNTRUNCATED style: PDFs are stored in numbered zip archives
under a two-level numeric folder hierarchy.

**Directory layout:**
```
zipfiles/
├── 0000-0999/
│   ├── 0000.zip      ← contains 0000000.pdf – 0000999.pdf
│   ├── 0001.zip      ← contains 0001000.pdf – 0001999.pdf
│   └── ...
├── 1000-1999/
│   ├── 1000.zip
│   └── ...
└── ...
manifest.jsonl
```

Path resolution: `0001234.pdf` → `zipfiles/0001-0001/0001.zip`

**Manifest format** (CC-MAIN style — multiple files per URL):
```json
{"cc_pdf_file_names": ["0000000.pdf", "0000001.pdf", ...], "url": "https://..."}
```

Or single-file style:
```json
{"file_name": "0000000.pdf", "url": "https://..."}
```

**Run command:**
```bash
python tutorials/interleaved/nemotron_parse_pdf/main.py \
    --manifest /path/to/manifest.jsonl \
    --zip-base-dir /path/to/zipfiles \
    --output-dir /path/to/output \
    --backend vllm \
    --enforce-eager
```

> **Dataset source**: [CC-MAIN-2021-31-PDF-UNTRUNCATED](https://github.com/tballison/CC-MAIN-2021-31-PDF-UNTRUNCATED)

---

### 3. GitHub JSONL (`--jsonl-base-dir`)

GitHub PDF datasets where each PDF is base64-encoded inside a JSONL file.
Each record has a `content` field (base64 PDF bytes) and metadata fields.

**Directory layout:**
```
github_pdfs/
├── 0001/
│   ├── 20260220_03h02m32s_xxx.jsonl
│   └── ...
├── 0002/
│   └── ...
manifest.jsonl
```

**JSONL record format:**
```json
{
  "repo_id": 979220528,
  "full_name": "user/repo",
  "file_name": "document.pdf",
  "encoding": "base64",
  "content": "<base64-encoded PDF bytes>",
  ...
}
```

**Manifest format** (with byte offsets for O(1) seeks):
```json
{"file_name": "document.pdf", "jsonl_file": "0001/xxx.jsonl", "byte_offset": 0, "url": "https://github.com/user/repo"}
```

**Generate manifest** (see `slurm/count_github_pdfs.py` for the full script):
```python
with open(jsonl_path, 'rb') as f:
    while True:
        offset = f.tell()
        line = f.readline()
        if not line: break
        rec = json.loads(line)
        manifest.append({
            "file_name": rec["file_name"],
            "jsonl_file": f"{folder}/{jsonl_name}",
            "byte_offset": offset,
        })
```

**Run command:**
```bash
python tutorials/interleaved/nemotron_parse_pdf/main.py \
    --manifest /path/to/manifest.jsonl \
    --jsonl-base-dir /path/to/github_pdfs \
    --output-dir /path/to/output \
    --backend vllm \
    --enforce-eager
```

---

## Output Schema

Each row in the output parquet represents one **document element** in reading order.

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Unique identifier (PDF filename without extension) |
| `position` | int | Element position within the document |
| `modality` | string | `text`, `image`, `table`, or `metadata` |
| `content_type` | string | `text/markdown`, `image/png`, or `application/json` |
| `text_content` | string | Extracted text (markdown for text/tables; empty for images) |
| `binary_content` | bytes | Image bytes (for image elements; null for text) |
| `source_ref` | string | Path to source file |
| `url` | string | Original URL (from manifest) |
| `page_number` | int | Page within PDF (0-indexed) |
| `pdf_name` | string | PDF filename |
| `element_class` | string | Nemotron-Parse class label |
| `materialize_error` | bool | True if image crop extraction failed |

**Reading output in Python:**
```python
import pandas as pd

df = pd.read_parquet("output/abc123.parquet")
print(df[["sample_id", "modality", "content_type", "text_content"]].head(10))

# Get all text elements
text = df[df["modality"] == "text"]["text_content"]

# Get all images
from PIL import Image
import io
images = [Image.open(io.BytesIO(b)) for b in df[df["modality"] == "image"]["binary_content"]]
```

---

## Performance

Benchmarked on NVIDIA H100 80GB with `--backend vllm --enforce-eager`:

| Scale | PDFs | GPUs | Throughput | Per-GPU | Pipeline Efficiency |
|-------|------|------|------------|---------|---------------------|
| 1 node | 500 | 8 | 12.8 pages/s | 1.60 pages/s/GPU | 80% |
| 4 nodes | 83K | 32 | 80.7 pages/s | 2.52 pages/s/GPU | 98.1% |
| 40 nodes | 985K | 320 | 986 pages/s | 3.08 pages/s/GPU | 90.5% |

Per-stage breakdown (40-node run, 10 PDFs/task):

| Stage | Avg time/task |
|-------|--------------|
| PDFPreprocessStage (CPU) | 30.7s |
| NemotronParseInferenceStage (GPU) | 22.3s |
| NemotronParsePostprocessStage (CPU) | 9.2s |

The pipeline runs all stages concurrently in streaming mode, hiding CPU preprocessing
behind GPU inference (**98% pipeline efficiency** at 4-node scale).

### vLLM vs HuggingFace Transformers

| Metric | HF Transformers | vLLM | Speedup |
|--------|-----------------|------|---------|
| Throughput | 4.3 pages/s | 12.8 pages/s | **3.0×** |
| GPU inference/page | 1.449s | 0.389s | **3.7×** |

> vLLM's continuous batching reduces GPU inference time by 3.7×. Use `--backend vllm`
> for production. HF backend requires `transformers==4.51.3` (newer versions break due to
> `DynamicCache` API changes).

---

## Full CLI Reference

```
python tutorials/interleaved/nemotron_parse_pdf/main.py [OPTIONS]

Required:
  --manifest PATH        Path to JSONL manifest listing PDFs
  --output-dir PATH      Output directory for parquet files

Source (exactly one required):
  --pdf-dir PATH         Directory containing PDF files
  --zip-base-dir PATH    Root of CC-MAIN zip archive hierarchy
  --jsonl-base-dir PATH  Root of JSONL-based PDF dataset (GitHub style)

Model:
  --model-path PATH      HuggingFace model ID or local path
                         (default: nvidia/NVIDIA-Nemotron-Parse-v1.2)
  --backend {vllm,hf}    Inference backend (default: vllm)
  --enforce-eager        Skip vLLM CUDA graph capture (~35min savings on first run)
  --max-num-seqs N       Max concurrent sequences for vLLM (default: 64)
  --text-in-pic          Predict text inside pictures (v1.2+ feature)

Processing:
  --pdfs-per-task N      PDFs per processing batch (default: 10)
  --max-pdfs N           Limit total PDFs (for testing)
  --dpi N                PDF rendering resolution (default: 300)
  --max-pages N          Max pages per PDF (default: 50)

Resume:
  --resume               Skip already-processed PDFs (checks output dir)
```

---

## Running at Scale with SLURM

For large-scale processing, use the SLURM submit scripts in `tutorials/slurm/`:

```bash
# Container-based (recommended — uses official NGC nemo-curator:26.02 image)
sbatch tutorials/slurm/submit_container.sh

# Bare-metal with uv
sbatch tutorials/slurm/submit.sh
```

Both scripts use `SlurmRayClient` which automatically handles head/worker node
coordination across SLURM nodes.

**Key SLURM considerations:**
- Set `RAY_PORT_BROADCAST_DIR` to a shared filesystem path (not `/tmp`, which is node-local)
- Use `--resume` for long-running jobs that may hit the SLURM time limit
- The pipeline checkpoints at the task level — completed tasks are not re-processed

---

## Sample Datasets

The `benchmarking/pdf_tutorial_dataset/` directory contains 15-PDF sample datasets
in all three input formats, along with pre-computed output parquets for reference.

```
benchmarking/pdf_tutorial_dataset/
├── pdf_dir/
│   ├── pdfs/           ← 15 CC PDFs (0000000.pdf – 0000014.pdf)
│   ├── manifest.jsonl
│   └── output/         ← 3 parquets + perf stats
├── cc_zip/
│   ├── 0000-0999/
│   │   └── 0000.zip    ← 15 PDFs packed
│   ├── manifest.jsonl
│   └── output/
└── github_jsonl/
    ├── 0001/
    │   └── tutorial_sample.jsonl   ← 15 records with base64 PDFs
    ├── manifest.jsonl
    └── output/
```
