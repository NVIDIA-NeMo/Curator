# PDF Extraction Pipeline using Nemotron-Parse

Convert PDFs into structured, interleaved parquet — text blocks, tables, images, and captions in reading order — using **Nemotron-Parse v1.2**.

For production PDF pipelines, use NeMo Curator's Dynamo-backed
`InferenceServer` with the HTTP client stage instead of loading vLLM inside the
pipeline stage. Internal comparisons found this to be the better default
because the serving layer can batch requests across pipeline tasks and keep
model replicas fed while PDF rendering and postprocessing scale independently.

Use this starting configuration:

- One inference-server replica per GPU.
- A fixed HTTP stage pool of `4 * num_gpus` workers.
- `inference_batch_size=32` or `64`, which is the maximum number of concurrent
  page requests sent by each HTTP client worker. Start with 32 and try 64 when
  the server still has scheduling headroom.

## Setup

```bash
git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
pip install uv
uv sync --extra interleaved_cuda12 --extra inference_server
```

The NeMo Curator container includes the `etcd` and `nats-server` binaries that
Dynamo starts. When using a source environment outside the container, install
those binaries before running the Dynamo entry point.

## Local smoke test

**Step 1 — Create a manifest listing your PDFs:**

```bash
# One JSON line per PDF
for f in /path/to/pdfs/*.pdf; do
    echo "{\"file_name\": \"$(basename $f)\"}" >> manifest.jsonl
done
```

**Step 2 — Run the in-process pipeline:**

```bash
uv run python tutorials/interleaved/nemotron_parse_pdf/main.py \
    --manifest manifest.jsonl \
    --pdf-dir /path/to/pdfs \
    --output-dir /path/to/output \
    --backend vllm \
    --enforce-eager
```

## Recommended inference-server pipeline

Run the Dynamo tutorial entry point:

```bash
uv run python tutorials/interleaved/nemotron_parse_pdf/dynamo.py \
    --manifest manifest.jsonl \
    --pdf-dir /path/to/pdfs \
    --output-dir /path/to/output \
    --model-path nvidia/NVIDIA-Nemotron-Parse-v1.2 \
    --inference-batch-size 32
```

`dynamo.py` detects the Ray-visible GPUs, starts one Dynamo model replica per
GPU, waits for the OpenAI-compatible endpoint to become healthy, and calls
`create_nemotron_parse_pdf_pipeline`. It fixes the HTTP stage pool at
`4 * num_gpus` workers and stops Dynamo after the pipeline finishes. Use
`CUDA_VISIBLE_DEVICES` or your Ray cluster resources to control which GPUs are
used. The default inference batch size for this entry point is 32; pass 64 to
compare the higher per-worker request concurrency on your corpus.

See the [Inference Server guide](https://docs.nvidia.com/nemo/curator/latest/curate-text/synthetic/inference-server)
for details about the `InferenceServer`, `DynamoVLLMModelConfig`, and
`DynamoServerConfig` objects used by the entry point.

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
| `--inference-batch-size` | 4 | Pages per HF pass or concurrent requests per HTTP client; use 32 or 64 with an inference server |
| `--pdfs-per-task` | 10 | PDFs batched per processing task |
| `--max-pdfs` | — | Cap total PDFs (for testing) |
| `--dpi` | 300 | PDF rendering resolution |
| `--max-pages` | 50 | Max pages per PDF |
| `--text-in-pic` | off | Predict text inside images (v1.2+ feature) |
