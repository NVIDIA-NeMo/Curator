# PDF Extraction Pipeline using Nemotron-Parse

Convert PDFs into structured, interleaved parquet — text blocks, tables, images, and captions in reading order — using **Nemotron-Parse v1.2**.

We recommend using NeMo Curator's Dynamo-backed
`InferenceServer` with the HTTP client stage instead of loading vLLM inside the
pipeline stage. Internal comparisons found this to be the better default
because the serving layer can batch requests across pipeline tasks and keep
model replicas fed while PDF rendering and postprocessing scale independently.

We tested the tutorial on 8 H100 GPUs with request concurrency values of 32 and
64. Use this starting configuration:

- One inference-server replica per GPU.
- A fixed HTTP stage pool of `4 * num_gpus` workers.
- `inference_batch_size=32`, which is the maximum number of concurrent page
  requests sent by each HTTP client worker. Because the best value depends on
  the GPU and corpus, benchmark 64 on the target workload and keep it only if it
  improves throughput without request failures or out-of-memory errors.

## Setup

```bash
git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
pip install uv
uv sync --extra interleaved_cuda12 --extra inference_server --extra cv2
```

OpenCV is required for PDF page rendering and postprocessing but is not part of
the `all` extra, so the NeMo Curator container does not include it. Inside the
container, install it with `uv pip install "nemo-curator[cv2]"`. Verify with
`python -c "import cv2"`.

The NeMo Curator container includes the `etcd` and `nats-server` binaries that
Dynamo starts. For a source environment outside the container, install them
with [`docker/common/install_etcd_nats.sh`](https://github.com/NVIDIA-NeMo/Curator/blob/main/docker/common/install_etcd_nats.sh)
before running the recommended entry point.

## Run the tutorial

**Step 1 — Create a manifest listing your PDFs:**

```bash
# One JSON line per PDF
for f in /path/to/pdfs/*.pdf; do
    echo "{\"file_name\": \"$(basename $f)\"}" >> manifest.jsonl
done
```

**Step 2 — Start Dynamo and run the pipeline (recommended):**

```bash
uv run python tutorials/interleaved/nemotron_parse_pdf/main.py \
    --manifest manifest.jsonl \
    --pdf-dir /path/to/pdfs \
    --output-dir /path/to/output \
    --model-path nvidia/NVIDIA-Nemotron-Parse-v1.2 \
    --inference-batch-size 32
```

`main.py` detects the Ray-visible GPUs, starts one Dynamo model replica per GPU,
waits for the OpenAI-compatible endpoint to become healthy, runs the pipeline,
and stops Dynamo. No separately managed server process is required. The HTTP
stage defaults to `4 * num_gpus` workers; override it with `--client-num-workers`.
Use `CUDA_VISIBLE_DEVICES` or your Ray cluster resources to control which GPUs are used.

Without `--inference-server-endpoint`, `main.py` starts Dynamo with vLLM.
This includes served Parse 3.5 when `--parse-mode` is selected and its model
package is installed in the driver environment. `main.py` does not expose a
`--backend` option.

**Alternative — Run inference in process:**

```bash
uv pip install albumentations==2.0.8  # required by the model's remote processor code
uv run python tutorials/interleaved/nemotron_parse_pdf/inprocess.py \
    --manifest manifest.jsonl \
    --pdf-dir /path/to/pdfs \
    --output-dir /path/to/output \
    --backend vllm \
    --enforce-eager
```

Use `inprocess.py` for local validation and debugging when you do not want a
separate serving topology.

The in-process and Ray Serve paths use Triton attention automatically on
Ampere GPUs, following the model's A100/A10 guidance. On Blackwell, they also
use Triton with vLLM versions before 0.23 to avoid the affected
FlashInfer/TRTLLM implementation. Ray Serve bases this choice on the
driver-visible GPU and assumes its architecture matches the serving replicas.
Explicit settings are preserved.

The entry point uses `create_nemotron_parse_inference_server`, which keeps the
Nemotron-Parse vLLM, Dynamo, and runtime-environment settings shared with the
benchmark. See the [Inference Server guide](https://docs.nvidia.com/nemo/curator/latest/curate-text/synthetic/inference-server)
for details about the underlying configuration objects.

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
| `content_type` | string | `text/markdown`, `text/html`, `image/png`, or `application/json` |
| `text_content` | string | Extracted text (Markdown; served tables use HTML) |
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
| `--backend` | `vllm` | In-process engine (`inprocess.py` only); also supports `hf`. |
| `--enforce-eager` | off | Skip vLLM CUDA graph capture (~35 min savings on first run) |
| `--max-num-seqs` | 64 | Max concurrent sequences for vLLM |
| `--inference-batch-size` | 32 (`main.py`), 4 (`inprocess.py`) | Concurrent requests per HTTP worker, or pages per in-process HF pass |
| `--pdfs-per-task` | 10 | PDFs batched per processing task |
| `--max-pdfs` | — | Cap total PDFs (for testing) |
| `--dpi` | 300 | PDF rendering resolution |
| `--max-pages` | 50 | Max pages per PDF |
| `--text-in-pic` | off | Predict text inside images (v1.2+ feature) |

## Served Nemotron Parse 3.5

Set `--parse-mode` to read pages with Nemotron Parse 3.5 through an inference server
that you start and stop. Install the model team's package from the model repository,
without the weights or its inference dependencies, at the revision that your server runs:

```bash
uv pip install "nemo-curator[interleaved_cpu,text_cpu,cv2]"
hf download nvidia/Nemotron-3.5-Parse --revision 62cb1e4ba153d113ff64aaf2be6e068c377cf53c \
    --exclude "*.safetensors" --local-dir nemotron-parse
uv pip install --no-deps ./nemotron-parse
```

```bash
python tutorials/interleaved/nemotron_parse_pdf/main.py \
    --manifest manifest.jsonl --pdf-dir /path/to/pdfs --output-dir /path/to/output \
    --inference-server-endpoint http://localhost:8000/v1 \
    --model-path nvidia/Nemotron-3.5-Parse \
    --inference-server-model-name nemotron-parse \
    --parse-mode single --inference-batch-size 32 --client-num-workers 4 \
    --executor ray-data
```

Omit `--inference-server-endpoint` to let the tutorial manage the same model
through Dynamo. CUDA 13 environments must set
`NEMO_CURATOR_DYNAMO_CUDA_TAG=cu130`; the default remains Curator's `cu129`
actor environment.

- `single` reads the whole page.
- `double` also rereads table and chart crops from that answer.
- `triple` adds a boxes-only detection request before reading the crops;
  `--triple-regions` selects `union` (default) or `detector` regions.

Use `--executor xenna` to run the same pipeline on Xenna. A page that fails to render
or parse is written as one `text` row with `materialize_error` set, which the tutorial
writer keeps with `on_materialize_error="warn"`. See the
[Nemotron Parse PDF guide](../../../fern/versions/main/pages/curate-text/load-data/nemotron-parse-pdf.mdx)
for serving limits and parser behavior.
