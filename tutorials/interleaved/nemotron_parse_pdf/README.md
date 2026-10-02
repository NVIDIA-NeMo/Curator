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
and stops Dynamo. No separately managed server process is required. It fixes
the HTTP stage pool at `4 * num_gpus` workers. Use `CUDA_VISIBLE_DEVICES` or
your Ray cluster resources to control which GPUs are used.

`main.py` always starts Dynamo with vLLM and does not expose a `--backend`
option.

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
| `--backend` | `vllm` | In-process engine (`inprocess.py` only); also supports `hf`. |
| `--enforce-eager` | off | Skip vLLM CUDA graph capture (~35 min savings on first run) |
| `--max-num-seqs` | 64 | Max concurrent sequences for vLLM |
| `--inference-batch-size` | 32 (`main.py`), 4 (`inprocess.py`) | Concurrent requests per HTTP worker, or pages per in-process HF pass |
| `--pdfs-per-task` | 10 | PDFs batched per processing task |
| `--max-pdfs` | — | Cap total PDFs (for testing) |
| `--dpi` | 300 | PDF rendering resolution |
| `--max-pages` | 50 | Max pages per PDF |
| `--text-in-pic` | off | Predict text inside images (v1.2+ feature) |

## Extract with NeMo Retriever Library (experimental)

The [`nrl_lance.py`](nrl_lance.py) recipe runs Nemotron-Parse through
[NeMo Retriever Library](https://github.com/NVIDIA/NeMo-Retriever) (NRL) and
hands the parsed elements to native Curator stages through a Lance dataset. It
extracts elements only; it does not chunk, embed, or index content.

NRL and Curator pin different GPU stacks, so the two commands run in separate
environments:

| Command | Environment | Work |
|---------|-------------|------|
| `ingest` | NRL, with a GPU visible to Ray | Deduplicate PDFs, run NRL's PDF graph with Nemotron-Parse v1.2, validate every page, and write the `pdf_elements` Lance table |
| `consume` | Curator, CPU only | Read the pinned Lance version with `InterleavedLanceReader`, decode every image, write Parquet, reconcile it with the source, and publish completion |

**Requirements:**

- An NRL release that records `raw_output` and finish reasons in the
  `nemotron_parse_v1_2` page metadata and supports the `strict_rows_per_block`
  executor option, installed with its local vLLM dependencies. `ingest` checks
  both before extracting, fails when Ray cannot see a GPU, and never falls back
  to a remote endpoint.
- A Curator environment with the Lance and OpenCV extras:

```bash
uv sync --extra interleaved_cpu --extra lance --extra cv2
```

**Inputs:** pass `--input-dir` to process every PDF under a directory, or
`--manifest` with one JSON object per line:

```json
{"path": "pdfs/report.pdf", "url": "https://example.com/report.pdf", "valid_blank_pages": [3]}
```

`path` may be relative to the manifest, and symlinks are resolved. `url` and
`valid_blank_pages` are optional, and page numbers are zero-based. Byte-identical
PDFs are parsed once and recorded as aliases of one document.

**Run:**

```bash
# NRL environment
python tutorials/interleaved/nemotron_parse_pdf/nrl_lance.py ingest \
    --manifest manifest.jsonl \
    --output-root /path/to/runs \
    --run-id run-001

# Curator environment
python tutorials/interleaved/nemotron_parse_pdf/nrl_lance.py consume \
    --handoff-manifest /path/to/runs/run-001/handoff_manifest.json \
    --output-dir /path/to/export
```

`ingest` writes `run-001/pdf_elements.lance` and, after validating the stored
table, a sealed `handoff_manifest.json`. `consume` writes Parquet and a sealed
`consume_validation.json` to the fresh output directory, then creates
`completion_manifest.json` in the run directory. The completion manifest is the
only signal that a run is published.

`consume --consume-executor` selects the executor for the consume stages and
the Parquet reread. `in_process` runs the existing CPU stages sequentially in
the consume process; `ray` uses Ray Data. The default, `auto`, selects
`in_process` for tables with at most 50,000 rows and `ray` for larger tables.
This row-count threshold is a heuristic: compare both modes on representative
inputs and hardware before selecting an override. Both modes retain image
checks, schema and content reconciliation, and publication validation. The
consume report records the selected executor. This option does not change
`ingest` or run Parse again.

`ingest` scheduling flags tune Ray without changing extraction or adding Parse
replicas: `--parse-batch-size` (pages per Parse batch, default 64; every batch
except the last is full),
`--parse-cpus` (CPUs reserved for the Parse actor, default 1),
`--projection-workers` (CPU projection actors, 1 to 8, default 8), and
`--projection-block-rows` (page rows per block before projection).

### Page outcomes

Each page is published whole or not at all. A page fails when NRL reports an
error for it, including a vLLM finish reason other than `stop`, when the model
response contains anything outside complete elements, when an element has no
class or an invalid box, or when a `Picture` crop cannot be produced or is smaller
than 10 pixels. An empty response counts as blank only on pages listed in
`valid_blank_pages`; otherwise the page fails with `unexpected_empty_output`.

A document is `success` or `valid_blank` when every page validates, `partial`
when some pages validate but the document has an issue such as a failed,
missing, or unexpected page, and `failed` when none validate. Failed documents produce
no rows. Each document's metadata row records its `extraction_status`,
`page_outcomes`, and issues, so the export carries its own coverage. These
statuses describe structural validation, not extraction accuracy.

Run directories, Lance versions, and output directories are never reused. Retry
a failed `ingest` with a new `--run-id`, and a failed `consume` with the same
handoff and a new `--output-dir`.

### `pdf_elements` schema

The table extends the interleaved schema with provenance columns:

| Column | Description |
|--------|-------------|
| `sample_id` | SHA-256 of the PDF bytes |
| `position` | `-1` for the metadata row, then document-wide element order |
| `modality` | `metadata`, `text`, `table`, or `image` |
| `content_type` | `application/json`, `text/markdown`, or `image/png` |
| `text_content` | Metadata JSON, element text, or model-native table markup |
| `binary_content` | Inline PNG bytes for `image` rows |
| `source_ref`, `materialize_error` | Always null |
| `url` | Source URL from the manifest |
| `page_number` | Zero-based source page; null for the metadata row |
| `pdf_name`, `source_path`, `source_aliases` | Representative file name and path, and every byte-identical alias |
| `element_class` | Nemotron-Parse class, such as `Text`, `Table`, or `Picture` |
| `content_sha256` | Same digest as `sample_id` |
| `bbox_xyxy_norm`, `bbox_coordinate_space` | Element box, normalized to Nemotron-Parse's 1664x2048 padded canvas |
| `run_id` | Ingest run identifier |

`Picture` maps to `image`, `Table` to `table`, and every other class, including
`Chart` and `Infographic`, to `text`, matching the native postprocessor.

### Limitations

- Each `ingest` runs one Nemotron-Parse model on one GPU.
- `ingest` collects every parsed element, including picture crops, in the
  driver before writing Lance, so memory grows with the input. Split large
  corpora across runs.
- `consume` reads the run directory and re-hashes the source PDFs at the
  absolute paths that `ingest` recorded, so both commands must see the same
  paths.
- The recipe and the native pipeline differ, so they do not produce identical
  output:
  - NRL renders pages at 200 DPI; native renders at 300 DPI by default.
  - NRL's local model allows 9,000 output tokens per page; native allows 8,192.
  - The recipe parses every page; native parses at most `--max-pages` (default
    50) pages per PDF.
  - The recipe keeps element text as NRL post-processes it, including empty
    elements, and NRL renames `Inline-formula` to `Formula`. Native strips
    `<...>` tags and drops empty non-`Picture` elements.
  - The recipe fails a page whose response does not parse completely or whose
    `Picture` crop fails. Native keeps the elements it can parse, falls back to
    one `Text` element when none parse, and skips failed crops.
  - Recipe crops come from the full-resolution page render; native crops come
    from the padded model canvas.

### Tests

The recipe tests run in both environments. Curator CI runs the Curator
environment set; tests that import NRL run only in the NRL environment:

```bash
# Curator environment; the NRL graph tests are skipped
pytest tests/tutorials -m "not gpu"

# NRL environment
pytest tests/tutorials/interleaved/nemotron_parse_pdf/test_nrl_graph.py \
    tests/tutorials/interleaved/nemotron_parse_pdf/test_nrl_lance_contract.py \
    --confcutdir=tests/tutorials/interleaved/nemotron_parse_pdf
```
