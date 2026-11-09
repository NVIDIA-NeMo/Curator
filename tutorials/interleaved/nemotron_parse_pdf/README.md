# PDF Extraction Pipeline using Nemotron-Parse

Convert PDFs into structured, interleaved parquet — text blocks, tables, images,
and captions in reading order — using **Nemotron Parse 3.5** through an
OpenAI-compatible endpoint or managed Dynamo. `main.py` defaults to the 3.5 model
and `single` mode. Explicit legacy selection and `inprocess.py` retain Parse v1.2
support; the public reader defaults remain unchanged. New Parse 3.5 and PNG-reuse
options are keyword-only, preserving existing positional reader and HTTP-client
calls.

We recommend using NeMo Curator's Dynamo-backed
`InferenceServer` with the HTTP client stage instead of loading vLLM inside the
pipeline stage. Internal comparisons found this to be the better default
because the serving layer can batch requests across pipeline tasks and keep
model replicas fed while PDF rendering and postprocessing scale independently.

The original tutorial validation used 8 H100 GPUs with request concurrency
values of 32 and 64. Use this as a starting configuration and validate it for
your Parse 3.5 workload:

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
uv venv
source .venv/bin/activate
uv pip install -e ".[interleaved_cpu,text_cpu,cv2]"
```

OpenCV is required for PDF page rendering and postprocessing but is not part of
the `all` extra, so the NeMo Curator container does not include it. Inside the
container, install it with `uv pip install "nemo-curator[cv2]"`. Verify with
`python -c "import cv2"`.

Install the matching model-team parser package as described in
[Served Nemotron Parse 3.5](#served-nemotron-parse-35). The first example below
reuses an existing server; the CPU environment does not load the model.
Managed Dynamo additionally requires a compatible driver stack and local
`etcd` and NATS binaries, as described in that section.

## Run the tutorial

**Step 1 — Create a manifest listing your PDFs:**

```bash
# One JSON line per PDF
for f in /path/to/pdfs/*.pdf; do
    echo "{\"file_name\": \"$(basename $f)\"}" >> manifest.jsonl
done
```

**Step 2 — Run against an existing Parse 3.5 endpoint:**

```bash
python tutorials/interleaved/nemotron_parse_pdf/main.py \
    --manifest manifest.jsonl \
    --pdf-dir /path/to/pdfs \
    --output-dir /path/to/output \
    --inference-server-endpoint http://localhost:8000/v1 \
    --model-path nvidia/Nemotron-3.5-Parse \
    --inference-server-model-name nemotron-parse \
    --inference-batch-size 32
```

`main.py` starts no model server when given an existing endpoint. It uses four
HTTP client workers by default; set `--client-num-workers` for your topology.
The default parse mode is `single`; choose `--parse-mode double` or `triple` for
additional crop/detection passes.

To let Curator manage Dynamo, provide a complete local repository with
`--nemotron-parse-path` and `--model-path`; see
[Served Nemotron Parse 3.5](#served-nemotron-parse-35). Curator detects the
Ray-visible GPUs, starts one replica per GPU, waits for readiness, runs the
pipeline, and stops Dynamo. Managed runs use `4 * num_gpus` HTTP workers unless
`--client-num-workers` is set. Use `CUDA_VISIBLE_DEVICES` or your Ray cluster
resources to select the GPUs. `main.py` does not expose a `--backend` option.

**Legacy alternative — Serve Parse v1.2:**

In a compatible v1.2 environment, explicitly select the legacy parser and model:

```bash
python tutorials/interleaved/nemotron_parse_pdf/main.py \
    --manifest manifest.jsonl --pdf-dir /path/to/pdfs --output-dir /path/to/output-v12 \
    --legacy-parse --model-path nvidia/NVIDIA-Nemotron-Parse-v1.2
```

Add `--inference-server-endpoint` to reuse an existing v1.2 endpoint. The
`--legacy-parse` and `--parse-mode` selectors are mutually exclusive.

**Legacy alternative — Run Parse v1.2 in process:**

```bash
uv sync --extra interleaved_cuda12 --extra inference_server --extra cv2
uv pip install albumentations==2.0.8  # required by the model's remote processor code
uv run python tutorials/interleaved/nemotron_parse_pdf/inprocess.py \
    --manifest manifest.jsonl \
    --pdf-dir /path/to/pdfs \
    --output-dir /path/to/output \
    --model-path nvidia/NVIDIA-Nemotron-Parse-v1.2 \
    --backend vllm \
    --enforce-eager
```

Use `inprocess.py` for Parse v1.2 local validation and debugging when you do not
want a separate serving topology. It keeps the v1.2 default and rejects served
Parse 3.5 modes.

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

For a page-level subset, add `selected_page_indices` to a manifest entry:

```json
{"file_name": "document.pdf", "selected_page_indices": [0, 4, 12]}
```

Use a nonempty list of unique, nonnegative integers. The values identify physical
pages in the original PDF, starting at zero. Explicit selection replaces the
first-`--max-pages` selection for that entry and keeps the original page numbers.
With served Parse 3.5, failed selected pages retain error rows.

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
| `--model-path` | `nvidia/Nemotron-3.5-Parse` (`main.py`, benchmark); v1.2 (`inprocess.py`) | Served model or local weights; legacy main/benchmark runs require an explicit model. |
| `--parse-mode` | `single` (`main.py`, benchmark) | Served Parse 3.5 mode: `single`, `double`, or `triple`. |
| `--legacy-parse` | off (`main.py`, benchmark) | Select the v1.2 path with an explicit `--model-path`; mutually exclusive with `--parse-mode`. |
| `--backend` | `vllm` | In-process engine (`inprocess.py`); benchmark also supports `hf` for explicit legacy in-process runs. |
| `--enforce-eager` | off | Skip vLLM CUDA graph capture (~35 min savings on first run) |
| `--max-num-seqs` | 64 | Max concurrent sequences for vLLM |
| `--inference-batch-size` | 32 (`main.py`), 4 (`inprocess.py`) | Concurrent requests per HTTP worker, or pages per in-process HF pass |
| `--pdfs-per-task` | 10 | PDFs per task; ignored with `--pages-per-task` |
| `--pages-per-task` | off | Optional selected-page budget across PDFs; requires served single mode and Ray Data |
| `--page-budget-extraction-config` | — | Resolved extraction-identity JSON required with `--pages-per-task` |
| `--max-pdfs` | — | Cap total PDFs (for testing) |
| `--dpi` | 300 | PDF rendering resolution |
| `--max-pages` | 50 | Legacy per-PDF limit; page mode processes the complete manifest selection |
| `--text-in-pic` | off | Predict text inside images (v1.2+ feature) |
| `--nemotron-parse-path` | — | Complete local Parse 3.5 repository required for managed Dynamo in `main.py` or the benchmark |
| `--reuse-encoded-images` | off | Reuse verified rendered PNG bytes for served Parse 3.5 `single` mode only |

## Served Nemotron Parse 3.5

The served CLI defaults to Parse 3.5 `single` mode. Use an existing endpoint or
let Curator manage Dynamo with a complete local model repository. Install the model team's package from the model repository,
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

To let the tutorial manage the model through Dynamo instead, download the complete
repository and pass it as both `--model-path` and `--nemotron-parse-path`. The
driver environment needs `ai-dynamo==1.4.2`, which pins vLLM 0.26; Curator's
locked release pins a vLLM that is older than Parse 3.5 supports. Each serving
worker installs the repository package with its own dependencies, and Dynamo's
frontend applies the repository's `chat_template.jinja`. Dynamo needs local
`etcd` and `nats-server` binaries: repository container images include them;
source environments can use
[`install_etcd_nats.sh`](https://github.com/NVIDIA-NeMo/Curator/blob/main/docker/common/install_etcd_nats.sh).
CUDA 13 environments must
set `NEMO_CURATOR_DYNAMO_CUDA_TAG=cu130`; the default remains Curator's `cu129`
actor environment. A CPU-only driver can keep CPU PyTorch: serving workers replace
it with the selected CUDA build.

```bash
hf download nvidia/Nemotron-3.5-Parse --revision 62cb1e4ba153d113ff64aaf2be6e068c377cf53c \
    --local-dir /path/to/nemotron-parse
uv pip install --no-deps /path/to/nemotron-parse
NEMO_CURATOR_DYNAMO_CUDA_TAG=cu130 python tutorials/interleaved/nemotron_parse_pdf/main.py \
    --manifest manifest.jsonl --pdf-dir /path/to/pdfs --output-dir /path/to/output \
    --model-path /path/to/nemotron-parse \
    --nemotron-parse-path /path/to/nemotron-parse \
    --parse-mode single --executor ray-data
```

- `single` reads the whole page.
- `double` also rereads table and chart crops from that answer.
- `triple` adds a boxes-only detection request before reading the crops;
  `--triple-regions` selects `union` (default) or `detector` regions.

Without page-based tasking, use `--executor xenna` to run the pipeline on Xenna.
Page-based tasking requires Ray Data. A page that fails to render
or parse is written as one `text` row with `materialize_error` set, which the tutorial
writer keeps with `on_materialize_error="warn"`. A response with no decoded blocks
also becomes an error row. The served path recovers missing or timed-out render
results with smaller page windows: the initial attempt has a 60-second timeout,
then recovery shares a 120-second budget and uses at most seven attempts including
the initial attempt. Unrecovered pages keep their physical page numbers and error
rows. The explicit v1.2 path keeps its single render attempt. See the
[Nemotron Parse PDF guide](../../../fern/versions/main/pages/curate-text/load-data/nemotron-parse-pdf.mdx)
for serving limits and parser behavior.

### Optional PNG reuse

Add `--reuse-encoded-images` to a served `--parse-mode single` command to avoid a
second PNG encode when preparing full-page requests. The matching Python option is
`NemotronParsePDFReader.inference_server_reuse_encoded_images=True`; directly
composed HTTP stages use `NemotronParseHTTPClientStage.reuse_encoded_images=True`.
All default to `False`, and enabling reuse requires served single-pass mode.

Reuse verifies the ordered producer batch, Pillow and zlib versions, dimensions,
and unchanged RGB pixels. Crops, altered images, image metadata, and unverified
inputs fall back to normal PNG encoding. It preserves rendering, parser settings,
page references, and output handling, and does not depend on page-based tasking
or a separate persistent HTTP client.

The existing inference-stage custom metrics include `image_reuse_count`,
`image_encode_count`, and `image_reuse_fallback_count`. The fallback count is a
subset of normal encodes. These count image preparations for logical parser
backend calls, rather than HTTP retry attempts, successful pages, or unique physical
pages. Sum them across tasks to check eligibility; compare pipeline wall time to
assess the benefit on a CPU-constrained workload.

### Optional page-based tasking

Use `--pages-per-task 64` with `--parse-mode single --executor ray-data` to pack
64 selected pages per task, except the final tail, across document boundaries.
Long PDFs are split for processing and regrouped into complete documents before
writing. This is a task budget, not an inference batch size or page cap:
`--pdfs-per-task` and `--max-pages` are ignored. Without the option, the existing
PDF-based tasking remains unchanged.

Page mode requires one PDF per manifest record with `source_locator`, an immutable
`source_version` for its actual bytes, and either `num_pages` (all pages) or
`selected_page_indices` (an explicit ordered selection). `source_version` is
caller-declared provenance, not automatic PDF-hash verification; keep inputs
immutable throughout every chunk read. Duplicate document identities, ambiguous
sample IDs, and grouped filename lists are rejected.

It also requires `--page-budget-extraction-config extraction.json`, containing a
resolved `pdf-page-extraction.v1` contract for parser identity, model artifacts,
requests, rendering/recovery, transport, and output projection. This records
settings rather than overriding them. Prepare it from the installed parser and
verified serving configuration; see the
[page-based tasking guide](../../../fern/versions/main/pages/curate-text/load-data/nemotron-parse-pdf.mdx#page-based-tasking)
for the required sections and source helpers.

```bash
python tutorials/interleaved/nemotron_parse_pdf/main.py \
    --manifest counted-versioned.jsonl --pdf-dir /path/to/pdfs \
    --output-dir /path/to/fresh-output \
    --inference-server-endpoint http://localhost:8000/v1 \
    --model-path nvidia/Nemotron-3.5-Parse \
    --inference-server-model-name nemotron-parse \
    --parse-mode single --executor ray-data \
    --pages-per-task 64 --page-budget-extraction-config extraction.json
```

Use a fresh, locally addressed output directory for each run and verify its backing
mount; a local filesystem API does not prove SSD storage or exclude NFS. The
builder writes the fixed page-output schema, rejects undeclared columns, retains
failed-page rows, and runs a mandatory final inventory check after writing. That
check detects missing whole PDFs as well as missing, duplicate, or substituted
pages; coverage passing does not establish extraction quality or all-page success.

Output filenames derive from verified document/selection/extraction identities.
The writer publishes each closed temporary Parquet with atomic replacement: a
complete retry can replace that file, and the last complete concurrent replacement
wins. It is not crash durability, no-clobber publication, exactly-once inference,
or a multi-document transaction. Interrupted temporary files are retained.
Regrouping is a global barrier, with materialized planning/inventory/results and
no regrouping checkpoint support; the page budget does not bound total memory.

## Reproducing the Parse 3.5 serving benchmarks

These scripts preserve fixed eight-GPU B200 workflows for CUDA 13 Dynamo and
direct vLLM. Their replica counts, FlashAttention 3 settings, and memory budgets
are B200 experiment settings, not portable defaults. These recipes are untested
on RTX GPUs. Use the parameterized `main.py` entry point for another topology
and validate its dependencies, memory headroom, output coverage, and failures.
The benchmark launchers refuse to overwrite existing result directories.

The generic `benchmarking/scripts/nemotron_parse_pdf_benchmark.py` also defaults
to Parse 3.5 single mode: supply `--nemotron-parse-path` for managed Dynamo, or
`--inference-server-endpoint` to reuse an existing server, and always provide
`--benchmark-results-path`. For v1.2 use `--legacy-parse` with an explicit
`--model-path`; the pinned v1.2 regression configurations retain that selection.

For an external endpoint, the generic benchmark requires explicit positive
`--num-replicas` and `--inference-server-num-gpus`. Replicas determine the HTTP
client-worker count; GPU count is only the throughput-normalization denominator.
It does not use client-side GPU discovery for external endpoints, or discover or
verify the remote topology. Declare the actual serving topology; managed-server
GPU discovery is unchanged.

For two replicas sharing one remote GPU:

```bash
python benchmarking/scripts/nemotron_parse_pdf_benchmark.py \
    --manifest manifest.jsonl --pdf-dir /path/to/pdfs --output-dir ./benchmark-output \
    --benchmark-results-path ./benchmark-results \
    --inference-server-endpoint http://parse-server:8000/v1 \
    --model-id nemotron-parse --executor ray_data \
    --num-replicas 2 --inference-server-num-gpus 1
```

The default four client workers per replica produce eight client workers here;
per-GPU throughput is divided by one. These flags apply to the benchmark only.

### Included files

| File | Purpose |
|---|---|
| `run_dynamo_uncapped.py` | Start Curator-managed Dynamo and run eager mode on, off, or both |
| `run_extractions.sh` | Run capped and uncapped DC10K cases through direct vLLM |
| `run_8gpu.sh` | Run the 16-replica direct-vLLM topology on eight GPUs |
| `run_8gpu_x100_direct.sh` | Run that topology on the physically repeated corpus |
| `create_x100_dataset.sh` | Create and validate the 100x physical corpus |
| `round_robin_proxy.py` | Distribute OpenAI-compatible requests across local vLLM replicas |
| `benchmark_run.py` | Record wall time and one-second GPU telemetry around a command |
| `benchmarking/scripts/nemotron_parse_pdf_benchmark.py` | Collect shared Parse throughput and output metrics |

### Prerequisites

- A Curator checkout with the Parse 3.5 and CUDA 13 Dynamo changes.
- A CUDA 13 Python environment containing Curator's PDF dependencies, vLLM,
  Dynamo, Ray, `aiohttp`, and the model's Python dependencies.
- Dynamo's `etcd` and NATS executables on `PATH` for managed-Dynamo runs.
- Linux with Bash, `setsid`, `nvidia-smi`, and `curl` on `PATH`.
- A local Nemotron Parse 3.5 model directory.
- A JSONL PDF manifest and the corresponding PDF root.
- All eight physical GPU indexes 0–7 assigned to the experiment for the preserved
  16-replica topology. The shell launchers target those indexes explicitly.

The shell launchers derive `CURATOR_ROOT` from their location and default
`PYTHON` to `${CURATOR_ROOT}/.venv-parse35-cuda13/bin/python`. Set either value
explicitly when the environment lives elsewhere.

### Managed Dynamo: uncapped DC10K

Run both graph mode and eager mode in separate processes:

```bash
python tutorials/interleaved/nemotron_parse_pdf/run_dynamo_uncapped.py \
  --eager-mode both \
  --curator-root /path/to/Curator \
  --python /path/to/cuda13-env/bin/python \
  --manifest /path/to/dc10k-manifest.jsonl \
  --pdf-dir /path/to/dc10k-pdfs \
  --model-path /path/to/Nemotron-3.5-Parse \
  --output-root /path/to/results \
  --run-name dc10k-reproduction-01
```

Use `--eager-mode off` or `--eager-mode on` for one mode. The runner sets
`NEMO_CURATOR_DYNAMO_CUDA_TAG=cu130`, starts 16 half-GPU replicas, excludes
server startup from pipeline throughput, and samples all eight GPUs once per
second. It uses the experimental `DynamoVLLMModelConfig.gpu_fraction=0.5`
reservation with normal GPU capacity checks; the Ray fraction does not reserve
VRAM. The B200 profile uses 16 HTTP workers, four concurrent requests per worker,
16 PDFs per task, and an effectively uncapped page limit. It sets
`gpu_memory_utilization=0.45` and an 8-GiB KV cache per replica. A unique
`--run-name` without path separators is required for every repeat.

### Direct vLLM: eight-GPU corpus

```bash
export CURATOR_ROOT=/path/to/Curator
export PYTHON=/path/to/cuda13-env/bin/python
export ROOT=/path/to/results
export MODEL=/path/to/Nemotron-3.5-Parse
export MANIFEST=/path/to/manifest.jsonl
export PDF_DIR=/path/to/pdfs

tutorials/interleaved/nemotron_parse_pdf/run_8gpu.sh
```

This starts two vLLM replicas per GPU in two waves, waits for every endpoint,
starts the round-robin proxy on port 8000, and runs the Curator pipeline with
16 PDFs per task, 36 HTTP workers, concurrency 64, 300 DPI, and a 50-page cap.

### Direct vLLM: capped and uncapped DC10K

`run_extractions.sh` expects `manifest-capped50.jsonl` and
`manifest-uncapped-window50.jsonl` under `ROOT`:

```bash
export CURATOR_ROOT=/path/to/Curator
export PYTHON=/path/to/cuda13-env/bin/python
export ROOT=/path/to/dc10k-experiment
export MODEL=/path/to/Nemotron-3.5-Parse
export PDF_DIR=/path/to/dc10k-pdfs

tutorials/interleaved/nemotron_parse_pdf/run_extractions.sh both
```

Pass `capped50` or `uncapped` instead of `both` to run only one condition.

### Creating and running the 100x corpus

The dataset builder makes independent physical copies rather than reflinks or
hard links:

```bash
tutorials/interleaved/nemotron_parse_pdf/create_x100_dataset.sh \
  /path/to/source-corpus \
  /path/to/repeated-corpus
```

After it validates the PDF and manifest counts, run:

```bash
export CURATOR_ROOT=/path/to/Curator
export PYTHON=/path/to/cuda13-env/bin/python
export ROOT=/path/to/results
export MODEL=/path/to/Nemotron-3.5-Parse
export MANIFEST=/path/to/repeated-corpus/manifest.jsonl
export PDF_DIR=/path/to/repeated-corpus

tutorials/interleaved/nemotron_parse_pdf/run_8gpu_x100_direct.sh
```

### Outputs and timing boundaries

Each measured run writes a result directory containing:

- `output/` with extracted Parquet data;
- `command.log` with pipeline stdout and stderr;
- `gpu-samples.csv` with one row per GPU per second;
- `run-summary.json` with elapsed time and per-GPU utilization; and
- `topology.json` for the Dynamo runner.

The Dynamo summary reports server startup separately and measures GPU
utilization only around `pipeline.run()`. The direct launchers start vLLM before
invoking `benchmark_run.py`, so their recorded interval also excludes model
startup. Both paths include PDF rendering, inference, postprocessing, and
output writing in the measured pipeline interval. These are pipeline-window
utilization measurements; use a separate whole-job monitor to include model
startup and shutdown. Compare the reported timestamps and per-GPU sample counts,
and keep idle gaps inside the measured interval. Sampling or command failures
remain visible in `run-summary.json` instead of producing a successful timing.

The launchers trap normal exit and signals and terminate the processes they
start. After an interrupted Dynamo run, verify that no experiment-owned etcd,
NATS, frontend, or vLLM process remains before starting another run.
