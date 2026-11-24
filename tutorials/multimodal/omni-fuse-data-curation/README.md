# Omni-Fuse Data Curation

Omni-Fuse ([paper](https://arxiv.org/pdf/2605.01163v1)) curates paired multimodal datasets by tightening raw/annotation alignment and ranking records for a target data blend. This tutorial implements the pipeline with NeMo Curator stages:

1. **Validate (`0_validate_inputs.py`)**: check data manifests, API credentials, LanguageBind, and model weights.
2. **Symmetric Nucleus Subsampling (`1_sns.py`)**: trim raw inputs (forward SNS) and annotations (backward SNS) to mutually informative content.
3. **Expert Embedding Engine (`2_embed.py`)**: embed raw and annotation views with text-based, fusion, and end-to-end experts (or overlap Steps 1–2 on multiple GPUs with `1_2_parallel.py`).
4. **Projection (`3_project.py`)**: train an MLP with contrastive and modality-bias losses to align concatenated expert embeddings.
5. **Datablend Ranking (`4_datablend.py`)**: rank projected records against a target query and export the blend.

The tutorial uses an API-first hybrid setup: NVIDIA Build APIs handle media description and text embeddings (with an option to self-host), while local models handle multimodal similarity, forward extraction, and fusion/end-to-end embeddings.

| Component | Backend | Default Model / Asset |
| --- | --- | --- |
| Media descriptions (backward SNS, text-based EEE) | NVIDIA API (or self-hosted) | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` |
| Text embeddings (text-based EEE) | NVIDIA API (or self-hosted) | `nvidia/nemotron-3.5-embed-1b` |
| Multimodal similarity & end-to-end EEE | Local GPU | `nvidia/omni-embed-nemotron-3b` |
| Forward SNS (image / audio / video) | Local GPU | `IDEA-Research/grounding-dino-tiny`, `lighthouse-emnlp2024/AM-DETR`, Lighthouse CG-DETR (`model_files/best.ckpt`) |
| Fusion EEE | Local GPU | LanguageBind (image, audio, video towers) |

## Hardware Requirements

Local models run on GPU by default (CPU works only for tiny smoke tests and is very slow):

| Mode | GPUs / VRAM | Notes |
| --- | --- | --- |
| **Sequential** (`1_sns.py` → `2_embed.py` → `3_project.py` → `4_datablend.py`) | **1× GPU, ≥ 24 GiB VRAM** | Stages run one at a time. An SNS worker uses ~10–14 GiB (`omni-embed-nemotron-3b` + forward extractors); an EEE worker uses ~14–18 GiB (`omni-embed-nemotron-3b` + LanguageBind towers). |
| **Parallel** (`1_2_parallel.py`) | **2× GPUs, ≥ 24 GiB VRAM each** (or **2× 46+ GiB** for packed SNS workers) | Runs SNS and EEE concurrently. On 24 GiB GPUs, set `sns_workers: 1` and `sns_gpus_per_worker: 1.0`. The default `configs/omni_fuse_hybrid.yaml` packs 2 SNS workers onto one 46 GiB GPU (`sns_gpus_per_worker: 0.5`) and reserves a second GPU for EEE. |
| **Optional self-hosted API models** | **+3× 46+ GiB GPUs** | `Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` (`--tensor-parallel-size 2`) + `Nemotron-3.5-Embed-1B-BF16` (1 GPU), isolated from pipeline GPUs via `CUDA_VISIBLE_DEVICES`. |

- **Storage & RAM**: ~35 GB free disk for Hugging Face model caches and ≥ 32 GiB system RAM.
- **Development & test environment**: Developed and tested on NVIDIA A40 GPUs (46 GiB VRAM each, 8-GPU node with 128 CPU cores / 1 TiB RAM) — using 1 A40 for sequential runs, 2 A40s for `1_2_parallel.py`, and 3 additional A40s when testing self-hosted `vllm serve` endpoints.

## Setup

### Local Setup (`uv`)

Run the following from the Curator repository root:

```bash
cd tutorials/multimodal/omni-fuse-data-curation

# 1. Ensure ffmpeg is installed (Ubuntu/Debian example; skip if already on PATH)
sudo apt-get update && sudo apt-get install -y ffmpeg

# 2. Install Python dependencies (includes spaCy en_core_web_sm and the hf CLI)
uv sync
source .venv/bin/activate

# 3. Log in to Hugging Face and configure API credentials
hf auth login
cp .env.example .env
# Edit .env and set NV_BUILD_API_KEY=<key from https://build.nvidia.com>

# 4. Clone LanguageBind (required for the EEE fusion expert)
git clone --depth 1 https://github.com/PKU-YuanGroup/LanguageBind.git third_party/LanguageBind

# 5. Download the Lighthouse CG-DETR checkpoint (required for SNS video forward extraction)
mkdir -p model_files
curl -L -o model_files/best.ckpt \
  https://zenodo.org/records/13960580/files/clip_cg_detr_qvhighlight.ckpt
```

### Docker Setup

The provided `Dockerfile` pre-installs Python 3.12, `ffmpeg`, `LanguageBind` (at `/opt/LanguageBind`), and all dependencies from `pyproject.toml`. Prepare `.env` and `model_files/best.ckpt` in the tutorial directory, then build and run:

```bash
cd tutorials/multimodal/omni-fuse-data-curation
cp .env.example .env  # edit .env to set NV_BUILD_API_KEY
mkdir -p model_files
curl -L -o model_files/best.ckpt \
  https://zenodo.org/records/13960580/files/clip_cg_detr_qvhighlight.ckpt

docker build -t omnifuse-tutorial .
docker run --gpus all --ipc=host --shm-size=16g --rm -it \
  -v "$PWD:/workspace" \
  -v /path/to/data:/data \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  omnifuse-tutorial
```

Inside the container, run `hf auth login` once if your host `~/.cache/huggingface` is not already authenticated.

### Optional: Self-Hosting the API Models

Instead of `build.nvidia.com`, you can serve the describer and embedder locally with any OpenAI-compatible server (such as vLLM or NIM). Pin the servers to separate GPUs from the pipeline via `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=2,3 vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
  --served-model-name nvidia/nemotron-3-nano-omni-30b-a3b-reasoning \
  --trust-remote-code --tensor-parallel-size 2 --max-model-len 32768 --port 8000

CUDA_VISIBLE_DEVICES=4 vllm serve nvidia/Nemotron-3.5-Embed-1B-BF16 \
  --served-model-name nvidia/nemotron-3.5-embed-1b \
  --dtype bfloat16 --max-model-len 32768 --port 8001
```

Then set `NVIDIA_API_BASE_URL=http://localhost:8000/v1` and `NVIDIA_EMBEDDING_BASE_URL=http://localhost:8001/v1` in `.env` (no API key is required for non-NVIDIA URLs unless your server enforces one).

## Data Preparation

### Data Layout

This tutorial is bring-your-own data. Supported raw modalities are `text`, `image`, `audio`, and `video`. Each pool directory contains raw files, text annotations, and a `pair_mapping.jsonl` manifest:

```text
my_pool/
  raw/
  annotations/
  pair_mapping.jsonl
```

Each line in `pair_mapping.jsonl` must specify `data_path` plus either `annotation_path` or inline `annotation` text (any extra keys are preserved as record metadata):

```json
{"id": "sample-1", "data_path": "raw/sample.jpg", "annotation_path": "annotations/sample.txt"}
{"id": "sample-2", "data_path": "raw/sample.wav", "annotation": "A person speaks over background music."}
```

### Reference Datasets

The Omni-Fuse paper and tutorial runs use five public datasets across the four modalities. Each dataset must be formatted into the [Data Layout](#data-layout) above (`raw/`, `annotations/`, and `pair_mapping.jsonl`, with ALFRED trajectory frames encoded as `.mp4` videos) before pointing `configs/omni_fuse_hybrid.yaml` at it:

- **Image — TextCaps**: [lmms-lab-encoder/TextCaps](https://huggingface.co/datasets/lmms-lab-encoder/TextCaps)
- **Audio — AudioCaps**: [OpenSound/AudioCaps](https://huggingface.co/datasets/OpenSound/AudioCaps)
- **Video — ALFRED**: [KoeYe/ALFRED](https://huggingface.co/datasets/KoeYe/ALFRED) (trajectory frames; official full release at [askforalfred/alfred](https://github.com/askforalfred/alfred))
- **Text (code) — CoNaLa**: [neulab/conala](https://huggingface.co/datasets/neulab/conala)
- **Text (QA) — TriviaQA**: [mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa)

## Running the Pipeline

Update `data_pools` in `configs/omni_fuse_hybrid.yaml` to point to your dataset directories (the shipped config uses small smoke-test defaults; see [Configuration Reference](#configuration-reference) at the end for all tunable settings).

### End-to-End

Run all steps sequentially (prefix `python` with `uv run` if your virtualenv is not activated):

```bash
CONFIG=configs/omni_fuse_hybrid.yaml
(
  for step in 0_validate_inputs 1_sns 2_embed 3_project 4_datablend; do
    python "${step}.py" --config "${CONFIG}" || exit "$?"
  done
)
```

For multi-GPU runs, replace `1_sns` and `2_embed` with `1_2_parallel`:

```bash
CONFIG=configs/omni_fuse_hybrid.yaml
(
  for step in 0_validate_inputs 1_2_parallel 3_project 4_datablend; do
    python "${step}.py" --config "${CONFIG}" || exit "$?"
  done
)
```

### Step-by-Step Reference

- **Step 0 — Validate (`0_validate_inputs.py`)**: Checks data manifests, API key availability, LanguageBind checkout, and the CG-DETR checkpoint path, and prints a resolved summary.
- **Step 1 — SNS (`1_sns.py`)**: Runs forward and/or backward SNS (both starting from the original pair in `bidirectional` mode). Audio descriptions use inline `input_audio` payloads and automatically fall back to an opening preview if the endpoint rejects the full file size. Writes `sns/manifest.jsonl` and `sns/records.jsonl`.
- **Step 2 — EEE (`2_embed.py`)**: Computes `interleaved`, `raw`, and `annotation` `.npy` embeddings per expert under `embeddings/`, plus `metadata.json`, `errors.json`, and `records.jsonl`.
- **Parallel Steps 1–2 (`1_2_parallel.py`)**: Streams record shards from SNS workers directly into EEE workers while keeping each worker's model stack resident, then merges shard outputs into the same canonical `sns/` and `embeddings/` layout for Step 3 (plus `parallelism/summary.json`). With the default two-A40 profile:

  ```text
  GPU 0: SNS worker A: shard 0 -> shard 2 -> shard 4 -> ...
         SNS worker B: shard 1 -> shard 3 -> shard 5 -> ...
  GPU 1: EEE:                    shard 0 -> shard 1 -> ...
  ```

- **Step 3 — Projection (`3_project.py`)**: Trains the projection MLP over all merged expert embeddings (global barrier) and writes `model.json`, `model.pt`, `loss_history.json`, `metrics.json`, `projected_embeddings.npy`, `annotation_embeddings.npy`, and `records.jsonl`.
- **Step 4 — Datablend (`4_datablend.py`)**: Embeds `datablend.query` with the projection's anchor expert (`text-based`, or the first configured expert), ranks projected records by cosine similarity, and writes `datablend_ranked.jsonl` and `datablend_topk.jsonl` (text rows include the SNS-refined `raw_text` alongside the original `raw_path`).

## Output Layout

```text
outputs/<experiment_id>/
  config.resolved.json
  sns/
    manifest.jsonl
    records.jsonl
    media/
    shards/
  embeddings/
    metadata.json
    errors.json
    records.jsonl
    *_interleaved.npy
    *_raw.npy
    *_annotation.npy
    shards/
  parallelism/
    summary.json
  projection/
    model.json
    model.pt
    loss_history.json
    metrics.json
    projected_embeddings.npy
    annotation_embeddings.npy
    records.jsonl
  datablend/
    datablend_ranked.jsonl
    datablend_topk.jsonl
```

## Configuration Reference

`configs/omni_fuse_hybrid.yaml` ships with tiny smoke-test values (`n_samples: 1`, `num_epochs: 1`, `batch_size: 4`, `top_k: 4`) so you can validate the pipeline quickly. Update the pool paths first, then tune the sections below for full runs (all relative paths in the config resolve relative to the YAML file's directory):

- **Top-level (`experiment_id`, `output_dir`)**: Artifacts are written to `<output_dir>/<experiment_id>/`. Change `experiment_id` to avoid overwriting earlier runs.
- **`data_pools`**: List of input pools. Each entry takes:
  - `name`, `modality` (`text`, `image`, `audio`, or `video`), `root_dir`, and `mapping_file` (default `pair_mapping.jsonl`).
  - `n_samples`: cap records loaded per pool (omit or set `null` to load the full pool; start small while testing).
  - `shuffle` (default `false`): deterministically shuffle before applying `n_samples`.
  - `max_file_size_mb` (default `null`): skip raw files exceeding this size in MiB (useful to stay within inline API payload limits).
- **`sns`**: Controls Symmetric Nucleus Subsampling:
  - `enabled` (default `true`) and `direction` (`forward`, `backward`, or `bidirectional`).
  - `mi_ratio` (config: `0.75`, dataclass default: `0.95`) and `mi_eps` (`0.05`): mutual-information gate `sim(candidate) >= mi_ratio * max(sim(original), mi_eps)`. Lower `mi_ratio` accepts more aggressive trims; higher keeps originals unless similarity stays close.
  - `tau_forward_text`, `tau_forward_image`, `tau_forward_audio`, `tau_forward_video`, `tau_backward`: per-modality relevance thresholds. Higher values prune more aggressively; lower values retain more content.
  - `max_video_segments` (`5`), `max_audio_segments` (`5`), `min_segment_duration` (`2.0` s), `bbox_padding_px` (`0`): media extraction limits.
  - `use_ann_components` (`true`): split annotations into spaCy noun/verb phrases for forward extraction (`false` uses the whole annotation).
  - `embedding_batch_size` (`16`): batch size for local Omni-Embed text encoding.
  - `require_forward_models` (`true`) and `continue_on_error` (`true`): set `continue_on_error: false` for fail-fast debugging; otherwise per-record failures keep the original pair and log `status: "error"` to `sns/manifest.jsonl`.
  - Model overrides: `nvidia_model`, `grounding_dino_model_id`, `amdetr_repo_id`, `cg_detr_checkpoint`, `sns_output_dir`.
- **`eee`**: Controls the Expert Embedding Engine:
  - `experts` (default `["text-based", "fusion", "e2e"]`): subset of experts to run. Dropping `text-based` skips API calls; dropping `fusion` removes the LanguageBind dependency.
  - `embedding_dim` (`2048`): shared per-expert output dimension (padded/truncated and L2-normalized).
  - `continue_on_error` (`true`): on per-record failure, writes a zero vector and records details in `embeddings/errors.json` (`false` fails fast).
  - API & model overrides: `nvidia_api_base_url`, `nvidia_embedding_base_url`, `nvidia_api_key` (prefer `.env` over putting secrets in YAML), `nvidia_{image,audio,video}_describer_model`, `nvidia_embedding_model`, `nvidia_multimodal_model`.
- **`parallelism`** (used only by `1_2_parallel.py`):
  - `enabled` (`true` in config; required for `1_2_parallel.py`), `records_per_shard` (`25`; smaller shards start SNS→EEE overlap sooner with slightly more merge overhead).
  - `sns_workers`, `eee_workers`, `sns_gpus_per_worker`, `eee_gpus_per_worker`: match to your GPU count and VRAM (see [Hardware Requirements](#hardware-requirements)).
- **`projection`**: Controls the projection MLP:
  - `device` (`auto`, `cuda`, `cpu`) and `num_gpus` (`1`; set `0` for CPU, or `> 1` for PyTorch `DataParallel`).
  - Training hyperparameters: smoke-test config uses `num_epochs: 1`, `batch_size: 4`, `hidden_layer_size: 256`; for full datasets, increase toward the dataclass defaults (`num_epochs: 100`, `batch_size: 128`, `hidden_layer_size: 512`, `num_layers: 3`, `dropout: 0.1`, `learning_rate: 1e-3`).
  - Loss weights & eval: `contrastive_loss_weight` (`0.99`), `bias_loss_weight` (`0.01`), `scale_loss_weight` (`0.0001`), `contrastive_temperature` (`0.07`), `eval_recall_k` (`10`).
  - Logging & checkpoints: `verbose` (`false`), `log_every_n_epochs` (`100`), `save_weights_path` (defaults to `<run_dir>/projection/model.pt`).
- **`datablend`**: Controls final ranking and export:
  - `query` (required): natural-language description of the target blend (e.g., `"natural, real-world scenes containing objects, landscapes, subjects, or people"`).
  - `top_k` or `blend_fraction`: select either the top *K* rows (`top_k` takes precedence) or a fraction of the pool (`0 < blend_fraction <= 1`); if neither is set, all ranked rows are exported.
  - `include_metadata` (`true`): include original pool metadata columns in output JSONL rows.
- **`runtime`**: `device` (`auto`, `cuda`, `cpu`, or a specific CUDA device like `cuda:0`) for local SNS/EEE models.
