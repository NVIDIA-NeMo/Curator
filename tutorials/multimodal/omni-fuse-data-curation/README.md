# Omni-Fuse Data Curation

Omni-Fuse (see paper [here](https://arxiv.org/pdf/2605.01163v1)) curates paired multimodal datasets by improving pair alignment and
then ranking the resulting records for a target data blend. This tutorial uses
NeMo Curator task/stage abstractions to implement the curation pipeline:

1. Validate paired data manifests and model assets.
2. Apply Symmetric Nucleus Subsampling (SNS).
3. Run the Expert Embedding Engine (EEE).
4. Train/apply the projection network.
5. Export a query-ranked datablend.

The tutorial is API-first hybrid. It uses NVIDIA API models where hosted
endpoints preserve the intended Omni-Fuse role, and local models where the
current implementation needs local model execution.

## Setup

Install the tutorial dependencies from the tutorial directory:

```bash
cd tutorials/multimodal/omni-fuse-data-curation/
uv sync --extra dev
```

You will need to do the following before you're able to run the tutorial:
- Ensure `ffmpeg` is installed and added to `PATH`.
- Log in to Hugging Face using `hf auth login`
- Copy `.env.example` to `.env` in this tutorial directory.
- Create an API key at `build.nvidia.com` and set the `NV_BUILD_API_KEY` variable in the `.env` file.
- Clone the [LanguageBind](https://github.com/pku-yuangroup/languagebind) repository. Either clone it to `third_party/` or set the `LANGUAGEBIND_ROOT` variable in the `.env` file.
- Download pre-trained weights for CG-DETR model from [Lighthouse](https://github.com/line/lighthouse#pre-trained-weights) and save it to `model_files/best.ckpt` We use `cg_detr/qvhighlight/clip/best.ckpt`.
- We recommend using GPUs as we run several local models
- Set the paths to the datasets you want to use in `configs/omni_fuse_hybrid.yaml` and change other settings as you see fit.

### Data Layout

This tutorial is bring-your-own data. Each pool contains raw files, text
annotations, and a `pair_mapping.jsonl` file:

```text
my_pool/
  raw/
  annotations/
  pair_mapping.jsonl
```

Each mapping row must contain a raw path and either an annotation path or inline
annotation text:

```json
{"id": "sample-1", "data_path": "raw/sample.jpg", "annotation_path": "annotations/sample.txt"}
{"id": "sample-2", "data_path": "raw/sample.wav", "annotation": "A person speaks over background music."}
```

Supported raw modalities are `text`, `image`, `audio`, and `video`. Configure
each pool in `configs/omni_fuse_hybrid.yaml`:

```yaml
data_pools:
  - name: "image_caption_pool"
    modality: "image"
    root_dir: "/path/to/image_pool"
    mapping_file: "pair_mapping.jsonl"
    n_samples: 1
```

Use small `n_samples` values while validating the tutorial.

### Model Backends

The default config uses `sns.backend: hybrid` and `eee.backend: hybrid`.
If you wish to use strictly api-based or local models, you can change these to `api` or `local`. However, this won't work out of the box and you will have to modify code to fit your requirements.

API-backed components:

- Modality descriptions for backward SNS and the text-based EEE expert:
  - `nvidia/nemotron-nano-12b-v2-vl` for text, image, and video.
  - `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` for audio.
- Text embeddings:
  - `nvidia/llama-nemotron-embed-1b-v2`.

Local components:

- SNS multimodal similarity and MI gating:
  - `nvidia/omni-embed-nemotron-3b`.
- SNS image forward extraction:
  - `IDEA-Research/grounding-dino-tiny`.
- SNS audio forward extraction:
  - `lighthouse-emnlp2024/AM-DETR`.
- SNS video forward extraction:
  - CG-DETR from Lighthouse with `model_files/best.ckpt`.
- EEE fusion expert:
  - LanguageBind.
- EEE end-to-end expert:
  - `nvidia/omni-embed-nemotron-3b`.



## Step 0: Validate Inputs

```bash
python 0_validate_inputs.py --config configs/omni_fuse_hybrid.yaml
```

This checks the data manifests, API key availability, LanguageBind checkout,
and CG-DETR checkpoint path.

## Step 1: Symmetric Nucleus Subsampling

```bash
python 1_sns.py --config configs/omni_fuse_hybrid.yaml
```

SNS writes:

```text
outputs/<experiment_id>/sns/manifest.jsonl
outputs/<experiment_id>/sns/records.jsonl
```

Hybrid SNS follows the EmbedSim execution path: backward extraction keeps the
configured NVIDIA API describers, while text similarity, media similarity, and
MI gating use local Omni-Embed. Text lists are embedded in real batches
(`sns.embedding_batch_size`, default 16), and repeated media embeddings are
cached within each worker. Forward extraction for image/audio/video stays local
with Grounding-DINO/AM-DETR/CG-DETR.

Gemma audio descriptions use the same inline `input_audio` payload as EmbedSim.
If the endpoint rejects the payload size, the tutorial automatically falls back
to an NVCF asset upload and finally to an inline preview. Bidirectional forward
and backward extraction both start from the original pair, matching EmbedSim.
With `sns.continue_on_error: true` (the default), a per-record SNS failure keeps
the original pair, writes `status: "error"` and error details to the SNS
manifest, and continues with the remaining records. Set it to `false` for
fail-fast behavior.

## Step 2: Expert Embeddings

```bash
python 2_embed.py --config configs/omni_fuse_hybrid.yaml
```

With `eee.continue_on_error: true` (the default), a per-record expert failure
uses a zero-vector placeholder and continues. Details are written to
`outputs/<experiment_id>/embeddings/errors.json`; set the option to `false` for
fail-fast behavior.

EEE writes interleaved, raw, and annotation embeddings for each expert:

```text
outputs/<experiment_id>/embeddings/text_based_*.npy
outputs/<experiment_id>/embeddings/fusion_*.npy
outputs/<experiment_id>/embeddings/e2e_*.npy
outputs/<experiment_id>/embeddings/metadata.json
outputs/<experiment_id>/embeddings/records.jsonl
```

The text-based expert uses NVIDIA API descriptions and text embeddings. The
fusion and e2e experts use LanguageBind and Omni-Embed locally.

## Parallel Steps 1-2: Multi-GPU SNS and EEE

The numbered single-stage scripts remain useful for learning and debugging. For
larger datasets, `1_2_parallel.py` combines SNS and EEE into one streaming NeMo
Curator pipeline. It divides the ordered records into small tasks and keeps each
worker's model stack resident. The included two-A40 profile packs two
half-GPU SNS workers onto one 46 GiB GPU and reserves the other GPU for EEE:

```text
GPU 0: SNS worker A: shard 0 -> shard 2 -> shard 4 -> ...
       SNS worker B: shard 1 -> shard 3 -> shard 5 -> ...
GPU 1: EEE:                    shard 0 -> shard 1 -> ...
```

As soon as SNS completes one shard, Curator can send it to EEE while SNS begins
the next shard. Shard outputs use distinct paths, so concurrent workers do not
overwrite manifests or embedding arrays. After the streaming pipeline finishes,
the driver merges shards in original record order and writes the same canonical
`sns/` and `embeddings/` files consumed by Step 3.

Enable and tune the parallel path in the config:

```yaml
parallelism:
  enabled: true
  records_per_shard: 25
  sns_workers: 2
  eee_workers: 1
  sns_gpus_per_worker: 0.5
  eee_gpus_per_worker: 1.0
```

The half-GPU profile is intended for the tutorial machine's 46 GiB A40s; each
SNS worker loads its own model stack. On smaller GPUs, use one SNS worker with
`sns_gpus_per_worker: 1.0`. Smaller shards begin overlap sooner but create more
scheduling and merge overhead; 25 records is the tutorial default.

Run only the combined parallel steps:

```bash
uv run python 1_2_parallel.py --config configs/omni_fuse_hybrid.yaml
```

The combined stage timing and worker/shard configuration are written to:

```text
outputs/<experiment_id>/parallelism/summary.json
```

The projection step remains a global barrier because it trains over the merged
embedding set. Continue with `3_project.py` and `4_datablend.py` normally.

## Step 3: Projection

```bash
python 3_project.py --config configs/omni_fuse_hybrid.yaml
```

The projection stage trains a small MLP over concatenated expert embeddings
using contrastive, cluster-bias, and scale-bias losses. The stage reserves
`projection.num_gpus` GPUs, and PyTorch uses data parallelism when more than
one GPU is assigned. Use a batch size large enough to keep each GPU busy:

```yaml
projection:
  backend: "torch"
  device: "cuda"
  num_gpus: 2
  batch_size: 128
  verbose: true
  log_every_n_epochs: 100
```

`device: "auto"` uses CUDA when the stage has assigned GPUs. Set
`num_gpus: 0` to run the projection stage on CPU.

The stage writes:

```text
outputs/<experiment_id>/projection/model.json
outputs/<experiment_id>/projection/model.pt
outputs/<experiment_id>/projection/loss_history.json
outputs/<experiment_id>/projection/metrics.json
outputs/<experiment_id>/projection/projected_embeddings.npy
outputs/<experiment_id>/projection/annotation_embeddings.npy
```

## Step 4: Datablend Ranking

```bash
python 4_datablend.py --config configs/omni_fuse_hybrid.yaml
```

The datablend stage embeds the query through the text-based expert and ranks
projected records by cosine similarity:

```text
outputs/<experiment_id>/datablend/datablend_ranked.jsonl
outputs/<experiment_id>/datablend/datablend_topk.jsonl
```

## End-to-End Script

Run every step in order:

```bash
CONFIG=configs/omni_fuse_hybrid.yaml bash e2e.sh
```

Set `PYTHON_BIN` if you want to use a specific interpreter:

```bash
PYTHON_BIN="uv run python" CONFIG=configs/omni_fuse_hybrid.yaml bash e2e.sh
```

Run the multi-GPU variant, which replaces the separate SNS and EEE commands
with `1_2_parallel.py`:

```bash
PYTHON_BIN="uv run python" CONFIG=configs/omni_fuse_hybrid.yaml bash e2e_parallel.sh
```

For the five-dataset benchmark used in this tutorial, the matching two-GPU,
4,000-epoch config is `configs/omni_fuse_1000_4000ep_parallel.yaml`.

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
    records.jsonl
    *_interleaved.npy
    *_raw.npy
    *_annotation.npy
    shards/
  parallelism/
    summary.json
  projection/
    model.json
    loss_history.json
    metrics.json
    projected_embeddings.npy
    annotation_embeddings.npy
  datablend/
    datablend_ranked.jsonl
    datablend_topk.jsonl
```
