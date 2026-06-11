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

Install the tutorial dependencies from the Curator repository root:

```bash
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
  - `google/gemma-3n-e4b-it` for audio.
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

In hybrid mode, backward extraction uses API descriptions and API text
embeddings. Forward extraction for image/audio/video uses local
Grounding-DINO/AM-DETR/CG-DETR and local Omni-Embed MI gating.

## Step 2: Expert Embeddings

```bash
python 2_embed.py --config configs/omni_fuse_hybrid.yaml
```

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

## Step 3: Projection

```bash
python 3_project.py --config configs/omni_fuse_hybrid.yaml
```

The projection stage trains a small MLP over concatenated expert embeddings
using contrastive, cluster-bias, and scale-bias losses. It writes:

```text
outputs/<experiment_id>/projection/model.json
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

## Output Layout

```text
outputs/<experiment_id>/
  config.resolved.json
  sns/
    manifest.jsonl
    records.jsonl
    media/
  embeddings/
    metadata.json
    records.jsonl
    *_interleaved.npy
    *_raw.npy
    *_annotation.npy
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
