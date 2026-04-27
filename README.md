<div align="center">

  <a href="https://github.com/NVIDIA-NeMo/Curator/blob/main/LICENSE">![https://pypi.org/project/nemo-curator](https://img.shields.io/github/license/NVIDIA-NeMo/Curator)</a>
  <a href="https://codecov.io/github/NVIDIA-NeMo/Curator">![codecov](https://codecov.io/github/NVIDIA-NeMo/Curator/graph/badge.svg)</a>
  <a href="https://pypi.org/project/nemo-curator/">![https://pypi.org/project/nemo-curator/](https://img.shields.io/pypi/pyversions/nemo-curator.svg)</a>
  <a href="https://github.com/NVIDIA-NeMo/Curator/graphs/contributors">![NVIDIA-NeMo/Curator](https://img.shields.io/github/contributors/NVIDIA-NeMo/Curator)</a>
  <a href="https://github.com/NVIDIA-NeMo/Curator/releases">![https://github.com/NVIDIA-NeMo/Curator/releases](https://img.shields.io/github/release/NVIDIA-NeMo/Curator)</a>
  <a href="https://pypi.org/project/nemo-curator/">![https://github.com/Naereen/badges/](https://badgen.net/badge/open%20source/❤/blue?icon=github)</a>

</div>

> **`dev` branch** — This branch is a shared workspace where all teams contribute with immediate availability after push.
>
> **How to contribute:**
> 1. Open (or have) a PR to `main` — it does not need to be merged
> 2. Cherry-pick the commit(s) into `dev`:
>    ```bash
>    git checkout dev
>    git cherry-pick <commit-sha>
>    git push origin dev
>    ```
> 3. Reference the PR number in your commit message (e.g., `feat: add X (#1234)`)
>
> **When there is no PR to main**, mark your commit with `#NO_PR`. This applies when:
> - Your changes are to files or methods that only exist on `dev`
> - A cherry-pick from main produced complex merge conflicts requiring a separate resolution

# NVIDIA NeMo Curator

**GPU-accelerated data curation for training better AI models, faster.** Scale from laptop to multi-node clusters with modular pipelines for text, images, video, and audio.

> *Part of the [NVIDIA NeMo](https://www.nvidia.com/en-us/ai-data-science/products/nemo/) software suite for managing the AI agent lifecycle.*

## Updates

- 2026-04: NeMo Curator 26.04 released with Cosmos-Xenna 0.2.0 upgrade, simplified `Resources` API, and Ray 2.54. See the [release notes](https://docs.nvidia.com/nemo/curator/latest/about/release-notes).
- 2026-02: NeMo Curator 26.02 released with Ray-based pipeline architecture for all modalities — text, image, video, and audio.

## What You Can Do

| Modality | Key Capabilities | Get Started |
|----------|-----------------|-------------|
| **Text** | Deduplication • Classification • Quality Filtering • Language Detection | [Text Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/text.html) |
| **Image** | Aesthetic Filtering • NSFW Detection • Embedding Generation • Deduplication | [Image Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/image.html) |
| **Video** | Scene Detection • Clip Extraction • Motion Filtering • Deduplication | [Video Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/video.html) |
| **Audio** | ASR Transcription • Quality Assessment • WER Filtering | [Audio Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/audio.html) |

## Quick Start

```bash
# Install for your modality
uv pip install "nemo-curator[text_cuda12]"

# Run the quickstart example
python tutorials/quickstart.py
```

**Full setup:** [Installation Guide](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html) • [Docker](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator) • [Tutorials](tutorials/)

---

## Architecture

NeMo Curator uses a modular, Ray-based pipeline architecture. Data flows through composable processing stages — each stage handles a discrete curation task (loading, filtering, deduplication, etc.) and can be configured with independent resource requirements.

<p align="center">
  <img src="./fern/assets/images/architecture-diagram.png" alt="NeMo Curator architecture diagram showing modular pipeline stages" width="700"/>
</p>

---

## Features by Modality

### Text Curation

Process and curate high-quality text datasets for large language model (LLM) training with multilingual support.

| Category | Features | Documentation |
|----------|----------|---------------|
| **Data Sources** | Common Crawl • Wikipedia • ArXiv • Custom datasets | [Load Data](https://docs.nvidia.com/nemo/curator/latest/curate-text/load-data/index.html) |
| **Quality Filtering** | 30+ heuristic filters • fastText classification • GPU-accelerated classifiers for domain, quality, safety, and content type | [Quality Assessment](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/quality-assessment/heuristic.html) |
| **Deduplication** | Exact • Fuzzy (MinHash LSH) • Semantic (GPU-accelerated) | [Deduplication](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/index.html) |
| **Processing** | Text cleaning • Language identification | [Content Processing](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/content-processing/text-cleaning.html) |

---

### Image Curation

Curate large-scale image datasets for vision language models (VLMs) and generative AI training.

| Category | Features | Documentation |
|----------|----------|---------------|
| **Data Loading** | WebDataset format • Large-scale image-text pairs | [Load Data](https://docs.nvidia.com/nemo/curator/latest/curate-images/load-data/index.html) |
| **Embeddings** | CLIP embeddings for semantic analysis | [Embeddings](https://docs.nvidia.com/nemo/curator/latest/curate-images/process-data/embeddings/index.html) |
| **Filtering** | Aesthetic quality scoring • NSFW detection | [Filters](https://docs.nvidia.com/nemo/curator/latest/curate-images/process-data/filters/index.html) |

---

### Video Curation

Process large-scale video corpora with distributed, GPU-accelerated pipelines for world foundation models (WFMs).

| Category | Features | Documentation |
|----------|----------|---------------|
| **Data Loading** | Local paths • S3-compatible storage • HTTP(S) URLs | [Load Data](https://docs.nvidia.com/nemo/curator/latest/curate-video/load-data/index.html) |
| **Clipping** | Fixed-stride splitting • Scene-change detection (TransNetV2) | [Clipping](https://docs.nvidia.com/nemo/curator/latest/curate-video/process-data/clipping.html) |
| **Processing** | GPU H.264 encoding • Frame extraction • Motion filtering • Aesthetic filtering | [Processing](https://docs.nvidia.com/nemo/curator/latest/curate-video/process-data/filtering.html) |
| **Embeddings** | Cosmos-Embed1 for clip-level embeddings | [Embeddings](https://docs.nvidia.com/nemo/curator/latest/curate-video/process-data/embeddings.html) |
| **Deduplication** | K-means clustering • Pairwise similarity for near-duplicates | [Deduplication](https://docs.nvidia.com/nemo/curator/latest/curate-video/process-data/dedup.html) |

---

### Audio Curation

Prepare high-quality speech datasets for automatic speech recognition (ASR) and multimodal AI training.

| Category | Features | Documentation |
|----------|----------|---------------|
| **Data Loading** | Local files • Custom manifests • Public datasets (FLEURS) | [Load Data](https://docs.nvidia.com/nemo/curator/latest/curate-audio/load-data/index.html) |
| **ASR Processing** | NeMo Framework pretrained models • Automatic transcription | [ASR Inference](https://docs.nvidia.com/nemo/curator/latest/curate-audio/process-data/asr-inference/index.html) |
| **Quality Assessment** | Word Error Rate (WER) calculation • Duration analysis • Quality-based filtering | [Quality Assessment](https://docs.nvidia.com/nemo/curator/latest/curate-audio/process-data/quality-assessment/index.html) |
| **Integration** | Text curation workflow integration for multimodal pipelines | [Text Integration](https://docs.nvidia.com/nemo/curator/latest/curate-audio/process-data/text-integration/index.html) |

---

## Why Data Curation?

High-quality training data is the single most important factor in building performant AI models. Raw datasets contain noise, duplicates, low-quality content, and potentially harmful material that degrade model performance and increase training costs.

<p align="center">
  <img src="./fern/assets/images/data-curation-challenges.png" alt="Common data curation challenges: quality, deduplication, filtering, and scale" width="700"/>
</p>

At scale, data curation is a **throughput maximization problem**. A typical pipeline chains stages with very different compute profiles — lightweight CPU tokenization, small GPU classifiers, large GPU inference models — and a naive sequential approach leaves most hardware idle most of the time.

**Example:** Consider a pipeline with language identification (0.5B model, 1 GB VRAM, 2s/sample), tokenization (CPU-only, 1s/sample), and a 5B answer model (10 GB VRAM, 10s/sample) processing 1,000 questions on a single 102 GB GPU:

| Approach | How it works | Total runtime |
|----------|-------------|---------------|
| **Sequential** | Process each sample through all stages, one at a time | ~13,000 seconds |
| **NeMo Curator** | Stream batches, auto-scale replicas per stage, overlap CPU/GPU work | ~1,000 seconds |

NeMo Curator achieves this by streaming data through the pipeline so all stages run concurrently, auto-balancing replicas to match each stage's throughput (2× language ID, 1× tokenizer, 10× answer model), and keeping GPU workers busy over 99% of the time after an initial warm-up period. See the [scaling concepts](https://docs.nvidia.com/nemo/curator/latest/about/concepts/scaling) for details.

---

## Proven at Scale: Nemotron

NeMo Curator powers the data pipelines behind [NVIDIA Nemotron](https://developer.nvidia.com/nemotron) models. For example, the [Nemotron-4 pre-training dataset](https://arxiv.org/abs/2402.16819) was curated using NeMo Curator's text processing pipeline across 8+ trillion tokens of multilingual web data, applying quality filtering, deduplication, and domain classification at scale.

---

## Why NeMo Curator?

### Performance at Scale

NeMo Curator leverages NVIDIA RAPIDS™ libraries such as cuDF, cuML, and cuGraph along with Ray to scale workloads across multi-node, multi-GPU environments.

**Proven Results:**
- **16× faster** fuzzy deduplication on 8 TB RedPajama v2 (1.78 trillion tokens)
- **40% lower** total cost of ownership (TCO) compared to CPU-based alternatives
- **Near-linear scaling** from one to four H100 80 GB nodes (2.05 hrs → 0.50 hrs)

**Real-World Recipe:** The [Nemotron-CC curation pipeline](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/src/nemotron/recipes/data_curation/nemotron-cc) uses NeMo Curator end-to-end — from Common Crawl extraction through language identification, exact/fuzzy/substring deduplication, ensemble quality classification, and LLM-based synthetic data generation — to reproduce the [Nemotron-CC datasets](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2). The SDG stage is also available as an [in-repo tutorial](tutorials/synthetic/nemotron_cc/).

<p align="center">
  <img src="./fern/assets/images/text-benchmarks.png" alt="Performance benchmarks showing 16x speed improvement, 40% cost savings, and near-linear scaling" width="700"/>
</p>

### Quality Improvements

Data curation modules measurably improve model performance. In ablation studies using a 357M-parameter GPT model trained on curated Common Crawl data:

<p align="center">
  <img src="./fern/assets/images/ablation.png" alt="Model accuracy improvements across curation pipeline stages" width="700"/>
</p>

**Results:** Progressive improvements in zero-shot downstream task performance through text cleaning, deduplication, and quality filtering stages.

---

## Learn More

| Resource | Links |
|----------|-------|
| **Documentation** | [Main Docs](https://docs.nvidia.com/nemo/curator/latest/) • [API Reference](https://docs.nvidia.com/nemo/curator/latest/apidocs/index.html) • [Concepts](https://docs.nvidia.com/nemo/curator/latest/about/concepts/index.html) |
| **Tutorials** | [Text](tutorials/text/) • [Image](tutorials/image/) • [Video](tutorials/video/) • [Audio](tutorials/audio/) |
| **Recipes** | [Nemotron-CC: end-to-end web data curation](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/src/nemotron/recipes/data_curation/nemotron-cc) • [SDG tutorial (in-repo)](tutorials/synthetic/nemotron_cc/) |
| **Deployment** | [Installation](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html) • [Infrastructure](https://docs.nvidia.com/nemo/curator/latest/reference/infrastructure/index.html) |
| **Community** | [GitHub Discussions](https://github.com/NVIDIA-NeMo/Curator/discussions) • [Issues](https://github.com/NVIDIA-NeMo/Curator/issues) |

---

## Contribute

We welcome community contributions! Please refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for guidelines.

---

## Citation

If you find NeMo Curator useful in your research, please cite:

```bibtex
@misc{nemo_curator,
  title = {NeMo Curator: GPU-Accelerated Data Curation for Training AI Models},
  author = {NVIDIA},
  year = {2024},
  url = {https://github.com/NVIDIA-NeMo/Curator}
}
```

For the data curation pipeline behind Nemotron models, please also cite:

```bibtex
@article{parmar2024nemotron4,
  title = {Nemotron-4 15B Technical Report},
  author = {Parmar, Jupinder and Satheesh, Shrimai and others},
  journal = {arXiv preprint arXiv:2402.16819},
  year = {2024}
}
```
