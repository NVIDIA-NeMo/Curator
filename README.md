<div align="center">

  <a href="https://github.com/NVIDIA-NeMo/Curator/blob/main/LICENSE">![https://pypi.org/project/nemo-curator](https://img.shields.io/github/license/NVIDIA-NeMo/Curator)</a>
  <a href="https://codecov.io/github/NVIDIA-NeMo/Curator">![codecov](https://codecov.io/github/NVIDIA-NeMo/Curator/graph/badge.svg)</a>
  <a href="https://pypi.org/project/nemo-curator/">![https://pypi.org/project/nemo-curator/](https://img.shields.io/pypi/pyversions/nemo-curator.svg)</a>
  <a href="https://github.com/NVIDIA-NeMo/Curator/graphs/contributors">![NVIDIA-NeMo/Curator](https://img.shields.io/github/contributors/NVIDIA-NeMo/Curator)</a>
  <a href="https://github.com/NVIDIA-NeMo/Curator/releases">![https://github.com/NVIDIA-NeMo/Curator/releases](https://img.shields.io/github/release/NVIDIA-NeMo/Curator)</a>
  <a href="https://pypi.org/project/nemo-curator/">![https://github.com/Naereen/badges/](https://badgen.net/badge/open%20source/❤/blue?icon=github)</a>

</div>

# NVIDIA NeMo Curator

**NeMo Curator helps ML engineers and data teams build repeatable, GPU-accelerated pipelines that load, filter, deduplicate, and transform large text, image, video, and audio datasets for AI training.** Run the same pipeline on a laptop or across a multi-node Ray cluster.

> *Part of the [NVIDIA NeMo](https://www.nvidia.com/en-us/ai-data-science/products/nemo/) software suite for managing the AI agent lifecycle.*

## Updates

- **2026-04** — NeMo Curator 26.04: Cosmos-Xenna 0.2.0 upgrade, simplified `Resources` API, Ray runtime upgrade. See the [release notes](https://docs.nvidia.com/nemo/curator/latest/about/release-notes).
- **2026-02** — NeMo Curator 26.02: Ray-based pipeline architecture for all modalities — text, image, video, and audio.

---

## What You Can Build

| Modality | Common Operations | Guide |
|----------|-------------------|-------|
| **Text** | Deduplication, classification, quality filtering, language detection | [Text Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/text) |
| **Image** | Aesthetic filtering, NSFW detection, embedding generation, deduplication | [Image Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/image) |
| **Video** | Scene detection, clip extraction, motion filtering, deduplication | [Video Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/video) |
| **Audio** | ASR transcription, quality assessment, WER filtering | [Audio Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/audio) |

### Use NeMo Curator when…

- You need **repeatable curation pipelines** — not one-off notebooks or ad-hoc scripts.
- You need **GPU and distributed execution** for data-heavy stages (dedupe, classification, embedding, inference).
- You need **modality-aware building blocks** for text, image, video, or audio.
- You want **recipes that map to NVIDIA training workflows** like Nemotron and Nemotron-CC.

---

## Quick Start

Three paths, depending on what you're trying to do. Each path is self-contained.

NeMo Curator uses [`uv`](https://docs.astral.sh/uv/) for installation. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Path A — CPU smoke test (no GPU required)

Verify your environment and run a tiny text pipeline.

```bash
uv venv && source .venv/bin/activate
uv pip install "nemo-curator[text_cpu]"
python -c "import nemo_curator; print(nemo_curator.__version__)"
```

### Path B — GPU text pipeline (CUDA 12, supported Linux)

The bundled quickstart starts Ray, downloads a Hugging Face model, and runs a sentiment classification pipeline on GPU.

**Prerequisites:** CUDA 12 toolkit, NVIDIA driver supporting CUDA 12, Linux x86_64, ~16 GB GPU memory, network access to Hugging Face.

```bash
uv venv && source .venv/bin/activate
uv pip install "nemo-curator[text_cuda12]"
python tutorials/quickstart.py
```

### Path C — Docker (recommended for video and audio)

Video and audio pipelines depend on system codec libraries; the published container ships them preconfigured.

- Container: [nemo-curator on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator)
- Setup instructions: [Installation Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/installation)

**Full setup for all paths:** [Installation Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/installation) • [Tutorials](tutorials/)

---

## Why NeMo Curator

### Proven at scale: Nemotron

NeMo Curator powers the data pipelines behind [NVIDIA Nemotron](https://developer.nvidia.com/nemotron) models. The [Nemotron-4 pre-training dataset](https://arxiv.org/abs/2402.16819) was curated using NeMo Curator's text pipeline across 8+ trillion tokens of multilingual web data — quality filtering, deduplication, and domain classification at scale.

The [Nemotron-CC curation pipeline](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/src/nemotron/recipes/data/curation/nemotron-cc) uses NeMo Curator end-to-end — from Common Crawl extraction through language ID, exact/fuzzy/substring deduplication, ensemble quality classification, and LLM-based synthetic data generation — to reproduce the [Nemotron-CC datasets](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2). The SDG stage is available as an [in-repo tutorial](tutorials/synthetic/nemotron_cc/).

### Benchmark results

NeMo Curator leverages NVIDIA RAPIDS™ (cuDF, cuML, cuGraph) with Ray to scale across multi-node, multi-GPU environments. Numbers below are from the throughput study published in the [scaling docs](https://docs.nvidia.com/nemo/curator/latest/about/concepts/scaling/throughput); see the source for full methodology, software versions, and baselines.

| Metric | Workload | Hardware | Baseline | NeMo Curator |
|--------|----------|---------|----------|--------------|
| Fuzzy dedupe speedup | RedPajama v2 subset | 3× H100 80 GB nodes | CPU-based alternative | 10.7 h → 0.65 h (**~16×**) |
| Total cost of ownership | RedPajama v2 subset | 3× H100 80 GB nodes | CPU-based alternative | $315 → $190 (**~40% lower**) |
| GPU scaling (1→4 nodes) | RedPajama v2 subset | 1, 2, 4 × H100 80 GB nodes | Single-node run | 2.05 h → 1.01 h → 0.50 h |

> Token counts and exact subset sizes vary across published panels; treat per-panel labels in the source as authoritative.

### Quality improvements

In ablation studies using a 357M-parameter GPT model trained on curated Common Crawl data, NeMo Curator's pipeline stages — text cleaning, deduplication, and quality filtering — produced progressive improvements in zero-shot downstream task accuracy.

<p align="center">
  <img src="https://raw.githubusercontent.com/NVIDIA-NeMo/Curator/main/fern/assets/images/ablation.png" alt="Model accuracy improvements across curation pipeline stages" width="700"/>
</p>

---

## How It Works

NeMo Curator pipelines are composed of **stages**, each handling a discrete curation task (load, filter, dedupe, classify, transform, write). Stages stream **tasks** through the pipeline and are executed by a pluggable **executor**.

- **Stages** declare their own resource requirements (CPU cores, GPU memory, replicas).
- **Pipelines** chain stages; the executor auto-scales replicas per stage to match throughput across the chain.
- **Streaming execution** overlaps CPU and GPU work so all stages run concurrently — typical pipelines keep GPU workers >99% busy after warm-up.
- **Executors** run the pipeline: the [XennaExecutor](https://docs.nvidia.com/nemo/curator/latest/api/reference/api-reference/executors/xenna-executor) (Cosmos-Xenna) is the production default, with experimental Ray-based backends also available — same pipeline definition, different runtime.
- **Modality plug-ins** (text, image, video, audio) provide ready-made stages on top of the same core abstractions.

See the [scaling concepts](https://docs.nvidia.com/nemo/curator/latest/about/concepts/scaling) for an end-to-end walkthrough.

<p align="center">
  <img src="https://raw.githubusercontent.com/NVIDIA-NeMo/Curator/main/fern/assets/images/architecture-diagram.png" alt="NeMo Curator architecture diagram showing modular pipeline stages" width="700"/>
</p>

---

## Recipes and Tutorials

| Recipe | What it does |
|--------|--------------|
| [Nemotron-CC end-to-end](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/src/nemotron/recipes/data/curation/nemotron-cc) | Reproduces the Nemotron-CC dataset from Common Crawl |
| [Nemotron-CC SDG](tutorials/synthetic/nemotron_cc/) | Synthetic data generation stage as an in-repo tutorial |
| [Text tutorials](tutorials/text/) | Loading, filtering, dedupe, classification |
| [Image tutorials](tutorials/image/) | WebDataset loading, CLIP embeddings, aesthetic/NSFW filtering |
| [Video tutorials](tutorials/video/) | Scene detection, clipping, motion filtering, dedupe |
| [Audio tutorials](tutorials/audio/) | ASR transcription, WER filtering, multimodal handoff |

---

## Installation and Deployment

| Resource | Link |
|----------|------|
| Installation guide (CPU, GPU, Docker, source) | [docs.nvidia.com/nemo/curator/latest/get-started/installation](https://docs.nvidia.com/nemo/curator/latest/get-started/installation) |
| Container image | [nemo-curator on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator) |
| Infrastructure (Slurm, Kubernetes, multi-node) | [Infrastructure docs](https://docs.nvidia.com/nemo/curator/latest/reference/infra) |
| API reference | [API docs](https://docs.nvidia.com/nemo/curator/latest/api/reference/api-reference) |
| Concepts | [Concepts](https://docs.nvidia.com/nemo/curator/latest/about/concepts) |

Supported Python and dependency versions are defined in [`pyproject.toml`](pyproject.toml); the Python versions are also shown on the PyPI badge above. The README does not duplicate them to avoid drift.

---

## Roadmap

Shipped changes are documented in the [release notes](https://docs.nvidia.com/nemo/curator/latest/about/release-notes). Planned work and feature direction are tracked in [GitHub Issues](https://github.com/NVIDIA-NeMo/Curator/issues) and [Discussions](https://github.com/NVIDIA-NeMo/Curator/discussions).

---

## Getting Help

Pick the channel that matches your need — these are community channels staffed on a best-effort basis; there is no SLA.

| You want to… | Channel | Typical response |
|--------------|---------|------------------|
| Ask a usage question, share a recipe, get design feedback | [GitHub Discussions](https://github.com/NVIDIA-NeMo/Curator/discussions) | A few business days |
| Report a reproducible bug or regression | [GitHub Issues](https://github.com/NVIDIA-NeMo/Curator/issues) — use the bug template | A few business days for triage |
| Request a feature or new modality capability | [GitHub Issues](https://github.com/NVIDIA-NeMo/Curator/issues) — use the feature-request template | Triaged into a milestone when accepted |
| Read the docs | [docs.nvidia.com/nemo/curator/latest](https://docs.nvidia.com/nemo/curator/latest) | — |

Please do not use Issues for "how do I…" questions — they belong in Discussions so they remain searchable for other users.

---

## Contributing

Contributions are welcome — bug fixes, docs, tutorials, new stages, and tests. **See [CONTRIBUTING.md](CONTRIBUTING.md)** for the full guide, including how to pick a good first issue, set up your environment, and open a signed-off PR. All participants are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

NeMo Curator is released under the **Apache License 2.0**. See [LICENSE](LICENSE) for the full text.

## Citation

If you use NeMo Curator in your research, please cite:

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
