<div align="center">

  <a href="https://github.com/NVIDIA-NeMo/Curator/blob/main/LICENSE">![https://pypi.org/project/nemo-curator](https://img.shields.io/github/license/NVIDIA-NeMo/Curator)</a>
  <a href="https://pypi.org/project/nemo-curator/">![https://pypi.org/project/nemo-curator/](https://img.shields.io/pypi/pyversions/nemo-curator.svg)</a>
  <a href="https://github.com/NVIDIA-NeMo/Curator/graphs/contributors">![NVIDIA-NeMo/Curator](https://img.shields.io/github/contributors/NVIDIA-NeMo/Curator)</a>
  <a href="https://github.com/NVIDIA-NeMo/Curator/releases">![https://github.com/NVIDIA-NeMo/Curator/releases](https://img.shields.io/github/release/NVIDIA-NeMo/Curator)</a>
  <a href="https://pypi.org/project/nemo-curator/">![https://github.com/Naereen/badges/](https://badgen.net/badge/open%20source/❤/blue?icon=github)</a>

</div>

# Accelerate Data Processing and Streamline Synthetic Data Generation with NVIDIA NeMo Curator

NeMo Curator is a Python library specifically designed for fast and scalable data processing and curation for generative AI use cases such as foundation language model pretraining, text-to-image model training, domain-adaptive pretraining (DAPT), supervised fine-tuning (SFT) and parameter-efficient fine-tuning (PEFT).

It greatly accelerates data processing and curation by leveraging GPUs with [Dask](https://www.dask.org/) and [RAPIDS](https://developer.nvidia.com/rapids), resulting in significant time savings. The library provides a customizable and modular interface, simplifying pipeline expansion and accelerating model convergence through the preparation of high-quality tokens.

NeMo Curator also provides pre-built pipelines for synthetic data generation for customization and evaluation of generative AI systems. You can use any OpenAI API compatible model and plug it in NeMo Curator's synthetic data generation pipelines to process and curate high-quality synthetic data for various use cases.

## Getting Started

New to NeMo Curator? Start with our quickstart guides for hands-on experience:

- **[Text Curation Quickstart](https://docs.nvidia.com/nemo/curator/latest/get-started/text.html)** - Set up your environment and run your first text curation pipeline in under 30 minutes
- **[Image Curation Quickstart](https://docs.nvidia.com/nemo/curator/latest/get-started/image.html)** - Learn to curate large-scale image-text datasets for generative model training

For production deployments and advanced configurations, see our [Setup & Deployment documentation](https://docs.nvidia.com/nemo/curator/latest/admin/index.html).

---

## Key Features

With NeMo Curator, you can process raw data and curate high-quality data for training and customizing generative AI models such as LLMs, VLMs and WFMs. NeMo Curator provides a collection of scalable data processing modules for text and image curation.

### Text Curation
All of our text pipelines have great multilingual support. With NeMo Curator, you can pick and choose the features you want and build your data curation pipelines. Text curation follows a three-stage workflow: **Load** → **Process** → **Generate**. A typical pipeline starts by downloading raw data from public resources, then applies cleaning and filtering steps, and optionally generates synthetic data for training enhancement.

#### Load Data
- **[Download and Extraction](https://docs.nvidia.com/nemo/curator/latest/curate-text/load-data/index.html)** - Default implementations for Common Crawl, Wikipedia, and ArXiv sources with easy customization for other sources

#### Process Data  
- **Quality Assessment & Filtering**
  - [Heuristic Filtering](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/quality-assessment/heuristic.html) - 30+ heuristic filters for punctuation density, length, and repetition analysis
  - [fastText Classification](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/quality-assessment/classifier.html) - Fast language and quality classification
  - [GPU-Accelerated Classification](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/quality-assessment/distributed-classifier.html) - Domain, Quality, Safety, Educational Content, Content Type, and Prompt Task/Complexity Classification

- **Deduplication**
  - [Exact Deduplication](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/gpudedup.html) - Remove identical documents efficiently
  - [Fuzzy Deduplication](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/gpudedup.html) - MinHash Locality Sensitive Hashing with optional False Positive Check
  - [Semantic Deduplication](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/semdedup.html) - GPU-accelerated semantic deduplication using RAPIDS cuML, cuDF, and PyTorch

- **Content Processing & Cleaning**
  - [Text Cleaning](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/content-processing/text-cleaning.html) - Remove improperly decoded Unicode characters, inconsistent line spacing, and excessive URLs
  - [PII Redaction](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/content-processing/pii.html) - Identify and remove personally identifiable information from training datasets

- **Specialized Processing**
  - [Language Identification](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/language-management/index.html) - Accurate language detection using fastText
  - [Task Decontamination](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/specialized-processing/task-decontamination.html) - Remove potential evaluation data leakage from training datasets

#### Generate Data
- **[Synthetic Data Pipelines](https://docs.nvidia.com/nemo/curator/latest/curate-text/generate-data/pipelines/index.html)** - Pre-built pipelines for generating high-quality synthetic training data:
  - [Open Q&A Generation](https://docs.nvidia.com/nemo/curator/latest/curate-text/generate-data/pipelines/open-qa.html) - Create question-answer pairs for instruction tuning
  - [Math Problem Generation](https://docs.nvidia.com/nemo/curator/latest/curate-text/generate-data/pipelines/math.html) - Generate mathematical problems for educational content
  - [Coding Tasks](https://docs.nvidia.com/nemo/curator/latest/curate-text/generate-data/pipelines/python.html) - Create programming challenges and code examples
  - [Writing Prompts](https://docs.nvidia.com/nemo/curator/latest/curate-text/generate-data/pipelines/writing-task.html) - Generate creative writing and content creation tasks
  - [Dialogue Generation](https://docs.nvidia.com/nemo/curator/latest/curate-text/generate-data/pipelines/dialogue.html) - Create conversational data for chat models
  - [Nemotron Pipelines](https://docs.nvidia.com/nemo/curator/latest/curate-text/generate-data/pipelines/wikipedia.html) - Wikipedia-style rewriting and knowledge distillation

---

### Image Curation

NeMo Curator provides powerful image curation features to curate high-quality image data for training generative AI models such as LLMs, VLMs, and WFMs. Image curation follows a **Load** → **Process** workflow: download datasets in WebDataset format, create embeddings, apply quality filters (NSFW and Aesthetic), and remove duplicates using semantic deduplication.

#### Load Data
- **[WebDataset Loading](https://docs.nvidia.com/nemo/curator/latest/curate-images/load-data/index.html)** - Load large-scale image-text datasets in WebDataset format

#### Process Data
- **Embeddings & Feature Extraction**
  - [Image Embedding Creation](https://docs.nvidia.com/nemo/curator/latest/curate-images/process-data/embeddings/index.html) - Generate CLIP embeddings for image analysis

- **Quality Assessment & Filtering**
  - [Aesthetic Classification](https://docs.nvidia.com/nemo/curator/latest/curate-images/process-data/classifiers/index.html) - Filter images based on aesthetic quality
  - [NSFW Classification](https://docs.nvidia.com/nemo/curator/latest/curate-images/process-data/classifiers/index.html) - Remove inappropriate content from datasets

- **Deduplication**
  - [Semantic Deduplication](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/semdedup.html) - Remove visually similar images using embedding-based clustering

---

## Module Ablation and Compute Performance

The modules within NeMo Curator were primarily designed to process and curate high-quality documents at scale.  To evaluate the quality of the data, we curated Common Crawl documents and conducted a series of ablation experiments. In these experiments, we trained a 357M-parameter GPT-style model using datasets generated at various stages of our data curation pipeline, which was implemented in NeMo Curator.

The following figure shows that the use of different data curation modules implemented in NeMo Curator led to improved model zero-shot downstream task performance.

<p align="center">
  <img src="./docs/user-guide/assets/readme/chart.png" alt="drawing" width="700"/>
</p>

NeMo Curator leverages NVIDIA RAPIDS™ libraries like cuDF, cuML, and cuGraph along with Dask to scale workloads across multi-node, multi-GPU environments, significantly reducing data processing time. With NeMo Curator, developers can achieve 16X faster processing for text. Refer to the chart below to learn more details.

NeMo Curator scales near linearly which means that developers can accelerate their data processing by adding more compute. For  deduplicating the 1.96 Trillion token subset of the RedPajama V2 dataset, NeMo Curator took  0.5 hours with 32 NVIDIA H100 GPUs. Refer to the scaling chart below to learn more

## Contribute to NeMo Curator

We welcome community contributions! Please refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for the process.