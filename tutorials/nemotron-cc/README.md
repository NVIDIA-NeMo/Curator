# Nemotron-CC Data Curation Pipeline Tutorial using NeMo Curator

This directory contains a Jupyter notebook (`nemotron_cc.ipynb`) demonstrating how to build the data curation pipeline used to create the [Nemotron-CC dataset](https://arxiv.org/pdf/2412.02595) using the [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) library.

NeMo Curator is a Python library providing scalable data-mining modules for curating Natural Language Processing (NLP) data for training Large Language Models (LLMs).

## Notebook Overview (`nemotron_cc.ipynb`)

The notebook walks through the multi-stage Nemotron-CC data curation pipeline, which involves:

1.  **Environment Setup**: Initializing the environment, importing necessary libraries (including NeMo Curator modules, Dask, Pandas, and cuDF), and defining paths.
2.  **Data Extraction and Preprocessing**:
    *   Downloading specified Common Crawl snapshots using `download_common_crawl`.
    *   Extracting text content from WARC files using `JusTextExtractor`.
    *   Performing language identification with `FastTextLangId` and separating documents by language, focusing on English (`EN`).
    *   Applying Unicode normalization using `UnicodeReformatter`.
3.  **Data Deduplication and Quality Filtering**:
    *   Assigning unique document IDs using `AddId`.
    *   Performing **Exact Deduplication** based on document content hashes (MD5) using `ExactDuplicates`.
    *   Performing **Fuzzy Deduplication** using MinHashLSH via the `FuzzyDuplicates` module to remove near-duplicates.
    *   (Optional) Reference implementation for **Exact Substring Deduplication** using Google's `deduplicate-text-datasets` library (requires Rust setup).
    *   Applying **Heuristic Filtering** based on rules defined in a YAML configuration (`heuristic_filter_en.yaml`) using `build_filter_pipeline`.
    *   Applying **Perplexity Filtering** using a custom filter leveraging a pre-trained KenLM model.
4.  **Model-Based Quality Labeling**:
    *   Classifying documents into quality buckets using an ensemble of pre-trained classifier models (`FineWebNemotronEduClassifier`, `FineWebMixtralEduClassifier`, `FastTextQualityClassifier`).
    *   Computing quality score thresholds and assigning integer scores.
    *   Calculating a final ensemble score based on the maximum integer score.
    *   (Optional) Saving the classified data into partitioned directories based on the ensemble score.
5.  **Synthetic Data Generation (SDG)**:
    *   Splitting the dataset into low-quality (buckets 0-11) and high-quality (buckets 12-19) subsets based on the ensemble score.
    *   Applying a **Wikipedia Rephraser** pipeline to a sample of low-quality data using an external LLM (e.g., Gemma via NVIDIA API Catalog or OpenAI).
    *   Applying multiple SDG techniques (**Diverse QA**, **Distill**, **Extract Knowledge**, **Knowledge List**) to a sample of high-quality data using the same LLM. Helper functions for these SDG steps are provided in `nemotron_sdg_utilities.py`.
6.  **Dataset Compilation**:
    *   Merging the processed low-quality and high-quality datasets.
    *   Selecting the final relevant columns (ID, original text, language, generated synthetic texts, quality label).
    *   Saving the final curated dataset as a JSONL file.
7.  **Visualization**:
    *   Using the `text_comparison_widget` (from `viz/text_comparison_widget.py`) to visually compare the original text with the synthetically generated versions (rephrased or distilled) for selected examples.

## Prerequisites

The notebook assumes you are running within the NeMo Framework Training Container and have access to the necessary hardware (e.g., NVIDIA A100 GPU), CUDA drivers, and potentially API keys for external LLMs used in the SDG steps. It also uses Dask for distributed processing (both CPU and GPU clusters are demonstrated).

## Usage

Follow the steps within the `nemotron_cc.ipynb` notebook. Adjust parameters like file paths, snapshot ranges, sampling fractions, and API keys as needed for your environment. The notebook uses helper functions from `quality_labeling_utilities.py`, `kenlm_utility.py`, `nemotron_sdg_utilities.py`, and visualization utilities from the `viz/` directory.
