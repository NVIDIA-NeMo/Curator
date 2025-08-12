# Data curation for DAPT (Domain Adaptive Pre-Training)

[ChipNeMo](https://arxiv.org/pdf/2311.00176) is a chip design domain adapted LLM. Instead of directly deploying off-theshelf
commercial or open-source LLMs, the paper instead adopts the following domain adaptation techniques:
domain-adaptive tokenization, domain adaptive continued pretraining, model alignment with domain-specific instructions, and domain adapted retrieval models. Specifically, LLama 2 foundation models are continually pre-trained with 20B plus tokens on domain-specific chip design data, including code, documents, etc., and then fine-tuned with instruction datasets from design data as well as external sources. Evaluations on the resultant domain-adapted ChipNeMo model demonstrate that
domain-adaptive pretraining of language models, can lead to superior performance in domain related downstream tasks compared to their base LLaMA2 counterparts, without degradations in generic capabilities.

Here, we share a tutorial with best practices on data curation for DAPT (domain-adaptive pre-training) for a ChipNeMo-like code generation use case.

* In this tutorial, we will leverage chip domain/hardware datasets from open-source GitHub repositories (`./code/sources/github_repos.jsonl`), wiki URLs (`./code/sources/wikipedia_urls.jsonl`), and academic papers (`./sources/arxiv_urls.jsonl`).

* `./code/data` contains curated data after processing

The small size of this dataset makes it ideal for creating and validating data curation pipelines on a local machine or a cluster.
This playbook utilizes specific tools and techniques. First, we convert all files to Txt format (if not already in Txt), compress files on disk, add metadata, and convert them to JSON (`./data/raw/`). Then, we leverage [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator/tree/main) to mine high-quality text at scale from a massive code-generation corpus. We use its capabilities to extract text, identify code file types, fix unicode errors, filter quality through heuristics, deduplicate, and redact personal information. We finally also provide steps to blend and shuffle data sources for continued pre-training.


## Hardware Requirements
* This playbook can run on CPUs or GPUs. This playbook can run entirely on a CPU. If you have GPUs, the PII Redaction will be accelerated using them.


## Walkthrough

We will use the datasets in the `dapt-curation/code/data` folder to illustrate data curation through this pipeline. Specifically sample data collected in:
* `./data/raw/github` (we clone github repos, extract text from each file and convert to jsonl)
* `./data/raw/arxiv_pdfs` (we extract data from pdfs, convert to txt and store as jsonl files)
* `./data/raw/wikipedia` (we extract data from htmls, parse, convert to txt and store as json files)

The tutorial follows the steps below:<br>
- Step 1: Install requirements and import libraries<br>
- Step 2: Download the data from online sources (Github repos, wiki urls, arxiv pdfs), extract metadata and convert to JSONL<br>
- Step 3: Load the dataset <br>
- Step 4: Examine the file types and sizes (optional) <br>
- Step 5: Run the data curation pipeline with with Nemo Curator<br>
    - File type identification and separation
    - Document-level exact deduplication
    - Heuristic-based quality filtering (Number of lines, worc count, top N-grams, etc.)
    - Fix unicode errors via ftfy
    - PII redaction
    - GPU accelerated fuzzy and semantic deduplication
- Step 6: Save the filtered and curated data <br>
- Step 7: Blend datasets and shuffle


## Usage

Please follow the instructions in NeMo Curator's [README](https://github.com/NVIDIA/NeMo-Curator?tab=readme-ov-file#nemo-framework-container) to run the NeMo Framework Container and install the NeMo Curator package. Then, install the following dependencies for running the DAPT tutorial:

```bash
cd /opt/Curator/tutorials/dapt-curation/code
apt update
apt-get install poppler-utils
apt-get install tesseract-ocr
apt install libtesseract-dev
pip install -r requirements.txt
pip uninstall --yes $(pip list --format=freeze | grep opencv)
rm -rf /usr/local/lib/python3.10/dist-packages/cv2/
pip install "opencv-python-headless<4.9.0" # Since we want to use numpy<2
python -c "import nltk; nltk.download('punkt_tab')"
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
python main.py --device "gpu"
```

This will download chip-design related datasets and begin the data curation pipeline.

Please use `--device "gpu"` to enable semantic and fuzzy deduplication, which require the GPU.
