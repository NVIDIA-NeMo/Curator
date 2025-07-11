# Multi-Modal Data Curation from PDFs

## Overview
This is Part 2 of the tutorial that provides best practices for data curation in Domain-Adaptive Pre-Training (DAPT).
The dataset used in this tutorial is small, making it ideal for developing and validating data curation pipelines on either a local machine or a computing cluster. The playbook employs specialized tools and techniques for high-quality text curation and refinement.

## Hardware Requirements
This playbook is compatible with both CPUs and GPUs.
While most steps can run on a CPU, the semantic and fuzzy deduplication modules require a GPU.
If GPUs are available, the PII redaction and exact deduplication processes will be accelerated.

## Walkthrough
The datasets used in this tutorial are located in the `NeMo-Curator/tutorials/multimodal_dapt_curation/ingest/sources/separated_extracted_data/data_type_map.json` file.

The tutorial follows these steps:
1. Install requirements and import libraries
2. Convert extracted data: Transform data from `nv-ingest` into Dask DataFrames and convert them to `DocumentDataset`.
3. Examine file types and sizes (optional)
4. Run the data curation pipeline with NeMo Curator:
   - Identify and separate file types
   - Perform document-level exact deduplication
   - Apply heuristic-based quality filtering (e.g., number of lines, word count, top N-grams)
   - Fix Unicode errors using `ftfy`
   - Redact PII
   - Execute GPU-accelerated fuzzy and semantic deduplication
5. Convert images extracted from nv-ingest into webdataset format
6. Apply semantic deduplication to get rid of duplicate images extracted
7. Save the filtered and curated data

## Usage
After installing the NeMo Curator package, install the required dependencies and run the pipeline using the following command:
```sh
pip install -r requirements.txt
```

```sh
python main.py --protocol "tcp" --rmm-pool-size "1GB" --device "gpu"
```

## License
Refer to the relevant repository for licensing information.
