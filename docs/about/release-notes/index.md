---
description: "Release notes and version history for NeMo Curator platform updates and new features"
categories: ["reference"]
tags: ["release-notes", "changelog", "updates"]
personas: ["data-scientist-focused", "mle-focused", "admin-focused", "devops-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

(about-release-notes)=
# Release Notes 25.07

## 🚀 Major Features and Enhancements

### New How-to Data Recipes (Tutorials)

- [**Multimodal DAPT Curation w/ PDF Extraction**](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/multimodal_dapt_curation): New tutorial showcasing PDF data extraction using NV-Ingest for multimodal data curation workflows
- [**Llama Nemotron Data Curation**](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/llama-nemotron-data-curation): Step-by-step guide for curating data specifically for Llama Nemotron model training
- [**LLM NIM - PII Redaction**](https://github.com/NVIDIA-NeMo/Curator/tree/main/tutorials/curator-llm-pii): Tutorial demonstrating PII (Personally Identifiable Information) redaction capabilities using LLM NIM

### Container and Deployment Improvements

- **Docker Container Build**: New docker container for text/image curator with Dask support, matching the current OSS implementation
- Streamlined deployment process for both text and image curation workflows

### Performance and Code Optimizations

- **Simplified Clustering Logic**: Significantly improved semantic deduplication clustering performance
  - Removed convoluted backend switching logic that caused performance issues
  - Eliminated expensive length assertions that could cause timeouts on large datasets
  - Improved GPU utilization during KMeans clustering operations
  - Tested on 37M embedding dataset (80GB) across 7 GPUs with substantial performance gains

## 🐛 Bug Fixes

### FastText Download URL Fix

- **Fix**: Corrected the `fasttext` model download URL in nemotron-cc tutorial
- Changed from `dl.fbaipublicfiles.com/fastText/` to `dl.fbaipublicfiles.com/fasttext/`
- Ensures reliable model downloads for language identification

### NeMo Retriever Tutorial Bug Fix

- **Fix**: Fixed lambda function bug in `RetrieverEvalSetGenerator`
- Corrected score assignment: `df["score"] = df["question"].apply(lambda: 1)` → `df["score"] = 1`
- Improved synthetic data generation reliability

### API Usage Updates

- **Fix**: Updated examples and tutorials to use correct DocumentDataset API
- Key changes:
  - Replaced deprecated `write_to_disk(result, output_dir, output_type="parquet")` with `result.to_parquet(output_dir)`
  - Updated exact deduplication workflows: `deduplicator.remove()` now returns `DocumentDataset` directly
  - Removed unnecessary `DocumentDataset` wrapper in deduplication examples
  - Fixed `read_json` calls to remove deprecated `add_filename` parameter where appropriate
- Files updated:
  - `examples/exact_deduplication.py`
  - `examples/fuzzy_deduplication.py`
  - `tutorials/dapt-curation/code/utils.py`
  - `tutorials/multimodal_dapt_curation/curator/utils.py`
  - `tutorials/tinystories/main.py`

---

## 🔄 Migration Guide

### For Users of Exact Deduplication Examples

```python
# Old approach
from nemo_curator.utils.distributed_utils import write_to_disk
result = exact_dup.remove(input_dataset, duplicates)
write_to_disk(result, output_dir, output_type="parquet")

# New approach  
result = exact_dup.remove(input_dataset, duplicates)
result.to_parquet(output_dir)
```

### For FastText Users

- Update any manual `fasttext` model downloads to use the corrected URL:
  ```bash
  # Old URL
  wget https://dl.fbaipublicfiles.com/fastText/supervised-models/lid.176.bin
  
  # New URL  
  wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
  ```