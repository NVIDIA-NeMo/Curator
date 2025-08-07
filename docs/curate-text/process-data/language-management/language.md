---
description: "Identify document languages accurately using FastText models supporting 176 languages for multilingual text processing"
categories: ["how-to-guides"]
tags: ["language-identification", "fasttext", "multilingual", "176-languages", "detection", "classification"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-languages-id)=
# Language Identification and Unicode Fixing

Large unlabeled text corpora often contain a variety of languages. NVIDIA NeMo Curator provides tools to accurately identify the language of each document, which is essential for language-specific curation tasks and building high-quality monolingual datasets.

## Overview

Language identification is a critical step in text data curation for several reasons:

- Many data curation steps are language-specific (for example, quality filtering with language-tuned heuristics)
- Most curation pipelines focus on creating monolingual datasets
- Document language is important metadata for model training and evaluation

NeMo Curator provides utilities for language identification using fastText, which offers highly accurate language detection across 176 languages. While preliminary language identification may occur earlier in the pipeline (such as during Common Crawl extraction with pyCLD2), fastText provides more accurate results for a definitive classification.

## Usage

::::{tab-set}

:::{tab-item} Python

```python
import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.filters import FastTextLangId

# Load your dataset
files = get_all_files_paths_under("input_data/", keep_extensions="jsonl")
dataset = DocumentDataset.read_json(files)

# Create language identification filter
# IMPORTANT: Download lid.176.bin from https://fasttext.cc/docs/en/language-identification.html first
langid_filter = nc.ScoreFilter(
    FastTextLangId(
        model_path="/path/to/lid.176.bin", 
        min_langid_score=0.3  # Default confidence threshold (can be adjusted based on requirements)
    ),
    text_field="text",  # Field in your documents containing text to analyze
    score_field="language",  # Field to store language identification results
    score_type="object"  # The score is an object containing [score, language_code]
)

# Apply language identification
identified_dataset = langid_filter(dataset)

# The language field contains [score, lang_code]
# Extract just the language code if needed
identified_dataset.df["language"] = identified_dataset.df["language"].apply(
    lambda score: score[1]  # Extract language code from [score, lang_code]
)

# Now each document has a language code field
# You can filter for specific languages
english_docs = identified_dataset[identified_dataset.df.language == "EN"]

# Save the dataset with language information
identified_dataset.to_json("output_with_language/", write_to_filename=True)
```

:::

:::{tab-item} CLI

### Identifying Languages

```bash
filter_documents \
  --input-data-dir=/path/to/jsonl/files \
  --filter-config-file=./config/fasttext_langid.yaml \
  --log-scores \
  --log-dir=./log/lang_id
```

This command applies the fastText model to compute language scores and codes for each document, adding this information as additional fields in each JSON document.

### Separating Documents by Language

Once language information is added to your documents, you can separate them by language:

```bash
separate_by_metadata \
  --input-data-dir=/path/to/jsonl/files \
  --input-metadata-field=language \
  --output-data-dir=/path/to/output/by_language \
  --output-metadata-distribution=./data/lang_distro.json
```

After running this command, the output directory will contain one subdirectory per language, with each containing only documents in that language.

:::
::::

## Configuration

A typical configuration for language identification looks like:

```yaml
# Example fasttext_langid.yaml
input_field: text
filters:
  - name: nemo_curator.filters.classifier_filter.FastTextLangId
    log_score: True
    params:
      model_path: /path/to/lid.176.bin
      min_langid_score: 0.3  # Default confidence threshold (adjust based on precision/recall needs)
```

## Understanding Results

The language identification process adds a field to each document:

1. `language`: By default, this contains a list with two elements:
   - Element 0: The confidence score (between 0 and 1)
   - Element 1: The language code in fastText format (for example, "EN" for English, "ES" for Spanish)

:::{note}
FastText language codes are typically two-letter uppercase codes that may differ slightly from standard ISO 639-1 codes. The model supports 176 languages with high accuracy.
:::

As shown in the Python example, you can extract just the language code with a simple transform if needed.

A higher confidence score indicates greater certainty in the language identification. You can adjust the threshold based on your requirements for precision.

## Performance Considerations

- Language identification is computationally intensive but highly scalable across processors
- For large datasets, consider using a distributed Dask setup
- The fastText model file (`lid.176.bin`) is approximately 130MB and must be accessible to all worker nodes
- Processing speed depends on document length and available computational resources
- Memory usage scales with the number of worker processes and batch sizes

## Best Practices

:::{important}
**Model Download Required**: Download the fastText language identification model (`lid.176.bin`) from the [official fastText repository](https://fasttext.cc/docs/en/language-identification.html) before using this filter. The model file is approximately 130MB.
:::

- Set an appropriate confidence threshold based on your requirements:
  - **Default threshold (0.3)**: Balanced approach suitable for most use cases
  - **Higher threshold (0.7+)**: More precision but may discard borderline documents
  - **Lower threshold (0.1-0.2)**: Higher recall but may include misclassified documents
- Analyze the language distribution in your dataset to understand its composition
- Consider a two-pass approach: first filter with a lower threshold, then manually review edge cases
- For production workflows, validate language identification accuracy on a sample of your specific domain data 