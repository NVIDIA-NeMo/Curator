---
description: "Filter text using trained quality classifiers including FastText models and pre-trained language classification"
categories: ["how-to-guides"]
tags: ["classifier-filtering", "fasttext", "ml-models", "quality", "training", "scoring"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-filter-classifier)=
# Classifier-Based Filtering

Classifier-based filtering uses machine learning models to differentiate between high-quality and low-quality documents. NVIDIA NeMo Curator implements an approach similar to the one described in [Brown et al., 2020](https://arxiv.org/abs/2005.14165), which trains a binary skip-gram classifier to distinguish between curated high-quality data and lower-quality data.

## Supported Classifier Models

NeMo Curator supports a variety of classifier models for different filtering and classification tasks. The table below summarizes the main supported models, their backend, typical use case, and HuggingFace model link (if public):

```{list-table}
:header-rows: 1
:widths: 20 20 30 30
* - Classifier Name
  - Model Type
  - Typical Use Case / Description
  - HuggingFace Model/Link
* - FastTextQualityFilter
  - fastText (binary classifier)
  - Quality filtering, high/low quality document classification
  - https://fasttext.cc/
* - FastTextLangId
  - fastText (language identification)
  - Language identification
  - https://fasttext.cc/docs/en/language-identification.html
* - QualityClassifier
  - DeBERTa (transformers, HF)
  - Document quality classification (multi-class, e.g., for curation)
  - https://huggingface.co/nvidia/quality-classifier-deberta
* - DomainClassifier
  - DeBERTa (transformers, HF)
  - Domain classification (English)
  - https://huggingface.co/nvidia/domain-classifier
* - MultilingualDomainClassifier
  - mDeBERTa (transformers, HF)
  - Domain classification (multilingual, 52 languages)
  - https://huggingface.co/nvidia/multilingual-domain-classifier
* - ContentTypeClassifier
  - DeBERTa (transformers, HF)
  - Content type classification (11 speech types)
  - https://huggingface.co/nvidia/content-type-classifier-deberta
* - AegisClassifier
  - LlamaGuard-7b (LLM, PEFT, HF)
  - Safety classification (AI content safety, requires access to LlamaGuard-7b)
  - https://huggingface.co/meta-llama/LlamaGuard-7b
* - InstructionDataGuardClassifier
  - Custom neural net (used with Aegis)
  - Detects instruction data poisoning
  - https://huggingface.co/nvidia/instruction-data-guard
* - FineWebEduClassifier
  - SequenceClassification (transformers, HF)
  - Educational content quality scoring (FineWeb)
  - https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier
* - FineWebMixtralEduClassifier
  - SequenceClassification (transformers, HF)
  - Educational content quality scoring (Mixtral variant)
  - https://huggingface.co/nvidia/nemocurator-fineweb-mixtral-edu-classifier
* - FineWebNemotronEduClassifier
  - SequenceClassification (transformers, HF)
  - Educational content quality scoring (Nemotron-4 variant)
  - https://huggingface.co/nvidia/nemocurator-fineweb-nemotron-4-edu-classifier
```

## How It Works

Classifier-based filtering learns the characteristics of high-quality documents from training data, unlike heuristic filtering which relies on predefined rules and thresholds. This approach is particularly effective when:

- You have a reference dataset of known high-quality documents
- The distinction between high and low quality is complex or subtle
- You want to filter based on domain-specific characteristics

NVIDIA NeMo Curator uses [fastText](https://fasttext.cc/) for implementing classifier-based filtering, which offers excellent performance and scalability for text classification tasks.

:::{note}
fastText is the official name and capitalization used by the fastText library created by Facebook Research.
:::

The classifier-based filtering process involves:

1. Preparing training data by sampling from high-quality and low-quality datasets
2. Training a binary skip-gram classifier using fastText
3. Using the trained model to score documents in your dataset
4. Filtering documents based on the classifier scores, optionally using Pareto-based sampling

---

## Usage

:::{note}
Training fastText classifiers requires using CLI commands. The trained models can then be used with the Python API for filtering datasets.
:::

### 1. Prepare Training Data

First, you need to prepare training data by sampling from high-quality and low-quality datasets using the CLI command:

```bash
# Sample from low-quality (e.g., raw Common Crawl) dataset
prepare_fasttext_training_data \
  --input-data-dir=/path/to/common-crawl \
  --output-num-samples=10000 \
  --label='__label__cc' \
  --output-train-file=./cc_samples.txt

# Sample from high-quality (e.g., Wikipedia) dataset
prepare_fasttext_training_data \
  --input-data-dir=/path/to/wikipedia \
  --output-num-samples=10000 \
  --label='__label__hq' \
  --output-train-file=./hq_samples.txt
```

### 2. Train a Classifier

Next, train a fastText classifier using the prepared samples:

```bash
train_fasttext \
  --fasttext-files-dir=./ \
  --output-train-file=./fasttext_samples.train \
  --output-validation-file=./fasttext_samples.valid \
  --output-model=./quality_classifier.bin \
  --output-predictions=./predictions.jsonl
```

The training script will output validation metrics including accuracy, precision, recall, F1 score, and confusion matrix.

### 3. Apply the Classifier for Filtering

Finally, use the trained model to filter your dataset:

::::{tab-set}

:::{tab-item} Python

```python
import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import FastTextQualityFilter

# Load your dataset
dataset = DocumentDataset.read_json("input_data/*.jsonl")

# Create a quality filter using the trained model
filter_step = nc.ScoreFilter(
    FastTextQualityFilter(
        model_path="./quality_classifier.bin",
        label="__label__hq",  # High quality label
        alpha=3,              # Pareto distribution alpha parameter
        seed=42               # Random seed for reproducibility
    ),
    text_field="text",
    score_field="quality_score"
)

# Apply the filter
high_quality_data = filter_step(dataset)

# Save the results
high_quality_data.to_json("high_quality_output/", write_to_filename=True)
```

:::

:::{tab-item} CLI

```bash
filter_documents \
  --input-data-dir=/path/to/input/data \
  --filter-config-file=./config/fasttext_quality_filter.yaml \
  --output-retained-document-dir=/path/to/output/high_quality \
  --output-removed-document-dir=/path/to/output/low_quality \
  --log-dir=/path/to/logs/fasttext_classifier
```

Where the YAML configuration file looks like:

```yaml
input_field: text
filters:
  - name: nemo_curator.filters.FastTextQualityFilter
    params:
      model_path: /path/to/quality_classifier.bin
      alpha: 3
      label: "__label__hq"
      seed: 42
```

:::

::::

## Pareto-Based Sampling

NeMo Curator's implementation includes support for Pareto-based sampling, as described in Brown et al., 2020. This approach:

1. Scores documents using the trained classifier
2. Ranks documents based on their scores
3. Samples documents according to a Pareto distribution, favoring higher-ranked documents

This method helps maintain diversity in the dataset while still prioritizing higher-quality documents.

## FastTextQualityFilter Parameters

The `FastTextQualityFilter` accepts the following parameters:

- `model_path` (str, required): Path to the trained fastText model file
- `label` (str, default="__label__hq"): The label for high-quality documents
- `alpha` (float, default=3): Alpha parameter for Pareto distribution sampling
- `seed` (int, default=42): Random seed for reproducible sampling

## Configuration

A typical configuration for classifier-based filtering looks like:

```yaml
filters:
  - name: ScoreFilter
    filter:
      name: FastTextQualityFilter
      model_path: /path/to/quality_classifier.bin
      label: __label__hq
      alpha: 3
      seed: 42
    text_field: text
    score_field: quality_score
```

## Best Practices

For effective classifier-based filtering:

1. **Training data selection**: Use truly high-quality sources for positive examples
2. **Validation**: Manually review a sample of filtered results to confirm effectiveness
3. **Threshold tuning**: Adjust the threshold based on your quality requirements
4. **Combination with heuristics**: Consider using heuristic filters as a pre-filter
5. **Domain adaptation**: Train domain-specific classifiers for specialized corpora 