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
  - Quality filtering, high/low quality document classification (available as filter, not distributed classifier)
  - https://fasttext.cc/
* - FastTextLangId
  - fastText (language identification)
  - Language identification (available as filter, not distributed classifier)
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

First, you need to prepare training data by sampling from high-quality and low-quality datasets using the CLI command.

You can prepare training data using Python scripts:

```python
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.pipeline import Pipeline
import random

# Sample from low-quality dataset (e.g., raw Common Crawl)
def sample_documents(input_path, output_path, num_samples, label):
    pipeline = Pipeline(name="sample_data")
    reader = JsonlReader(file_paths=input_path, fields=["text"])
    pipeline.add_stage(reader)
    
    # Execute pipeline to load data
    results = pipeline.run()
    
    # Sample and save with labels for fastText format
    with open(output_path, 'w') as f:
        for result in results:
            data = result.to_pandas()
            sampled = data.sample(min(num_samples, len(data)))
            for _, row in sampled.iterrows():
                f.write(f"{label} {row['text'].replace(chr(10), ' ')}\n")

# Create training samples
sample_documents(
    "/path/to/common-crawl/*.jsonl", 
    "./cc_samples.txt", 
    10000, 
    "__label__cc"
)
sample_documents(
    "/path/to/wikipedia/*.jsonl", 
    "./hq_samples.txt", 
    10000, 
    "__label__hq"
)
```

#### Command Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--input-data-dir` | Directory containing JSONL files to sample from | `/data/common_crawl/` |
| `--output-num-samples` | Number of documents to sample | `10000` |
| `--label` | FastText label prefix for the category | `__label__hq` or `__label__cc` |
| `--output-train-file` | Output file for training samples | `./samples.txt` |

### 2. Use Pre-trained Quality Classifiers

For most quality filtering use cases, we recommend using the pre-trained quality classifiers available in NeMo Curator rather than training custom fastText models:

#### Option A: DeBERTa Quality Classifier (Recommended)

Use the pre-trained `nvidia/quality-classifier-deberta` model for robust quality assessment:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules import DistributedDataClassifier
from nemo_curator.stages.text.classifiers import QualityClassifier

# Create pipeline with DeBERTa quality classifier
pipeline = Pipeline(name="quality_classification_pipeline")

# Add stages
read_stage = JsonlReader("input_data/")
classify_stage = DistributedDataClassifier(
    QualityClassifier(
        model_path="nvidia/quality-classifier-deberta",
        labels=["Low", "Medium", "High"],  # Quality levels
        batch_size=256,
        max_chars=2000
    ),
    text_field="text",
    pred_column="quality_pred"
)
write_stage = JsonlWriter("classified_output/")

pipeline.add_stage(read_stage)
pipeline.add_stage(classify_stage)
pipeline.add_stage(write_stage)

# Execute pipeline
results = pipeline.run()
```

#### Option B: Custom fastText Models

If you need to train custom fastText models for specific domains or requirements, refer to the [fastText documentation](https://fasttext.cc/docs/en/supervised-tutorial.html) for comprehensive training guides.

:::{note}
**When to use each approach:**

- **DeBERTa Quality Classifier**: Higher accuracy, multi-class output (Low/Medium/High), better for general quality assessment
- **fastText**: Faster inference, lower memory usage, better for high-throughput scenarios or when you have domain-specific training data
:::

### 3. Filter Data Using Quality Scores

After obtaining quality predictions, you can filter your dataset based on the scores. Here are examples for both classifier types:

::::{tab-set}

:::{tab-item} DeBERTa Quality Filter

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import QualityFilter

# Create pipeline with DeBERTa quality filter
pipeline = Pipeline(name="deberta_quality_pipeline")

# Add stages
read_stage = JsonlReader("input_data/")
filter_stage = ScoreFilter(
    QualityFilter(
        model_path="nvidia/quality-classifier-deberta",
        filter_by=["High"],  # Keep only high-quality documents
        batch_size=256
    ),
    text_field="text",
    score_field="quality_pred"
)
write_stage = JsonlWriter("high_quality_output/")

pipeline.add_stage(read_stage)
pipeline.add_stage(filter_stage)
pipeline.add_stage(write_stage)

# Execute pipeline
results = pipeline.run()
```

:::{tab-item} FastText Quality Filter

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import FastTextQualityFilter

# Create pipeline with FastText filter (requires pre-trained model)
pipeline = Pipeline(name="fasttext_quality_pipeline")

# Add stages
read_stage = JsonlReader("input_data/")
filter_stage = ScoreFilter(
    FastTextQualityFilter(
        model_path="./quality_classifier.bin",  # Path to your fastText model
        label="__label__hq",  # High quality label
        alpha=3,              # Pareto distribution alpha parameter
        seed=42               # Random seed for reproducibility
    ),
    text_field="text",
    score_field="quality_score"
)
write_stage = JsonlWriter("high_quality_output/")

pipeline.add_stage(read_stage)
pipeline.add_stage(filter_stage)
pipeline.add_stage(write_stage)

# Execute pipeline
results = pipeline.run()
```

:::

:::{tab-item} Configuration

You can configure quality filters with different parameters:

```python
from nemo_curator.stages.text.filters import QualityFilter, FastTextQualityFilter

# DeBERTa quality filter configurations
basic_deberta_filter = QualityFilter(
    model_path="nvidia/quality-classifier-deberta",
    filter_by=["High"],          # Keep only high-quality documents
    batch_size=256,
    max_chars=2000
)

# More inclusive DeBERTa filter
inclusive_deberta_filter = QualityFilter(
    model_path="nvidia/quality-classifier-deberta",
    filter_by=["Medium", "High"], # Keep medium and high-quality documents
    batch_size=128
)

# FastText quality filter configurations
basic_fasttext_filter = FastTextQualityFilter(
    model_path="./quality_classifier.bin",
    label="__label__hq",         # High quality label
    alpha=3,                     # Pareto distribution alpha parameter
    seed=42                      # Random seed for reproducibility
)

# More selective FastText filter
selective_fasttext_filter = FastTextQualityFilter(
    model_path="./quality_classifier.bin",
    label="__label__hq",
    alpha=5,                     # Higher alpha for stricter filtering
    seed=42
)
```

:::

::::

## Pareto-Based Sampling

NeMo Curator's implementation includes support for Pareto-based sampling, as described in Brown et al., 2020. This approach:

1. Scores documents using the trained classifier
2. Ranks documents based on their scores
3. Samples documents according to a Pareto distribution, favoring higher-ranked documents

This method helps maintain diversity in the dataset while still prioritizing higher-quality documents.

## Quality Filter Parameters

### QualityFilter (DeBERTa)

The `QualityFilter` accepts the following parameters:

- `model_path` (str, required): HuggingFace model path (e.g., "nvidia/quality-classifier-deberta")
- `filter_by` (list, default=["High"]): Quality levels to keep (options: "Low", "Medium", "High")
- `batch_size` (int, default=256): Batch size for inference
- `max_chars` (int, default=2000): Max characters per document for processing

### FastTextQualityFilter

The `FastTextQualityFilter` accepts the following parameters:

- `model_path` (str, required): Path to the trained fastText model file
- `label` (str, default="__label__hq"): The label for high-quality documents
- `alpha` (float, default=3): Alpha parameter for Pareto distribution sampling
- `seed` (int, default=42): Random seed for reproducible sampling

## Configuration

Typical configurations for quality-based filtering:

### DeBERTa Quality Classifier

```yaml
filters:
  - name: ScoreFilter
    filter:
      name: QualityFilter
      model_path: nvidia/quality-classifier-deberta
      filter_by: ["High"]
      batch_size: 256
      max_chars: 2000
    text_field: text
    score_field: quality_pred
```

### FastText Quality Classifier

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

1. **Model selection**: Start with the DeBERTa quality classifier for general use cases; consider fastText for high-throughput scenarios
2. **Validation**: Manually review a sample of filtered results to confirm effectiveness
3. **Quality level tuning**: Adjust `filter_by` levels (DeBERTa) or `alpha` values (fastText) based on your quality requirements
4. **Batch size optimization**: Tune `batch_size` for DeBERTa models based on your available memory
5. **Combination with heuristics**: Consider using heuristic filters as a pre-filter to improve efficiency
6. **Domain adaptation**: For specialized corpora, consider training custom models using domain-specific data
