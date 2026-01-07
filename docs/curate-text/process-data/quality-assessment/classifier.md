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


NeMo Curator provides two approaches for quality assessment:

1. **Classification**: Use `QualityClassifier` to add quality predictions and optionally filter during classification
2. **Filtering**: Use `FastTextQualityFilter` with `ScoreFilter` for document-level filtering with Pareto sampling

:::{note}
If you need to train custom fastText models for specific domains or requirements, refer to the [fastText documentation](https://fasttext.cc/docs/en/supervised-tutorial.html) for comprehensive training guides.
:::

::::{tab-set}

:::{tab-item} DeBERTa Quality Classification

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.classifiers import QualityClassifier

# Create pipeline with DeBERTa quality classifier
pipeline = Pipeline(name="deberta_quality_pipeline")

# Add stages
read_stage = JsonlReader("input_data/")
classify_stage = QualityClassifier(
    filter_by=["High"],  # Keep only high-quality documents
    model_inference_batch_size=256,
    max_chars=6000  # Default value
)
write_stage = JsonlWriter("high_quality_output/")

pipeline.add_stage(read_stage)
pipeline.add_stage(classify_stage)
pipeline.add_stage(write_stage)

# Execute pipeline
results = pipeline.run()
```

:::

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

You can configure quality classifiers and filters with different parameters:

```python
from nemo_curator.stages.text.classifiers import QualityClassifier
from nemo_curator.stages.text.filters import FastTextQualityFilter

# DeBERTa quality classifier configurations
basic_deberta_classifier = QualityClassifier(
    filter_by=["High"],          # Keep only high-quality documents
    model_inference_batch_size=256,
    max_chars=6000               # Default value
)

# More inclusive DeBERTa classifier
inclusive_deberta_classifier = QualityClassifier(
    filter_by=["Medium", "High"], # Keep medium and high-quality documents
    model_inference_batch_size=128,
    max_chars=6000
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

## Quality Classifier and Filter Parameters

### QualityClassifier (DeBERTa)

The `QualityClassifier` accepts the following parameters:

- `filter_by` (list, default=None): Quality levels to keep (options: "Low", "Medium", "High")
- `model_inference_batch_size` (int, default=256): Batch size for inference
- `max_chars` (int, default=6000): Max characters per document for processing
- `label_field` (str, default="quality_pred"): Name of the prediction column
- `text_field` (str, default="text"): Name of the text field in input data

### FastTextQualityFilter

The `FastTextQualityFilter` accepts the following parameters:

- `model_path` (str, required): Path to the trained fastText model file
- `label` (str, default="__label__hq"): The label for high-quality documents
- `alpha` (float, default=3): Alpha parameter for Pareto distribution sampling
- `seed` (int, default=42): Random seed for reproducible sampling

## Best Practices

For effective classifier-based filtering:

1. **Model selection**: Start with the DeBERTa quality classifier for general use cases; consider fastText for high-throughput scenarios
2. **Validation**: Manually review a sample of filtered results to confirm effectiveness
3. **Quality level tuning**: Adjust `filter_by` levels (DeBERTa) or `alpha` values (fastText) based on your quality requirements
4. **Batch size optimization**: Tune `model_inference_batch_size` for DeBERTa models based on your available memory
5. **Combination with heuristics**: Consider using heuristic filters as a pre-filter to improve efficiency
6. **Domain adaptation**: For specialized corpora, consider training custom models using domain-specific data
