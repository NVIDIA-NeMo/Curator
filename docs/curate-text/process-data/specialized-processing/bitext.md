---
description: "Specialized filters for processing and filtering bilingual text data for machine translation and multilingual tasks"
categories: ["how-to-guides"]
tags: ["bitext-filtering", "translation", "bilingual", "quality-estimation", "alignment", "multilingual"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-filter-bitext)=
# Bitext Filtering

:::{note}
**Documentation Status**: This page has been verified against NeMo Curator source code for accuracy (January 2025).
:::

NVIDIA NeMo Curator provides specialized filters for processing and filtering bilingual text (bitext) data. These filters are designed specifically for parallel corpora used in machine translation and other multilingual applications.

Bitext filtering addresses specific challenges related to parallel text:

- Ensuring source and target texts are proper translations of each other
- Filtering out low-quality machine translations
- Maintaining reasonable length ratios between language pairs
- Removing pairs with mismatched content or formatting
- Quality estimation of translations using pretrained models

NeMo Curator's bitext filters operate on `ParallelDataset` objects, which contain paired source and target texts.

## Supported Quality Estimation Models

NeMo Curator supports several Quality Estimation (QE) models for assessing translation quality without reference translations. The table below summarizes the supported models, their architecture, and intended use:

```{list-table}
:header-rows: 1
:widths: 20 20 30 30
* - Model Name
  - Architecture
  - Description
  - Best For
* - comet-qe
  - XLM-RoBERTa
  - Reference-free quality estimation model trained on human judgments
  - General-purpose translation QE, fast inference
* - cometoid-wmt23
  - COMET-based
  - Fine-tuned on WMT23 Direct Assessment data
  - State-of-the-art performance on modern translations
* - cometoid-wmt23-mqm
  - COMET-based
  - Fine-tuned on Multidimensional Quality Metrics (MQM) data
  - High correlation with human quality judgments
```

## How It Works

Bitext filters analyze parallel text pairs using various metrics to evaluate their quality:

1. **Quality Estimation Models**: Advanced pretrained models assess translation quality without reference translations
2. **Length Ratio Analysis**: Filters check if the source and target texts have reasonable length proportions
3. **Content Matching**: Algorithms verify that key entities like numbers, dates, and named entities are preserved across translations
4. **Language-Specific Adaptation**: The filters can be tuned based on specific language pair characteristics

These filters can be combined in sequence to create comprehensive filtering pipelines, allowing for precise control over parallel data quality.

---

## Usage

Here's an example of applying bitext filters to a translation dataset:

```python
import nemo_curator as nc
from nemo_curator.datasets import ParallelDataset
from nemo_curator.filters import QualityEstimationFilter

# Load your parallel dataset
dataset = ParallelDataset.read_json("translations/*.jsonl")

# Create a quality estimation filter
qe_filter = QualityEstimationFilter(
    model_name="comet-qe",  # Quality estimation model
    cutoff=0.5,             # Quality threshold
    mode="always_en_x",     # Inference mode
    gpu=True,               # Use GPU for inference
    src_field="source",     # Field name for source text
    tgt_field="target",     # Field name for target text
    metadata_fields=["src_lang", "tgt_lang"],  # Additional fields
    score_field="translation_quality"  # Field to store scores
)

# Apply the filter
filtered_dataset = qe_filter(dataset)

# Save the results
filtered_dataset.to_json("filtered_translations/", write_to_filename=True)
```

:::{warning}
**Important**: The `QualityEstimationFilter` requires `src_lang` and `tgt_lang` fields in your dataset to properly determine translation direction. Make sure your data includes these language code fields.
:::

## Available Filters

| Filter | Description | Key Parameters |
|--------|-------------|----------------|
| **QualityEstimationFilter** | Assesses translation quality using pretrained models | `model_name`, `cutoff`, `mode`, `gpu`, `src_field`, `tgt_field` |
| **LengthRatioFilter** | Ensures reasonable length ratios between source and target | `max_ratio`, `src_lang`, `tgt_lang`, `src_field`, `tgt_field` |

### BitextFilter Base Class

The `BitextFilter` abstract base class provides the foundation for all bitext filters:

```python
from nemo_curator.filters import BitextFilter
import pandas as pd

class MyBitextFilter(BitextFilter):
    def __init__(self, threshold=0.7, **kwargs):
        super().__init__(**kwargs)
        self._threshold = threshold
        
    def score_bitext(self, src, tgt, **kwargs):
        # Compute score based on source and target
        # Can access additional fields via kwargs if specified in metadata_fields
        return compute_quality_score(src, tgt, **kwargs)
        
    def keep_bitext(self, scores):
        # Determine which pairs to keep based on scores
        return scores >= self._threshold
```

### Key Parameters

When creating a bitext filter, you can specify:

- `src_field`: Name of the field containing source text (default: "src")
- `tgt_field`: Name of the field containing target text (default: "tgt")
- `metadata_fields`: Additional fields to pass to the scoring function (default: [])
- `metadata_field_name_mapping`: Mapping between field names and function arguments (default: {})
- `score_field`: Field to store the quality scores (default: None, scores are discarded)
- `score_type`: Data type for scores (default: None)
- `invert`: Whether to invert the filter logic (keep low scores instead of high) (default: False)

### Quality Estimation Filters

The `QualityEstimationFilter` uses pretrained models to assess translation quality:

```python
from nemo_curator.filters import QualityEstimationFilter

# COMET-QE quality estimation
comet_filter = QualityEstimationFilter(
    model_name="comet-qe",
    cutoff=0.5,            # Quality threshold
    mode="always_en_x",    # Always use English as source if present
    gpu=True,              # Use GPU acceleration
    src_field="source",
    tgt_field="target",
    metadata_fields=["src_lang", "tgt_lang"]
)

# Apply to dataset
high_quality = comet_filter(parallel_dataset)
```

### Translation Direction Modes

The `QualityEstimationFilter` supports different modes for handling translation direction:

- `simple`: Use source and target as provided
- `always_en_x`: Always use English as source if present (better for some models)
- `bidi`: Score in both directions and average (more accurate but slower)

```python
# Use bidirectional scoring for highest accuracy
bidi_filter = QualityEstimationFilter(
    model_name="comet-qe",
    cutoff=0.6,
    mode="bidi",  # Score in both directions
    src_field="source",
    tgt_field="target",
    metadata_fields=["src_lang", "tgt_lang"]
)
```

## Custom Bitext Filters

You can implement custom bitext filters for specific use cases:

### Custom Length Ratio Filter

Here's how you could implement a custom length ratio filter (note that NeMo Curator already provides a `LengthRatioFilter`):

```python
from nemo_curator.filters.bitext_filter import BitextFilter
import pandas as pd

class CustomLengthRatioFilter(BitextFilter):
    """Custom implementation showing length ratio filtering concept"""
    
    def __init__(self, min_ratio=0.5, max_ratio=2.0, **kwargs):
        super().__init__(**kwargs)
        self._min_ratio = min_ratio
        self._max_ratio = max_ratio
        
    def score_bitext(self, src, tgt, **kwargs):
        # Handle batched input
        if isinstance(src, pd.Series) and isinstance(tgt, pd.Series):
            return pd.Series([len(t) / len(s) if len(s) > 0 else float('inf') 
                             for s, t in zip(src, tgt)])
        # Handle single input
        return len(tgt) / len(src) if len(src) > 0 else float('inf')
        
    def keep_bitext(self, score):
        if isinstance(score, pd.Series):
            return (score >= self._min_ratio) & (score <= self._max_ratio)
        return self._min_ratio <= score <= self._max_ratio
```

### Custom Entity Preservation Filter

You can create custom filters to verify that key entities are preserved across translations:

```python
import re
from nemo_curator.filters.bitext_filter import BitextFilter
from nemo_curator.utils.decorators import batched

class EntityPreservationFilter(BitextFilter):
    """Custom filter to check if important entities are preserved in translation"""
    
    def __init__(self, min_entities_match=0.5, **kwargs):
        super().__init__(**kwargs)
        self._threshold = min_entities_match
        
    def score_bitext(self, src, tgt, **kwargs):
        # Extract entities like numbers, dates, and URLs
        src_entities = self._extract_entities(src)
        tgt_entities = self._extract_entities(tgt)
        
        # Calculate match ratio
        if len(src_entities) == 0:
            return 1.0
        
        matches = len(set(src_entities) & set(tgt_entities))
        return matches / len(src_entities)
    
    def keep_bitext(self, score):
        return score >= self._threshold
        
    def _extract_entities(self, text):
        # Simple entity extraction (numbers, URLs, etc.)
        if isinstance(text, str):
            numbers = re.findall(r'\d+', text)
            urls = re.findall(r'https?://\S+', text)
            return numbers + urls
        return []
```

## Combining Bitext Filters

You can combine multiple bitext filters in sequence:

```python
from nemo_curator.filters import QualityEstimationFilter, LengthRatioFilter

# Create a pipeline of bitext filters
quality_filter = QualityEstimationFilter(
    model_name="comet-qe",
    cutoff=0.5,
    src_field="source",
    tgt_field="target",
    metadata_fields=["src_lang", "tgt_lang"]
)

length_filter = LengthRatioFilter(
    max_ratio=2.0,
    src_lang="en",
    tgt_lang="de", 
    src_field="source",
    tgt_field="target"
)

# Apply filters sequentially
filtered_dataset = quality_filter(parallel_dataset)
filtered_dataset = length_filter(filtered_dataset)
```

## Filter Configuration

A typical YAML configuration for bitext filtering:

```yaml
filters:
  - name: QualityEstimationFilter
    model_name: comet-qe
    cutoff: 0.5
    mode: always_en_x
    gpu: true
    src_field: source
    tgt_field: target
    metadata_fields: [src_lang, tgt_lang]
    score_field: translation_quality
  
  - name: LengthRatioFilter
    max_ratio: 2.0
    src_lang: en
    tgt_lang: de
    src_field: source
    tgt_field: target
    score_field: length_ratio
```

## Best Practices

### Language-Specific Considerations

Different language pairs have different characteristics that affect filtering:

```python
# For English-Japanese translation
en_ja_filter = QualityEstimationFilter(
    model_name="comet-qe",
    cutoff=0.4,  # Lower threshold for challenging language pairs
    mode="always_en_x",
    src_field="source",
    tgt_field="target",
    metadata_fields=["src_lang", "tgt_lang"]
)

# For English-German translation
en_de_filter = QualityEstimationFilter(
    model_name="comet-qe",
    cutoff=0.6,  # Higher threshold for easier language pairs
    mode="always_en_x",
    src_field="source",
    tgt_field="target",
    metadata_fields=["src_lang", "tgt_lang"]
)
```

### Filter Tuning

When tuning bitext filters, consider these strategies:

1. **Start with conservative thresholds**: Begin with lower quality thresholds to avoid over-filtering
   ```python
   # Start with a more permissive threshold
   initial_filter = QualityEstimationFilter(
       model_name="comet-qe",
       cutoff=0.3  # Lower initial threshold
   )
   ```

2. **Analyze score distributions**: Understand the distribution of quality scores
   ```python
   # Apply filter without filtering, just to get scores
   scored_dataset = QualityEstimationFilter(
       model_name="comet-qe",
       cutoff=0.0,  # Keep all examples
       score_field="quality"
   )(parallel_dataset)
   
   # Analyze distribution
   import matplotlib.pyplot as plt
   plt.hist(scored_dataset.df["quality"], bins=50)
   plt.title("Translation Quality Distribution")
   plt.savefig("quality_hist.png")
   ```

3. **Consider language pair difficulty**: Use different thresholds for different language pairs
   ```python
   # Apply language-specific filtering
   def get_threshold(src_lang, tgt_lang):
       if src_lang == "en" and tgt_lang == "ja":
           return 0.4  # Lower threshold for English-Japanese
       elif src_lang == "en" and tgt_lang == "de":
           return 0.6  # Higher threshold for English-German
       return 0.5  # Default threshold
   
   # Filter with dynamic thresholds
   filtered_pairs = []
   for langs, subset in parallel_dataset.groupby(["src_lang", "tgt_lang"]):
       src_lang, tgt_lang = langs
       threshold = get_threshold(src_lang, tgt_lang)
       lang_filter = QualityEstimationFilter(
           model_name="comet-qe",
           cutoff=threshold
       )
       filtered_pairs.append(lang_filter(subset))
   
   # Combine results
   import pandas as pd
   combined = pd.concat([pair.df for pair in filtered_pairs])
   filtered_dataset = ParallelDataset(combined)
   ```

## Use Cases

::::{tab-set}

:::{tab-item} Cleaning Web-Mined Translations
```python
# Filter for web-mined translations
web_filter = QualityEstimationFilter(
    model_name="comet-qe",
    cutoff=0.6,  # Stricter threshold for noisy web data
    mode="always_en_x",
    src_field="source",
    tgt_field="target"
)

# Filter out likely incorrect translations
clean_translations = web_filter(web_mined_dataset)
```
:::

:::{tab-item} Preparing Data for Multilingual Models
```python
from nemo_curator.filters import QualityEstimationFilter, LengthRatioFilter

# Process each language pair with appropriate settings
filtered_pairs = {}

for lang_pair, subset in parallel_dataset.groupby(["src_lang", "tgt_lang"]):
    src_lang, tgt_lang = lang_pair
    
    # Apply QE filtering
    qe_filter = QualityEstimationFilter(
        model_name="comet-qe",
        cutoff=0.5,
        mode="always_en_x",
        src_field="source",
        tgt_field="target",
        metadata_fields=["src_lang", "tgt_lang"]
    )
    
    # Apply length ratio filtering
    length_filter = LengthRatioFilter(
        max_ratio=2.0,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        src_field="source",
        tgt_field="target"
    )
    
    # Apply filters sequentially
    filtered = qe_filter(subset)
    filtered = length_filter(filtered)
    
    filtered_pairs[lang_pair] = filtered

# Combine all filtered pairs
import pandas as pd
combined = pd.concat([pair.df for pair in filtered_pairs.values()])
multilingual_dataset = ParallelDataset(combined)
```
:::

::::

By applying these specialized bitext filters, you can significantly improve the quality of your parallel corpora, leading to better machine translation models and multilingual applications.
