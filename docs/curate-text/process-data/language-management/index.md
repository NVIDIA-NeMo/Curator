---
description: "Handle multilingual content and language-specific processing including language identification and stop word management"
categories: ["workflows"]
tags: ["language-management", "multilingual", "fasttext", "stop-words", "language-detection"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "text-only"
---

(text-process-data-languages)=
# Language Management

Handle multilingual content and language-specific processing requirements using NeMo Curator's tools and utilities.

NeMo Curator provides robust tools for managing multilingual text datasets through language detection, stop word management, and specialized handling for non-spaced languages. These tools are essential for creating high-quality monolingual datasets and applying language-specific processing.

## How it Works

Language management in NeMo Curator typically follows this pattern:

```python
import ray
from ray_curator.stages.filters.fasttext_filter import FastTextLangId
from ray_curator.backends.experimental.ray_data import RayDataExecutor

# Initialize Ray
ray.init()

# Load your dataset with Ray
dataset = ray.data.read_json("input_data/*.jsonl")

# Identify languages using FastText with Ray
lang_filter = FastTextLangId(
    model_path="lid.176.bin",
    min_langid_score=0.8
)

# Apply language identification using Ray backend
executor = RayDataExecutor()
dataset = executor.run_filter(dataset, lang_filter, text_field="text")

# Apply language-specific processing
# Group by language and process each group
def process_by_language(dataset):
    if lang in ["zh", "ja", "th", "ko"]:
        # Special handling for non-spaced languages
        processor = get_word_splitter(lang)
        # Note: Word splitter returns a function for processing text
        # Apply as needed for your specific use case
    
    # Apply language-specific stop words
    stop_filter = ScoreFilter(
        HistogramFilter(lang=lang),
        text_field="text",
        score_field="quality"
    )
    subset = stop_filter(subset)
```

---

## Language Processing Capabilities

- **Language detection** using FastText and CLD2 (176+ languages)
- **Stop word management** with built-in lists and customizable thresholds
- **Special handling** for non-spaced languages (Chinese, Japanese, Thai, Korean)
- **Language-specific** text processing and quality filtering

## Available Tools

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Language Identification
:link: language
:link-type: doc
Identify document languages and separate multilingual datasets
+++
{bdg-secondary}`fasttext`
{bdg-secondary}`176-languages`
{bdg-secondary}`detection`
{bdg-secondary}`classification`
:::

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Stop Words
:link: stopwords
:link-type: doc
Manage high-frequency words to enhance text extraction and content detection
+++
{bdg-secondary}`preprocessing`
{bdg-secondary}`filtering`
{bdg-secondary}`language-specific`
{bdg-secondary}`nlp`
:::

::::

## Usage

### Quick Start Example

```python
import ray
from ray_curator.stages.filters.fasttext_filter import FastTextLangId
from ray_curator.backends.experimental.ray_data import RayDataExecutor

# Initialize Ray and load multilingual dataset
ray.init()
dataset = ray.data.read_json("multilingual_data/*.jsonl")

# Identify languages
langid_filter = FastTextLangId(
    model_path="/path/to/lid.176.bin",  # Download from fasttext.cc
    min_langid_score=0.3
)

# Apply language identification using Ray
executor = RayDataExecutor()
identified_dataset = executor.run_filter(dataset, langid_filter, text_field="text")

# Extract language codes
def extract_language_code(batch):
    import ast
    batch["language"] = [ast.literal_eval(score)[1] for score in batch["language"]]
    return batch

identified_dataset = identified_dataset.map_batches(extract_language_code)

# Filter for specific languages
english_docs = identified_dataset.filter(lambda row: row["language"] == "EN")
spanish_docs = identified_dataset.filter(lambda row: row["language"] == "ES")

# Save by language
english_docs.write_json("output/english/")
spanish_docs.write_json("output/spanish/")
```

### Supported Languages

NeMo Curator supports language identification for **176 languages** through FastText, including:

- **Major languages**: English, Spanish, French, German, Chinese, Japanese, Arabic, Russian
- **Regional languages**: Many local and regional languages worldwide
- **Special handling**: Non-spaced languages (Chinese, Japanese, Thai, Korean)

## Best Practices

1. **Download the FastText model**: Get `lid.176.bin` from [fasttext.cc](https://fasttext.cc/docs/en/language-identification.html)
2. **Set appropriate thresholds**: Balance precision vs. recall based on your needs
3. **Handle non-spaced languages**: Use special processing for Chinese, Japanese, Thai, Korean
4. **Check on your domain**: Test language detection accuracy on your specific data

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Language Identification <language>
Stop Words <stopwords>
``` 