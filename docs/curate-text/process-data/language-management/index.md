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

Identify document languages, filter multilingual content, and apply language-specific processing to create high-quality monolingual or multilingual text datasets.

## Overview

NeMo Curator provides robust tools for managing multilingual text datasets through language detection, stop word management, and specialized handling for language-specific requirements. These capabilities are essential for:

- **Monolingual Dataset Creation**: Filter documents by language to create single-language training datasets
- **Multilingual Dataset Curation**: Identify and tag languages for balanced multilingual corpora
- **Quality Filtering**: Apply language-specific quality checks and stop word filtering
- **Non-Spaced Language Support**: Handle Chinese, Japanese, Thai, and Korean text with specialized tokenization

## Language Processing Capabilities

### Language Detection

- **FastText Model**: Supports 176 languages with confidence scores
- **CLD2 Integration**: Used automatically in Common Crawl text extraction pipeline
- **Configurable Thresholds**: Filter documents by minimum confidence scores

### Stop Word Management

- **Built-in Stop Word Lists**: Pre-configured lists for common languages
- **Customizable Filtering**: Adjust thresholds for stop word density
- **Content Quality Enhancement**: Remove low-information documents

### Special Language Handling

- **Non-Spaced Languages**: Specialized tokenization for Chinese, Japanese, Thai, Korean
- **Script Detection**: Identify and process different writing systems
- **Language-Specific Processing**: Apply custom rules per language

## Prerequisites

Before implementing language management in your pipeline:

### Required Resources

* **FastText Model File**: Download the language identification model
  - Model options: `lid.176.bin` (full model, ~131MB) or `lid.176.ftz` (compressed model, ~917KB)
  - Download from: [FastText Language Identification](https://fasttext.cc/docs/en/language-identification.html)
  - Save to an accessible location (local path or shared storage)

* **Data Format**: JSONL (JSON Lines) input with text content
  - Default field name: `text`
  - Custom field support: Specify with `text_field` parameter

* **Cluster Setup** (if applicable):
  - Ensure FastText model file is accessible to all workers
  - Use shared filesystem, network storage, or object storage (S3, GCS, etc.)

### Installation Dependencies

## Basic Language Filtering

### Quick Start Example

Filter documents by language using FastText:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import FastTextLangId

# Create language filtering pipeline
pipeline = Pipeline(name="language_filtering")

# 1. Read JSONL input files
pipeline.add_stage(
    JsonlReader(
        file_paths="input_data/",
        files_per_partition=2  # Process 2 files per partition
    )
)

# 2. Identify languages and filter by confidence threshold
pipeline.add_stage(
    ScoreFilter(
        FastTextLangId(
            model_path="/path/to/lid.176.bin",  # Path to FastText model
            min_langid_score=0.3                # Minimum confidence (0.0-1.0)
        ),
        score_field="language"  # Output field for language code
    )
)

# 3. Write filtered results
pipeline.add_stage(
    JsonlWriter(path="output_filtered/")
)

# Execute pipeline (uses XennaExecutor by default)
results = pipeline.run()
```
**Parameters explained:**
- `model_path`: Absolute path to FastText model file (`lid.176.bin` or `lid.176.ftz`)
- `min_langid_score`: Minimum confidence score (0.0 to 1.0). Documents below this threshold are filtered out
- `score_field`: Field name to store detected language code (e.g., "en", "es", "zh")
- `files_per_partition`: Number of files to process per partition (tune based on file sizes)

**Output format:**
Each document will include a `language` field with the detected language code:
```json
{"text": "This is an English document.", "language": "en"}
{"text": "Este es un documento en español.", "language": "es"}
```
## Integration with HTML Extraction

When processing HTML content (e.g., Common Crawl), CLD2 provides language hints automatically:

```python
from nemo_curator.stages.text.download import CommonCrawlDownloadExtractStage

# HTML extraction automatically uses CLD2 for language hints
pipeline.add_stage(CommonCrawlWarcDownloader(...))

# Additional FastText filtering for refined language detection
pipeline.add_stage(
    ScoreFilter(
        FastTextLangId(model_path="/path/to/lid.176.bin", min_langid_score=0.5),
        score_field="language"
    )
)
```

**CLD2 vs FastText:**
- **CLD2**: Fast, lightweight, used for initial hints during HTML extraction
- **FastText**: More accurate, supports 176 languages, recommended for final filtering

## Complete Language Management Example

Here's a comprehensive pipeline demonstrating language detection and filtering:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import FastTextLangId
from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.tasks import DocumentBatch

# Create comprehensive language management pipeline
pipeline = Pipeline(name="language_management_complete")

# 1. Load input data
pipeline.add_stage(
    JsonlReader(file_paths="raw_data/", files_per_partition=4)
)

# 2. Detect languages with FastText
pipeline.add_stage(
    ScoreFilter(
        FastTextLangId(
            model_path="/models/lid.176.bin",
            min_langid_score=0.6  # Medium-high confidence
        ),
        score_field="language"
    )
)

# 3. Filter to English documents only
@processing_stage(name="keep_english")
@processing_stage(name="keep_english")
def filter_english(batch: DocumentBatch) -> DocumentBatch:
    import ast
    df = batch.data
    parsed = df["language"].apply(lambda v: ast.literal_eval(v) if isinstance(v, str) else v)
    df["lang_code"] = parsed.apply(lambda p: str(p[1]))
    df = df[df['lang_code'] == 'EN']
    return DocumentBatch(data=df, task_id=batch.task_id, dataset_name=batch.dataset_name)

pipeline.add_stage(filter_english)

# 4. Export filtered, high-quality English documents
pipeline.add_stage(JsonlWriter(path="curated_english/"))

# Execute pipeline (uses XennaExecutor by default)
results = pipeline.run()
print("Language management pipeline completed!")
```

**Expected workflow:**
1. Load multilingual JSONL documents
2. Detect language with 60% minimum confidence
3. Keep only English documents
4. Export high-quality English dataset

## Troubleshooting

### Common Issues

**FastText model not found:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/lid.176.bin'
```
**Solution:** Download the model from [FastText Language Identification](https://fasttext.cc/docs/en/language-identification.html) and provide the correct absolute path.

**Low detection accuracy:**
```
Many documents classified incorrectly
```
**Solution:** 
- Increase `min_langid_score` to filter low-confidence predictions
- Ensure input text is clean (remove HTML tags, special characters)
- Check for very short documents (<50 chars) which are harder to classify

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

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Language Identification <language>
Stop Words <stopwords>
```
