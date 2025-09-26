---
description: "Clean, normalize, and transform text content to meet specific requirements including PII removal and text cleaning"
categories: ["workflows"]
tags: ["content-processing", "text-cleaning", "pii-removal", "unicode", "normalization"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "text-only"
---

(text-process-data-format)=
# Content Processing & Cleaning

Clean, normalize, and transform text content to meet specific requirements for training language models using NeMo Curator's tools and utilities.

Content processing involves transforming your text data while preserving essential information. This includes fixing encoding issues, removing sensitive information, and standardizing text format to ensure high-quality input for model training.

## How it Works

Content processing transformations typically modify documents in place or create new versions with specific changes. Most processing tools follow this pattern:

1. Load your dataset using `DocumentDataset`
2. Configure and apply the appropriate processor
3. Save the transformed dataset for further processing

You can combine processing tools in sequence or use them alongside other curation steps like filtering and language management.

---

## Available Processing Tools

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`number;1.5em;sd-mr-1` Document IDs
:link: add-id
:link-type: doc
Add unique identifiers to documents for tracking and deduplication
+++
{bdg-secondary}`identifiers`
{bdg-secondary}`tracking`
{bdg-secondary}`preprocessing`
{bdg-secondary}`deduplication`
:::

:::{grid-item-card} {octicon}`shield-lock;1.5em;sd-mr-1` PII Removal
:link: pii
:link-type: doc
Identify and remove personal identifiable information from text
+++
{bdg-secondary}`privacy`
{bdg-secondary}`regex`
{bdg-secondary}`masking`
{bdg-secondary}`compliance`
:::

:::{grid-item-card} {octicon}`typography;1.5em;sd-mr-1` Text Cleaning
:link: text-cleaning
:link-type: doc
Fix Unicode issues, standardize spacing, and remove URLs
+++
{bdg-secondary}`unicode`
{bdg-secondary}`normalization`
{bdg-secondary}`preprocessing`
{bdg-secondary}`urls`
:::

::::

## Usage

Here's an example of a typical content processing pipeline:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modifiers import UnicodeReformatter, UrlRemover, NewlineNormalizer
from nemo_curator.stages.text.modules import Modify

# Create a comprehensive cleaning pipeline
processing_pipeline = Pipeline(
    name="content_processing_pipeline",
    description="Comprehensive text cleaning and processing"
)

# Load dataset
reader = JsonlReader(file_paths="input_data/*.jsonl")
processing_pipeline.add_stage(reader)

# Fix Unicode encoding issues
processing_pipeline.add_stage(
    Modify(modifier=UnicodeReformatter(), text_field="text")
)

# Standardize newlines
processing_pipeline.add_stage(
    Modify(modifier=NewlineNormalizer(), text_field="text")
)

# Remove URLs
processing_pipeline.add_stage(
    Modify(modifier=UrlRemover(), text_field="text")
)

# Save the processed dataset
writer = JsonlWriter(path="processed_output/")
processing_pipeline.add_stage(writer)

# Execute pipeline
results = processing_pipeline.run()
```

## Common Processing Tasks

### Text Normalization
- Fix broken Unicode characters (mojibake)
- Standardize whitespace and newlines
- Remove or normalize special characters

### Content Sanitization
- Remove personally identifiable information (PII)
- Strip unwanted URLs or links
- Remove boilerplate text or headers

### Format Standardization
- Ensure consistent text encoding
- Normalize punctuation and spacing
- Standardize document structure

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Document IDs <add-id>
PII Removal <pii>
Text Cleaning <text-cleaning>
```
