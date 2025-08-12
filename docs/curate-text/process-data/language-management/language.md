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

## How it Works

NeMo Curator's language identification system works through a three-step process:

1. **Text Preprocessing**: The system normalizes input text by stripping whitespace and converting newlines to spaces to prepare it for fastText analysis.

2. **FastText Language Detection**: The pre-trained fastText language identification model (`lid.176.bin`) analyzes the preprocessed text and returns:
   - A confidence score (0.0 to 1.0) indicating certainty of the prediction
   - A language code (e.g., "EN", "ES", "FR") in fastText's two-letter uppercase format

3. **Filtering and Scoring**: Documents are filtered based on a configurable confidence threshold (`min_langid_score`), with results stored as metadata containing both the confidence score and language code.

### Language Detection Process

The `FastTextLangId` filter implements this workflow by:

- Loading the fastText language identification model on worker initialization
- Processing text through `model.predict()` with `k=1` to get the top language prediction
- Extracting the language code from fastText labels (e.g., `__label__en` becomes "EN")
- Comparing confidence scores against the threshold to determine document retention
- Returning results as `[confidence_score, language_code]` for downstream processing

This approach supports **176 languages** with high accuracy, making it suitable for large-scale multilingual dataset curation where language-specific processing and monolingual dataset creation are critical.

## Before You Start

- Language identification requires NeMo Curator with distributed backend support. For installation instructions, see the {ref}`admin-installation` guide.

---

## Usage

The following example demonstrates how to create a language identification pipeline using `ray_curator` with distributed processing on a cluster.

::::{tab-set}

:::{tab-item} Python

```python
"""Language identification using ray_curator."""

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.filters.classifier_filter import FastTextLangId
from ray_curator.stages.io.reader.jsonl import JsonlReader
from ray_curator.stages.modules.filter import ScoreFilter

def create_language_identification_pipeline(data_dir: str) -> Pipeline:
    """Create a pipeline for language identification."""
    
    # Define pipeline
    pipeline = Pipeline(
        name="language_identification", 
        description="Identify document languages using FastText"
    )
    
    # Add stages
    # 1. Reader stage - creates tasks from JSONL files
    pipeline.add_stage(
        JsonlReader(
            file_paths=data_dir,
            files_per_partition=2,  # Each task processes 2 files
            reader="pandas"
        )
    )
    
    # 2. Language identification with filtering
    # IMPORTANT: Download lid.176.bin or lid.176.ftz from https://fasttext.cc/docs/en/language-identification.html
    fasttext_model_path = "/path/to/lid.176.bin"  # or lid.176.ftz (compressed)
    pipeline.add_stage(
        ScoreFilter(
            FastTextLangId(model_path=fasttext_model_path, min_langid_score=0.3), 
            score_field="language"
        )
    )
    
    return pipeline

def main():
    # Create pipeline
    pipeline = create_language_identification_pipeline("./data")
    
    # Print pipeline description
    print(pipeline.describe())
    
    # Create executor and run
    executor = XennaExecutor()
    results = pipeline.run(executor)
    
    # Process results
    print(f"Pipeline completed! Processed {len(results)} batches")
    
    total_documents = sum(task.num_items for task in results) if results else 0
    print(f"Total documents processed: {total_documents}")
    
    # Access language scores
    for i, batch in enumerate(results):
        if batch.num_items > 0:
            df = batch.to_pandas()
            print(f"Batch {i} columns: {list(df.columns)}")
            # Language scores are now in the 'language' field

if __name__ == "__main__":
    main()
```

:::
::::

## Configuration

Currently, `ray_curator` language identification is configured programmatically using the Python API as shown in the previous section. 

:::{note}
YAML pipeline configuration support for `ray_curator` is planned for future releases but not yet available. Use the programmatic Python API for now.
:::

## Understanding Results

The language identification process adds a score field to each document batch:

1. **`language` field**: Contains the FastText language identification results as a list with two elements:
   - Element 0: The confidence score (between 0 and 1)
   - Element 1: The language code in fastText format (for example, "EN" for English, "ES" for Spanish)

2. **Task-based processing**: `ray_curator` processes documents in batches (tasks), and results are available through the task's pandas DataFrame:

```python
# Access results from pipeline execution
for batch in results:
    df = batch.to_pandas()
    # Language scores are in the 'language' column
    print(df[['text', 'language']].head())
```

:::{note}
FastText language codes are typically two-letter uppercase codes that may differ slightly from standard ISO 639-1 codes. The model supports 176 languages with high accuracy.
:::

### Processing Language Results

You can extract and work with language identification results:

```python
# Extract language codes from results
for batch in results:
    df = batch.to_pandas()
    if 'language' in df.columns:
        # Parse the [score, language_code] format
        df['lang_score'] = df['language'].apply(lambda x: eval(x)[0] if isinstance(x, str) else x[0])
        df['lang_code'] = df['language'].apply(lambda x: eval(x)[1] if isinstance(x, str) else x[1])
        
        # Filter by confidence threshold
        high_confidence = df[df['lang_score'] > 0.7]
        print(f"High confidence documents: {len(high_confidence)}")
```

A higher confidence score indicates greater certainty in the language identification. The `ScoreFilter` automatically filters documents below your specified `min_langid_score` threshold.

## Performance Considerations

- Language identification with `ray_curator` is computationally intensive but highly scalable with Ray's distributed processing
- `ray_curator` automatically handles distributed execution across cluster nodes using the XennaExecutor backend
- The fastText model file (`lid.176.bin` or compressed `lid.176.ftz`) must be accessible to all worker nodes
- Processing speed depends on document length, cluster size, available computational resources, and the `files_per_partition` setting
- Memory usage scales with cluster configuration and task batch sizes
- ray_curator provides efficient resource allocation and autoscaling for large-scale processing
- **Task-based processing**: Documents are grouped into tasks based on the `JsonlReader` configuration, enabling parallel processing across the cluster

## Best Practices

:::{important}
**Model Download Required**: Download the fastText language identification model from the [official fastText repository](https://fasttext.cc/docs/en/language-identification.html) before using this filter. Available formats:
- `lid.176.bin` (full model, ~130MB)
- `lid.176.ftz` (compressed model, smaller file size)
:::

### `ray_curator` Specific Recommendations

- **Task partitioning**: Adjust `files_per_partition` in `JsonlReader` based on your data size and cluster capacity. Smaller values create more tasks for better parallelization
- **Pipeline design**: Use `ScoreFilter` for combined scoring and filtering, or separate `Score` and `Filter` stages for more control
- **Model accessibility**: Ensure the fastText model file is accessible to all worker nodes in your Ray cluster
- **Resource allocation**: Configure appropriate resources for the `FastTextLangId` stage based on your model size and processing requirements

### Confidence Threshold Guidelines

- **Default threshold (0.3)**: Balanced approach suitable for most use cases
- **Higher threshold (0.7+)**: More precision but may discard borderline documents
- **Lower threshold (0.1-0.2)**: Higher recall but may include misclassified documents

### Production Workflow Tips

- Analyze the language distribution in your dataset using the pipeline results to understand composition
- Consider a two-pass approach: first filter with a lower threshold, then manually review edge cases
- For production workflows, validate language identification accuracy on a sample of your specific domain data
- Monitor task completion and resource usage through Ray's monitoring capabilities
- Save pipeline results to Parquet format for efficient downstream processing
