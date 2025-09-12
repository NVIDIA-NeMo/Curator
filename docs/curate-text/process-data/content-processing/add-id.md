---
description: "Add unique identifiers to documents in your text dataset for tracking and deduplication workflows"
categories: ["text-curation"]
tags: ["preprocessing", "identifiers", "document-tracking", "pipeline"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-add-id)=

# Adding Document IDs

The `AddId` module provides a reliable way to add unique identifiers to each document in your text dataset. These IDs are essential for tracking documents through processing pipelines and duplicate removal workflows require them.

## Overview

The `AddId` stage generates unique identifiers by combining a task UUID with sequential indices, ensuring uniqueness across distributed processing environments. This approach guarantees that each document receives a distinct ID even when processing large datasets across distributed workers.

### Key Features

- **Guaranteed Uniqueness**: Combines task universally unique identifiers with sequential indices to prevent ID collisions
- **Configurable Field Names**: Specify custom field names for storing generated IDs
- **Optional Prefixes**: Add custom prefixes to make IDs more meaningful
- **Overwrite Protection**: Prevents accidental overwriting of existing ID fields
- **Distributed Processing**: Works seamlessly in distributed environments

## Before You Start

- Ensure you have NeMo Curator installed with Ray backend support
- Load your dataset as `DocumentBatch` objects in the processing pipeline
- Consider whether you need custom prefixes for your use case

---

## Usage

### Basic Usage

Add unique IDs to documents using default settings:

```python
from nemo_curator.stages.text.modules import AddId
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader

# Create pipeline
pipeline = Pipeline(
    name="add_document_ids",
    description="Add unique IDs to documents"
)

# Add reader stage
pipeline.add_stage(
    JsonlReader(
        file_paths="./data/*.jsonl",
        files_per_partition=2
    )
)

# Add ID generation stage
pipeline.add_stage(
    AddId(
        id_field="doc_id"  # Field name where IDs will be stored
    )
)
```

### Advanced Configuration

Customize ID generation with prefixes and overwrite behavior:

```python
from nemo_curator.stages.text.modules import AddId

# Configure AddId with custom settings
add_id_stage = AddId(
    id_field="document_id",        # Custom field name
    id_prefix="corpus_v2",         # Add meaningful prefix
    overwrite=True                 # Allow overwriting existing IDs
)

pipeline.add_stage(add_id_stage)
```

### Integration with Duplicate Removal

Use `AddId` before duplicate removal workflows that require document identifiers:

```python
from nemo_curator.stages.text.modules import AddId
from nemo_curator.stages.text.filters import ExactDuplicateFilter

# Add IDs before duplicate removal
pipeline.add_stage(
    AddId(id_field="doc_id")
)

# Use IDs in duplicate removal
pipeline.add_stage(
    ExactDuplicateFilter(
        id_field="doc_id",
        text_field="text"
    )
)
```

---

## Configuration Parameters

### Constructor Parameters

```{list-table} AddId Parameters
:header-rows: 1
:widths: 25 15 15 45

* - Parameter
  - Type
  - Default
  - Description
* - `id_field`
  - `str`
  - Required
  - Field name where generated IDs will be stored
* - `id_prefix`
  - `str | None`
  - `None`
  - Optional prefix to add to generated IDs
* - `overwrite`
  - `bool`
  - `False`
  - Whether to overwrite existing ID fields
```

### ID Generation Format

Generated IDs follow this pattern:

- **Without prefix**: `{task_uuid}_{sequential_index}`
- **With prefix**: `{id_prefix}_{task_uuid}_{sequential_index}`

**Example IDs:**

```text
# Without prefix
a1b2c3d4-e5f6-7890-abcd-ef1234567890_0
a1b2c3d4-e5f6-7890-abcd-ef1234567890_1

# With prefix "corpus_v1"
corpus_v1_a1b2c3d4-e5f6-7890-abcd-ef1234567890_0
corpus_v1_a1b2c3d4-e5f6-7890-abcd-ef1234567890_1
```

---

## Input and Output

### Input Requirements

- **Data Format**: `DocumentBatch` containing text documents
- **Required Fields**: None (works with any document structure)
- **Optional Fields**: Existing ID field (if `overwrite=True`)

### Output Format

The stage adds a new field containing unique identifiers:

```json
{
  "text": "Sample document content...",
  "doc_id": "corpus_v1_a1b2c3d4-e5f6-7890-abcd-ef1234567890_0",
  "other_field": "existing data preserved"
}
```

---

## Error Handling

### Existing ID Fields

By default, `AddId` prevents overwriting existing ID fields:

```python
# This will raise ValueError if 'doc_id' already exists
add_id = AddId(id_field="doc_id", overwrite=False)

# This will overwrite existing 'doc_id' field with warning
add_id = AddId(id_field="doc_id", overwrite=True)
```

### Common Error Messages

```{list-table} Error Scenarios
:header-rows: 1
:widths: 40 60

* - Error
  - Solution
* - `Column 'doc_id' already exists`
  - Set `overwrite=True` or choose different `id_field`
* - `ValueError: Column name required`
  - Provide valid `id_field` parameter
```

---

## Best Practices

### Field Naming

- Use descriptive field names: `doc_id`, `document_id`, `unique_id`
- Avoid conflicts with existing fields unless intentional
- Consider downstream processing requirements

### Prefix Usage

- Use prefixes to identify dataset versions: `v1_`, `v2_`
- Include corpus names for multi-source datasets: `wiki_`, `news_`
- Keep prefixes short to reduce storage overhead

### Pipeline Placement

- **Initial Stage**: Add IDs at the beginning of pipeline for consistent tracking
- **Before Duplicate Removal**: Required for most duplicate removal workflows
- **After Loading**: Place after data loading but before filtering

### Performance Considerations

- ID generation is lightweight and adds minimal processing overhead
- Universally unique identifiers ensure uniqueness but increase storage requirements
- Consider ID field data types in downstream processing

---

## Examples

### Complete Pipeline Example

```python
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.modules import AddId
from nemo_curator.stages.text.io.writer import JsonlWriter

def create_id_pipeline(input_path: str, output_path: str) -> Pipeline:
    """Create pipeline to add IDs to documents."""
    
    pipeline = Pipeline(
        name="document_id_pipeline",
        description="Add unique IDs to text documents"
    )
    
    # Load documents
    pipeline.add_stage(
        JsonlReader(
            file_paths=input_path,
            files_per_partition=4
        )
    )
    
    # Add unique IDs
    pipeline.add_stage(
        AddId(
            id_field="doc_id",
            id_prefix="dataset_v1"
        )
    )
    
    # Save results
    pipeline.add_stage(
        JsonlWriter(
            output_path=output_path
        )
    )
    
    return pipeline

# Execute pipeline
executor = XennaExecutor()
pipeline = create_id_pipeline("./input/*.jsonl", "./output/")
result = executor.run(pipeline)
```

### Batch Processing Example

```python
from nemo_curator.stages.text.modules import AddId

# Process multiple datasets with consistent ID prefixes
datasets = [
    ("news_data/*.jsonl", "news"),
    ("wiki_data/*.jsonl", "wiki"),
    ("books_data/*.jsonl", "books")
]

for input_path, prefix in datasets:
    pipeline = Pipeline(name=f"{prefix}_id_pipeline")
    
    pipeline.add_stage(JsonlReader(file_paths=input_path))
    pipeline.add_stage(AddId(id_field="doc_id", id_prefix=prefix))
    pipeline.add_stage(JsonlWriter(output_path=f"./output/{prefix}/"))
    
    executor.run(pipeline)
```

---

## Related Topics

- {ref}`text-process-data-deduplication-exact` - Exact duplicate removal requiring document IDs
- {ref}`text-process-data-deduplication-semantic` - Semantic duplicate removal workflows
- {ref}`text-process-data-quality-heuristic` - Quality filtering that benefits from document tracking
- {ref}`about-concepts-text-document-batch` - Understanding DocumentBatch data structures
