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

Generate unique document identifiers for tracking and duplicate removal workflows.

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

Minimal configuration for adding IDs:

```python
from nemo_curator.stages.text.modules import AddId

pipeline.add_stage(AddId(id_field="doc_id"))
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

### ID Generation Format

Generated IDs follow this pattern:

- Without prefix: `{task_uuid}_{sequential_index}`
- With prefix: `{id_prefix}_{task_uuid}_{sequential_index}`

Example:

```text
a1b2c3d4-e5f6-7890-abcd-ef1234567890_0
corpus_v1_a1b2c3d4-e5f6-7890-abcd-ef1234567890_1
```

### Integration with Duplicate Removal

Use `AddId` before duplicate removal workflows that require document identifiers. For exact duplicate removal, NeMo Curator provides a workflow that can automatically assign IDs, or you can add them explicitly with `AddId`:

```python
from nemo_curator.stages.text.modules import AddId
from nemo_curator.stages.text.deduplication.removal import TextDuplicatesRemovalStage

# Add IDs before duplicate identification/removal
pipeline.add_stage(
    AddId(id_field="doc_id")
)

# Later, when applying a removal list of duplicate IDs (written by a prior
# identification workflow), use the TextDuplicatesRemovalStage:
pipeline.add_stage(
    TextDuplicatesRemovalStage(
        ids_to_remove_path="/path/to/duplicate_ids.parquet",
        id_field="doc_id",
        duplicate_id_field="id"
    )
)
```

See also: {ref}`text-process-data-dedup`.

You can also use the exact duplicate workflow, which can assign IDs for you:

```python
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow

exact_workflow = ExactDeduplicationWorkflow(
    input_path="/path/to/input",
    output_path="/path/to/output",
    text_field="text",
    assign_id=True,   # Automatically assign unique IDs if not present
    id_field="doc_id"
)

exact_workflow.run()
```

### Advanced: Using Reader-Based ID Generation (ID Generator)

NeMo Curator readers can generate or assign monotonically increasing IDs during load using an ID Generator actor. This is useful when preparing datasets for duplicate removal workflows that require stable integer IDs.

```python
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.deduplication.id_generator import create_id_generator_actor, write_id_generator_to_disk

# Start a detached ID Generator actor (optionally from a saved state)
create_id_generator_actor()  # or create_id_generator_actor(filepath="/path/to/id_state.json")

# Configure readers to generate or assign IDs
jsonl_reader = JsonlReader(
    file_paths="/data/*.jsonl",
    _generate_ids=True  # set _assign_ids=True to reuse previously generated ranges
)

# ... run your pipeline that uses jsonl_reader ...

# Optionally persist the IdGenerator state for later reuse
write_id_generator_to_disk("/path/to/id_state.json")
```

Notes:

- `JsonlReader` and `ParquetReader` support `_generate_ids` (create new IDs) and `_assign_ids` (reassign using saved state).
- These flags add the `_curator_dedup_id` field to outputs. Duplicate-removal stages depend on this field.
- Refer to the step-by-step semantic duplicate removal tutorial for end-to-end usage.

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
pipeline = create_id_pipeline("./input/*.jsonl", "./output/")
result = pipeline.run()
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
    
    pipeline.run(executor)
```
