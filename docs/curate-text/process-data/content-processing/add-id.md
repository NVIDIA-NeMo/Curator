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

Add unique identifiers to each document in your text dataset for tracking and deduplication workflows.

## How It Works

Document IDs are required for:
- **Deduplication workflows** - Track which documents are duplicates
- **Pipeline tracking** - Monitor documents through processing stages
- **Dataset versioning** - Identify documents across different versions

---

## Usage

### Basic Usage

```python
from nemo_curator.stages.text.modules import AddId

# Add to your pipeline
pipeline.add_stage(AddId(id_field="doc_id"))
```

### Configuration Options

```python
# Customize ID generation
pipeline.add_stage(AddId(
    id_field="document_id",        # Field name for IDs
    id_prefix="corpus_v2",         # Optional prefix
    overwrite=True                 # Overwrite existing IDs
))
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id_field` | `str` | Required | Field name where IDs will be stored |
| `id_prefix` | `str` | `None` | Optional prefix for IDs |
| `overwrite` | `bool` | `False` | Whether to overwrite existing ID fields |

#### ID Format

Generated IDs follow this pattern:
- Without prefix: `{task_uuid}_{index}`
- With prefix: `{prefix}_{task_uuid}_{index}`

**Examples:**
```text
a1b2c3d4-e5f6-7890-abcd-ef1234567890_0
corpus_v1_a1b2c3d4-e5f6-7890-abcd-ef1234567890_1
```

### Complete Example

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.modules import AddId
from nemo_curator.stages.text.io.writer import JsonlWriter

# Create pipeline
pipeline = Pipeline(name="add_ids")

# Add stages
pipeline.add_stage(JsonlReader(file_paths="input/*.jsonl"))
pipeline.add_stage(AddId(id_field="doc_id", id_prefix="v1"))
pipeline.add_stage(JsonlWriter("output/"))

# Run pipeline
result = pipeline.run()
```

### Alternative: Reader-Based ID Generation

For deduplication workflows, you can generate IDs during data loading:

```python
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.deduplication.id_generator import create_id_generator_actor

# Create ID generator
create_id_generator_actor()

# Reader generates IDs automatically
reader = JsonlReader(
    file_paths="data/*.jsonl",
    _generate_ids=True  # Adds '_curator_dedup_id' field
)
```

This approach:
- Generates monotonically increasing integer IDs
- Required for some deduplication workflows
- Persists ID state across pipeline runs

---

## Error Handling

**Existing ID field:**
```python
# This raises ValueError if 'doc_id' already exists
AddId(id_field="doc_id", overwrite=False)

# This overwrites existing field with warning
AddId(id_field="doc_id", overwrite=True)
```

---

## Best Practices

- **Place early in pipeline** - Add IDs after loading, before filtering
- **Use descriptive field names** - `doc_id`, `document_id`, `unique_id`
- **Choose appropriate method**:
  - Use `AddId` for general document tracking
  - Use reader-based generation for deduplication workflows

---

For deduplication workflows, see {ref}`text-process-data-dedup`.