---
description: "Identify and remove exact duplicates using MD5 hashing in a Ray-based workflow"
categories: ["how-to-guides"]
tags: ["exact-dedup", "hashing", "md5", "gpu", "ray"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-dedup-exact)=

# Exact Duplicate Removal

Remove character-for-character duplicate documents using NeMo Curator's exact duplicate removal workflow. This method computes MD5 hashes for each document's text and identifies documents with identical hashes as duplicates.

For an overview of all duplicate removal options, refer to {ref}`Deduplication <text-process-data-dedup>`.

## How It Works

Exact deduplication uses MD5 hashing to identify identical documents:

1. Computes MD5 hash for each document's text content
2. Groups documents by identical hash values
3. Identifies duplicates and saves IDs for removal

This method targets character-for-character duplicates and is recommended for removing exact copies of documents.

## Before You Start

**Prerequisites**:

- Ray cluster with GPU support (required for distributed processing)
- Stable document identifiers for removal (either existing IDs or IDs assigned by the workflow)

## Quick Start

Get started with exact deduplication using the following example of identifying duplicates, then remove them:

```python
from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow

ray_client = RayClient()
ray_client.start()

# Step 1: Identify duplicates
exact_workflow = ExactDeduplicationWorkflow(
    input_path="input_data/",
    output_path="./results",
    text_field="text",
    assign_id=True,
    perform_removal=False,
    input_filetype="parquet"
)
exact_workflow.run()
# Duplicate IDs saved to ./results/ExactDuplicateIds/

# Step 2: Remove duplicates
removal_workflow = TextDuplicatesRemovalWorkflow(
    input_path="input_data/",
    ids_to_remove_path="./results/ExactDuplicateIds",
    output_path="./deduplicated",
    input_filetype="parquet",
    id_field="_curator_dedup_id",
    duplicate_id_field="_curator_dedup_id",
    id_generator_path="./results/exact_id_generator.json"
)
removal_workflow.run()
# Clean dataset saved to ./deduplicated/
```

## Configuration

Configure exact deduplication using these key parameters:

```{list-table} Key Configuration Parameters
:header-rows: 1
:widths: 25 15 20 40

* - Parameter
  - Type
  - Default
  - Description
* - `input_path`
  - str | list[str]
  - None
  - Path(s) to input files or directories
* - `output_path`
  - str
  - Required
  - Directory to write duplicate IDs and ID generator
* - `text_field`
  - str
  - "text"
  - Name of the text field in input data
* - `assign_id`
  - bool
  - True
  - Whether to automatically assign unique IDs
* - `id_field`
  - str | None
  - None
  - Existing ID field name (if assign_id=False)
* - `input_filetype`
  - str
  - "parquet"
  - Input file format ("parquet" or "jsonl")
* - `input_blocksize`
  - str | int
  - "2GiB"
  - Size of input blocks for processing
* - `perform_removal`
  - bool
  - False
  - Reserved; must remain `False`. Exact removal is performed with `TextDuplicatesRemovalWorkflow`.
```

## Removing Duplicates

After identifying duplicates, use `TextDuplicatesRemovalWorkflow` to remove them:

```python
from nemo_curator.core.client import RayClient
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow

ray_client = RayClient()
ray_client.start()

removal_workflow = TextDuplicatesRemovalWorkflow(
    input_path="/path/to/input/data",
    ids_to_remove_path="/path/to/output/ExactDuplicateIds",
    output_path="/path/to/deduplicated",
    input_filetype="parquet",
    id_field="_curator_dedup_id",
    duplicate_id_field="_curator_dedup_id",
    id_generator_path="/path/to/output/exact_id_generator.json"  # Required if assign_id=True
)
removal_workflow.run()
```

:::{dropdown} ID Field Configuration
:icon: info

**When `assign_id=True`**:

- Duplicate IDs file contains `_curator_dedup_id` column
- Set `duplicate_id_field="_curator_dedup_id"`
- `id_generator_path` is required

**When `assign_id=False`**:

- Duplicate IDs file contains the column specified by `id_field` (e.g., `"id"`)
- Set `duplicate_id_field` to match your `id_field` value
- `id_generator_path` not required
:::

## Output Format

The exact deduplication process produces the following directory structure:

```s
output_path/
├── ExactDuplicateIds/              # Duplicate identification results
│   └── *.parquet                   # Parquet files with document IDs to remove
└── exact_id_generator.json         # ID generator mapping (if assign_id=True)
```

### File Formats

The workflow produces these output files:

1. **Duplicate IDs** (`ExactDuplicateIds/*.parquet`):
   - Contains document IDs to remove
   - Format: Parquet files with a single ID column
   - Column name depends on `assign_id`:
     - When `assign_id=True`: Column is `"_curator_dedup_id"`
     - When `assign_id=False`: Column matches the `id_field` parameter (e.g., `"id"`)
   - **Important**: Contains only the IDs of documents to remove, not the full document content

2. **ID Generator** (`exact_id_generator.json`):
   - JSON file containing ID generator state
   - Required for removal workflow when `assign_id=True`
   - Ensures consistent ID mapping across workflow stages

:::{dropdown} Performance Considerations
:icon: zap

**Performance characteristics**:

- Uses MD5 hashing over the configured text field to derive duplicate groups
- Runs as a Ray-based workflow and writes duplicate IDs to the `ExactDuplicateIds/` directory
- Stores only document IDs to remove in the output files, not full document content

**Best practices**:

- Use `input_blocksize="2GiB"` for optimal performance
- Clear output directory between runs
- Use `assign_id=True` for consistent ID tracking
:::


For comparison with other deduplication methods and guidance on when to use exact deduplication, refer to the {ref}`Deduplication overview <text-process-data-dedup>`.
