---
description: "Identify exact duplicates using MD5 hashing with distributed GPU acceleration"
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

---

## How It Works

1. File partitioning for scalable, distributed processing
2. MD5 hashing of the configured `text_field`
3. Identification of duplicate groups by hash equality

---

## Usage (Python)

```python
# Import directly from the workflow module
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow

# Basic exact duplicate identification
exact_workflow = ExactDeduplicationWorkflow(
    input_path="/path/to/input/data",
    output_path="/path/to/output",
    text_field="text",
    perform_removal=False,  # Identification only
    assign_id=True,         # Auto-assign unique IDs if missing
    input_filetype="parquet",  # "parquet" or "jsonl"
    input_blocksize="2GiB"
)

exact_workflow.run()

# Advanced configuration with existing IDs and storage options
exact_workflow_advanced = ExactDeduplicationWorkflow(
    input_path="/path/to/input/data",
    output_path="/path/to/output",
    # Input configuration
    input_filetype="jsonl",
    input_blocksize="2GiB",
    input_file_extensions=[".jsonl"],
    read_kwargs={
        "storage_options": {"key": "<access_key>", "secret": "<secret_key>"}
    },
    write_kwargs={
        "storage_options": {"key": "<access_key>", "secret": "<secret_key>"}
    },
    # Processing configuration
    text_field="content",
    assign_id=False,        # Use existing ID field
    id_field="document_id",
    perform_removal=False,
    # Optional: environment variables for GPU comms
    env_vars={
        "UCX_TLS": "rc,cuda_copy,cuda_ipc",
        "UCX_IB_GPU_DIRECT_RDMA": "yes",
    },
)

exact_workflow_advanced.run()

# Integrate with existing pipelines using initial tasks
from nemo_curator.tasks import FileGroupTask

initial_tasks = [
    FileGroupTask(
        task_id="batch_0",
        dataset_name="my_dataset",
        data=["/path/to/file1.parquet", "/path/to/file2.parquet"],
        _metadata={"source_files": ["/path/to/file1.parquet", "/path/to/file2.parquet"]},
    )
]

exact_workflow.run(initial_tasks=initial_tasks)
```

```{note}
Removal is currently not implemented in the exact workflow (`perform_removal=True` raises an error). Use the duplicate ID outputs with the Text Duplicates Removal workflow. Refer to {ref}`Common Operations <text-process-data-dedup-common-ops>` for removal and outputs.
```

---

## Performance Recommendations

- Uses MD5 hashing for exact duplicate detection
- Requires a Ray-based distributed GPU execution environment
- Clear the output directory between runs to avoid conflicts

---

## Output Structure

- Output directory contains duplicate IDs and, when `assign_id=True`, an ID generator mapping
  - `ExactDuplicateIds/`: Parquet files with document IDs to remove
  - `exact_id_generator.json`: ID generator mapping

---

## Workflow Stages

1. File Partitioning: Groups input files for parallel processing
2. Exact Duplicate Identification: Computes MD5 hashes and identifies duplicates

---

## Ray Cluster Setup

The workflow runs on a Ray cluster with GPU support. Initialization and cleanup of the ID generator actor occur within the workflow when `assign_id=True`.

Refer to the Ray documentation (`https://docs.ray.io/en/latest/cluster/getting-started.html`) for distributed cluster setup.
