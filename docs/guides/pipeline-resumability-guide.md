# Pipeline-Specific Resumability Guide

This guide describes how individual pipeline authors can implement resumability
within their NeMo Curator pipelines. This is the recommended approach when
framework-wide resumability is not yet available.

## Overview

Resumability allows a pipeline to restart after a failure or interruption and
skip work that was already completed. NeMo Curator does not currently provide
built-in framework-level resumability, but pipeline authors can implement it
using patterns already proven in production pipelines (PDF parsing, video
processing, download stages).

The core idea: **scan output artifacts before processing starts, then filter
the input to exclude already-completed items.**

## Patterns by Layer

### Pattern 1: Input Filtering via Output Scan (Recommended)

**Used by**: PDF Nemotron-Parse pipeline  
**Not used by**: Video pipeline (which relies only on Pattern 2)

This is the most robust pattern. Before the pipeline starts, scan the output
directory for completed work and filter the input list accordingly.

#### How It Works

1. **Before pipeline starts**: Scan all output files (e.g., parquet) and
   extract the set of completed item IDs.
2. **At the partitioning stage**: Filter out input items whose IDs are already
   in the completed set.
3. **Pipeline runs normally**: Only unprocessed items flow through the pipeline.

#### Implementation Steps

**Step 1: Write a completion scanner**

The scanner reads output files and extracts the identifying column:

```python
def load_completed_ids(output_dir: str, id_column: str = "sample_id") -> set[str]:
    """Scan existing output files for already-processed IDs."""
    import glob
    import pyarrow.parquet as pq
    from loguru import logger

    completed: set[str] = set()
    for path in glob.glob(os.path.join(output_dir, "*.parquet")):
        try:
            table = pq.read_table(path, columns=[id_column])
            completed.update(table[id_column].to_pylist())
        except Exception as e:
            logger.warning(f"Could not read {path} for resume: {e}")

    logger.info(f"Resume scan: found {len(completed)} completed IDs in {output_dir}")
    return completed
```

For non-parquet outputs (e.g., JSON, images), scan filenames instead:

```python
def load_completed_ids_from_filenames(output_dir: str) -> set[str]:
    """Derive completed IDs from output filenames."""
    completed = set()
    for fname in os.listdir(output_dir):
        if fname.endswith(".json"):
            completed.add(fname.removesuffix(".json"))
    return completed
```

**Step 2: Pass completed IDs to the partitioning stage**

```python
from dataclasses import dataclass, field
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import FileGroupTask, _EmptyTask

@dataclass
class MyPartitioningStage(ProcessingStage[_EmptyTask, FileGroupTask]):
    input_manifest: str
    items_per_task: int = 10
    completed_ids: set[str] = field(default_factory=set)
    name: str = "my_partitioning"

    def process(self, _: _EmptyTask) -> list[FileGroupTask]:
        entries = []
        skipped = 0
        for item in self._read_manifest():
            if item["id"] in self.completed_ids:
                skipped += 1
                continue
            entries.append(item)

        logger.info(f"Resume: skipped {skipped}, processing {len(entries)}")

        tasks = []
        for i in range(0, len(entries), self.items_per_task):
            batch = entries[i : i + self.items_per_task]
            tasks.append(FileGroupTask(
                task_id=f"batch_{i // self.items_per_task:06d}",
                dataset_name="my_dataset",
                data=batch,
            ))
        return tasks
```

**Step 3: Wire it up in main()**

```python
def main():
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    completed_ids = load_completed_ids(output_dir)
    if completed_ids:
        logger.info(f"Resuming: {len(completed_ids)} items already processed")

    pipeline = Pipeline(
        name="my_pipeline",
        stages=[
            MyPartitioningStage(
                input_manifest=args.manifest,
                completed_ids=completed_ids,
            ),
            # ... processing stages ...
            MyWriterStage(path=output_dir),
        ],
    )
    pipeline.run()
```

#### Requirements for This Pattern

- **Deterministic ID column**: Your output must contain an identifying column
  (e.g., `sample_id`, `url`, `video_name`) that can be matched back to input
  items. Design your writer stage to include this.
- **Atomic writes**: Each output file should represent a complete unit of work.
  If a file is partially written (crash mid-write), the scanner should either
  handle it gracefully or the file should be detectable as corrupt.
- **Deterministic file naming**: Use `get_deterministic_hash()` from
  `nemo_curator.stages.text.io.writer.utils` to ensure the same input always
  produces the same output filename. This prevents duplicates on re-run.

#### Handling Partial Files

Partially-written parquet files are typically unreadable (pyarrow will throw an
exception). The scanner's `try/except` around `pq.read_table()` handles this
automatically — corrupt files are skipped, and those items will be reprocessed.

For better crash safety, consider writing to a `.tmp` file and atomically
renaming after completion (see Pattern 2).

---

### Pattern 2: Skip-If-Exists at Write Time

**Used by**: Video pipeline (`ClipWriterStage`), Download stages

This pattern checks whether each output file already exists before writing.
It is simpler to implement but less efficient — all compute stages still run;
only the final write is skipped.

#### How It Works

1. **Deterministic output paths**: Each item maps to a predictable output path.
2. **At write time**: Check if the output file exists. If yes, skip the write.
3. **On re-run**: Items that were fully written are skipped at the write stage,
   but all upstream compute (GPU inference, etc.) is still performed.

#### Implementation

```python
from nemo_curator.utils.writer_utils import write_bytes

# In your writer stage:
def process(self, task):
    for item in task.data:
        output_path = self._get_output_path(item)
        write_bytes(
            buffer=item.content,
            dest=output_path,
            desc="clip",
            source_video=item.source,
            verbose=True,
            overwrite=False,  # Default: skip if exists
        )
```

The `write_bytes()` utility in `nemo_curator/utils/writer_utils.py` already
implements skip-if-exists as the default behavior.

#### When to Use This Pattern

- When compute is cheap relative to I/O (e.g., simple transformations).
- When you cannot easily scan output to derive completed input IDs (e.g., cloud storage).
- As a safety net in combination with Pattern 1.

#### Limitations

- **Wasted compute**: All upstream stages (including expensive GPU inference)
  re-execute for already-processed items. Only the final write is saved.
- **No crash safety**: If the pipeline crashes between compute and write,
  the compute is lost and must be redone.

---

### Pattern 3: Atomic Downloads with Temp Files

**Used by**: All download stages (`DocumentDownloader`)

For stages that fetch external data, use atomic temp-file downloads to ensure
crash safety.

#### How It Works

1. Download to a `.tmp` file.
2. On success, atomically `os.rename()` to the final path.
3. On re-run, check if the final file exists and is non-empty → skip.

```python
def download(self, url: str) -> str | None:
    output_file = os.path.join(self.download_dir, self.get_filename(url))
    temp_file = output_file + ".tmp"

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        return output_file  # Already downloaded

    success = self._download_to_path(url, temp_file)
    if success:
        os.rename(temp_file, output_file)  # Atomic on same filesystem
        return output_file
    return None
```

This is already implemented in `nemo_curator.stages.text.download.base.download.DocumentDownloader`.

---

### Pattern 4: Deterministic File Naming for Idempotent Writes

**Used by**: All writer stages via `get_deterministic_hash()`

Ensure the same input data always produces the same output filename. This
makes re-runs idempotent (overwriting identical content) and enables
Pattern 1's output scanning to work correctly.

```python
from nemo_curator.stages.text.io.writer.utils import get_deterministic_hash

# In your writer stage:
source_files = task._metadata.get("source_files", [])
filename = get_deterministic_hash(source_files, task.task_id)
# Result: 12-char hex hash, deterministic for the same inputs
```

**Important**: For this to work, propagate `source_files` through the pipeline
in `task._metadata`. The built-in `FilePartitioningStage` already sets
`_metadata["source_files"]` at task creation time.

---

## Combining Patterns

The most robust approach combines Pattern 1 + Pattern 4:

1. **Deterministic file naming** (Pattern 4) ensures same input → same output
   file, preventing duplicates.
2. **Input filtering via output scan** (Pattern 1) avoids re-running expensive
   compute stages for already-completed items.
3. Optionally add Pattern 2 as a safety net at the writer stage.

```
┌─────────────────────────────────────────────────────┐
│ main()                                              │
│  1. Scan output dir → completed_ids                 │
│  2. Build pipeline with completed_ids               │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ PartitioningStage                                   │
│  - Filter out completed_ids from input manifest     │
│  - Create tasks for remaining items only            │
└────────┬────────────────────────────────────────────┘
         │ Only unprocessed items
         ▼
┌─────────────────────────────────────────────────────┐
│ Processing Stages (CPU/GPU)                         │
│  - Run normally on the reduced input set            │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Writer Stage                                        │
│  - Deterministic filenames (Pattern 4)              │
│  - Optionally skip-if-exists (Pattern 2)            │
└─────────────────────────────────────────────────────┘
```

## Checklist for Pipeline Authors

- [ ] **Include an ID column in output**: Your output format (parquet, JSON,
  etc.) should contain a column that uniquely identifies each input item.
- [ ] **Use deterministic file naming**: Ensure `_metadata["source_files"]` is
  propagated through the pipeline, or use your own deterministic naming.
- [ ] **Write a completion scanner**: Scan output directory before pipeline
  starts to build a set of completed IDs.
- [ ] **Filter at partitioning stage**: Pass `completed_ids` to the first stage
  and skip items that are already done.
- [ ] **Handle corrupt/partial files**: Use try/except when scanning output,
  and consider atomic writes (`.tmp` + rename) for crash safety.
- [ ] **Log resume statistics**: Log how many items were skipped vs. remaining
  so users can verify the resume is working.
- [ ] **Test resume behavior**: Verify that interrupting and restarting the
  pipeline produces correct, duplicate-free output.

## Limitations of Pipeline-Specific Resumability

- **No stage-level resume**: If the pipeline fails between Stage 2 and Stage 3,
  all items restart from Stage 1. Only fully-written outputs are recognized.
- **Scan overhead at scale**: Scanning millions of output files at startup can
  take minutes. Consider maintaining a separate manifest file for very large
  datasets.
- **No coordination**: In multi-node pipelines, the scan happens on the driver
  before distributing work. This is fine for current use cases but could become
  a bottleneck with very large completed sets.
- **Duplicates are possible**: If the pipeline crashes after partial output for
  a multi-item task (e.g., 5 PDFs per task), some items may be reprocessed.
  Deterministic file naming prevents duplicate files but not duplicate compute.
