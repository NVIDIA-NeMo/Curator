---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'NVIDIA Sans', 'Helvetica Neue', Arial, sans-serif;
    font-size: 1.1rem;
  }
  h1 { color: #76b900; }
  h2 { color: #1a1a1a; border-bottom: 2px solid #76b900; padding-bottom: 6px; }
  code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
  pre { background: #1e1e1e; color: #d4d4d4; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
  .highlight { background: #76b90020; border-left: 4px solid #76b900; padding: 0.5rem 1rem; }
---

# Pipeline Resumability
## NeMo Curator — Checkpointing Support

`onur/resumability-support` · April 2026

---

## The Problem

Long-running curation pipelines — downloading, filtering, deduplicating terabytes of data — can take **hours or days**.

When a run fails (node crash, preemption, OOM, network blip) the only option without checkpointing is:

> **Restart from scratch.**

We want:

> **Resume from where we left off.**

---

## User API — One Parameter

```python
pipeline.run(
    executor=executor,
    checkpoint_path="/shared/nfs/my_pipeline_ckpt",
    # or: "s3://my-bucket/checkpoints/my_pipeline"
)
```

That's it. No changes to stages. No changes to tasks. One optional parameter.

On the **next** `pipeline.run(...)` call with the same `checkpoint_path`, completed source partitions are automatically skipped.

---

## Architecture: Components

```
┌─────────────────────────────────────────────────────────────┐
│  Pipeline.run(checkpoint_path=...)                          │
│     └── _with_checkpoint_stages()                           │
│           Injects filter + recorder around user stages       │
└─────────────────────────────────────────────────────────────┘
         │                                       │
┌────────▼──────────┐               ┌────────────▼───────────┐
│ _CheckpointFilter │               │ _CheckpointRecorder    │
│ Stage             │               │ Stage                  │
│ (after each       │               │ (last stage)           │
│  source stage)    │               │                        │
│ Skips completed   │               │ marks_completed()      │
│ partitions        │               │ via Ray actor          │
└───────────────────┘               └────────────────────────┘
                                              │
         ┌────────────────────────────────────▼──────────────┐
         │  BaseStageAdapter (every stage)                   │
         │   _propagate_source_files() — threads metadata    │
         │   _record_checkpoint_events() — fan-out tracking  │
         └────────────────────────────────────┬──────────────┘
                                              │
         ┌────────────────────────────────────▼──────────────┐
         │  _CheckpointActorProxy  →  _CheckpointWriterActor │
         │  (fire-and-forget)          (named Ray actor,     │
         │                              serializes writes)    │
         └────────────────────────────────────┬──────────────┘
                                              │
                               ┌──────────────▼─────────────┐
                               │  {checkpoint_path}/        │
                               │    {sha256hash}.json  ×N   │
                               │  (shared FS: NFS / S3/GCS) │
                               └────────────────────────────┘
```

---

## Stage Injection

`Pipeline._with_checkpoint_stages()` wraps the user's stage list transparently:

```
User stages:                      Augmented stages:

  SourceStage                →      SourceStage
  Stage1                              _CheckpointFilterStage  ← injected
  Stage2                            Stage1
  ...                               Stage2
  StageN                            ...
                                    StageN
                                    _CheckpointRecorderStage  ← injected
```

Every stage also gets `_checkpoint_path` stamped on it so `BaseStageAdapter` can write fan-out increments without extra configuration.

---

## First Run — Full Execution

```
pipeline.run(checkpoint_path="/ckpt")

  1. CheckpointFilter.setup()
       → reads /ckpt/  (empty — fresh run)
       → 0 completed partitions

  2. For each source partition (e.g. file_a.tar, file_b.tar, ...):
       CheckpointFilter.process(task)
         → is_task_completed("file_a.tar")? → false → pass through

  3. Stage1 … StageN process the task
       (BaseStageAdapter propagates source_files metadata through each stage)

  4. If any stage fans out 1 → N:
       BaseStageAdapter._record_checkpoint_events()
         → write_expected_increment(source_key, trigger_id, N-1)
         → "I produced N outputs from 1 input, expect N completions"

  5. CheckpointRecorder.process(task)
       → mark_completed(task_id, source_files)
       → /ckpt/{sha256(file_a.tar)[:16]}.json updated
```

---

## Resume Run — Skip Completed Work

```
pipeline.run(checkpoint_path="/ckpt")  ← same path, same pipeline

  1. CheckpointFilter.setup()
       → reads /ckpt/*.json
       → finds 70 of 100 source partitions fully complete

  2. For the 70 completed partitions:
       CheckpointFilter.process(task)
         → is_task_completed? → true → return []  (skip, no downstream work)

  3. For the 30 remaining partitions:
       → pass through → Stage1 … StageN → mark_completed

  Result: Only 30% of the work is repeated.
  The 70 already-done partitions are never re-processed.
```

---

## Checkpoint File Format

One JSON file per source partition, keyed by `sha256(source_key)[:16]`:

```
/checkpoint_path/
  a3f7c2e1b9d84f20.json    ← sha256("file_a.tar")[:16]
  b2e1a4d3c8f97e51.json    ← sha256("file_b.tar|file_c.tar")[:16]
  ...
```

```json
{
  "source_key": "file_a.tar",
  "completed": ["leaf_task_0", "leaf_task_1", "leaf_task_2"],
  "filtered":  ["leaf_task_3"],
  "increments": [
    {"triggering_task_id": "file_group_abc", "increment": 3}
  ]
}
```

**Completion check:**
```
len(completed) + len(filtered)  >=  1 + sum(inc["increment"] for inc in increments)
         3      +      1         >=         1 + 3
                  4              >=    4    → complete ✓
```

---

## Fan-out Tracking

The hard case: one source file fans out into multiple leaf tasks across stages.

```
Source: file_a.tar  (source_key = "file_a.tar", expected = 1)

    Stage1: split archive → 5 chunks
    BaseStageAdapter: write_expected_increment(+4)
    → expected = 1 + 4 = 5

      chunk_0 → Stage2 → … → Recorder → mark_completed("chunk_0", ["file_a.tar"])
      chunk_1 → Stage2 → … → Recorder → mark_completed("chunk_1", ["file_a.tar"])
      chunk_2 → Stage2 → … → Recorder → mark_completed("chunk_2", ["file_a.tar"])
      chunk_3 → Stage2 → … → Recorder → mark_completed("chunk_3", ["file_a.tar"])
      chunk_4 → Stage2 → … → Recorder → mark_completed("chunk_4", ["file_a.tar"])

  completed = 5, expected = 5 → 5 >= 5 → partition complete ✓

On resume: file_a.tar skipped entirely (all 5 leaf tasks already done).
```

Chained fan-outs (Stage1 splits 1→5, Stage2 splits each 5→3, total 15 leaves) work correctly — each stage appends its own increment entry.

---

## Concurrency Safety

Multiple Ray workers process tasks in parallel. Without coordination, concurrent JSON writes from N workers would corrupt checkpoint files.

**Solution: named Ray actor serializes all writes.**

```
Worker 1 ──fire-and-forget──┐
Worker 2 ──fire-and-forget──┤→ _CheckpointActorProxy
Worker N ──fire-and-forget──┘        │
                                     │ queued .remote() calls
                               ┌─────▼────────────────────┐
                               │  _CheckpointWriterActor  │
                               │  (single-threaded actor)  │
                               │  serializes read-modify-  │
                               │  write per file           │
                               └──────────┬────────────────┘
                                          │
                                   /ckpt/*.json (no races)
```

- **`lifetime="detached"`** — actor survives worker restarts/crashes during autoscaling
- **`get_if_exists=True`** — multiple adapters calling `setup()` in parallel all find the same actor

---

## Filtered Tasks

Stages can drop tasks (filter stages returning `None`/`[]`). These must still count toward completion.

```python
# BaseStageAdapter._record_checkpoint_events()
if not results:
    # All inputs filtered — record each as filtered (not completed)
    for input_task in tasks:
        self._write_checkpoint_mgr.mark_filtered(input_task.task_id, src_files)
```

Both `completed` and `filtered` counts contribute to the completion check:

```
len(completed) + len(filtered) >= expected
```

So: if a source partition produces 10 leaf tasks and 3 are filtered by a quality filter, the partition is still complete when the remaining 7 are processed and 3 are filtered (7 + 3 = 10 ≥ 10).

---

## Deduplication Workflows

Dedup stages are multi-input (fan-in) and cannot use the normal leaf-task tracker. They have their own internal checkpointing:

| Workflow | Checkpoint approach |
|---|---|
| Exact dedup | Intermediate Parquet files written to `output_path/dedup_ckpt/` — stages check file existence |
| Fuzzy dedup | Same pattern — each sub-stage writes its output; checks existence before re-running |
| Semantic dedup | KMeans + pairwise similarity writes intermediate results; resumes from last written stage |

These are at the **workflow level**, not the task level — they checkpoint whole sub-stages of the dedup multi-step algorithm.

---

## Key Properties

| Property | Details |
|---|---|
| **Zero stage changes** | Add `checkpoint_path=...` to `pipeline.run()` — done |
| **Idempotent writes** | Task IDs deduplicated in JSON; retries are safe |
| **Fire-and-forget** | Checkpoint writes never block stage execution |
| **Fan-out aware** | Increment tracking handles 1→N splits at any stage |
| **Filter aware** | Filtered tasks counted toward partition completion |
| **Cloud-native** | Works with S3, GCS, NFS (any fsspec-compatible FS) |
| **Crash-safe** | Detached actor survives worker restarts |
| **Xenna-safe** | Handles preemption and task rescheduling (idempotent) |

---

## Usage Examples

**Text pipeline (TinyStories):**
```python
pipeline.run(checkpoint_path=args.checkpoint_dir)
```

**Video pipeline:**
```python
pipeline.run(checkpoint_path=args.checkpoint_dir)
```

**With S3 storage:**
```python
pipeline.run(
    checkpoint_path="s3://my-bucket/curator-checkpoints/my-run",
    checkpoint_storage_options={"key": "...", "secret": "..."},
)
```

**Resume is automatic** — just re-run the same script with the same `checkpoint_path`.

---

## Summary

```
Problem:   Long pipelines fail mid-run → restart from scratch

Solution:  Checkpoint completed source partitions to shared FS

API:       pipeline.run(checkpoint_path="...")  ← one parameter

Mechanism: - Inject _CheckpointFilterStage after source stages
           - Inject _CheckpointRecorderStage at end
           - BaseStageAdapter tracks fan-outs and filtered tasks
           - Named Ray actor serializes all writes (no races)
           - JSON files: one per source partition

Resumes:   Skip completed partitions, re-run only remaining work
```
