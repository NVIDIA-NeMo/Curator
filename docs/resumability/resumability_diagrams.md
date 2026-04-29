# Resumability — Workflow Diagrams

Companion diagrams for the resumability feature on `onur/resumability-support`.

---

## 1. Component Architecture

```
                        pipeline.run(checkpoint_path=...)
                                       │
                                       ▼
          ┌────────────────────────────────────────────────────────┐
          │         Pipeline._with_checkpoint_stages()             │
          │                                                        │
          │  ┌─────────────┐    ┌────────────────────────────┐    │
          │  │ SourceStage │───►│  _CheckpointFilterStage    │    │
          │  └─────────────┘    │  skips completed partitions│    │
          │                     └──────────────┬─────────────┘    │
          │                                    │                   │
          │                     ┌──────────────▼─────────────┐    │
          │                     │          Stage1             │    │
          │                     └──────────────┬─────────────┘    │
          │                                    │                   │
          │                     ┌──────────────▼─────────────┐    │
          │                     │           ...               │    │
          │                     └──────────────┬─────────────┘    │
          │                                    │                   │
          │                     ┌──────────────▼─────────────┐    │
          │                     │          StageN             │    │
          │                     └──────────────┬─────────────┘    │
          │                                    │                   │
          │                     ┌──────────────▼─────────────┐    │
          │                     │  _CheckpointRecorderStage  │    │
          │                     │  records completed tasks   │    │
          │                     └────────────────────────────┘    │
          └────────────────────────────────────────────────────────┘
                    │ each stage wrapped by BaseStageAdapter
                    │  · _propagate_source_files()
                    │  · _record_checkpoint_events()
                    │
                    │ fire-and-forget writes
                    ▼
          ┌─────────────────────────────────────────────────────┐
          │  _CheckpointActorProxy  ──►  _CheckpointWriterActor │
          │  (per worker, non-blocking)  (single named Ray      │
          │                              actor, serializes      │
          │                              all writes)            │
          └──────────────────────────────┬──────────────────────┘
                                         │ atomic read-modify-write
                                         ▼
                               ┌─────────────────────┐
                               │  checkpoint_path/   │
                               │    *.json files     │◄── read at startup by
                               │  (NFS / S3 / GCS)   │    CheckpointFilterStage
                               └─────────────────────┘
```

---

## 2. First Run — Full Execution

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  STARTUP                                                        │
  │                                                                 │
  │  CheckpointFilterStage.setup()                                  │
  │    │                                                            │
  │    └──► read checkpoint_path/   ──►  (empty — fresh run)       │
  │         0 completed partitions found                            │
  └─────────────────────────────────────────────────────────────────┘
                              │
                              │  for each source partition
                              ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  task { source_files: ["file_a.tar"] }                          │
  └────────────────────────────┬────────────────────────────────────┘
                               │
                               ▼
          ┌────────────────────────────────────┐
          │      _CheckpointFilterStage        │
          │  is_task_completed("file_a.tar")?  │
          │           → false                  │
          └────────────────┬───────────────────┘
                           │ pass through
                           ▼
          ┌────────────────────────────────────┐
          │         Stage1 … StageN            │
          │                                    │
          │  if fan-out (1 input → N outputs): │
          │    BaseStageAdapter calls          │
          │    write_expected_increment(+N-1)  │──► CheckpointActor ──► file_a.json
          │                                    │
          │  if task filtered (output = []):   │
          │    BaseStageAdapter calls          │
          │    mark_filtered(task_id)          │──► CheckpointActor ──► file_a.json
          └────────────────┬───────────────────┘
                           │ leaf tasks flow out
                           ▼
          ┌────────────────────────────────────┐
          │    _CheckpointRecorderStage        │
          │  mark_completed(task_id, files)    │──► CheckpointActor ──► file_a.json
          └────────────────────────────────────┘
```

---

## 3. Resume Run — Skip Completed Work

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  STARTUP  (same script, same checkpoint_path)                   │
  │                                                                 │
  │  CheckpointFilterStage.setup()                                  │
  │    │                                                            │
  │    └──► read checkpoint_path/*.json                             │
  │         70 of 100 source partitions fully complete              │
  └─────────────────────────────────────────────────────────────────┘
                              │
                              │  for each source partition
                              ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  task { source_files: ["file_x.tar"] }                          │
  └────────────────────────────┬────────────────────────────────────┘
                               │
                               ▼
          ┌────────────────────────────────────┐
          │      _CheckpointFilterStage        │
          │  is_task_completed("file_x.tar")?  │
          └──────┬─────────────────────┬───────┘
                 │                     │
              true (70×)           false (30×)
                 │                     │
                 ▼                     ▼
        ┌────────────────┐    ┌─────────────────────┐
        │  return []     │    │   pass through      │
        │  task dropped  │    │                     │
        │  no downstream │    │  Stage1 … StageN    │
        │  work          │    │    ↓                 │
        └────────────────┘    │  Recorder           │
                              │  mark_completed     │──► checkpoint
                              └─────────────────────┘

  Result: 70% of work skipped instantly. Only 30% re-processed.
```

---

## 4. Fan-out Tracking

One source file splits into multiple leaf tasks — all must complete before the partition is marked done.

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Source partition:  file_a.tar                                  │
  │  Initial expected:  1                                           │
  └────────────────────────────┬────────────────────────────────────┘
                               │
                               ▼
                   ┌───────────────────────┐
                   │  Stage: split archive │
                   │  1 input → 5 chunks   │
                   │                       │
                   │  BaseStageAdapter     │
                   │  detects len(out)=5   │
                   │        > len(in)=1    │
                   └──────┬──────────┬─────┘
                          │          │
              write_expected_increment(+4)
                          │          │
                          ▼          │
              ┌──────────────────┐   │
              │  file_a.json     │   │  expected = 1 + 4 = 5
              │  increments:     │   │
              │  [{id, +4}]      │   │
              └──────────────────┘   │
                                     │ 5 leaf tasks flow downstream
                                     ▼
          ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
          │ chunk_0  │  │ chunk_1  │  │ chunk_2  │  │ chunk_3  │  │ chunk_4  │
          │Stage2…N  │  │Stage2…N  │  │Stage2…N  │  │Stage2…N  │  │Stage2…N  │
          └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
               │             │             │              │              │
               └─────────────┴─────────────┴──────────────┴──────────────┘
                                           │ each calls mark_completed
                                           ▼
                               ┌───────────────────────┐
                               │  file_a.json          │
                               │  completed: 5 tasks   │
                               │  expected:  5         │
                               │                       │
                               │  5 >= 5  →  done ✓   │
                               │  skipped on resume    │
                               └───────────────────────┘
```

---

## 5. Completion Counting Formula

```
  ┌──────────────────────────────┐   ┌──────────────────────────────┐
  │  completed list              │   │  filtered list               │
  │  task IDs that reached       │   │  task IDs dropped by         │
  │  the recorder stage          │   │  a filter stage              │
  └──────────────┬───────────────┘   └───────────────┬──────────────┘
                 │                                    │
                 └──────────────┬─────────────────────┘
                                │
                                ▼
                   len(completed) + len(filtered)
                                │
                                │  compared to
                                ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  increments list                                                 │
  │  one entry per fan-out event, written by BaseStageAdapter        │
  │                                                                  │
  │  expected  =  1  +  sum(inc["increment"] for inc in increments)  │
  └──────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
               ┌─────────────────────────────────────┐
               │  len(completed) + len(filtered)      │
               │            >= expected?              │
               └──────────┬──────────────┬────────────┘
                          │              │
                         yes             no
                          │              │
                          ▼              ▼
             ┌────────────────┐   ┌─────────────────────┐
             │  Partition     │   │  Partition           │
             │  complete ✓   │   │  incomplete          │
             │  skipped on    │   │  re-processed on     │
             │  next resume   │   │  next resume         │
             └────────────────┘   └─────────────────────┘
```

---

## 6. Concurrency Safety — Ray Actor as Serializer

```
  ┌────────────────┐
  │    Worker 1    │──┐
  │ (stage adapter)│  │  fire-and-forget
  └────────────────┘  │  .remote() calls
                      │
  ┌────────────────┐  │  ┌────────────────────────────────────────┐
  │    Worker 2    │──┼─►│         _CheckpointActorProxy          │
  │ (stage adapter)│  │  │  one instance per worker, non-blocking │
  └────────────────┘  │  └────────────────────┬───────────────────┘
                      │                       │ queued async calls
  ┌────────────────┐  │                       ▼
  │    Worker N    │──┘  ┌────────────────────────────────────────┐
  │ (stage adapter)│     │       _CheckpointWriterActor           │
  └────────────────┘     │  · single named Ray actor              │
                         │  · lifetime="detached" (survives       │
                         │    worker crashes and autoscaling)     │
                         │  · get_if_exists=True (all workers     │
                         │    share the same actor instance)      │
                         │  · single-threaded → writes serialized │
                         └──────────────────┬─────────────────────┘
                                            │ atomic read-modify-write
                                            ▼
                               ┌────────────────────────┐
                               │   checkpoint_path/     │
                               │     *.json files       │
                               │   no concurrent-write  │
                               │   races possible       │
                               └────────────────────────┘
```

---

## 7. Checkpoint File Layout

```
  checkpoint_path/
  ├── a3f7c2e1b9d84f20.json    ← sha256("file_a.tar")[:16]
  ├── b2e1a4d3c8f97e51.json    ← sha256("file_b.tar|file_c.tar")[:16]
  └── ...

  ┌──────────────────────────────────────────────────────────────────┐
  │  {                                                               │
  │    "source_key":  "file_a.tar",          ← sorted, joined by |  │
  │    "completed":   ["task_0", "task_1"],  ← deduplicated         │
  │    "filtered":    ["task_2"],            ← dropped by filters   │
  │    "increments":  [                      ← one per fan-out      │
  │      {"triggering_task_id": "g_abc", "increment": 3}            │
  │    ]                                                             │
  │  }                                                               │
  └──────────────────────────────────────────────────────────────────┘

  Completion check:

    len(completed) + len(filtered)  >=  1 + sum(inc["increment"])
          2        +       1         >=       1 + 3
                3                    >=    4   →  incomplete, re-process
```

---

## 8. Two-Run Lifecycle

```
  RUN 1 (fails mid-way)
  ──────────────────────────────────────────────────────────────────►
  │                                                                  │
  t=0                          t=1h                           t=1h30m
  │                            │                              │
  ▼                            ▼                              ▼
  ┌──────────────┐   ┌──────────────────────┐   ┌────────────────────┐
  │ Pipeline     │   │ 70 / 100 partitions  │   │ Process crash      │
  │ starts fresh │   │ complete             │   │ (OOM / preemption) │
  │              │   │                      │   │                    │
  │ FilterStage  │   │ Each writes          │   │ 30 partitions      │
  │ loads empty  │   │ {sha256}.json to     │   │ never completed    │
  │ checkpoint   │   │ /ckpt/               │   │                    │
  └──────────────┘   └──────────────────────┘   └────────────────────┘


  RUN 2 (resumes)
  ──────────────────────────────────────────────────────────────────►
  │                                        │
  t=0                                    t=45m
  │                                        │
  ▼                                        ▼
  ┌──────────────────────────────┐   ┌────────────────────────────┐
  │ Pipeline re-launched         │   │ All 100 partitions done    │
  │                              │   │                            │
  │ FilterStage loads 70 entries │   │ Total wall-clock time for  │
  │ from /ckpt/                  │   │ run 2: ~45m instead of     │
  │                              │   │ 1h30m from scratch         │
  │ 70 partitions skipped        │   │                            │
  │ instantly on first check     │   │                            │
  │                              │   │                            │
  │ 30 partitions re-processed   │   │                            │
  └──────────────────────────────┘   └────────────────────────────┘
```
