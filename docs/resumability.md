# Pipeline Resumability

When `pipeline.run(checkpoint_path=...)` is set, completed source partitions are tracked in an LMDB database and skipped on subsequent runs. This allows interrupted pipelines to continue from where they left off rather than reprocessing everything from scratch.

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│  pipeline.run(checkpoint_path=...)                                   │
│                                                                      │
│  ┌──────────────────┐     ┌──────────────────┐     ┌─────────────┐  │
│  │  Source Stage    │────▶│ CheckpointFilter │────▶│  Stage 1…N  │  │
│  │                  │     │                  │     │             │  │
│  │ sets             │     │ already done?    │     │  fan-out?   │  │
│  │ resumability_key │     │  yes → drop ✗   │     │  ┌────────┐ │  │
│  └──────────────────┘     │  no  → pass  ✓  │     │  │child 1│ │  │
│                           └──────────────────┘     │  │child 2│ │  │
│                                    │               │  │child N│ │  │
│                                    │ stamp uuid    │  └────────┘ │  │
│                                    │ reset counter │             │  │
│                                    │               │  full drop? │  │
│                                    │               │  mark done  │  │
│                                    │               └─────────────┘  │
│                                    │                      │         │
│                                    ▼                      ▼         │
│                           ┌──────────────────────────────────────┐  │
│                           │         CheckpointRecorder           │  │
│                           │         mark_completed in LMDB       │  │
│                           └──────────────────────────────────────┘  │
│                                                                      │
│                    ┌─────────────────────────────┐                  │
│                    │           LMDB              │                  │
│                    │  partitions: expected count │                  │
│                    │  completions: uuid set      │                  │
│                    └─────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘

Fan-out: add_expected runs synchronously (ray.get) before child tasks
are dispatched — prevents recorder from satisfying the check early.
```

## Counting With Fan-Out Stages

`FilePartitioningStage` fans out into file groups — each becomes its own tracked partition. When a downstream stage (e.g. `ImageReaderStage`) further fans out a file group into multiple batches, the expected completion count for that partition grows accordingly.

```
FilePartitioningStage fans out into file groups — each becomes its own tracked partition.

┌─────────────────────────────────────────────────────────────────────────────────┐
│  FilePartitioningStage  (source, fan-out)                                       │
│                                                                                  │
│  files: [a.tar, b.tar, c.tar, d.tar]  →  2 file groups (files_per_partition=2) │
│                                                                                  │
│  task_id="file_group_0"    resumability_key = sha256("a.tar|b.tar::0")          │
│  task_id="file_group_1"    resumability_key = sha256("c.tar|d.tar::1")          │
└────────────────┬────────────────────────────────┬───────────────────────────────┘
                 │                                │
                 ▼                                ▼
┌───────────────────────────┐      ┌───────────────────────────┐
│  CheckpointFilter         │      │  CheckpointFilter         │
│                           │      │                           │
│  key = rk_0               │      │  key = rk_1               │
│  uuid = uuid(rk_0, 0)     │      │  uuid = uuid(rk_1, 0)     │
│  LMDB: expected=1         │      │  LMDB: expected=1         │
│         completed={}      │      │         completed={}      │
└───────────────┬───────────┘      └───────────────┬───────────┘
                │                                  │
                ▼                                  ▼
┌───────────────────────────┐      ┌───────────────────────────┐
│  ImageReaderStage (fan-out)│      │  ImageReaderStage (fan-out)│
│                           │      │                           │
│  fg_0 → [batch_0, batch_1,│      │  fg_1 → [batch_0, batch_1]│
│           batch_2]        │      │                           │
│                           │      │  add_expected(rk_1, +1)   │
│  add_expected(rk_0, +2)   │      │  LMDB: expected=2         │
│  LMDB: expected=3         │      │                           │
│                           │      │  uuid(rk_1,1)  uuid(rk_1,2)│
│  uuid(rk_0,1) uuid(rk_0,2)│      └──────┬────────────┬───────┘
│  uuid(rk_0,3)             │             │            │
└───┬──────────┬────────┬───┘             ▼            ▼
    │          │        │        ┌──────────────┐  ┌──────────────┐
    ▼          ▼        ▼        │   Recorder   │  │   Recorder   │
┌────────┐ ┌────────┐ ┌────────┐ │ mark(rk_1,1) │  │ mark(rk_1,2) │
│Recorder│ │Recorder│ │Recorder│ │ done: 1/2    │  │ done: 2/2 ✓  │
│mark    │ │mark    │ │mark    │ └──────────────┘  └──────────────┘
│(rk_0,1)│ │(rk_0,2)│ │(rk_0,3)│
│1/3     │ │2/3     │ │3/3 ✓   │          rk_1 partition complete → LMDB sealed
└────────┘ └────────┘ └────────┘
                                  On re-run: CheckpointFilter drops fg_1 entirely.
   rk_0 partition complete → LMDB sealed                    │
                                                            ▼
   On re-run: CheckpointFilter drops fg_0 entirely.       (skipped)
```

Each file group is an independent partition. The expected count starts at 1 (the source task itself) and grows by `N-1` on each fan-out of size N. The partition is sealed when `completions == expected`.
