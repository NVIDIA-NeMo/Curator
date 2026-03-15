# Audio Stages Developer Guide

All audio processing stages inherit from `AudioEntryStage` (defined in
`common.py`), which itself inherits from the framework's `ProcessingStage`.

Each `AudioEntry` wraps a single manifest entry as a plain `dict`.  Stages
read keys from that dict, add or modify keys, and return the result.

## Writing a CPU stage

Override **one** method: `process_dataset_entry`.

```python
from dataclasses import dataclass
from nemo_curator.stages.audio.common import AudioEntryStage


@dataclass
class ComputeSNRStage(AudioEntryStage):
    """Compute signal-to-noise ratio for an audio file."""

    name: str = "ComputeSNRStage"
    audio_filepath_key: str = "audio_filepath"
    snr_key: str = "snr"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.snr_key]

    def process_dataset_entry(self, data: dict) -> dict | None:
        # data is the manifest entry dict, e.g. {"audio_filepath": "/a.wav", ...}
        data[self.snr_key] = _compute_snr(data[self.audio_filepath_key])
        return data          # return dict to keep, or None to drop this entry
```

That is it.  The base class handles:

- **Input validation** — checks that `audio_filepath` exists in the entry
  before your code runs (via `inputs()`).
- **Task unwrapping / rewrapping** — you receive a plain `dict`, not an
  `AudioEntry`.  The base rebuilds the `AudioEntry` with propagated
  `_stage_perf`, `_metadata`, `task_id`, etc.
- **Filtering** — return `None` to drop an entry from the pipeline.

### Lazy imports and `setup()`

If your stage depends on a heavy library (e.g. `soundfile`, `torch`), import
it inside `setup()` so it is only loaded on workers, not on the driver:

```python
def setup(self, worker_metadata=None) -> None:
    import soundfile
    self._soundfile = soundfile
```

`setup()` is called once per worker before any processing begins.

## Writing a GPU stage

Override **one** method: `_process_validated`.

```python
from dataclasses import dataclass, field
from nemo_curator.stages.audio.common import AudioEntryStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioEntry


@dataclass
class InferenceSpeakerIDStage(AudioEntryStage):
    """Speaker identification using a GPU model."""

    name: str = "SpeakerID_inference"
    model_name: str = "nvidia/speakerverification_en_titanet_large"
    filepath_key: str = "audio_filepath"
    speaker_key: str = "speaker_id"
    batch_size: int = 32
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpus=1.0))

    def setup(self, _worker_metadata=None) -> None:
        import nemo.collections.asr as nemo_asr
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=self.model_name
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.filepath_key, self.speaker_key]

    def _process_validated(self, tasks: list[AudioEntry]) -> list[AudioEntry]:
        # tasks are already validated — all have self.filepath_key
        files = [t.data[self.filepath_key] for t in tasks]
        speaker_ids = self.model.get_label(files)       # one batched GPU call
        results = []
        for task, sid in zip(tasks, speaker_ids, strict=True):
            results.append(
                AudioEntry(
                    data={**task.data, self.speaker_key: sid},
                    task_id=task.task_id,
                    dataset_name=task.dataset_name,
                    filepath_key=task.filepath_key,
                    _stage_perf=list(task._stage_perf),
                    _metadata=task._metadata.copy(),
                )
            )
        return results
```

Key differences from a CPU stage:

| | CPU stage | GPU stage |
|---|---|---|
| Override | `process_dataset_entry` | `_process_validated` |
| Receives | One `dict` | `list[AudioEntry]` (the whole batch) |
| Returns | `dict \| None` | `list[AudioEntry]` |
| `batch_size` | Default `1` | Set to match GPU throughput (e.g. `16`, `32`) |
| `resources` | Default `cpus=1.0` | Set `gpus=1.0` (or fractional) |

## What you must always declare

Every stage (CPU or GPU) should declare:

- **`inputs()`** — which dict keys must be present.  The base class
  validates these before your code runs.
- **`outputs()`** — which dict keys your stage produces.  Used for
  pipeline introspection and documentation.
- **`name`** — a human-readable stage name for logging and metrics.

## Filtering entries

To drop an entry from the pipeline:

- **CPU stage**: return `None` from `process_dataset_entry`.
- **GPU stage**: omit the entry from the returned list in `_process_validated`.

## Method reference

`AudioEntryStage` provides four methods.  Each exists for a reason that
cannot be collapsed further:

```
process_dataset_entry(dict) -> dict | None
    CPU-stage hook.  Dict in, dict out.  Eliminates unwrap/rewrap
    boilerplate for CPU stages.

process(AudioEntry) -> AudioEntry | list[AudioEntry]
    Required by the abstract ProcessingStage base.  Delegates to
    process_batch.  Cannot hold logic itself because backends call
    process_batch directly with N tasks; putting logic here would
    force process_batch to loop through process per task, preventing
    GPU stages from batching N tasks in one kernel call.

process_batch(list[AudioEntry]) -> list[AudioEntry]
    The real entry point called by backends.  Validates every task up
    front (fail-fast), then delegates to _process_validated.  Not meant
    to be overridden — it is the single validation gateway.

_process_validated(list[AudioEntry]) -> list[AudioEntry]
    Post-validation batch hook.  Default loops via process_dataset_entry
    for CPU stages.  GPU stages override only this method — they never
    need to repeat the validation loop.
```

## How backends parallelise your stage

Both Xenna and Ray follow the same high-level pattern: they create
multiple **workers** (Ray Actors), each holding its own copy of your
stage, and distribute task batches across them.  The differences are in
scheduling and resource management.

### Lifecycle on every worker

```
1.  setup_on_node()   — called once per node (shared across all workers
                        on that node).  Use for one-time node-level setup
                        like downloading a shared model file to local disk.

2.  setup()           — called once per worker.  Use for loading models
                        into memory / onto the assigned GPU.

3.  process_batch()   — called repeatedly with batches of tasks.
      → validates all tasks
      → calls _process_validated(tasks)

4.  teardown()        — called once when the worker shuts down.
```

### CPU stage parallelism

For a CPU stage with default `resources=Resources(cpus=1.0)` and
`batch_size=1`:

```
                        ┌─────────────────────────────────────────┐
                        │            Backend (Xenna / Ray)        │
                        │                                         │
1000 AudioEntry tasks   │   Determines worker count from          │
       │                │   available CPUs / stage.resources.cpus  │
       │                │                                         │
       ▼                │   e.g. 32 CPUs → 32 workers             │
  ┌─────────┐           │                                         │
  │ batch=1 │──────────►│   Worker 0: process_batch([task_0])     │
  │ batch=1 │──────────►│   Worker 1: process_batch([task_1])     │
  │ batch=1 │──────────►│   Worker 2: process_batch([task_2])     │
  │  ...    │           │   ...                                   │
  │ batch=1 │──────────►│   Worker 31: process_batch([task_31])   │
  └─────────┘           │                                         │
                        │   Work-stealing: as each worker finishes│
                        │   it picks up the next unprocessed task │
                        └─────────────────────────────────────────┘
```

Each `process_batch([single_task])` call goes through:
`AudioEntryStage.process_batch` → validate → `_process_validated` →
`process_dataset_entry(data)` → your code.

### GPU stage parallelism

For a GPU stage with `resources=Resources(cpus=1.0, gpus=1.0)` and
`batch_size=16`:

```
                        ┌─────────────────────────────────────────┐
                        │            Backend (Xenna / Ray)        │
                        │                                         │
1000 AudioEntry tasks   │   Determines worker count from          │
       │                │   available GPUs / stage.resources.gpus  │
       │                │                                         │
       ▼                │   e.g. 4 GPUs → 4 workers               │
  ┌──────────┐          │                                         │
  │ batch=16 │─────────►│   Worker 0 (GPU 0): process_batch(16)   │
  │ batch=16 │─────────►│   Worker 1 (GPU 1): process_batch(16)   │
  │ batch=16 │─────────►│   Worker 2 (GPU 2): process_batch(16)   │
  │ batch=16 │─────────►│   Worker 3 (GPU 3): process_batch(16)   │
  │  ...     │          │                                         │
  └──────────┘          │   63 total batches distributed across   │
                        │   4 workers via work-stealing           │
                        └─────────────────────────────────────────┘
```

Each `process_batch([16 tasks])` call goes through:
`AudioEntryStage.process_batch` → validate all 16 → `_process_validated`
→ your code receives all 16 tasks → **one** batched GPU call.

### Xenna specifics

Xenna is the production backend built on Cosmos-Xenna (which uses Ray
under the hood).

- **Worker creation**: Xenna creates Ray Actors.  Each actor gets
  `stage.resources.cpus` CPUs and `stage.resources.gpus` GPUs.  Xenna
  manages `CUDA_VISIBLE_DEVICES` directly.
- **Batching**: Xenna reads `stage.batch_size` via the adapter's
  `stage_batch_size` property and groups that many tasks per
  `process_data()` call.
- **Streaming**: In the default streaming execution mode, Xenna feeds
  batches to workers as they become idle — no worker waits while others
  are busy.
- **Multi-node**: Xenna's scheduler places actors across all nodes in
  the Ray cluster.  A 2-node cluster with 4 GPUs each = 8 workers, each
  with its own model copy.
- **Autoscaling**: Xenna can adjust worker counts based on measured
  throughput (`autoscale_interval_s` in executor config).
- **Call chain**:
  `Xenna scheduler → XennaStageAdapter.process_data(tasks)`
  `→ BaseStageAdapter.process_batch(tasks)` (timing + metrics)
  `→ AudioEntryStage.process_batch(tasks)` (validation)
  `→ _process_validated(tasks)` (your code)

### Ray ActorPool specifics

Ray ActorPool is an experimental backend that uses Ray's `ActorPool`
directly.

- **Worker count**: Calculated from available cluster resources:
  `min(num_batches, available_gpus // stage.resources.gpus)`.
  For 4 GPUs and `gpus=1.0`, that is 4 actors.
- **Batching**: The executor splits all tasks into batches of
  `stage.batch_size` and sends them via
  `actor_pool.map_unordered(actor.process_batch.remote, batches)`.
- **Work-stealing**: `map_unordered` assigns each batch to the next
  idle actor — results arrive in completion order.
- **Setup**: `setup_on_node()` runs once per node *before* actors are
  created (via a pinned Ray task).  `setup()` runs in each actor's
  `__init__`.
- **Multi-node**: Ray's cluster scheduler bin-packs actors across all
  nodes based on resource requests.
- **Call chain**:
  `ActorPool.map_unordered → actor.process_batch.remote(tasks)`
  `→ BaseStageAdapter.process_batch(tasks)` (timing + metrics)
  `→ AudioEntryStage.process_batch(tasks)` (validation)
  `→ _process_validated(tasks)` (your code)

### Two levels of parallelism

| Level | What it controls | Who sets it |
|---|---|---|
| **Worker count** | How many parallel copies of your stage run (one per CPU core or GPU) | The backend, based on `stage.resources` and available hardware |
| **`batch_size`** | How many tasks each worker processes per call | The stage author (domain knowledge about optimal GPU batch size) |

Total in-flight = `num_workers x batch_size`.  For 4 GPUs with
`batch_size=16`, that is 64 audio files being processed concurrently.
