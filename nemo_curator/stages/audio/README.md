# Audio Stages Developer Guide

All audio processing stages subclass `ProcessingStage[AudioTask, AudioTask]`
directly — the same base class used by video, text, and image modalities.
There is no audio-specific intermediate base class.

Each `AudioTask` wraps a single manifest entry as a plain `dict` (backed by
`_AttrDict` for attribute-style access).  Stages read keys from that dict,
mutate it in-place, and return the same task object.

## Writing a CPU stage

Override **one** method: `process`.

```python
from dataclasses import dataclass
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class ComputeSNRStage(ProcessingStage[AudioTask, AudioTask]):
    """Compute signal-to-noise ratio for an audio file."""

    name: str = "ComputeSNRStage"
    audio_filepath_key: str = "audio_filepath"
    snr_key: str = "snr"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.snr_key]

    def process(self, task: AudioTask) -> AudioTask | None:
        # task.data is the manifest entry dict, e.g. {"audio_filepath": "/a.wav", ...}
        task.data[self.snr_key] = _compute_snr(task.data[self.audio_filepath_key])
        return task          # return task to keep, or None to drop this entry
```

That is it.  The base class handles:

- **Input validation** — `ProcessingStage.process_batch` checks that
  `audio_filepath` exists in the entry before your code runs (via `inputs()`).
- **Filtering** — return `None` from `process()` to drop an entry from
  the pipeline (matching the text-modality filter convention).

### Lazy imports and `setup()`

If your stage depends on a heavy library (e.g. `soundfile`, `torch`), import
it inside `setup()` so it is only loaded on workers, not on the driver:

```python
def setup(self, worker_metadata=None) -> None:
    import soundfile
    self._soundfile = soundfile
```

`setup()` is called once per worker before any processing begins.

## Writing a GPU or IO stage

Override **`process_batch`** for batched processing.  `process()` should
raise `NotImplementedError` — matching the pattern used by deduplication
stages (`ConnectedComponentsStage`, `KMeansReadFitWriteStage`, etc.).

```python
from dataclasses import dataclass, field
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class InferenceSpeakerIDStage(ProcessingStage[AudioTask, AudioTask]):
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

    def process(self, task: AudioTask) -> AudioTask:
        msg = "InferenceSpeakerIDStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task.task_id} missing required columns for {type(self).__name__}: {self.inputs()}"
                raise ValueError(msg)
        files = [t.data[self.filepath_key] for t in tasks]
        speaker_ids = self.model.get_label(files)       # one batched GPU call
        for task, sid in zip(tasks, speaker_ids, strict=True):
            task.data[self.speaker_key] = sid            # mutate in-place
        return tasks
```

Key differences from a CPU stage:

| | CPU stage | GPU / IO stage |
|---|---|---|
| Override | `process` | `process_batch` (+ `process` raising `NotImplementedError`) |
| Receives | One `AudioTask` | `list[AudioTask]` (the whole batch) |
| Returns | `AudioTask \| None` | `list[AudioTask]` (or `list[DocumentBatch]` for IO) |
| Validation | Automatic (base `process_batch`) | Call `self.validate_input(task)` in a loop |
| `batch_size` | Default `1` | Set to match GPU throughput or IO aggregation (e.g. `16`, `64`) |
| `resources` | Default `cpus=1.0` | Set `gpus=1.0` for GPU stages; cpus for IO |

**Other stages that override `process_batch`:**

- `AudioToDocumentStage` (`io/convert.py`) — aggregates N `AudioTask`
  dicts into a single multi-row `pd.DataFrame` in one `DocumentBatch`,
  avoiding N single-row DataFrame allocations.  Not a GPU stage, but
  benefits from batched processing.
- `ManifestWriterStage` (`common.py`) — writes
  entries to JSONL, returns `AudioTask`.

### Setting `batch_size` for GPU inference

The `batch_size` field on a GPU stage controls how many `AudioTask` tasks
the backend groups into a single `process_batch()` call. The stage can then
make one or more model calls from that finite candidate window. For example,
`ASRStage` can segment long parents, locally regroup model items by duration,
and enforce a padded-audio budget for each adapter call.

**Defining batch_size in the stage class:**

```python
@dataclass
class ASRStage(ProcessingStage[AudioTask, AudioTask]):
    batch_size: int = 32
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
```

The default is a sensible starting point; pipeline authors can override it
at pipeline construction time without modifying the stage class:

**Overriding batch_size at pipeline construction:**

```python
pipeline.add_stage(
    ASRStage(
        adapter_target="nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
        model_id="nvidia/parakeet-tdt-0.6b-v2",
        max_audio_sec_per_actor=240,
        max_inference_duration_s=120,
        local_bucketing=True,
        audio_filepath_key="audio_filepath",
    )
    .with_(resources=Resources(gpus=1), batch_size=32)
)
```

The `.with_()` method supports common execution overrides such as `resources`
and `batch_size`. Here it sets `batch_size` to `32` and assigns 1 GPU. Put
stage- or model-specific fields in the stage constructor.

**Overriding batch_size via Hydra YAML:**

```yaml
stages:
  - _target_: nemo_curator.stages.audio.inference.asr.stage.ASRStage
    adapter_target: nemo_curator.models.asr.nemo_asr.NeMoASRAdapter
    model_id: nvidia/parakeet-tdt-0.6b-v2
    max_audio_sec_per_actor: 240
    max_inference_duration_s: 120
    local_bucketing: true
    audio_filepath_key: audio_filepath
    batch_size: 32
```

For Hydra to accept `batch_size` from YAML, it must be a dataclass field
on the stage (which it already is).

**How batch_size flows through the backend:**

```
Backend reads stage.batch_size
    → groups N tasks into batches of batch_size
    → sends each batch to a worker
    → worker calls stage.process_batch(tasks)
        → your override receives that finite candidate window
        → stage-specific code makes one or more model calls
```

For variable-duration audio, `batch_size` is not necessarily the exact number
of items in one model call. See
[Local Duration Bucketing for Audio GPU Inference](inference/README.md) for the
current `ASRStage` behavior, the duration-packing theory, and the integration
contract for other GPU stages.

**Choosing a good batch_size:**

- **Too small** (e.g. `1`) — gives the stage little opportunity to form useful
  model batches or duration-coherent groups.
- **Too large** (e.g. `1024`) — can increase waveform preparation and host
  memory pressure before the stage makes any model calls.
- **Sweet spot** — depends on the model, audio distribution, GPU memory, and
  the stage-level audio budget. Tune with representative inputs rather than
  treating the backend window as the model batch size.

## What you must always declare

Every stage (CPU or GPU) should declare:

- **`inputs()`** — which dict keys must be present.  The base class
  validates these before your code runs.
- **`outputs()`** — which dict keys your stage produces.  Used for
  pipeline introspection and documentation.
- **`name`** — a human-readable stage name for logging and metrics.

## Filtering entries

To drop an entry from the pipeline:

- **CPU filter stage**: return `None` from `process()`.  The base
  `ProcessingStage.process_batch` will include `None` in the results list.
- **Batch filter stage**: override `process_batch` to return only the
  entries that pass the filter (omit entries that should be dropped).
  This avoids `None` reaching the backend adapter, which calls
  `task.add_stage_perf()` on every element.  See `PreserveByValueStage`
  in `common.py` for the canonical pattern — its `process()` raises
  `NotImplementedError` and all logic lives in `process_batch`.
- **GPU / IO stage**: omit the entry from the returned list in `process_batch`.

## Method reference

Audio stages use two methods from `ProcessingStage`:

```
process(AudioTask) -> AudioTask | None
    The primary hook for CPU stages.  Receives a single task, mutates
    task.data in-place, and returns the task (or None to filter).
    GPU/IO stages raise NotImplementedError here — all work goes
    through process_batch.

process_batch(list[AudioTask]) -> list[AudioTask]
    The entry point called by backends.  The base ProcessingStage
    implementation validates each task via validate_input(), then
    loops calling process() per task.
    GPU stages and IO stages override this entirely for batched
    processing.
```

**Why both?**

- `process_batch` is the backend entry point — backends always call it
  with N tasks.
- `process` is the natural single-task hook for CPU stages — no
  boilerplate to handle lists.
- GPU/IO stages override `process_batch` to receive the full backend batch and
  organize its work efficiently. A stage may issue one or more bounded model
  calls from that candidate window. Their `process()` raises
  `NotImplementedError`, matching the dedup-stage convention
  (`ConnectedComponentsStage`, `KMeansReadFitWriteStage`, etc.).

## Optimizations in the base class

1. **Aggregated IO conversion** — `AudioToDocumentStage` overrides
   `process_batch` to combine N `AudioTask` dicts into one multi-row
   `pd.DataFrame` in a single `DocumentBatch`, avoiding N single-row
   DataFrame allocations.

2. **Ray Data compatibility** — empty-batch guards use `len(tasks) == 0`
   instead of `not tasks` because Ray Data's `map_batches` passes
   `tasks` as a numpy array, and `not ndarray` raises `ValueError`
   for arrays with more than one element.  This applies to
   `process_batch` in `ASRStage` and
   `AudioToDocumentStage`.

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

4.  teardown()        — called once when the worker shuts down.
```

### CPU stage parallelism

For a CPU stage with default `resources=Resources(cpus=1.0)` and
`batch_size=1`:

```
                        ┌─────────────────────────────────────────┐
                        │            Backend (Xenna / Ray)        │
                        │                                         │
1000 AudioTask tasks   │   Determines worker count from          │
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
`ProcessingStage.process_batch` → `validate_input(task)` →
`stage.process(task)` → your code.

### GPU stage parallelism

For a GPU stage with `resources=Resources(cpus=1.0, gpus=1.0)` and
`batch_size=16`:

```
                        ┌─────────────────────────────────────────┐
                        │            Backend (Xenna / Ray)        │
                        │                                         │
1000 AudioTask tasks   │   Determines worker count from          │
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
`ASRStage.process_batch` → validate, load, and model-safely segment the current
waveforms → plan capacity-bounded adapter calls across all segments in this
window → `NeMoASRAdapter.transcribe_batch` once per planned call → stitch
segments and restore parent-task order. `batch_size=16` therefore defines the
planning window, not a guarantee of exactly one 16-item NeMo call.

### Xenna specifics

Xenna is the default backend built on Cosmos-Xenna (which uses Ray
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
  `→ stage.process_batch(tasks)` (your override or base default)

### Ray Data

Ray Data is an alternative backend that uses Ray's Dataset
API.  It wraps each stage in a `RayDataStageAdapter` and applies stage
transformations as Ray Data `map_batches` operations.  Audio stages
work with Ray Data without modification.

```python
from nemo_curator.backends.ray_data import RayDataExecutor

executor = RayDataExecutor()
pipeline.run(executor)
```

> **Note**: Ray ActorPool is a separate backend used
> primarily for deduplication workloads.  It is **not** a recommended
> backend for audio pipelines.

### Two levels of parallelism

| Level | What it controls | Who sets it |
|---|---|---|
| **Worker count** | How many parallel copies of your stage run (one per CPU core or GPU) | The backend, based on `stage.resources` and available hardware |
| **`batch_size`** | Maximum candidate tasks supplied to each worker's `process_batch` call | The stage author |

Maximum candidate tasks in flight = `num_workers x batch_size`. For 4 GPUs
with `batch_size=16`, up to 64 audio files can be inside stage calls at once.
The stage may split each candidate window into smaller model calls.

## How `batch_size` travels from your stage to the backend

When you set `batch_size = 16` on a GPU stage, this is the exact path
the value takes until it controls how many tasks land in your
`process_batch` call:

```
ASRStage                                  (generic stage dataclass)
    batch_size: int = 16                  ← defined as a dataclass field
        │
        │  ProcessingStage (base class)
        │    stages/base.py                batch_size = 1  (default)
        │    stages/base.py                @property _batch_size → self.batch_size
        │    stages/base.py                with_(batch_size=N) → deepcopy + override
        │
        ▼
┌─── Xenna path ──────────────────────────────────────────────────────────┐
│                                                                         │
│  XennaStageAdapter wraps your stage                                     │
│    backends/xenna/adapter.py             @property stage_batch_size     │
│      → self.processing_stage.batch_size  → 16                           │
│                                                                         │
│  Cosmos-Xenna runtime reads adapter.stage_batch_size                    │
│    → groups incoming tasks into batches of 16                           │
│    → calls adapter.process_data(batch_of_16)                            │
│      backends/xenna/adapter.py                                          │
│      → BaseStageAdapter.process_batch(batch_of_16)                      │
│        backends/base.py                  stage.process_batch(batch_of_16)│
│          → your process_batch override receives 16 tasks                │
└─────────────────────────────────────────────────────────────────────────┘
```

Key takeaways:
- `batch_size` is a plain dataclass field on `ProcessingStage` (default `1`).
- Subclasses override it as a field (e.g. `batch_size: int = 16`).
- Pipeline authors can further override via `.with_(batch_size=32)` or Hydra YAML.
- The backend adapter reads `stage.batch_size` and groups tasks *before*
  calling `process_batch`.
- A stage can still split model-unsafe inputs and plan one or more adapter
  calls within that finite backend-provided batch, as `ASRStage` does.

## Exact call chains

Every file reference below is relative to the repo root
(`nemo_curator/` prefix).  The two chains differ only in the
stage-level override; everything above and below is shared.

### CPU stage (e.g. `GetAudioDurationStage`, `batch_size=1`)

**Xenna backend:**

```
pipeline.run(executor)
│   nemo_curator/pipeline/pipeline.py              executor.execute(self.stages, initial_tasks)
│
├─ XennaExecutor.execute()
│   backends/xenna/executor.py                     wraps each stage in XennaStageAdapter
│                                                  create_named_xenna_stage_adapter(stage)
│                                                  builds pipelines_v1.StageSpec with:
│                                                    - required_resources from adapter
│                                                    - stage_batch_size from adapter
│                                                  pipelines_v1.run_pipeline(pipeline_spec)
│
│   ── Xenna scheduler creates N Ray Actor workers (N = available_cpus / stage.resources.cpus) ──
│
├─ Per worker — one-time setup:
│   backends/xenna/adapter.py                      XennaStageAdapter.setup_on_node()
│     → backends/base.py                             stage.setup_on_node(node_info, worker_metadata)
│   backends/xenna/adapter.py                      XennaStageAdapter.setup()
│     → backends/base.py                             stage.setup(worker_metadata)
│       → stages/audio/common.py                       GetAudioDurationStage.setup() imports soundfile
│
├─ Per batch (batch_size=1, so 1 AudioTask per call):
│   backends/xenna/adapter.py                      XennaStageAdapter.process_data(tasks)
│     → backends/base.py                             BaseStageAdapter.process_batch(tasks)
│         ├─ start perf timer
│         ├─ stage.process_batch(tasks)                                                ──────────┐
│         ├─ log stats, attach _stage_perf                                                       │
│         └─ return results                                                                      │
│                                                                                                │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────┘
│   │  ProcessingStage.process_batch()              (base class — NOT overridden for CPU stages)
│   │    stages/base.py                             for task in tasks:
│   │      if not self.validate_input(task): raise ValueError(...)
│   │      result = self.process(task)
│   │        │
│   │    ┌───┘
│   │    │  GetAudioDurationStage.process(task)
│   │    │    stages/audio/common.py
│   │    │    audio_filepath = task.data[self.audio_filepath_key]
│   │    │    raw, samplerate = soundfile.read(audio_filepath)
│   │    │    task.data[self.duration_key] = raw.shape[0] / samplerate
│   │    │    return task                           (mutated in-place)
│   │    │
│   │    append result to results list
│   └─ return results
```

### GPU stage (e.g. `ASRStage` + `NeMoASRAdapter`, `batch_size=16`)

**Xenna backend:**

```
pipeline.run(executor)
│   nemo_curator/pipeline/pipeline.py              executor.execute(self.stages, initial_tasks)
│
├─ XennaExecutor.execute()
│   backends/xenna/executor.py                     same wrapping as CPU
│                                                  StageSpec with:
│                                                    - required_resources: gpus=1.0
│                                                    - stage_batch_size: 16
│
│   ── Xenna creates N workers (N = available_gpus / stage.resources.gpus) ──
│
├─ Per worker — one-time setup:
│   backends/xenna/adapter.py                      XennaStageAdapter.setup_on_node()
│     → stages/audio/inference/asr/stage.py           ASRStage.setup_on_node()
│       → models/asr/nemo_asr.py                      NeMoASRAdapter.download_weights_on_node()
│         ASRModel.from_pretrained(model_name=model_id, return_model_file=True)
│       (downloads model to shared cache — one download per node)
│   backends/xenna/adapter.py                      XennaStageAdapter.setup()
│     → backends/base.py                             stage.setup(worker_metadata)
│       → stages/audio/inference/asr/stage.py           ASRStage.setup()
│         → models/asr/nemo_asr.py                      NeMoASRAdapter.load_model(num_gpus=1)
│           ASRModel.from_pretrained(model_name=model_id, map_location=cuda)
│
├─ Per backend batch (batch_size=16, so up to 16 candidate tasks per call):
│   backends/xenna/adapter.py                      XennaStageAdapter.process_data(tasks)
│     → backends/base.py                             BaseStageAdapter.process_batch(tasks)
│         ├─ start perf timer
│         ├─ stage.process_batch(tasks)                                                ──────────┐
│         ├─ log stats, attach _stage_perf                                                       │
│         └─ return results                                                                      │
│                                                                                                │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────┘
│   │  ASRStage.process_batch()                    (generic batched GPU stage)
│   │    stages/audio/inference/asr/stage.py
│   │    validate_input(task) per task              schema check
│   │    load and normalize up to 16 current waveforms
│   │    split every waveform at max_inference_duration_s
│   │    plan calls bounded by max_audio_sec_per_actor
│   │    for each planned call:
│   │      adapter.transcribe_batch(items)
│   │        → models/asr/nemo_asr.py
│   │          self._model.transcribe(audio=waveforms, batch_size=len(items))
│   │    restore segment order, stitch parent transcripts, and update tasks
│   └─ return tasks                                 → same 16 AudioTask objects
```

## Memory characteristics of `AudioTask`

An `AudioTask` is a thin dataclass wrapping a single manifest-entry
`dict`.  The wrapper itself adds **~350 bytes** of overhead regardless
of entry size.  All memory is in `task.data`:

| Entry type | JSON on disk | `dict` in memory | `AudioTask` total | Wrapper overhead |
|---|---|---|---|---|
| Simple FLEURS (2 keys) | ~120 B | 394 B | 741 B | 347 B |
| Median ALM manifest row | ~1.2 MB | ~4 MB | ~4 MB | 347 B |
| Largest ALM manifest row | 10.8 MB | 39.3 MB | 39.3 MB | 349 B |

The largest entry observed in production (`fused_ia_top3.jsonl`) is a
6.3-hour podcast with 6 616 segments and 54 912 word-level timestamps:

```json
{
  "id": "podcasts_non_stream_eng_only_234154",
  "dataset_source": "internet_archive",
  "audio_filepath": "/local/.../podcasts_non_stream_eng_only_234154.mp3",
  "audio_sample_rate": 44100,
  "audio_num_channels": 1,
  "audio_size": 361070627,
  "actual_duration": 22618.81,
  "duration": 22618.15,
  "language": "en",
  "sample_rate": 16000,
  "resampled_audio_filepath": "/local/.../M_podcasts_non_stream_eng_only_234154.wav",
  "segments": [
    {
      "speaker": "podcasts_non_stream_eng_only_234154_SPEAKER_17",
      "start": 20.85,
      "end": 40.99,
      "text": "Well it's the last fan name I've ever won ...",
      "text_ITN": "Well it's the last fan name ...",
      "metrics": {
        "pesq_squim": 1.15,
        "stoi_squim": 0.56,
        "sisdr_squim": -7.491,
        "bandwidth": 15848,
        "hallucination": false
      },
      "words": [
        {"word": "Well", "start": 20.8, "end": 20.96},
        {"word": "it's", "start": 20.96, "end": 21.12}
      ]
    }
  ],
  "swift_audio_filepath": "IA_Audio_Datasets/podcasts_non_stream/...",
  "dataset_name": "ia_non_streaming_batch1",
  "num_speakers": null,
  "split_number": "00168"
}
```

*(54 top-level keys total; 6 616 segments shown as one; 54 912 words
across all segments.  This single entry is 10.8 MB on disk / 39.3 MB
in memory.)*

### Peak memory by stage type

**CPU stages** (e.g. `GetAudioDurationStage`, `ALMDataBuilderStage`):
Peak memory ≈ `num_workers × entry_size`.  With 32 CPU workers
processing median ALM entries (~4 MB each), that is ~128 MB of task
data in flight.  The worker process itself uses minimal additional
memory (soundfile, editdistance, etc. are lightweight).

**GPU stages** (e.g. `ASRStage` + `NeMoASRAdapter`): Peak memory is
dominated by **model VRAM**, not task data.  A NeMo ASR
FastConformer-TDT model uses ~2–4 GB of VRAM.  The task data
(`batch_size × entry_size`) is negligible in comparison — 16 FLEURS
entries is 16 × 741 B ≈ 12 KB, while even 16 large ALM entries is
16 × 4 MB ≈ 64 MB (still small vs the model).

## End-to-end `AudioTask` trace (FLEURS pipeline)

Below is a single English FLEURS entry flowing through every stage in
`tutorials/audio/fleurs/main.py`. All values are **real output** from running
`lang=en_us stages.1.model_id=nvidia/parakeet-tdt-0.6b-v2
data_split=dev wer_threshold=75`.

Pipeline: download → ASR → WER → duration → filter → convert → write.

### Stage 1: `CreateInitialManifestFleursStage`

Downloads the FLEURS `dev` split, parses the TSV transcript, and emits
one `AudioTask` per line (394 entries for `en_us` dev).

**Output** (one of 394 entries):

```
AudioTask(
  task_id      = "task_id_/home/user/example_audio/fleurs_en/dev/10146705666908229607.wav",
  dataset_name = "Fleurs_en_us_dev_./example_audio/fleurs_en",
  filepath_key = "audio_filepath",
  data = {
    "audio_filepath": "/home/user/example_audio/fleurs_en/dev/10146705666908229607.wav",
    "text": "The major religion in Moldova is Orthodox Christian."
  }
)
```

*(Only 2 keys: `audio_filepath` and `text`.)*

### Stage 2: `ASRStage` + `NeMoASRAdapter` (GPU)

The generic stage loads and normalizes each current-batch waveform, performs
model-safe segmentation, and plans capacity-bounded calls from the current
candidate window. The NeMo adapter loads `nvidia/parakeet-tdt-0.6b-v2` onto the
GPU and runs one batched `transcribe()` per planned call. The stage stitches
segments, restores parent order, and writes predictions back **in-place**.

**Output** — `data` gains `pred_text`:

```json
{
  "audio_filepath": "/home/user/example_audio/fleurs_en/dev/10146705666908229607.wav",
  "text": "The major religion in Moldova is Orthodox Christian.",
  "pred_text": "The major religion in Moldova is Orthodox Christian."
}
```

*(3 keys now.  `pred_text` is the model's hypothesis — perfect match here.)*

### Stage 3: `GetPairwiseWerStage`

Computes word-error-rate between `text` and `pred_text`.

**Output** — `data` gains `wer`:

```json
{
  "audio_filepath": "...",
  "text": "The major religion in Moldova is Orthodox Christian.",
  "pred_text": "The major religion in Moldova is Orthodox Christian.",
  "wer": 0.0
}
```

*(4 keys.  WER is in percent — 0.0% means a perfect transcription.)*

### Stage 4: `GetAudioDurationStage`

Opens the WAV file with `soundfile`, reads `shape[0] / samplerate`.

**Output** — `data` gains `duration`:

```json
{
  "audio_filepath": "...",
  "text": "The major religion in Moldova is Orthodox Christian.",
  "pred_text": "The major religion in Moldova is Orthodox Christian.",
  "wer": 0.0,
  "duration": 4.92
}
```

*(5 keys.  Duration is 4.92 seconds.)*

### Stage 5: `PreserveByValueStage`

Filters: keep only entries where `wer <= 75.0`.

- This entry has `wer = 0.0` → **kept** (returns same task).
- An entry with `wer = 88.5` would be **dropped** (returns `None`).

In this run all 394 entries passed (max WER was 50.0% — the Parakeet
model transcribes English FLEURS very accurately).

**Output**: unchanged task, or entry removed from pipeline.

### Stage 6: `AudioToDocumentStage`

Converts the `AudioTask` into a `DocumentBatch` for downstream text
stages (e.g. `JsonlWriter`).  With `batch_size=1` the output is a
single-row `pd.DataFrame`:

```
DocumentBatch(
  task_id      = "task_id_/home/user/.../10146705666908229607.wav,...",
  dataset_name = "Fleurs_en_us_dev_./example_audio/fleurs_en",
  data = pd.DataFrame({
    "audio_filepath": ["/home/user/example_audio/fleurs_en/dev/10146705666908229607.wav"],
    "text":           ["The major religion in Moldova is Orthodox Christian."],
    "pred_text":      ["The major religion in Moldova is Orthodox Christian."],
    "wer":            [0.0],
    "duration":       [4.92]
  })
)
```

### Stage 7: `JsonlWriter`

Writes each row of the DataFrame as one JSON line to
`./example_audio/fleurs_en/result/`:

```json
{"audio_filepath": "/home/user/example_audio/fleurs_en/dev/10146705666908229607.wav", "text": "The major religion in Moldova is Orthodox Christian.", "pred_text": "The major religion in Moldova is Orthodox Christian.", "wer": 0.0, "duration": 4.92}
```

### Summary table

| Stage | Keys in `data` | Type out |
|---|---|---|
| `CreateInitialManifestFleursStage` | `audio_filepath`, `text` | `AudioTask` |
| `ASRStage` + `NeMoASRAdapter` | + `pred_text` | `AudioTask` |
| `GetPairwiseWerStage` | + `wer` | `AudioTask` |
| `GetAudioDurationStage` | + `duration` | `AudioTask` |
| `PreserveByValueStage` | (unchanged or dropped) | `AudioTask` |
| `AudioToDocumentStage` | (all 5 keys) | `DocumentBatch` |
| `JsonlWriter` | — | file on disk |

### Contrast: high-WER entry

For comparison, here is a real entry where the model struggled (WER = 50%):

```json
{
  "text": "The Tibetan Buddhism is based on the teachings of Buddha, but were extended by the mahayana path of love and by a lot of techniques from Indian Yoga.",
  "pred_text": "The Tibetan Buddhism is based on the teachings of Buddha but were extended by Mahayana by the Mahayana Deputy Buddha.",
  "wer": 50.0,
  "duration": 8.88
}
```

With `--wer_threshold 75`, this entry still passes.  At a stricter
threshold like `--wer_threshold 30`, it would be dropped by
`PreserveByValueStage`.

## End-to-end `AudioTask` trace (ALM pipeline)

Below is a real entry from `fused_ia_top3.jsonl` (a 2-speaker, 1041s
Internet Archive podcast) flowing through every stage in
`tutorials/audio/alm/pipeline.yaml`.

Pipeline: read manifest → build windows → filter overlap → write.

### Stage 0: `ManifestReader` (CompositeStage)

Decomposes into `FilePartitioningStage` + `ManifestReaderStage`.
Reads the JSONL line-by-line (no Pandas), emits one `AudioTask` per
entry.

**Output**:

```
AudioTask(
  task_id      = <auto>,
  dataset_name = <auto>,
  data = {
    "id": "podcasts_non_stream_eng_only_686",
    "audio_filepath": "/local/.../0300-FDR_300_Guest_Host.mp3",
    "duration": 1040.688,
    "audio_sample_rate": 22050,
    "sample_rate": 16000,
    "language": "en",
    "segments": [ ... 77 dicts ... ],
    "text": "Good evening everybody, it's Steph, and I'd like ...",
    "alignment": [ ... 2843 items ... ],
    ...                            ← 54 top-level keys total
  }
)
```

Each segment in the input has `speaker`, `start`, `end`, `text`,
`text_ITN`, `metrics` (PESQ, STOI, SI-SDR, bandwidth), and `words`
(word-level timestamps).  Here are the 5 original segments (indices
33–37) that become the first training window:

```json
[
  {
    "speaker": "..._SPEAKER_00",
    "start": 510.97, "end": 516.79,
    "text": "Sure. I mean if you feel that it would be helpful I'd be more than happy to approach. Well",
    "metrics": {"pesq_squim": 3.429, "stoi_squim": 0.991, "sisdr_squim": 25.066, "bandwidth": 11000, "hallucination": false},
    "words": [{"word": "Sure.", "start": 510.88, "end": 511.12}, {"word": "I", "start": 511.20, "end": 511.36}, ... ]
  },
  {
    "speaker": "..._SPEAKER_01",
    "start": 516.84, "end": 534.19,
    "text": "no actually I don't think it's going to be very helpful because I think you've made yourself perfectly clear in the podcasts and perfectly clear to the listeners that you just really don't value the family. You think that everybody is corrupt or immoral or amoral or even to use the term",
    "metrics": {"pesq_squim": 3.015, "stoi_squim": 0.96, "sisdr_squim": 17.852, "bandwidth": 11062, "hallucination": false},
    "words": [ ... 51 words ... ]
  },
  {
    "speaker": "..._SPEAKER_01",
    "start": 534.56, "end": 540.20,
    "text": "evil which a lot of people have a hard time with, I mean, evil is such a really, really strong term",
    "metrics": {"pesq_squim": 3.272, "stoi_squim": 0.974, "sisdr_squim": 21.181, "bandwidth": 10812, "hallucination": false},
    "words": [ ... 21 words ... ]
  },
  {
    "speaker": "..._SPEAKER_01",
    "start": 540.44, "end": 541.70,
    "text": "and yet you think of",
    "metrics": {"pesq_squim": 2.497, "stoi_squim": 0.953, "sisdr_squim": 17.39, "bandwidth": 11062, "hallucination": false},
    "words": [{"word": "and", "start": 540.48, "end": 540.64}, {"word": "yet", "start": 540.72, "end": 540.96}, {"word": "you", "start": 541.04, "end": 541.20}, {"word": "think", "start": 541.20, "end": 541.36}, {"word": "of", "start": 541.36, "end": 541.52}]
  },
  {
    "speaker": "..._SPEAKER_00",
    "start": 625.30, "end": 627.58,
    "text": "no, I'd be happy to.",
    "metrics": {"pesq_squim": 2.946, "stoi_squim": 0.978, "sisdr_squim": 17.429, "bandwidth": 11062, "hallucination": false},
    "words": [{"word": "no,", "start": 625.36, "end": 625.52}, {"word": "I'd", "start": 626.16, "end": 626.56}, {"word": "be", "start": 626.56, "end": 626.64}, {"word": "happy", "start": 626.64, "end": 626.96}, {"word": "to.", "start": 626.96, "end": 627.04}]
  }
]
```

### Stage 1: `ALMDataBuilderStage`

Filters segments by bandwidth (≥ 8000), sample rate (≥ 16000), and
speaker count (2–5).  Creates sliding windows of 120s ± 10%.  Drops
`words` from window segments and `words`/`segments` from the top level.

From 77 segments (639.2s total), the builder produces **3 windows**:
- 5 lost to low bandwidth
- 17 lost to speaker-count constraints (single speaker)
- 52 lost to duration not fitting the 108–132s target
- 12 truncation events (segments cut at window boundary)

**Output** — `data` changes:

- Top-level `segments` and `words` **removed** (per `drop_fields_top_level`)
- `windows`, `stats`, `truncation_events` **added**
- 55 keys total (was 54; lost 2, gained 3)

First window (in its entirety):

```json
{
  "segments": [
    {
      "speaker": "podcasts_non_stream_eng_only_686_SPEAKER_00",
      "start": 510.97221875,
      "end": 516.79409375,
      "text": "Sure. I mean if you feel that it would be helpful I'd be more than happy to approach. Well",
      "text_ITN": "Sure. I mean if you feel that it would be helpful I'd be more than happy to approach. Well",
      "metrics": {
        "pesq_squim": 3.429, "stoi_squim": 0.991,
        "sisdr_squim": 25.066, "bandwidth": 11000, "hallucination": false
      }
    },
    {
      "speaker": "podcasts_non_stream_eng_only_686_SPEAKER_01",
      "start": 516.8447187500001,
      "end": 534.1922187499999,
      "text": "no actually I don't think it's going to be very helpful because I think you've made yourself perfectly clear in the podcasts and perfectly clear to the listeners that you just really don't value the family. You think that everybody is corrupt or immoral or amoral or even to use the term",
      "text_ITN": "no actually I don't think it's going to be very helpful ...",
      "metrics": {
        "pesq_squim": 3.015, "stoi_squim": 0.96,
        "sisdr_squim": 17.852, "bandwidth": 11062, "hallucination": false
      }
    },
    {
      "speaker": "podcasts_non_stream_eng_only_686_SPEAKER_01",
      "start": 534.5634687500001,
      "end": 540.1997187500001,
      "text": "evil which a lot of people have a hard time with, I mean, evil is such a really, really strong term",
      "text_ITN": "evil which a lot of people have a hard time with, I mean, evil is such a really, really strong term",
      "metrics": {
        "pesq_squim": 3.272, "stoi_squim": 0.974,
        "sisdr_squim": 21.181, "bandwidth": 10812, "hallucination": false
      }
    },
    {
      "speaker": "podcasts_non_stream_eng_only_686_SPEAKER_01",
      "start": 540.43596875,
      "end": 541.70159375,
      "text": "and yet you think of",
      "text_ITN": "and yet you think of",
      "metrics": {
        "pesq_squim": 2.497, "stoi_squim": 0.953,
        "sisdr_squim": 17.39, "bandwidth": 11062, "hallucination": false
      }
    },
    {
      "speaker": "podcasts_non_stream_eng_only_686_SPEAKER_00",
      "start": 625.3003437500001,
      "end": 627.57846875,
      "text": "no, I'd be happy to.",
      "text_ITN": "no, I'd be happy to.",
      "metrics": {
        "pesq_squim": 2.946, "stoi_squim": 0.978,
        "sisdr_squim": 17.429, "bandwidth": 11062, "hallucination": false
      }
    }
  ],
  "speaker_durations": [24.25, 8.10, 0.0, 0.0, 0.0]
}
```

Note: `words` arrays are **gone** from the window segments (dropped
by `drop_fields="words"`).  The window spans 510.97s–627.58s
(116.6s duration, within the 108–132s target).  Two speakers
contributed 24.25s and 8.10s of speech respectively.

Stats produced for this entry:

```json
{
  "total_segments": 77,
  "total_dur": 639.19,
  "audio_sample_rate": 22050,
  "lost_bw": 5,      "dur_lost_bw": 4.08,
  "lost_sr": 0,      "dur_lost_sr": 0.0,
  "lost_spk": 17,    "dur_lost_spk": 227.44,
  "lost_win": 52,    "dur_lost_win": 382.22,
  "lost_no_spkr": 0, "dur_lost_no_spkr": 0.0,
  "lost_next_seg_bm": 38, "dur_lost_next_seg_bm": 256.01
}
```

### Stage 2: `ALMDataOverlapStage`

Filters overlapping windows (threshold = 50%).  Of the 3 input
windows, 1 is removed due to overlap, leaving **2 filtered windows**
with a combined duration of 248.5s.

**Output** — `data` gains 9 new keys (64 total):

- `filtered_windows`: the 2 surviving windows (same structure as above)
- `filtered_dur`: `248.5` (seconds)
- `filtered_dur_list`: `[116.6, 131.9]`
- `total_dur_window`: `359.2` (all 3 windows before filtering)
- `filtered`, `total_dur_list_window`, `total_dur_list_window_timestamps`
- `manifest_filepath`, `swift_filepath`

The two surviving windows:

| Window | Time range | Duration | Segments |
|---|---|---|---|
| 0 | 510.97s – 627.58s | 116.6s | 5 |
| 1 | 625.30s – 757.20s | 131.9s | 4 |

### Stage 3: `ManifestWriterStage`

Appends the entry as a single JSON line to
`./alm_output/alm_output.jsonl` (351 KB for this entry).

### ALM summary table

| Stage | Keys in `data` | Notable changes | Mutates in-place? |
|---|---|---|---|
| `ManifestReader` | 54 (all original) | N/A (creates from JSONL line) | N/A |
| `ALMDataBuilderStage` | 55 (−2, +3) | Drops `segments`/`words`; adds `windows`, `stats`, `truncation_events` | Yes (clear + update) |
| `ALMDataOverlapStage` | 64 (+9) | Adds `filtered_windows`, `filtered_dur`, overlap metadata | Yes (clear + update) |
| `ManifestWriterStage` | — | Writes JSON line to disk, returns `AudioTask` | N/A |

---

## Quick checklist for adding a new audio stage

1. Subclass `ProcessingStage[AudioTask, AudioTask]`
2. Order dataclass fields: `name` first, stage-specific params, then `resources`, then `batch_size`
3. Implement `inputs()` and `outputs()` to declare required/produced keys
4. For CPU stages: override `process(task: AudioTask) -> AudioTask | None`
   — mutate `task.data` in-place and return `task` (or `None` to filter)
5. For filtering stages: override `process_batch` to return only passing
   entries (see `PreserveByValueStage` in `common.py`); `process()` should
   raise `NotImplementedError`
6. For GPU / IO stages: override `process_batch(tasks) -> list[AudioTask]`,
   call `self.validate_input(task)` per task at the top, guard with
   `if len(tasks) == 0: return []`.  `process()` should raise
   `NotImplementedError` (matching the dedup-stage convention).
7. Declare GPU resources via `.with_(resources=Resources(gpus=1.0))`
8. Add tests in `tests/stages/audio/` using `AudioTask` for fixtures

---

## Local duration bucketing: complete theory and current contract

This section describes the local duration-bucketing algorithm currently
implemented by [`ASRStage`](inference/asr/stage.py), why it is structured this
way, and the contract another audio GPU inference stage must preserve to adopt
the same approach. This is the top-level conceptual contract; the dedicated
[audio inference guide](inference/README.md) adds ASR-specific configuration,
implementation, and test references.

Local bucketing solves one specific problem: candidate audio inputs in one GPU
batch can have very different durations, while the model usually pads every
input to the longest input in that batch. The stage therefore uses duration to
reorder only the finite set of model inputs already present in one
`process_batch()` call. It does not buffer globally or change which parent
rows belong to that call.

### The four controls are independent

| Control | Current `ASRStage` default | Boundary it controls |
|---|---:|---|
| `batch_size` | `32` | Number of parent `AudioTask` rows the backend normally offers to one `process_batch()` planning window |
| `max_inference_duration_s` | `2400` | User-supplied maximum duration of one segment derived from a parent waveform |
| `max_audio_sec_per_actor` | required | Maximum padded-audio proxy cost of one adapter call |
| `local_bucketing` | `false` | Whether segments are stable-sorted by duration and DP-partitioned over contiguous spans instead of greedily packed in input order |

These settings are deliberately not aliases for one another:

- `batch_size` bounds the backend candidate window; it is not an adapter item
  limit. `ASRStage.process_batch()` itself does not reject a directly supplied
  list merely because it contains more than `batch_size` parents.
- `max_inference_duration_s` enforces the configured single-input ceiling; it
  applies with bucketing both on and off. The stage does not discover or
  verify the selected model's true limit.
- `max_audio_sec_per_actor` bounds every adapter invocation produced from the
  current window; it is not the total audio accepted by `process_batch()`.
- `local_bucketing` changes planning order and boundary selection only; it
  does not enable or disable segmentation.

There is no separate `adapter_batch_size`, maximum-items-per-bucket map,
static set of bucket edges, timer, queue, flush hook, or generic item-cost
feature. Despite its name, `max_audio_sec_per_actor` is a per-adapter-call
budget planned by the actor. It is not accumulated over the actor's lifetime
or across multiple `process_batch()` calls.

Configuration validation matches those boundaries:

- `max_audio_sec_per_actor` is required, must be a numeric `Real` other than
  `bool`, and must be finite and greater than zero;
- `local_bucketing` must be an actual Boolean, so integer `1` is rejected;
- `max_inference_duration_s` must be finite and greater than zero, and cannot
  exceed `max_audio_sec_per_actor`;
- `batch_size` and `target_sample_rate` are converted to integers and must be
  greater than zero.

The planner consumes durations generated internally from prepared segments;
it does not expose an API for arbitrary caller-provided cost features.

### Parent rows, segments, and adapter calls

Three different units pass through the stage:

1. A **parent row** is one `AudioTask` received in `process_batch(tasks)`.
2. A **segment** is one model-safe waveform prepared from a parent. A short
   parent normally produces one segment; a long parent produces several.
3. An **adapter call** is one capacity-bounded group of segments passed to
   `adapter.transcribe_batch()`. Its segments may come from multiple parents.

The complete data flow is:

```text
backend candidate rows (`batch_size`)
    -> skip/eligibility checks
    -> load, downmix, and resample each eligible waveform
    -> model-safe segmentation of every eligible parent
    -> one flat segment list for the current `process_batch()` call
    -> optional stable duration sort
    -> padded-seconds partitioning
    -> one adapter invocation per planned group
    -> scatter results to original segment positions
    -> stitch each parent's segments in temporal order
    -> return parent rows in their original order
```

Rows with a reused output, rows rejected by a configured language allowlist
(including a missing code while that allowlist is active), and rows whose
waveforms fail preparation without raising do not enter the segment plan.
Every other segment from every eligible row in the same
`process_batch()` participates in one shared local planning window. No segment
is retained for a later backend batch, actor, or worker.

### Why the budget measures padded audio

For a nonempty adapter call with segment durations
`d[0], ..., d[n - 1]`, define:

```text
useful audio seconds = sum(d)
padded audio seconds = n * max(d)
padding waste        = n * max(d) - sum(d)
padding efficiency   = sum(d) / (n * max(d)), when max(d) > 0
```

For an all-zero-duration call, padded cost and padding waste are both zero,
while this efficiency ratio is undefined rather than `100%`.

The feasibility condition for every planned call is:

```text
n * max(d) <= max_audio_sec_per_actor
```

This is intentionally not `sum(d)`. If durations `[1, 10]` are sent together,
the shorter waveform is normally padded to 10 seconds and the model processes
a tensor representing approximately `2 * 10 = 20` audio seconds, not 11.
Grouping similar durations reduces work that carries no useful audio.

Padded seconds are a useful proxy, not a proof of GPU-memory safety. Actual
memory and runtime also depend on model architecture, precision, decoder
state, framework workspaces, fixed per-item costs, and adapter-specific
behavior. The budget must therefore be measured and tuned for each model,
adapter, and hardware configuration. The proxy is most meaningful when the
adapter forms a jointly padded batch. A serial, ragged, or packed adapter may
receive the same planned groups without realizing the expected padding
benefit.

### Model-safe segmentation always precedes bucketing

After waveform preparation, `ASRStage` calls
[`plan_audio_segments`](model_input_segmentation.py) for every eligible parent.
For target sample rate `s` and duration ceiling `D`, the segment limit in
samples is:

```text
max_samples = int(D * s)
```

`D` must cover at least one sample. The current planner creates contiguous,
nonoverlapping ranges from sample zero to the end of the waveform. Every
remainder becomes a segment, an exact multiple creates no empty tail, and a
zero-sample waveform remains representable as one zero-duration segment.
`audio_seconds` is calculated from the actual prepared segment length and
target sample rate, rather than trusted from manifest metadata.

Construction requires:

```text
max_inference_duration_s <= max_audio_sec_per_actor
```

This guarantees that every permitted individual segment can at least fit in a
singleton adapter call.

The current ASR segmentation has an important semantic limitation: the cuts
are hard and have no shared audio context. Each segment is decoded
independently, and the current stitcher joins nonempty transcript strings with
a space. A word crossing a cut can consequently be omitted, duplicated, or
decoded differently. Local bucketing neither causes nor fixes that behavior;
adding overlap would also require a model-appropriate transcript
reconciliation algorithm. Other audio stages must define their own safe
segmentation and reconstruction semantics rather than copying ASR text
concatenation blindly.

### Planning when `local_bucketing=true`

Each prepared segment enters the planner as:

```text
(original_segment_index, adapter_item, audio_seconds)
```

The enabled planner then performs these steps:

1. Stable-sort all segments by ascending `audio_seconds`. Equal-duration
   segments retain their original relative order.
2. Restrict each adapter call to a contiguous span of that sorted sequence.
3. Consider every span whose padded cost fits the actor budget.
4. Use dynamic programming to select the complete partition with the fewest
   adapter calls.
5. Among partitions with that minimum call count, select the one with the
   fewest total padded seconds.
6. Execute calls in planned order and scatter every result through the saved
   original segment index.

For sorted durations `d[0] <= ... <= d[N - 1]`, the cost of span
`[start, stop)` is:

```text
span_count = stop - start
span_max   = d[stop - 1]
span_cost  = span_count * span_max
```

A span is feasible when `span_cost` is within the budget. The implementation
accepts representation-level equality using `math.isclose` with
`rel_tol=1e-12` and `abs_tol=1e-9`, avoiding a spurious split for values such
as three 0.1-second segments under a 0.3-second budget.

The dynamic-programming suffix state can be written as:

```text
best[N] = (0 calls, 0 padded seconds)

best[start] = lexicographic minimum over every feasible stop:
    (
        1 + best[stop].calls,
        cost(start, stop) + best[stop].padded_seconds,
    )
```

The selected `stop` is stored for each `start`, then those boundaries are
followed from zero to reconstruct the calls. Python tuple comparison gives the
required calls-first, padded-seconds-second ordering. Exact score ties retain
the first feasible boundary encountered. Score comparison uses the raw float
totals; the small tolerance described above applies only to budget
feasibility.

This is an exact optimum for the implementation's stable duration-sorted,
contiguous-span planning space. It is not an optimizer over arbitrary
noncontiguous subsets, other `process_batch()` calls, actors, workers, or the
whole dataset.

### Why sorted greedy filling is not enough

Greedily extending a sorted call until the next segment would breach the
budget makes the current call as full as possible, but can make the complete
plan do unnecessary padded work.

For sorted durations `[1, 2, 2]` and budget `4`, greedy produces:

```text
[1, 2] -> 2 * 2 = 4 padded seconds
[2]    -> 1 * 2 = 2 padded seconds
score  -> 2 calls, 6 padded seconds
```

The current dynamic program chooses:

```text
[1]    -> 1 * 1 = 1 padded second
[2, 2] -> 2 * 2 = 4 padded seconds
score  -> 2 calls, 5 padded seconds
```

It deliberately leaves room unused in the first call because doing so keeps
the same two-call count and eliminates one padded second overall. Three
singleton calls would also use five padded seconds, but lose on the primary
call-count objective. Prioritizing call count prevents the padding objective
from degenerating into one adapter call per segment and minimizes adapter
launches before optimizing the padded-work proxy; it is an objective, not a
guarantee of the lowest wall-clock time.

### Example across several parent rows

Suppose one `process_batch()` receives three parents with prepared durations
`250`, `10`, and `1` seconds, with:

```text
max_inference_duration_s = 120
max_audio_sec_per_actor  = 240
local_bucketing           = true
```

Segmentation happens first:

```text
parent A -> [120, 120, 10]
parent B -> [10]
parent C -> [1]
```

The shared sorted planning sequence is `[1, 10, 10, 120, 120]`, and the
selected adapter calls are:

```text
[1, 10, 10] -> 3 * 10  = 30 padded seconds
[120, 120]  -> 2 * 120 = 240 padded seconds
```

The 10-second remainder from parent A can share a call with segments from B
and C. After inference, saved indices restore temporal and parent ownership:
parent A receives its `120`, `120`, and `10` results in that order, while B
and C each receive their own result. Adapter-call order never becomes output
row order.

### Planning when `local_bucketing=false`

The disabled mode is the input-order control. It keeps the flattened segment
order and greedily extends the current call while its candidate padded cost
fits. When the next segment would breach the budget, it emits the current call
and starts another with that segment.

Disabling local bucketing therefore disables the duration sort and the
whole-window dynamic-programming boundary optimization, but it does not
disable either segmentation or the
`max_audio_sec_per_actor` constraint. The same adapter and result-restoration
contract applies in both modes.

For example, original durations `[8, 2, 7, 3]` under budget `16` produce:

```text
input-order greedy:
    [8, 2] -> 16 padded seconds
    [7, 3] -> 14 padded seconds
    total  -> 30 padded seconds

local bucketing, sorted order [2, 3, 7, 8]:
    [2, 3] ->  6 padded seconds
    [7, 8] -> 16 padded seconds
    total  -> 22 padded seconds
```

Both plans process the same 20 useful seconds in two calls, but local
bucketing presents substantially less padding to the model.

### Adapter execution, result restoration, and ASR stitching

Every selected group causes exactly one `adapter.transcribe_batch(items)`
invocation. `NeMoASRAdapter` filters zero-length waveforms while preserving
their result slots. If any nonempty waveforms remain, it then makes one
`model.transcribe(audio=waveforms, batch_size=len(waveforms), ...)` call for
that planned group. Consequently, one `process_batch()` can make zero, one, or
many model calls.

The adapter must return exactly one result for every submitted item. The stage
places each result into an array indexed by the segment's position before
bucketing and rejects incomplete or wrong-sized result sets. It then groups
those aligned segment results by parent, preserving the parent's temporal
segment order. This restoration relies on the adapter's ordered contract of
one result per input; matching result counts alone cannot detect an adapter
that internally permutes its outputs.

For a multi-segment ASR parent, the current stitcher:

- strips and space-joins nonempty segment transcripts;
- marks the parent skipped if any segment was skipped;
- retains the first available skip reason and unsupported-language value;
- merges adapter `extras` dictionaries in segment order, with a later value
  replacing an earlier value for the same key.

Finally, predictions are written to the same parent `AudioTask` objects and
the original parent-row order is returned. Local execution order is therefore
an internal optimization, not an externally visible row reorder.

### Correctness invariants

The implementation is correct only if all of the following remain true:

1. Model-safe segmentation runs whether bucketing is enabled or disabled.
2. Duration is calculated once from each final prepared segment.
3. Every eligible segment enters exactly one adapter call.
4. Every nonempty call satisfies
   `len(call) * max(segment_duration) <= max_audio_sec_per_actor`, allowing
   only the documented representation-level floating-point tolerance.
5. Disabled mode greedily preserves original segment order.
6. Enabled mode uses stable ascending-duration order and chooses the
   contiguous partition with minimum call count and then minimum total padded
   seconds.
7. Every adapter call returns one result per submitted segment.
8. Results are scattered to original segment positions before parent
   assembly.
9. Each parent's segment results are reconstructed in temporal order.
10. For successfully returned, ordered adapter results, scatter preserves
    parent association and parent-row order. The planner cannot guarantee that
    a stateful adapter's outputs or exceptions are independent of call
    composition or execution order.
11. No queued audio, timer, flush obligation, or planner state survives the
    current `process_batch()` call.

The adapter and loaded model remain worker-local and may persist across calls;
that lifecycle is separate from the deliberately stateless local planner.

### Applying the same design to another audio GPU stage

The planner currently lives in `ASRStage`; it is not a generic base-class
hook. It is appropriate for another stage only when independent audio inputs
can be grouped safely, each result can be mapped back unambiguously, and
`item_count * longest_duration` meaningfully approximates that adapter's
jointly padded work. Serial, ragged, or packed execution may need a different
cost model or may gain nothing from duration bucketing.

To apply the pattern:

1. Keep the backend `batch_size`, required padded-seconds budget, and
   `local_bucketing` Boolean as distinct controls.
2. Validate and prepare all eligible parents within one finite
   `process_batch()` call.
3. Apply any model-specific maximum-input segmentation unconditionally.
4. Flatten the resulting model items while saving original item index, parent
   index, segment ordinal, timestamps, output paths, and any other metadata
   required to reconstruct results.
5. Measure duration from the final resampled or sliced waveform.
6. Preserve input order and greedily pack when bucketing is disabled.
7. Stable-sort by duration and use the calls-first, padded-seconds-second
   dynamic program over contiguous spans when bucketing is enabled.
8. Call the adapter once per planned group, require a complete ordered
   one-to-one result mapping, scatter by saved index, and only then assemble
   parent outputs. How many native model calls the adapter makes is
   adapter-specific.
9. Test both modes against the same correctness oracle before measuring
   performance.

Reordering independent whole inputs is often safe; segmentation is
model-specific:

- ASR must define how boundary transcripts are reconciled.
- SED must restore valid-frame counts, frame offsets, timestamps, and sidecar
  paths.
- VAD splits can change onset/offset decisions and merged speech regions.
- Diarization may require whole-recording speaker clustering and identity.
- Alignment must keep every hypothesis paired with the right segment
  metadata.
- File-producing stages must derive paths from saved parent metadata rather
  than reordered call positions.

Do not add buffering across `process_batch()` calls merely to improve the
duration distribution. That would change latency, ownership, failure, and
end-of-stream semantics and would require a separate queue-and-flush design.

### Tuning and measurement

There is no universal safe or optimal audio-seconds budget:

1. Choose `max_inference_duration_s` from the model's semantic and technical
   single-input limit first.
2. Start `max_audio_sec_per_actor` from a known-safe uniform model call. If
   `k` inputs of duration `d` are safe, `k * d` is a reasonable initial proxy
   budget. Keep it at least as large as `max_inference_duration_s`.
3. Exercise the longest allowed singleton, then increase the budget gradually
   while observing GPU memory, throughput, latency, and failures.
4. Benchmark representative short, medium, long, exact-boundary, and remainder
   inputs. Average duration hides padding and tail effects.
5. Compare enabled and disabled modes using identical candidate windows,
   model settings, software, input order, and hardware. Establish output
   parity before comparing performance.
6. Tune backend `batch_size` separately. A small window offers few regrouping
   choices; a very large window increases waveform-preparation latency and
   host memory before inference begins.

A duration budget cannot represent fixed per-item memory. A finite candidate
window containing many very short or zero-duration items can therefore
produce a large item-count call. If a model has a hard native item-count
limit, enforce or expose that adapter-specific constraint at the model
boundary rather than silently redefining `batch_size`.

### Complexity, guarantees, and non-goals

For `N` prepared segments:

- bucketing disabled: `O(N)` greedy planning;
- bucketing enabled: `O(N log N)` stable sorting plus `O(N^2)` dynamic
  programming, for `O(N^2)` total time and `O(N)` auxiliary storage.

Given the same prepared segment sequence and configuration, planning is
deterministic. Enabled mode guarantees the minimum number of calls and then
minimum padded seconds within its sorted contiguous-span search space. It
does not guarantee lower end-to-end latency or higher throughput for every
model and duration distribution. The model-time or resource savings from
reduced padding must exceed the added sorting and dynamic-programming cost for
the optimization to pay off.

The current design intentionally does not provide:

- global, partition-wide, cross-worker, or cross-`process_batch()` bucketing;
- queues, timers, flush hooks, or end-of-stream state;
- static duration buckets or per-bucket configuration;
- a separate adapter batch size or per-bucket item-count map;
- feature-weighted cost estimation;
- arbitrary noncontiguous grouping after the stable duration sort;
- adaptive OOM retries or automatic budget tuning;
- a GPU-memory-safety proof from the duration proxy;
- a throughput win for every workload;
- boundary-safe ASR overlap and transcript reconciliation.

### Tests a reusable implementation needs

At minimum, tests should cover:

- missing, Boolean, nonnumeric, zero, negative, `NaN`, and infinite budgets;
- a model limit larger than the actor budget and a limit shorter than one
  sample;
- empty input, zero-duration audio, exact fills, and floating-point boundary
  equality;
- original-order packing, stable duration ordering, and equal-duration
  stability;
- the `[1, 2, 2]` under budget `4` case that distinguishes the dynamic program
  from sorted greedy filling;
- segments from multiple parent rows sharing one call;
- strict isolation between separate `process_batch()` calls;
- exact model-duration boundaries, long parents, and every final remainder;
- exact-once submission, wrong adapter result counts, scatter, stitching,
  skip, and error behavior;
- proof that backend `batch_size` is not treated as an adapter item cap;
- parity of user-visible outputs, using exact equality or the stage's
  documented tolerance, between enabled and disabled modes before comparing
  padding efficiency, throughput, latency, RAM, or VRAM.
