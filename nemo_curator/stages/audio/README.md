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

Override **`process_batch`** for batched processing, and provide a thin
**`process`** that delegates to it (to satisfy the ABC contract).

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
        return self.process_batch([task])[0]

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
| Override | `process` | `process_batch` (+ thin `process` delegating to it) |
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
- `ALMManifestWriterStage` (`alm/alm_manifest_writer.py`) — writes
  entries to JSONL, returns `FileGroupTask`.

### Setting `batch_size` for GPU inference

The `batch_size` field on a GPU stage controls how many `AudioTask` tasks
the backend groups into a single `process_batch()` call.  This directly
determines how many files are passed to your model in one batched GPU
inference call.

**Defining batch_size in the stage class:**

```python
@dataclass
class InferenceAsrNemoStage(ProcessingStage[AudioTask, AudioTask]):
    batch_size: int = 16      # default for this stage
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
```

The default is a sensible starting point; pipeline authors can override it
at pipeline construction time without modifying the stage class:

**Overriding batch_size at pipeline construction:**

```python
pipeline.add_stage(
    InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2")
    .with_(resources=Resources(gpus=1), batch_size=32)
)
```

The `.with_()` method sets any stage field.  Here it bumps `batch_size`
from the default `16` to `32` and assigns 1 GPU.

**Overriding batch_size via Hydra YAML:**

```yaml
pipeline:
  stages:
    - _target_: nemo_curator.stages.audio.inference.asr_nemo.InferenceAsrNemoStage
      model_name: nvidia/parakeet-tdt-0.6b-v2
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
        → your override receives exactly batch_size tasks
          (or fewer for the last batch)
```

**Choosing a good batch_size:**

- **Too small** (e.g. `1`) — GPU is underutilised; kernel launch overhead
  dominates.  Each call processes one file, losing the benefit of batching.
- **Too large** (e.g. `1024`) — may exceed GPU memory (OOM), especially
  with long audio files or large models.
- **Sweet spot** — depends on model size, audio length, and GPU VRAM.
  Start with `16` and increase until you see OOM or throughput plateaus.
  For NeMo ASR FastConformer models, `16–64` is typical on a single GPU.

## What you must always declare

Every stage (CPU or GPU) should declare:

- **`inputs()`** — which dict keys must be present.  The base class
  validates these before your code runs.
- **`outputs()`** — which dict keys your stage produces.  Used for
  pipeline introspection and documentation.
- **`name`** — a human-readable stage name for logging and metrics.

## Filtering entries

To drop an entry from the pipeline:

- **CPU filter stage**: return `None` from `process()` **and** override
  `process_batch` to filter out `None` results before returning.  This is
  necessary because the base `ProcessingStage.process_batch` appends all
  results (including `None`) to the list, and the backend adapter calls
  `task.add_stage_perf()` on every element.  See `PreserveByValueStage`
  in `common.py` for the canonical pattern.
- **GPU / IO stage**: omit the entry from the returned list in `process_batch`.

## Method reference

Audio stages use two methods from `ProcessingStage`:

```
process(AudioTask) -> AudioTask | None
    The primary hook for CPU stages.  Receives a single task, mutates
    task.data in-place, and returns the task (or None to filter).
    GPU stages provide a thin wrapper that delegates to process_batch.

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
- GPU stages override `process_batch` to receive the full batch for one
  batched kernel call.  Their `process()` simply delegates to
  `process_batch([task])[0]`.

## Optimizations in the base class

1. **Aggregated IO conversion** — `AudioToDocumentStage` overrides
   `process_batch` to combine N `AudioTask` dicts into one multi-row
   `pd.DataFrame` in a single `DocumentBatch`, avoiding N single-row
   DataFrame allocations.

2. **Ray Data compatibility** — empty-batch guards use `len(tasks) == 0`
   instead of `not tasks` because Ray Data's `map_batches` passes
   `tasks` as a numpy array, and `not ndarray` raises `ValueError`
   for arrays with more than one element.  This applies to
   `process_batch` in `InferenceAsrNemoStage` and
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

Each `process_batch([16 tasks])` call goes directly to:
`InferenceAsrNemoStage.process_batch` → `validate_input` per task →
extract filepaths → **one** batched GPU call → mutate each task in-place.

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
| **`batch_size`** | How many tasks each worker processes per call | The stage author (domain knowledge about optimal GPU batch size) |

Total in-flight = `num_workers x batch_size`.  For 4 GPUs with
`batch_size=16`, that is 64 audio files being processed concurrently.

## How `batch_size` travels from your stage to the backend

When you set `batch_size = 16` on a GPU stage, this is the exact path
the value takes until it controls how many tasks land in your
`process_batch` call:

```
InferenceAsrNemoStage                     (your stage dataclass)
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
  calling `process_batch`.  Your stage never has to split or batch tasks itself.

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

### GPU stage (e.g. `InferenceAsrNemoStage`, `batch_size=16`)

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
│     → stages/audio/inference/asr_nemo.py           InferenceAsrNemoStage.setup_on_node()
│       nemo_asr.models.ASRModel.from_pretrained(model_name, return_model_file=True)
│       (downloads model to shared cache — one download per node)
│   backends/xenna/adapter.py                      XennaStageAdapter.setup()
│     → backends/base.py                             stage.setup(worker_metadata)
│       → stages/audio/inference/asr_nemo.py           InferenceAsrNemoStage.setup()
│         map_location = self.check_cuda()           → "cuda"
│         self.asr_model = ASRModel.from_pretrained(model_name, map_location=cuda)
│
├─ Per batch (batch_size=16, so 16 AudioTask tasks per call):
│   backends/xenna/adapter.py                      XennaStageAdapter.process_data(tasks)
│     → backends/base.py                             BaseStageAdapter.process_batch(tasks)
│         ├─ start perf timer
│         ├─ stage.process_batch(tasks)                                                ──────────┐
│         ├─ log stats, attach _stage_perf                                                       │
│         └─ return results                                                                      │
│                                                                                                │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────┘
│   │  InferenceAsrNemoStage.process_batch()       (OVERRIDDEN — batched GPU)
│   │    stages/audio/inference/asr_nemo.py
│   │    validate_input(task) per task              schema check
│   │    files = [t.data[self.filepath_key] for t in tasks]
│   │                                                → list of 16 audio file paths
│   │    texts = self.transcribe(files)
│   │      → stages/audio/inference/asr_nemo.py
│   │        self.asr_model.transcribe(files)
│   │          → ONE batched GPU kernel call for all 16 files
│   │        return [output.text for output in outputs]
│   │    for task, text in zip(tasks, texts):
│   │      task.data[self.pred_text_key] = text     (mutate in-place)
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

**GPU stages** (e.g. `InferenceAsrNemoStage`):  Peak memory is
dominated by **model VRAM**, not task data.  A NeMo ASR
FastConformer-TDT model uses ~2–4 GB of VRAM.  The task data
(`batch_size × entry_size`) is negligible in comparison — 16 FLEURS
entries is 16 × 741 B ≈ 12 KB, while even 16 large ALM entries is
16 × 4 MB ≈ 64 MB (still small vs the model).

## End-to-end `AudioTask` trace (FLEURS pipeline)

Below is a single English FLEURS entry flowing through every stage in
`tutorials/audio/fleurs/pipeline.py`.  All values are **real output**
from running `--lang en_us --model_name nvidia/parakeet-tdt-0.6b-v2
--split dev --wer_threshold 75`.

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

### Stage 2: `InferenceAsrNemoStage` (GPU)

Loads `nvidia/parakeet-tdt-0.6b-v2` onto the GPU.  Receives a batch
of 16 `AudioTask`s, extracts file paths, runs one batched
`transcribe()` call, and writes predictions back **in-place**.

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
| `InferenceAsrNemoStage` | + `pred_text` | `AudioTask` |
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

### Stage 0: `ALMManifestReader` (CompositeStage)

Decomposes into `FilePartitioningStage` + `ALMManifestReaderStage`.
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

### Stage 3: `ALMManifestWriterStage`

Appends the entry as a single JSON line to
`./alm_output/alm_output.jsonl` (351 KB for this entry).

### ALM summary table

| Stage | Keys in `data` | Notable changes | Mutates in-place? |
|---|---|---|---|
| `ALMManifestReader` | 54 (all original) | N/A (creates from JSONL line) | N/A |
| `ALMDataBuilderStage` | 55 (−2, +3) | Drops `segments`/`words`; adds `windows`, `stats`, `truncation_events` | Yes (clear + update) |
| `ALMDataOverlapStage` | 64 (+9) | Adds `filtered_windows`, `filtered_dur`, overlap metadata | Yes (clear + update) |
| `ALMManifestWriterStage` | — | Writes JSON line to disk, returns `FileGroupTask` | N/A |

---

## Quick checklist for adding a new audio stage

1. Subclass `ProcessingStage[AudioTask, AudioTask]`
2. Order dataclass fields: `name` first, stage-specific params, then `resources`, then `batch_size`
3. Implement `inputs()` and `outputs()` to declare required/produced keys
4. For CPU stages: override `process(task: AudioTask) -> AudioTask | None`
   — mutate `task.data` in-place and return `task` (or `None` to filter)
5. For filtering stages: also override `process_batch` to filter out `None`
   results (see `PreserveByValueStage` in `common.py`)
6. For GPU / IO stages: override `process_batch(tasks) -> list[AudioTask]`,
   call `self.validate_input(task)` per task at the top, guard with
   `if len(tasks) == 0: return []`.  Add a thin
   `process(task) -> AudioTask` that delegates to `process_batch([task])[0]`.
7. Declare GPU resources via `.with_(resources=Resources(gpus=1.0))`
8. Add tests in `tests/stages/audio/` using `AudioTask` for fixtures
