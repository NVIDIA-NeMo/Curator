# Audio Checkpointing

This package provides the first checkpointing foundation for `AudioTask`-based
pipelines in NeMo Curator.

The main goal is to make audio pipelines resumable without reprocessing already
completed samples. Instead of changing executor internals, checkpointing is
implemented as an audio-specific orchestration layer on top of the existing
`Pipeline` abstraction.

## What is included

- Stable `sample_key` identity for `AudioTask`
- JSONL manifest serialization for checkpointed task payloads
- Per-stage checkpoint store with:
  - stage metadata
  - per-sample records
  - output manifests
- `AudioCheckpointRunner` for stage-by-stage resumable execution

## Current execution model

`AudioCheckpointRunner`:

1. Calls `pipeline.build()` to flatten composite stages
2. Splits the pipeline into:
   - checkpointed audio stages
   - a finalization tail starting at `AudioToDocumentStage`
3. Runs audio stages one by one
4. Saves stage outputs and per-sample records under `checkpoint_dir`
5. Reuses saved outputs on rerun when a stage is already complete
6. Runs the finalization tail from the last saved `AudioTask` manifest

`AudioToDocumentStage` and `JsonlWriter` are intentionally not checkpointed.
They are treated as finalization stages that can be rerun from the last saved
audio manifest.

## Stable sample identity

Checkpointing relies on stable sample identity rather than runtime `task_id`.

The audio identity helpers live in `nemo_curator.tasks.audio_task`:

- `build_audio_sample_key(...)`
- `ensure_sample_key(...)`
- `carry_sample_key(...)`
- `derive_child_sample_key(...)`

Use them as follows:

- Readers create root identity with `build_audio_sample_key(...)`
- `1:1` transform stages keep identity with `carry_sample_key(...)`
- Fan-out stages create deterministic child identity with
  `derive_child_sample_key(...)`

## Supported input paths

The current foundation supports root `sample_key` generation for:

- `JsonlReader(task_type="audio")` / `JsonlAudioReaderStage`
- `TarredAudioManifestReaderStage`
- `ALMManifestReaderStage`

It also supports identity propagation for several `AudioTask` stages that create
new tasks, including:

- `InferenceSortformerStage`
- `VADSegmentationStage`
- `SpeakerSeparationStage`
- `SegmentConcatenationStage`

## AudioCheckpointRunner API

Import:

```python
from nemo_curator.checkpointing.audio import AudioCheckpointRunner
```

Constructor:

```python
AudioCheckpointRunner(
    *,
    pipeline: Pipeline,
    checkpoint_dir: str,
    executor: BaseExecutor | None = None,
    ignore_failed: bool = False,
)
```

### Arguments

- `pipeline`
  The existing `Pipeline` definition. The runner will call `build()` internally.

- `checkpoint_dir`
  Root directory for checkpoint state. This is required.

- `executor`
  Executor used to run each checkpointed stage. If omitted, the pipeline's
  normal default executor behavior applies when the internal single-stage
  sub-pipelines are executed.

- `ignore_failed`
  If `False`, stage failures stop execution immediately.

  If `True`, a failed batch is retried one task at a time. Successfully retried
  tasks are kept; failed tasks are recorded as `failed_retriable` in the stage
  store and excluded from downstream processing.

## Pipeline config

The runner also reads one optional execution policy from `pipeline.config`:

```python
pipeline = Pipeline(
    name="my_pipeline",
    config={"link_stages_via_io": True},
)
```

### `link_stages_via_io`

- `False` by default
- If `False`, successful audio stage outputs are passed to the next stage
  in-memory, while also being checkpointed to disk
- If `True`, stage outputs are checkpointed to disk and the next stage reloads
  from the saved output manifest

This is useful when stage outputs are too large to keep comfortably in memory.

## Note on `materialization_dir`

`materialization_dir` is configured on `MaterializeTarredAudioStage`, not on
`AudioCheckpointRunner`.

Example:

```python
pipeline.add_stage(
    MaterializeTarredAudioStage(
        materialization_dir="/path/to/checkpoints/artifacts/materialized_audio",
    )
)
```

Use this when tarred-audio materialization must survive process restarts. This
is especially important if:

- the pipeline uses tarred input
- downstream stages consume materialized local audio paths
- you want restart-safe behavior after the materialization stage

When set, durable artifacts are placed under shard-aware subdirectories such as:

```text
materialized_audio/shard_0/...
materialized_audio/shard_1/...
```

If `materialization_dir` is not set, `MaterializeTarredAudioStage` keeps its
temporary-file behavior.

## Checkpoint directory layout

Checkpoint state is stored under `checkpoint_dir`:

```text
checkpoints/
  pipeline.json
  00_tarred_audio_manifest_reader/
    stage.json
    records/
      shard_0.jsonl
    outputs/
      shard_0.jsonl
  01_materialize_tarred_audio/
    stage.json
    records/
      shard_0.jsonl
    outputs/
      shard_0.jsonl
```

### Files

- `pipeline.json`
  Pipeline-level metadata and execution policy

- `stage.json`
  Stage-level metadata including:
  - stage name
  - config fingerprint
  - status
  - done/failed/filtered counts

- `records/shard_0.jsonl`
  Per-sample checkpoint records with fields such as:
  - `sample_key`
  - `status`
  - `task`
  - `error_type`
  - `error_message`

- `outputs/shard_0.jsonl`
  Serialized `AudioTask` outputs for one checkpoint shard

Shard file names correspond to checkpoint shard identity:

- tarred pipelines typically use names like `shard_0`, `shard_1`, ...
- JSONL / manifest-based pipelines use source-based names derived from input file
  identities whenever possible

## Example: tarred ASR pipeline

```python
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.checkpointing.audio import AudioCheckpointRunner
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import (
    CleanupTemporaryAudioStage,
    MaterializeTarredAudioStage,
    TarredAudioManifestReader,
)
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.io import AudioToDocumentStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.text.io.writer import JsonlWriter

pipeline = Pipeline(
    name="tarred_asr_pipeline",
    config={"link_stages_via_io": False},
)

pipeline.add_stage(
    TarredAudioManifestReader(
        manifest_paths="/path/to/manifests/manifest__OP_0..255_CL_.json",
        tar_paths="/path/to/audio_shards/audio__OP_0..255_CL_.tar",
        skip_missing_entries=True,
    )
)
pipeline.add_stage(MaterializeTarredAudioStage())
pipeline.add_stage(
    InferenceAsrNemoStage(
        model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"
    )
)
pipeline.add_stage(
    GetPairwiseWerStage(
        text_key="text",
        pred_text_key="pred_text",
        wer_key="wer",
    )
)
pipeline.add_stage(CleanupTemporaryAudioStage())
pipeline.add_stage(AudioToDocumentStage())
pipeline.add_stage(JsonlWriter(path="/path/to/output"))

runner = AudioCheckpointRunner(
    pipeline=pipeline,
    checkpoint_dir="/path/to/checkpoints",
    executor=XennaExecutor(config={"execution_mode": "streaming"}),
    ignore_failed=False,
)

runner.run()
```

## Example: JSONL audio manifest

```python
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.checkpointing.audio import AudioCheckpointRunner
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.io import AudioToDocumentStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

pipeline = Pipeline(
    name="jsonl_asr_pipeline",
    config={"link_stages_via_io": True},
)

pipeline.add_stage(
    JsonlReader(
        file_paths="/path/to/manifests",
        task_type="audio",
    )
)
pipeline.add_stage(
    InferenceAsrNemoStage(
        model_name="nvidia/stt_en_fastconformer_hybrid_large_pc"
    )
)
pipeline.add_stage(
    GetPairwiseWerStage(
        text_key="text",
        pred_text_key="pred_text",
        wer_key="wer",
    )
)
pipeline.add_stage(AudioToDocumentStage())
pipeline.add_stage(JsonlWriter(path="/path/to/output"))

runner = AudioCheckpointRunner(
    pipeline=pipeline,
    checkpoint_dir="/path/to/checkpoints",
    executor=XennaExecutor(config={"execution_mode": "streaming"}),
    ignore_failed=True,
)

runner.run()
```

## Current limitations

- Checkpointing is currently audio-specific
- The checkpoint boundary is the last stage that still returns `AudioTask`
- `AudioToDocumentStage` and writer stages are not checkpointed
- This is designed as a runner/orchestration layer on top of existing executors,
  not as checkpointing inside `XennaExecutor` or `RayDataExecutor`

## Practical guidance

- Use `link_stages_via_io=False` if intermediate audio outputs fit comfortably in
  memory and you want less I/O overhead
- Use `link_stages_via_io=True` if intermediate stage outputs are very large
- Use `materialization_dir` when tarred audio materialization needs to survive
  restarts
- Use `ignore_failed=True` when best-effort progress is preferable to failing the
  whole run on a small number of bad samples
