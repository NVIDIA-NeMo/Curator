# Duration-Aware Bucketing Execution Flow

This note describes the current PR code on disk for a large input, e.g.
100,000 audio rows with mixed durations such as 5 seconds, 10 minutes,
50 minutes, 2 hours, and 4 hours.

The user-facing switch is singular:

```yaml
# tutorials/audio/qwen_omni_inprocess/qwen_omni_inprocess.yaml:112-123
duration_aware_bucketing:
  enabled: true
  strategy: duration_bucketed
  buckets_sec: [0, 600, 1200, 2400]
  max_items_per_batch_by_bucket: [32, 16, 8, 4]
  max_audio_sec_per_batch: 2400
  prebatching_window_size: null
  flush_interval_ms: 250
```

The ASR stage receives that policy here:

```yaml
# tutorials/audio/qwen_omni_inprocess/qwen_omni_inprocess.yaml:141-168
- stage_id: qwen_omni
  _target_: nemo_curator.stages.audio.inference.asr.ASRStage
  batch_size: ${candidate_batch_size}
  batch_policy:
    _target_: nemo_curator.stages.audio.inference.batch_policy.BatchPolicy
    enabled: ${duration_aware_bucketing.enabled}
    strategy: ${duration_aware_bucketing.strategy}
    buckets_sec: ${duration_aware_bucketing.buckets_sec}
    max_items_per_batch_by_bucket: ${duration_aware_bucketing.max_items_per_batch_by_bucket}
    max_audio_sec_per_batch: ${duration_aware_bucketing.max_audio_sec_per_batch}
    prebatching_window_size: ${duration_aware_bucketing.prebatching_window_size}
```

There is no second boolean. `enabled: false` means no scheduler chunking, no
duration bucketing, and no extra planning path.

## Common Startup

1. Hydra enters `main()` at
   `tutorials/audio/qwen_omni_inprocess/main.py:285-297`.
2. `build_granary_v2_pipeline(cfg)` constructs the `Pipeline` at
   `main.py:258-260`.
3. `_instantiate_configured_stages(cfg)` reads the `stages:` list, instantiates
   each Hydra target, applies `stage_with`, and appends enabled stages at
   `main.py:201-238`.
4. `_create_executor(cfg)` chooses:
   - `backend: ray_data` -> `RayDataExecutor`, `main.py:263-282`
   - `backend: xenna` -> `XennaExecutor`, `main.py:263-279`
5. `Pipeline.run()` decomposes composite stages and calls
   `executor.execute(self.stages, initial_tasks)` at
   `nemo_curator/pipeline/pipeline.py:177-215`.

For a JSONL manifest reader, each line becomes one `AudioTask`:

```python
# nemo_curator/stages/audio/common.py:145-173
def process(self, task: FileGroupTask) -> list[AudioTask]:
    ...
    for line in f:
        if line.strip():
            results.append(AudioTask(data=json.loads(line.strip()), ...))
    return results
```

The Qwen tutorial YAML uses `NemoTarredAudioReader` instead of
`ManifestReaderStage`, but the backend behavior below starts from the same
shape: many `AudioTask` rows, each carrying decoded waveform/sample-rate data by
the time ASR runs.

## Shared Bucketing Decision

The central gate is `stage_uses_centralized_batching()`:

```python
# nemo_curator/backends/base.py:92-113
def _enabled_batch_policy(stage):
    policy = getattr(stage, "batch_policy", None)
    if policy is None or not getattr(policy, "enabled", False):
        return None
    return policy

def stage_uses_centralized_batching(stage):
    policy = _enabled_batch_policy(stage)
    if policy is None:
        return False
    build_tasks = getattr(stage, "build_prebucketed_tasks", None)
    assemble_results = getattr(stage, "assemble_prebucketed_task_results", None)
    return callable(build_tasks) and callable(assemble_results) and _scheduler_task_cost_fn(stage) is not None
```

For ASR with `enabled: true`, this returns true because ASR exposes
`build_prebucketed_tasks()`, `assemble_prebucketed_task_results()`, and
`scheduler_task_cost()`.

For ASR with `enabled: false`, `_enabled_batch_policy()` returns `None`.
Everything falls back to ordinary backend batches.

## ASR Bucketing On: Chunk Then Bucket Once

The executor builds scheduler tasks before any ASR worker call:

```python
# nemo_curator/backends/base.py:167-201
def build_scheduled_task_batch_plan(stage, tasks):
    if not tasks or not stage_uses_centralized_batching(stage):
        return None
    scheduler_tasks = build_tasks(tasks)
    ...
    ready_batches = [
        SchedulerReadyTaskBatch(tasks=list(sub_tasks), total_cost=total_cost)
        for source_indices, sub_tasks, total_cost in policy.bucketize_with_costs(...)
    ]
    return ScheduledTaskBatchPlan(parent_tasks=list(tasks), ready_batches=ready_batches)
```

ASR builds chunk tasks here:

```python
# nemo_curator/stages/audio/inference/asr/stage.py:479-493
def build_prebucketed_tasks(self, tasks):
    policy = self.batch_policy
    if policy is None or not policy.enabled:
        return None
    chunk_plan = self._build_prebucket_chunk_plan(tasks)
    return [chunk.task for chunk in chunk_plan]
```

The actual split is duration based:

```python
# nemo_curator/stages/audio/inference/asr/stage.py:604-643
def _build_chunk_specs(self, tasks):
    slice_ceiling = float(self.max_inference_duration_s or self.ideal_inference_segment_s)
    for parent_idx, task in enumerate(tasks):
        chunks = self._chunk_waveform(waveform, sr, slice_ceiling)
        for chunk_idx, chunk in enumerate(chunks):
            specs.append(_ChunkSpec(parent_idx=parent_idx, chunk_idx=chunk_idx, cost=...))
```

Then each virtual chunk becomes a minimal dispatch `AudioTask`:

```python
# nemo_curator/stages/audio/inference/asr/stage.py:657-703
def _build_prebucket_chunk_plan(self, tasks):
    return [
        _PrebucketChunk(task=self._make_prebucket_chunk_task_from_spec(spec), ...)
        for spec in self._build_chunk_specs(tasks)
    ]
```

`BatchPolicy.bucketize_with_costs()` owns bucket formation and longest-first
ordering:

```python
# nemo_curator/stages/audio/inference/batch_policy.py:355-381
def bucketize_with_costs(self, items, cost_fn):
    if not self.enabled:
        return [(list(range(len(items))), list(items), 0.0)]
    scheduler = BucketQueueScheduler(self, enable_timer=False)
    for i, it in enumerate(items):
        ready_batches.extend(scheduler.enqueue(i, it, float(cost_fn(it))))
    ready_batches.extend(scheduler.flush_all())
    return sorted(..., key=lambda batch: batch.total_cost, reverse=True)
```

Example for 100,000 varied rows:

```text
100,000 parent rows
  -> ASR chunks long rows by max_inference_duration_s
  -> maybe 130,000 scheduler chunk tasks
  -> chunks enter duration buckets:
       [0, 600) seconds     cap 32 items
       [600, 1200) seconds  cap 16 items
       [1200, 2400) seconds cap 8 items
       [2400, +inf) seconds cap 4 items
  -> scheduler-ready GPU batches run longest-first
  -> processed chunk tasks stitch back to 100,000 parent rows
```

## Ray Data, Bucketing On

Mock config:

```yaml
backend: ray_data
duration_aware_bucketing:
  enabled: true
```

Flow:

1. `RayDataExecutor.execute()` initializes Ray and converts initial tasks to a
   dataset at `nemo_curator/backends/ray_data/executor.py:50-91`.
2. For each stage, `_process_stage_dataset()` checks centralized bucketing at
   `ray_data/executor.py:105-111`.
3. ASR is centralized, so Ray Data takes the current dataset to parent tasks at
   `ray_data/executor.py:113-116`.
4. `build_scheduled_task_batch_plan()` chunks and buckets at
   `base.py:167-201`.
5. The ready batches become a Ray dataset at
   `ray_data/executor.py:120-142`.
6. `RayDataStageAdapter.process_scheduler_ready_dataset()` marks the dataset as
   scheduler-ready at `ray_data/adapter.py:121-127`.
7. `_process_dataset()` keeps `map_batch_size = 1` for scheduler-ready rows at
   `ray_data/adapter.py:140-173`.
8. `create_actor_from_stage().__call__()` or `create_task_from_stage()` routes
   scheduler-ready rows to `_process_scheduler_ready_batch_internal()` at
   `ray_data/adapter.py:246-251` and `ray_data/adapter.py:282-288`.
9. `_process_scheduler_ready_batch_internal()` calls
   `process_scheduler_ready_batch()` at `ray_data/adapter.py:92-101`.
10. `BaseStageAdapter.process_scheduler_ready_batch()` calls exactly one
    `stage.process_batch()` without recursive planning at `base.py:285-299`.
11. ASR recognizes prebucketed chunk tasks and runs `_process_prebucketed_chunk_batch()`
    at `asr/stage.py:382-439`.
12. After workers return processed chunks, Ray Data collects them and stitches
    parent rows at `ray_data/executor.py:125-126`.

Diagram:

```text
Ray Dataset[parent AudioTask rows]
  -> take_all() at centralized ASR boundary
  -> build_scheduled_task_batch_plan()
  -> Ray Dataset[SchedulerReadyTaskBatch rows]
  -> map_batches(batch_size=1)
  -> ASR process_batch(prebucketed chunk tasks)
  -> processed chunk tasks
  -> assemble_prebucketed_task_results()
  -> Ray Dataset[parent AudioTask rows]
```

## Xenna Streaming, Bucketing On

Mock config:

```yaml
backend: xenna
execution_mode: streaming
duration_aware_bucketing:
  enabled: true
```

Flow:

1. `XennaExecutor.execute()` sees at least one centralized stage and routes to
   `_run_pipeline_with_scheduler_ready_stages()` at
   `nemo_curator/backends/xenna/executor.py:70-83`.
2. `_run_pipeline_with_scheduler_ready_stages()` accumulates ordinary stages in
   `xenna_segment` until it reaches ASR at `xenna/executor.py:85-107`.
3. Any segment before ASR runs through normal Xenna at
   `xenna/executor.py:99-101`.
4. ASR runs through `_run_scheduler_ready_stage()` at
   `xenna/executor.py:109-120`.
5. `_run_scheduler_ready_stage()` calls `build_scheduled_task_batch_plan()` at
   `xenna/executor.py:111`.
6. The one-stage scheduler-ready Xenna pipeline receives `plan.ready_batches` at
   `xenna/executor.py:115-119`.
7. `_run_xenna_pipeline()` creates a `SchedulerReady` adapter at
   `xenna/executor.py:138-146`.
8. Because `execution_mode` is not `batch`, `_run_xenna_pipeline()` chooses
   `ExecutionMode.STREAMING` and adds `StreamingSpecificSpec` at
   `xenna/executor.py:165-177`.
9. Xenna runs the one-stage scheduler-ready pipeline at
   `xenna/executor.py:192-211`.
10. `XennaSchedulerReadyStageAdapter.stage_batch_size` returns `1`, preserving
    one scheduler-ready dispatch per worker call at `xenna/adapter.py:159-165`.
11. `XennaSchedulerReadyStageAdapter.process_data()` calls
    `process_scheduler_ready_batch()` for each ready row at
    `xenna/adapter.py:167-172`.
12. ASR processes the prebucketed chunks at `asr/stage.py:382-439`.
13. Xenna returns processed chunk tasks; `_run_scheduler_ready_stage()` stitches
    them to parent rows at `xenna/executor.py:120`.

Diagram:

```text
Xenna streaming segment before ASR
  -> current parent tasks returned
  -> Curator scheduler chunks + buckets
  -> Xenna streaming one-stage SchedulerReady ASR
  -> stitch chunks back to parents
  -> Xenna streaming downstream segment
```

Streaming changes Xenna execution behavior and autoscaling. It does not change
the bucketing plan: ASR workers still receive scheduler-ready duration-coherent
chunk batches.

## Xenna Batch, Bucketing On

Mock config:

```yaml
backend: xenna
execution_mode: batch
duration_aware_bucketing:
  enabled: true
```

The call chain is the same as Xenna streaming through
`xenna/executor.py:70-146`. The difference is execution mode:

```python
# nemo_curator/backends/xenna/executor.py:165-168
exec_mode = pipelines_v1.ExecutionMode.STREAMING
if self._get_pipeline_config("execution_mode") == "batch":
    exec_mode = pipelines_v1.ExecutionMode.BATCH
```

With `batch`, no `StreamingSpecificSpec` is created because
`xenna/executor.py:170-177` only applies that config to streaming mode.

Diagram:

```text
Xenna batch segment before ASR
  -> current parent tasks returned
  -> Curator scheduler chunks + buckets
  -> Xenna batch one-stage SchedulerReady ASR
  -> stitch chunks back to parents
  -> Xenna batch downstream segment
```

The GPU dispatch shape is identical to streaming: one scheduler-ready row maps
to one ASR worker call, and each row already contains duration-coherent chunks.

## Bucketing Off: Current-Main-Compatible Path

Mock config:

```yaml
duration_aware_bucketing:
  enabled: false
```

What happens:

1. `_enabled_batch_policy()` returns `None` at `base.py:92-96`.
2. `stage_uses_centralized_batching()` returns false at `base.py:108-113`.
3. Ray Data does not enter `_process_centralized_stage_dataset()` and instead
   calls the ordinary adapter path at `ray_data/executor.py:105-111`.
4. Xenna does not enter `_run_pipeline_with_scheduler_ready_stages()` and
   instead calls `_run_xenna_pipeline()` at `xenna/executor.py:80-83`.
5. If ASR receives a normal backend batch, `_requires_centralized_scheduler()`
   is false at `asr/stage.py:475-477`, and `_process_plain_batch()` builds one
   full item per parent row at `asr/stage.py:402-437`.

So off mode for a mixed-duration backend batch is:

```text
[5s parent, 10m parent, 4h parent, 50m parent]
  -> one process_batch call
  -> one adapter item per parent
  -> no scheduler chunk tasks
  -> no duration buckets
  -> one output row per input row
```

## Current Main Comparison

The current-main checkout has no backend duration-aware scheduler symbols. A
grep for `BatchPolicy`, `duration_aware_bucketing`,
`SchedulerReadyTaskBatch`, `stage_uses_centralized_batching`, and `prebucketed`
under `references/Curator-main/nemo_curator` and audio tutorials returns no
matches.

Main `BaseStageAdapter` directly invokes `stage.process_batch(tasks)`:

```python
# references/Curator-main/nemo_curator/backends/base.py:65-96
def process_batch(self, tasks):
    ...
    with self._timer.time_process(input_size):
        results = self.stage.process_batch(tasks)
    ...
    return results
```

Main Ray Data maps fixed backend batches directly:

```python
# references/Curator-main/nemo_curator/backends/ray_data/adapter.py:68-117
tasks = batch["item"]
results = self.process_batch(tasks)
...
processed_dataset = dataset.map_batches(map_batches_fn, batch_size=self.batch_size, ...)
```

Main Xenna uses the stage `batch_size` and directly processes data:

```python
# references/Curator-main/nemo_curator/backends/xenna/adapter.py:78-102
def stage_batch_size(self):
    batch_size = self.processing_stage.batch_size
    return batch_size if batch_size is not None else 1

def process_data(self, tasks):
    return self.process_batch(tasks)
```

Main NeMo ASR similarly transcribes the backend-provided batch as-is:

```python
# references/Curator-main/nemo_curator/stages/audio/inference/asr/asr_nemo.py:112-130
def process_batch(self, tasks):
    files = [t.data[self.filepath_key] for t in tasks]
    texts = self.transcribe(files)
    ...
    return tasks
```

That is the behavior `duration_aware_bucketing.enabled: false` now preserves in
this PR: fixed backend batches are sent through without scheduler chunking,
without duration bucketing, and without a second partial/local bucketing mode.
