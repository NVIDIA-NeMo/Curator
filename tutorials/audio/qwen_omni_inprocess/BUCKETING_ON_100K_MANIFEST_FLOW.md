# Bucketing-On Flow For A 100k-Row Manifest

This document is based on the current PR checkout on disk:

```text
/home/aaftabv/grananary-v2/references/CuratorPR1967
```

It explains the `duration_aware_bucketing.enabled: true` path for an input
manifest with roughly 100,000 mixed-duration audio rows. The three execution
types covered are:

- Ray Data
- Xenna streaming
- Xenna batch

The important contract is:

```text
parent AudioTask rows
  -> ASR builds chunk work units once
  -> shared BatchPolicy scheduler buckets those work units
  -> backend workers receive SchedulerReadyTaskBatch rows
  -> ASR processes already-planned chunk batches
  -> processed chunks stitch back to original parent rows
```

RAM annotations below use these symbols:

```text
N = parent input rows, about 100,000 here
C = ASR chunk work units after splitting long audio
B = SchedulerReadyTaskBatch rows emitted by the bucket scheduler
W = concurrent worker dispatches in the backend
audio_bytes = sum(waveform.nbytes for retained waveform arrays)
```

For mono float32 audio, raw waveform RAM is roughly:

```text
bytes_per_second = sample_rate * 4
16 kHz audio ~= 64 KB/sec
40 min chunk ~= 153.6 MB decimal / 146.5 MiB
4 hr parent clip ~= 921.6 MB decimal / 879 MiB
```

So the dominant RAM term is usually waveform retention and backend
serialization/object-store copies, not the Python `AudioTask` wrapper itself.

## Mock Config

For a manifest-style input, an illustrative stage graph looks like this:

```yaml
backend: ray_data  # or xenna
execution_mode: streaming  # Xenna only: streaming or batch

stages:
  - stage_id: reader
    _target_: nemo_curator.stages.audio.common.ManifestReader
    manifest_path: /data/audio_manifest_100k.jsonl
    files_per_partition: 1

  - stage_id: qwen_omni
    _target_: nemo_curator.stages.audio.inference.asr.ASRStage
    name: QwenOmni_inference
    adapter_target: nemo_curator.models.asr.qwen_omni.QwenOmniASRAdapter
    model_id: Qwen/Qwen3-Omni
    source_lang_key: source_lang
    pred_text_key: qwen3_prediction_s1
    ideal_inference_segment_s: 2400
    max_inference_duration_s: 2400
    keep_waveform: true
    batch_size: 32
    batch_policy:
      _target_: nemo_curator.stages.audio.inference.batch_policy.BatchPolicy
      enabled: true
      strategy: duration_bucketed
      buckets_sec: [0, 600, 1200, 2400]
      max_items_per_batch_by_bucket: [32, 16, 8, 4]
      max_audio_sec_per_batch: 2400
      prebatching_window_size: null
      flush_interval_ms: 250
```

The checked-in Qwen tutorial uses `NemoTarredAudioReader` instead of
`ManifestReader`, but its ASR config has the same bucketing structure:
`tutorials/audio/qwen_omni_inprocess/qwen_omni_inprocess.yaml:112-168`.

RAM annotation:

- `keep_waveform: true` matches the checked-in tutorial default at
  `qwen_omni_inprocess.yaml:81` and `qwen_omni_inprocess.yaml:153`. With that
  setting, parent rows keep waveform arrays after ASR stitch-back. Setting it
  to `false` reduces post-ASR parent RAM, but not the pre-ASR planning peak.
- `max_inference_duration_s: 2400` caps ASR chunk work units at 40 minutes.
  It limits GPU dispatch size, but the original parent waveform is still held
  while the scheduler plan exists.
- `max_items_per_batch_by_bucket` and `max_audio_sec_per_batch` bound the RAM
  of one GPU-facing dispatch batch. They do not by themselves bound the driver
  RAM needed to hold all parent rows plus the global scheduler plan.
- The mock `ManifestReader` emits metadata rows only. ASR requires
  `waveform` and `sample_rate`, so a real manifest pipeline needs an earlier
  audio-loading stage, or it should use the checked-in tutorial reader
  `NemoTarredAudioReader`, which emits decoded waveform arrays.

## Common Driver Flow

This part is identical before the backend-specific split.

1. Hydra enters the tutorial at
   `tutorials/audio/qwen_omni_inprocess/main.py:285-296`.

2. `main()` calls `build_granary_v2_pipeline(cfg)` at
   `main.py:290`.

3. `build_granary_v2_pipeline()` creates a `Pipeline` at
   `main.py:258-260`.

4. `_instantiate_configured_stages(cfg)` reads the YAML `stages:` list at
   `main.py:201-215`.

5. For each enabled stage, it resolves and instantiates the Hydra target at
   `main.py:231-238`.

6. `_create_executor(cfg)` reads `backend` at `main.py:263-282`.

7. `main()` calls `pipeline.run(executor=executor)` at `main.py:293-296`.

8. `Pipeline.run()` calls `self.build()` at
   `nemo_curator/pipeline/pipeline.py:177-187`.

9. `Pipeline.build()` decomposes composite stages at
   `pipeline.py:64-80`.

10. `Pipeline._decompose_stages()` calls `stage.decompose_and_apply_with()` for
    composite stages at `pipeline.py:83-125`.

11. `ManifestReader.decompose()` expands the manifest reader into
    `FilePartitioningStage` plus `ManifestReaderStage` at
    `nemo_curator/stages/audio/common.py:211-221`.

12. `Pipeline.run()` finally calls `executor.execute(self.stages, initial_tasks)`
    at `pipeline.py:215`.

Manifest row creation happens here:

```python
# nemo_curator/stages/audio/common.py:145-173
def process(self, task: FileGroupTask) -> list[AudioTask]:
    paths = task.data
    results: list[AudioTask] = []
    for manifest in paths:
        fs, resolved = url_to_fs(manifest)
        with fs.open(resolved, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(AudioTask(data=json.loads(line.strip()), ...))
    return results
```

For 100,000 JSONL rows, the reader emits about 100,000 parent `AudioTask`
objects before ASR.

RAM annotation:

- `ManifestReaderStage.process()` streams the file line by line, but it appends
  every parsed row in the current `FileGroupTask` to the local `results` list
  before returning it. With `files_per_partition: 1`, a single 100k-line
  manifest can therefore create a 100k-element `results` list in that reader
  worker before the next stage sees those rows.
- If those rows are metadata-only, this is roughly
  `O(rows in partition * metadata dict size)`. If an earlier audio loader has
  already placed waveform arrays into each row, add `audio_bytes` for those
  retained arrays.
- The checked-in tutorial reader is `NemoTarredAudioReader`, not
  `ManifestReader`. Its shard reader explicitly decodes audio into in-memory
  mono float32 arrays at
  `nemo_curator/stages/audio/io/nemo_tarred_reader.py:532-560`, then appends
  one `AudioTask` per utterance at `nemo_tarred_reader.py:565-573`.
- `NemoTarShardReaderStage.num_workers()` returns `1` to bound per-reader
  shard RAM at `nemo_tarred_reader.py:414-416`; its Ray/Xenna specs keep that
  one-shard-at-a-time shape at `nemo_tarred_reader.py:418-425`. That bounds
  reader-worker RAM, but once ASR planning starts the executor still retains
  all emitted parent tasks for the centralized ASR boundary.

## Shared Bucketing-On Decision

Every backend uses the same structural gate:

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

With the mock ASR config above:

```text
BatchPolicy.enabled == true
ASRStage.build_prebucketed_tasks exists
ASRStage.assemble_prebucketed_task_results exists
ASRStage.scheduler_task_cost exists
```

So ASR is routed through centralized chunking + bucketing before worker
dispatch.

RAM annotation:

- This gate is structural and cheap. It only checks stage attributes and does
  not allocate row data, chunk data, or waveform buffers.
- The practical memory consequence is that ASR cannot be called through normal
  backend `process_batch()` with parent tasks. The executor must first build the
  scheduler plan, which creates the central materialization point described in
  the next section.

## Shared ASR Scheduler Plan

This is the core of bucketing-on behavior, independent of backend.

1. The backend calls `build_scheduled_task_batch_plan(stage, parent_tasks)`.

2. `build_scheduled_task_batch_plan()` checks
   `stage_uses_centralized_batching(stage)` at
   `nemo_curator/backends/base.py:167-173`.

3. It calls `stage.build_prebucketed_tasks(parent_tasks)` at
   `base.py:175-179`.

4. ASR receives that call at
   `nemo_curator/stages/audio/inference/asr/stage.py:479-493`.

```python
# asr/stage.py:479-493
def build_prebucketed_tasks(self, tasks):
    policy = self.batch_policy
    if policy is None or not policy.enabled:
        return None
    chunk_plan = self._build_prebucket_chunk_plan(tasks)
    return [chunk.task for chunk in chunk_plan]
```

5. `_build_prebucket_chunk_plan()` calls `_build_chunk_specs(tasks)` at
   `asr/stage.py:657-668`.

6. `_build_chunk_specs()` slices long waveforms by
   `max_inference_duration_s` or `ideal_inference_segment_s` at
   `asr/stage.py:604-643`.

```python
# asr/stage.py:604-643
slice_ceiling = float(self.max_inference_duration_s or self.ideal_inference_segment_s)
for parent_idx, task in enumerate(tasks):
    chunks = self._chunk_waveform(waveform, sr, slice_ceiling)
    for chunk_idx, chunk in enumerate(chunks):
        specs.append(_ChunkSpec(parent_idx=parent_idx, chunk_idx=chunk_idx, cost=...))
```

7. `_make_prebucket_chunk_task()` materializes minimal chunk tasks and stamps:
   `_curator_asr_parent_idx`, `_curator_asr_chunk_idx`,
   `_curator_asr_chunk_count`, and `_curator_asr_chunk_cost` at
   `asr/stage.py:682-712`.

8. Back in `build_scheduled_task_batch_plan()`, the shared policy buckets those
   scheduler tasks at `base.py:183-197`.

```python
# base.py:189-197
ready_batches = [
    SchedulerReadyTaskBatch(
        tasks=list(sub_tasks),
        total_cost=total_cost,
        source_indices=list(source_indices),
    )
    for source_indices, sub_tasks, total_cost in policy.bucketize_with_costs(...)
]
```

9. `BatchPolicy.bucketize_with_costs()` creates finite bucket queues at
   `nemo_curator/stages/audio/inference/batch_policy.py:355-381`.

10. `BucketQueueScheduler.enqueue()` assigns each chunk to a bucket and flushes
    when adding a chunk would exceed item or audio-second caps at
    `batch_policy.py:83-105`.

11. `BucketQueueScheduler.flush_all()` drains remaining queues at
    `batch_policy.py:124-131`.

12. `bucketize_with_costs()` sorts ready batches by descending total cost at
    `batch_policy.py:378-381`.

For 100,000 parent rows:

```text
100,000 parent AudioTask rows
  -> chunk specs
  -> scheduler chunk tasks
  -> bucket queues
  -> SchedulerReadyTaskBatch rows sorted heavy-first
```

RAM annotation for the scheduler plan:

```text
Before planning:
  parent_tasks list
  + parent AudioTask objects
  + parent waveform arrays

During _build_chunk_specs():
  + _ChunkSpec list of length C
  + references to parent task and waveform slices

During _build_prebucket_chunk_plan():
  + _PrebucketChunk list of length C
  + one minimal chunk AudioTask per chunk
  + chunk data dict with waveform slice, sample rate, language/path metadata,
    parent index, chunk index, chunk count, and chunk cost

During bucketize_with_costs():
  + BucketQueueScheduler queues
  + ReadyBatch objects/lists
  + returned list of (source_indices, sub_tasks, total_cost)

After build_scheduled_task_batch_plan():
  + ScheduledTaskBatchPlan.parent_tasks
  + SchedulerReadyTaskBatch rows, each holding a tasks list
```

Important details:

- `_chunk_waveform()` slices long arrays with `waveform[start : start +
  max_samples]` at `asr/stage.py:296-299`. For NumPy arrays this is usually a
  view, so chunking itself normally does not copy the raw sample buffer.
  However, the slice object still keeps the original parent array alive.
- `_make_prebucket_chunk_task()` intentionally uses a minimal `chunk_data` dict
  at `asr/stage.py:692-712`; it does not shallow-copy the full parent
  `task.data`. This keeps metadata overhead lower than duplicating every
  manifest field.
- `bucketize_with_costs()` creates a finite scheduler with `enable_timer=False`
  at `batch_policy.py:372`, queues every chunk at `batch_policy.py:374-375`,
  drains remaining queues at `batch_policy.py:376`, and sorts ready batches at
  `batch_policy.py:378-381`.
- Planning RAM is therefore roughly:

```text
parent waveform bytes
+ parent task/list overhead O(N)
+ chunk descriptor/task/list overhead O(C)
+ ready-batch/list overhead O(B + C references)
```

- The scheduler metadata is much smaller than waveform arrays for long audio,
  but it is not zero. With hundreds of thousands of chunks, the Python object
  overhead can become visible even when waveform slices are views.

Example duration movement:

```text
Input parent rows:
  row A: 5 sec
  row B: 10 min
  row C: 50 min
  row D: 4 hr

After ASR chunking with max_inference_duration_s=2400 sec:
  A -> [5s]
  B -> [600s]
  C -> [2400s, 600s]
  D -> [2400s, 2400s, 2400s, 2400s, 2400s, 2400s]

After bucketing:
  [0, 600) bucket:       [5s]
  [600, 1200) bucket:    [600s, 600s]
  [2400, +inf) bucket:   [2400s, 2400s, 2400s, ...]
```

## Ray Data: Bucketing On

Mock config:

```yaml
backend: ray_data
duration_aware_bucketing:
  enabled: true
```

Call chain:

1. `Pipeline.run()` calls `RayDataExecutor.execute()` at
   `pipeline.py:215`.

2. `RayDataExecutor.execute()` initializes Ray at
   `nemo_curator/backends/ray_data/executor.py:50-77`.

3. It converts initial tasks to a Ray Dataset at
   `ray_data/executor.py:79-80`; `_tasks_to_dataset()` is implemented at
   `ray_data/executor.py:128-138`.

4. It runs setup and loops through stages at `ray_data/executor.py:82-91`.

5. Each stage enters `_process_stage_dataset(stage, dataset)` at
   `ray_data/executor.py:105-111`.

6. Reader stages are not centralized, so they go through
   `RayDataStageAdapter.process_dataset()` at `ray_data/executor.py:110-111`
   and `nemo_curator/backends/ray_data/adapter.py:129-138`.

7. `ManifestReaderStage.process()` reads JSONL lines into `AudioTask`s at
   `nemo_curator/stages/audio/common.py:145-173`.

8. When the loop reaches ASR, `_process_stage_dataset()` sees
   `stage_uses_centralized_batching(stage) == true` at
   `ray_data/executor.py:105-108`.

9. It calls `_process_centralized_stage_dataset()` at
   `ray_data/executor.py:113-126`.

10. `_process_centralized_stage_dataset()` materializes the upstream dataset to
    parent tasks using `_dataset_to_tasks(dataset)` at
    `ray_data/executor.py:115`; `_dataset_to_tasks()` calls `dataset.take_all()`
    at `ray_data/executor.py:144-160`.

11. It calls `build_scheduled_task_batch_plan(stage, parent_tasks)` at
    `ray_data/executor.py:116`.

12. The shared plan follows the ASR scheduler plan described above:
    `base.py:167-201`, `asr/stage.py:479-493`,
    `asr/stage.py:604-712`, `batch_policy.py:355-381`.

13. `_scheduler_ready_batches_to_dataset(plan.ready_batches)` creates one Ray
    row per scheduler-ready batch at `ray_data/executor.py:120` and
    `ray_data/executor.py:140-142`.

14. `RayDataStageAdapter(stage).process_scheduler_ready_dataset(...)` is called
    at `ray_data/executor.py:121-124`.

15. `process_scheduler_ready_dataset()` calls `_process_dataset(...,
    scheduler_ready_batches=True)` at `ray_data/adapter.py:121-127`.

16. `_process_dataset()` blocks accidental parent-task processing for
    centralized stages unless `scheduler_ready_batches=True` at
    `ray_data/adapter.py:149-155`.

17. `_process_dataset()` sets `map_batch_size = 1` for scheduler-ready rows at
    `ray_data/adapter.py:172`.

18. It calls `dataset.map_batches(...)` at `ray_data/adapter.py:173`.

19. If Ray Data uses actor workers, `create_actor_from_stage().__call__()` sees
    `scheduler_ready_batches=True` and calls
    `_process_scheduler_ready_batch_internal()` at
    `ray_data/adapter.py:223-251`.

20. If Ray Data uses task workers, `create_task_from_stage().stage_map_fn()` does
    the same at `ray_data/adapter.py:261-288`.

21. `_process_scheduler_ready_batch_internal()` unwraps each
    `SchedulerReadyTaskBatch` and calls `self.process_scheduler_ready_batch()`
    at `ray_data/adapter.py:92-101`.

22. `BaseStageAdapter.process_scheduler_ready_batch()` calls `_process_batch_once`
    without re-entering planning at `base.py:285-299`.

23. `_process_batch_once()` calls `ASRStage.process_batch(tasks)` at
    `base.py:297-299`.

24. `ASRStage.process_batch()` detects prebucketed chunk tasks at
    `asr/stage.py:382-400` and calls `_process_prebucketed_chunk_batch()` at
    `asr/stage.py:439-473`.

25. `_process_prebucketed_chunk_batch()` builds model items and calls
    `run_inference()` at `asr/stage.py:455-473`.

26. `run_inference()` calls the model adapter’s `transcribe_batch(items)` at
    `asr/stage.py:784-797`.

27. `assemble()` writes prediction fields on processed chunk tasks at
    `asr/stage.py:799-840`.

28. Ray Data collects processed chunk tasks at `ray_data/executor.py:125`.

29. It calls `assemble_scheduled_task_batch_results(stage, plan, processed_tasks)`
    at `ray_data/executor.py:126`.

30. `assemble_scheduled_task_batch_results()` calls
    `ASRStage.assemble_prebucketed_task_results(...)` at `base.py:204-214`.

31. ASR validates/stitches chunks at `asr/stage.py:519-527` and writes final
    parent outputs in `_assemble_prebucketed_chunks()` at `asr/stage.py:714-737`.

Ray Data shape:

```text
Ray Dataset[~100k parent rows]
  -> driver take_all at ASR boundary
  -> global ASR chunk plan
  -> Ray Dataset[SchedulerReadyTaskBatch rows]
  -> map_batches(batch_size=1)
  -> GPU workers process chunk batches
  -> driver take_all processed chunks
  -> stitch back to ~100k parent rows
```

RAM and overlap annotation:

- Ray Data builds lazy dataset transformations stage by stage at
  `ray_data/executor.py:86-91`. For regular non-centralized stages, Ray may
  execute upstream `map_batches` work concurrently when an action is triggered.
- Centralized ASR is an action boundary. `_process_centralized_stage_dataset()`
  calls `_dataset_to_tasks(dataset)` at `ray_data/executor.py:115`, and
  `_dataset_to_tasks()` calls `dataset.take_all()` at
  `ray_data/executor.py:153-160`. That `take_all()` triggers the upstream
  reader/prep dataset and materializes all parent tasks on the driver before
  ASR planning starts.
- Because of that barrier, in this PR's bucketing-on path, Qwen ASR inference
  does not start while the manifest/tar reader is still producing rows for the
  same ASR plan. The reader/prep stages may be active while the `take_all()` is
  collecting parents, but ASR chunking/bucketing begins only after the parent
  list exists.
- Driver RAM at the ASR boundary is at least:

```text
parent_tasks from take_all()
+ all parent waveform arrays retained by those tasks
+ Ray object-store memory for upstream blocks until Ray releases them
```

- `build_scheduled_task_batch_plan()` then adds the shared scheduler-plan RAM
  described above.
- `_scheduler_ready_batches_to_dataset()` converts every
  `SchedulerReadyTaskBatch` into a Ray Dataset row at
  `ray_data/executor.py:120` and `ray_data/executor.py:140-142`. This can add
  Ray object-store/serialization memory. If NumPy chunk slices are serialized
  as independent buffers, object-store pressure can approach the sum of chunk
  waveform bytes, even though the Python chunking step used views.
- `process_scheduler_ready_dataset()` uses `map_batch_size = 1` for
  scheduler-ready rows at `ray_data/adapter.py:172-173`, so each Ray worker call
  receives one already-planned `SchedulerReadyTaskBatch`.
- During GPU processing, each worker holds only its current scheduler-ready
  chunk batch plus model-side tensors/activations. That worker RAM is bounded by
  the bucket caps (`max_items_per_batch_by_bucket` and
  `max_audio_sec_per_batch`) and the number of concurrent Ray workers.
- After GPU workers finish, `_process_centralized_stage_dataset()` calls
  `_dataset_to_tasks(processed_dataset)` at `ray_data/executor.py:125`, which
  materializes processed chunk tasks on the driver before stitch-back.
- Peak driver/object-store RAM can therefore look like:

```text
parent tasks + parent waveforms
+ scheduler chunk tasks / ready batches
+ scheduler-ready Ray Dataset/object-store rows
+ processed chunk tasks collected by take_all()
+ final stitched parent dataset from _tasks_to_dataset()
```

- `ASRStage.assemble()` only removes waveform data from processed chunk tasks or
  parent tasks when `keep_waveform` is `false` at `asr/stage.py:826-827` and
  `asr/stage.py:735-736`. The checked-in tutorial default is `true`, so the
  final parent rows retain waveform RAM.

## Xenna Streaming: Bucketing On

Mock config:

```yaml
backend: xenna
execution_mode: streaming
duration_aware_bucketing:
  enabled: true
```

Call chain:

1. `Pipeline.run()` calls `XennaExecutor.execute()` at `pipeline.py:215`.

2. `XennaExecutor.execute()` sets initial tasks at
   `nemo_curator/backends/xenna/executor.py:70-80`.

3. It checks `any(stage_uses_centralized_batching(stage) for stage in stages)`
   at `xenna/executor.py:81`.

4. Because ASR is centralized, it calls
   `_run_pipeline_with_scheduler_ready_stages()` at `xenna/executor.py:82`.

5. `_run_pipeline_with_scheduler_ready_stages()` walks the stage list at
   `xenna/executor.py:85-107`.

6. Non-ASR stages are appended into `xenna_segment` at
   `xenna/executor.py:94-97`.

7. When ASR is reached, any preceding segment runs through
   `_run_xenna_pipeline(xenna_segment, current_tasks)` at
   `xenna/executor.py:99-101`.

8. `_run_xenna_pipeline()` creates regular Xenna stage adapters at
   `xenna/executor.py:138-146`.

9. For streaming mode, `_run_xenna_pipeline()` keeps
   `ExecutionMode.STREAMING` at `xenna/executor.py:165-168`.

10. It creates `StreamingSpecificSpec` at `xenna/executor.py:170-177`.

11. It builds `PipelineConfig` and `PipelineSpec` at
    `xenna/executor.py:179-193`.

12. It calls `pipelines_v1.run_pipeline(pipeline_spec)` at
    `xenna/executor.py:198-211`.

13. For regular stages, `XennaStageAdapter.stage_batch_size` returns
    `upstream_prebatching_batch_size(...)` at
    `nemo_curator/backends/xenna/adapter.py:82-86`.

14. Regular Xenna worker calls enter `XennaStageAdapter.process_data()` and then
    `BaseStageAdapter.process_batch()` at `xenna/adapter.py:99-107`.

15. After the pre-ASR segment returns parent tasks,
    `_run_pipeline_with_scheduler_ready_stages()` calls
    `_run_scheduler_ready_stage(stage, current_tasks)` at
    `xenna/executor.py:103`.

16. `_run_scheduler_ready_stage()` calls
    `build_scheduled_task_batch_plan(stage, tasks)` at
    `xenna/executor.py:109-111`.

17. The shared plan follows `base.py:167-201`, `asr/stage.py:479-493`,
    `asr/stage.py:604-712`, and `batch_policy.py:355-381`.

18. `_run_scheduler_ready_stage()` calls `_run_xenna_pipeline([stage],
    plan.ready_batches, scheduler_ready_stage=True)` at
    `xenna/executor.py:115-119`.

19. `_run_xenna_pipeline()` uses
    `create_named_xenna_scheduler_ready_stage_adapter(stage)` for this one-stage
    scheduler pipeline at `xenna/executor.py:142-146`.

20. Streaming mode is again selected at `xenna/executor.py:165-177`.

21. `XennaSchedulerReadyStageAdapter.stage_batch_size` returns `1` at
    `xenna/adapter.py:159-165`, so each `SchedulerReadyTaskBatch` remains one
    worker dispatch row.

22. `XennaSchedulerReadyStageAdapter.process_data()` iterates ready batches and
    calls `process_scheduler_ready_batch()` at `xenna/adapter.py:167-172`.

23. `BaseStageAdapter.process_scheduler_ready_batch()` calls `_process_batch_once`
    at `base.py:285-299`.

24. ASR sees prebucketed chunk tasks and runs
    `_process_prebucketed_chunk_batch()` at `asr/stage.py:382-473`.

25. `run_inference()` calls `transcribe_batch(items)` at `asr/stage.py:784-797`.

26. `_run_scheduler_ready_stage()` stitches processed chunks back to parent tasks
    with `assemble_scheduled_task_batch_results()` at `xenna/executor.py:120`,
    `base.py:204-214`, and `asr/stage.py:519-527`.

27. Any downstream Xenna segment runs at `xenna/executor.py:105-107`.

Xenna streaming shape:

```text
Xenna streaming reader/prep segment
  -> list[parent AudioTask rows]
  -> Curator ASR scheduler plan
  -> Xenna streaming one-stage SchedulerReady ASR pipeline
  -> list[processed chunk tasks]
  -> stitch back to parent rows
  -> Xenna streaming downstream segment
```

RAM and overlap annotation:

- Xenna streaming can overlap stages inside one Xenna pipeline segment. In this
  PR, however, centralized ASR causes executor segmentation at
  `xenna/executor.py:85-107`: non-centralized stages are accumulated into
  `xenna_segment`, then `_run_xenna_pipeline(xenna_segment, current_tasks)` must
  return before `_run_scheduler_ready_stage()` starts ASR.
- `_run_xenna_pipeline()` is configured with `return_last_stage_outputs=True`
  at `xenna/executor.py:179-185`, and the returned `results` list becomes
  `current_tasks` at `xenna/executor.py:99-103`.
- Therefore, for bucketing-on ASR, reader/prep rows can stream through the
  pre-ASR Xenna segment, but Qwen ASR does not begin chunking/bucketing until
  that segment has returned its full parent-task output list.
- Chunking and bucketing then happen once in `_run_scheduler_ready_stage()`:
  `build_scheduled_task_batch_plan(stage, tasks)` at
  `xenna/executor.py:109-112`.
- Driver RAM before ASR scheduler dispatch is:

```text
current_tasks returned by the pre-ASR Xenna segment
+ parent waveform arrays in current_tasks
+ shared scheduler-plan RAM after build_scheduled_task_batch_plan()
```

- `_run_scheduler_ready_stage()` then calls `_run_xenna_pipeline([stage],
  plan.ready_batches, scheduler_ready_stage=True)` at
  `xenna/executor.py:115-119`. This passes scheduler-ready rows as the input
  data for a second, one-stage Xenna pipeline.
- The scheduler-ready adapter forces `stage_batch_size == 1` at
  `xenna/adapter.py:159-165`, so each Xenna worker call receives one
  `SchedulerReadyTaskBatch` rather than allowing Xenna to merge unrelated
  scheduler rows.
- Xenna/Ray may still serialize the `SchedulerReadyTaskBatch` rows to workers.
  As with Ray Data, that can add object-store/transport memory beyond the
  driver references, especially if waveform slice views are serialized as
  independent buffers.
- Worker RAM during ASR is bounded by one scheduler-ready chunk batch plus
  model tensors. With `asr_num_workers: 4` and `omni_resource_gpus: 2.0`, the
  YAML is asking for up to four ASR workers, each reserving two GPUs; actual
  concurrency still depends on cluster resources.
- `_run_scheduler_ready_stage()` keeps the original `plan.parent_tasks` until
  processed chunks return, because `assemble_scheduled_task_batch_results()`
  needs them for stitch-back at `xenna/executor.py:120`. Peak memory during the
  one-stage ASR pipeline is therefore parent tasks plus the scheduler plan plus
  in-flight/returned processed chunk tasks.

## Xenna Batch: Bucketing On

Mock config:

```yaml
backend: xenna
execution_mode: batch
duration_aware_bucketing:
  enabled: true
```

The call chain is identical to Xenna streaming until execution mode selection:

1. `Pipeline.run()` calls `XennaExecutor.execute()` at `pipeline.py:215`.

2. `XennaExecutor.execute()` detects centralized ASR and calls
   `_run_pipeline_with_scheduler_ready_stages()` at `xenna/executor.py:70-83`.

3. `_run_pipeline_with_scheduler_ready_stages()` segments the graph around ASR
   at `xenna/executor.py:85-107`.

4. `_run_scheduler_ready_stage()` builds the ASR chunk/bucket plan at
   `xenna/executor.py:109-120`.

5. The shared plan again uses:
   - `base.py:167-201`
   - `asr/stage.py:479-493`
   - `asr/stage.py:604-712`
   - `batch_policy.py:355-381`

The difference from streaming is here:

```python
# nemo_curator/backends/xenna/executor.py:165-168
exec_mode = pipelines_v1.ExecutionMode.STREAMING
if self._get_pipeline_config("execution_mode") == "batch":
    exec_mode = pipelines_v1.ExecutionMode.BATCH
```

Because `exec_mode` is batch, Xenna does not create the streaming-specific
config at `xenna/executor.py:170-177`.

Then:

6. `_run_xenna_pipeline()` builds `PipelineConfig` and `PipelineSpec` at
   `xenna/executor.py:179-193`.

7. `pipelines_v1.run_pipeline(pipeline_spec)` runs the finite batch pipeline at
   `xenna/executor.py:198-211`.

8. The scheduler-ready ASR stage still uses
   `XennaSchedulerReadyStageAdapter.stage_batch_size == 1` at
   `xenna/adapter.py:159-165`.

9. Scheduler-ready ASR dispatch still enters
   `XennaSchedulerReadyStageAdapter.process_data()` at
   `xenna/adapter.py:167-172`.

10. ASR processing and stitch-back are the same:
    `base.py:285-299`, `asr/stage.py:382-473`,
    `asr/stage.py:784-840`, `xenna/executor.py:120`,
    `base.py:204-214`, and `asr/stage.py:519-527`.

Xenna batch shape:

```text
Xenna batch reader/prep segment
  -> list[parent AudioTask rows]
  -> Curator ASR scheduler plan
  -> Xenna batch one-stage SchedulerReady ASR pipeline
  -> list[processed chunk tasks]
  -> stitch back to parent rows
  -> Xenna batch downstream segment
```

RAM and overlap annotation:

- Xenna batch uses the same Curator segmentation as streaming:
  `_run_pipeline_with_scheduler_ready_stages()` runs the pre-ASR segment first,
  then `_run_scheduler_ready_stage()` builds the ASR scheduler plan, then any
  downstream segment runs after stitch-back.
- The main memory difference from streaming is that `_run_xenna_pipeline()`
  selects `ExecutionMode.BATCH` at `xenna/executor.py:165-168` and does not
  create `StreamingSpecificSpec` at `xenna/executor.py:170-177`.
- Because batch mode is finite by design, it is even more natural to think of
  this as phase-separated RAM:

```text
Phase 1, pre-ASR batch segment:
  Xenna owns input/output buffering for regular stages.
  return_last_stage_outputs=True returns all parent tasks to the driver.

Phase 2, Curator scheduler plan:
  driver holds parent_tasks + parent waveforms + O(C) chunk/scheduler metadata.

Phase 3, one-stage ASR batch segment:
  Xenna consumes plan.ready_batches.
  workers hold one SchedulerReadyTaskBatch each.
  driver still holds parent_tasks and the plan for stitch-back.

Phase 4, stitch-back:
  processed chunk tasks are collected.
  parent tasks receive final prediction fields.
```

- As in streaming mode, reader and Qwen ASR are not active together for the same
  centralized ASR plan. The pre-ASR Xenna batch pipeline returns before the ASR
  scheduler pipeline starts.
- Peak RAM terms are very similar to Xenna streaming:

```text
parent tasks + parent waveform arrays
+ scheduler chunk tasks / ready batches
+ Xenna/Ray serialization of ready batches
+ in-flight and returned processed chunk tasks
+ final parent tasks, with waveform retained unless keep_waveform=false
```

## Backend Comparison For 100k Rows

| Execution type | Where the 100k parent rows become scheduler work | Worker input shape | Stitch-back point |
|---|---|---|---|
| Ray Data | Driver collects dataset at `ray_data/executor.py:115` and plans at `ray_data/executor.py:116` | Ray Dataset rows of `SchedulerReadyTaskBatch`, `map_batches(batch_size=1)` at `ray_data/adapter.py:172-173` | `ray_data/executor.py:125-126` |
| Xenna streaming | `_run_scheduler_ready_stage()` plans after the pre-ASR Xenna segment returns tasks at `xenna/executor.py:109-120` | One-stage Xenna streaming pipeline over `SchedulerReadyTaskBatch`; batch size 1 at `xenna/adapter.py:159-165` | `xenna/executor.py:120` |
| Xenna batch | Same scheduler plan as streaming at `xenna/executor.py:109-120`; only `ExecutionMode.BATCH` differs at `xenna/executor.py:165-168` | One-stage Xenna batch pipeline over `SchedulerReadyTaskBatch`; batch size 1 at `xenna/adapter.py:159-165` | `xenna/executor.py:120` |

RAM/overlap comparison:

| Execution type | Can reader and Qwen ASR overlap with bucketing on? | Main RAM peak before/during ASR | What bounds GPU-worker RAM |
|---|---|---|---|
| Ray Data | No for the centralized ASR plan. Upstream reader/prep runs when `dataset.take_all()` is triggered, then ASR planning starts after all parents are collected. | Driver `parent_tasks` plus parent waveforms, scheduler plan, scheduler-ready Ray Dataset/object-store rows, processed chunk `take_all()` results. | One `SchedulerReadyTaskBatch` per worker call, bounded by `max_items_per_batch_by_bucket` and `max_audio_sec_per_batch`. |
| Xenna streaming | Pre-ASR Xenna stages can stream/overlap with each other, but ASR is split into a later scheduler-ready segment, so reader and Qwen ASR do not overlap for the same plan. | Returned pre-ASR `current_tasks`, parent waveforms, scheduler plan, Xenna/Ray ready-batch serialization, processed chunks. | Scheduler-ready adapter `stage_batch_size == 1`; each row is already bucket-capped. |
| Xenna batch | No. The pre-ASR batch segment completes and returns all parent rows before scheduler-ready ASR starts. | Same terms as streaming, with batch-mode Xenna buffering instead of streaming-specific scheduling. | Same as streaming: one already-planned scheduler row per worker call. |

The GPU-facing ASR batch is therefore not the original manifest order. It is a
duration-coherent list of already-chunked ASR work units produced by the shared
Curator scheduler.

Bottom line for the user's overlap question:

```text
With duration_aware_bucketing.enabled=true in this PR:
  reader/prep may overlap internally inside a regular backend segment
  but centralized ASR chunking + bucketing is a barrier
  so chunking/bucketing happens once over the materialized parent list
  then workers consume scheduler-ready batches
```

That is why RAM is higher at the ASR boundary than a fully persistent
cross-stage streaming scheduler would be: the current design maximizes
duration-coherent GPU dispatch for the materialized ASR window, but it does not
let Qwen ASR consume rows while the same 100,000-row parent window is still
being read.
