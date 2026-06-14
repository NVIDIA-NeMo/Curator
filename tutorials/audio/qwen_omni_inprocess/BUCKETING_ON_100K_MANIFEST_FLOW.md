# Bucketing-On Flow For A 100k-Row Manifest

This document describes the current PR checkout:

```text
/home/aaftabv/grananary-v2/references/CuratorPR1967
```

It covers `duration_aware_bucketing.enabled: true` for roughly 100,000 mixed
duration audio rows across:

- Ray Data
- Xenna streaming
- Xenna batch

The important current contract is **not** a separate scheduler-ready backend
stream. The active Ray Data/Xenna path is:

```text
parent rows arrive in a backend window
  -> ASR builds chunk work units for that window
  -> BatchPolicy buckets those chunks
  -> BaseStageAdapter processes ready batches inside the same worker call
  -> ASR stitches chunks back to parent rows
  -> backend continues with parent rows
```

The older design materialized at the Ray Data ASR boundary and split Xenna into
multiple `_run_xenna_pipeline()` segments. That is now stale.

## Symbols And RAM Terms

```text
N = parent input rows, about 100,000
C = ASR chunk work units after splitting long audio
B = ready batches emitted by BatchPolicy for one backend window
W = concurrent backend workers
audio_bytes = sum(waveform.nbytes for retained waveform arrays)
```

For mono float32 audio:

```text
bytes_per_second = sample_rate * 4
16 kHz audio ~= 64 KB/sec
40 min chunk ~= 153.6 MB decimal
4 hr parent clip ~= 921.6 MB decimal
```

The dominant RAM term is usually waveform retention and backend transport,
not Python list or dataclass overhead.

## Mock Config

Illustrative config:

```yaml
backend: ray_data            # or xenna
execution_mode: streaming    # Xenna only: streaming or batch

chunking:
  enabled: true

duration_aware_bucketing:
  enabled: true
  strategy: duration_bucketed
  buckets_sec: [0, 600, 1200, 2400]
  max_items_per_batch_by_bucket: [32, 16, 8, 4]
  max_audio_sec_per_batch: 2400
  prebatching_window_size: null

stages:
  - stage_id: qwen_omni
    _target_: nemo_curator.stages.audio.inference.asr.ASRStage
    chunking_enabled: ${chunking.enabled}
    batch_size: ${model_batch_size}
    batch_policy:
      _target_: nemo_curator.stages.audio.inference.batch_policy.BatchPolicy
      enabled: ${duration_aware_bucketing.enabled}
      strategy: ${duration_aware_bucketing.strategy}
      buckets_sec: ${duration_aware_bucketing.buckets_sec}
      max_items_per_batch_by_bucket: ${duration_aware_bucketing.max_items_per_batch_by_bucket}
      max_audio_sec_per_batch: ${duration_aware_bucketing.max_audio_sec_per_batch}
      prebatching_window_size: ${duration_aware_bucketing.prebatching_window_size}
```

The checked-in tutorial wiring is at
`tutorials/audio/qwen_omni_inprocess/qwen_omni_inprocess.yaml:120-176`.

RAM notes:

- `chunking.enabled=true` bounds individual model requests.
- `duration_aware_bucketing.enabled=true` enables duration-coherent mixing.
- `prebatching_window_size: null` means the backend candidate window defaults
  to `sum(max_items_per_batch_by_bucket)`.
- `keep_waveform=true` keeps parent waveform arrays after ASR. Setting it false
  lowers post-ASR RAM, but not pre-ASR planning RAM.

## Common Entry Flow

1. Hydra enters `main()` at
   `tutorials/audio/qwen_omni_inprocess/main.py:285-296`.
2. `build_granary_v2_pipeline(cfg)` constructs the `Pipeline` at
   `main.py:258-260`.
3. `_create_executor(cfg)` picks `RayDataExecutor` or `XennaExecutor` at
   `main.py:263-282`.
4. `Pipeline.run()` calls `executor.execute(self.stages, initial_tasks)` at
   `nemo_curator/pipeline/pipeline.py:177-215`.

For the checked-in Qwen tutorial, `NemoTarredAudioReader` decodes audio into
waveform arrays before ASR. By the time ASR runs, the row shape is one
`AudioTask` per utterance with waveform/sample-rate fields.

RAM note:

```text
before ASR:
  parent AudioTask rows
  + decoded waveform arrays for the active backend window/block/segment
```

Exactly how much of the 100,000 rows can coexist before ASR depends on backend
windowing and reader behavior. The fixed bucket-on path no longer intentionally
pulls all 100,000 rows to the driver just to plan ASR.

## Shared ASR Planning Inside A Backend Window

The centralized gate is structural:

```python
# nemo_curator/backends/base.py:99-113
def stage_uses_centralized_batching(stage):
    policy = _enabled_batch_policy(stage)
    if policy is None:
        return False
    build_tasks = getattr(stage, "build_prebucketed_tasks", None)
    assemble_results = getattr(stage, "assemble_prebucketed_task_results", None)
    return callable(build_tasks) and callable(assemble_results) and _scheduler_task_cost_fn(stage) is not None
```

For each backend-visible ASR window, `BaseStageAdapter.process_batch()` plans:

```python
# nemo_curator/backends/base.py:267-283
def process_batch(self, tasks):
    centralized_plan = build_scheduled_task_batch_plan(self.stage, tasks)
    if centralized_plan is not None:
        return self._process_scheduled_task_batch_plan(centralized_plan)
    return self._process_batch_after_worker_planning(tasks)
```

`build_scheduled_task_batch_plan()` builds chunk tasks and ready batches:

```python
# nemo_curator/backends/base.py:167-201
scheduler_tasks = build_tasks(tasks)
ready_batches = [
    SchedulerReadyTaskBatch(tasks=list(sub_tasks), total_cost=total_cost)
    for source_indices, sub_tasks, total_cost in policy.bucketize_with_costs(scheduler_tasks, cost_fn=cost_fn)
    if sub_tasks
]
```

ASR creates chunk work units once:

```python
# nemo_curator/stages/audio/inference/asr/stage.py:548-562
def build_prebucketed_tasks(self, tasks):
    chunk_plan = self._build_prebucket_chunk_plan(tasks)
    return [chunk.task for chunk in chunk_plan]
```

Chunk descriptors are shared by direct and bucketed paths:

```python
# nemo_curator/stages/audio/inference/asr/stage.py:655-705
chunks = self._chunk_waveform(waveform, sr, slice_ceiling)
for chunk_idx, chunk in enumerate(chunks):
    specs.append(_ChunkSpec(parent_idx=parent_idx, chunk_idx=chunk_idx, cost=...))
```

The finite planner is at `batch_policy.py:355-381` and sorts ready batches by
descending cost.

RAM inside one backend ASR window:

```text
parent rows in window
+ parent waveform arrays in window
+ _ChunkSpec list for that window
+ minimal chunk AudioTask rows for that window
+ ready batch lists for that window
+ processed chunk tasks until stitch-back
```

The key improvement versus the stale path is the scope: this is now per backend
window/call, not a mandatory all-dataset driver plan.

## Ray Data Bucketing On

Ray Data executor path:

```python
# nemo_curator/backends/ray_data/executor.py:82-104
for stage in stages:
    current_dataset = self._process_stage_dataset(stage, current_dataset)

def _process_stage_dataset(self, stage, dataset):
    adapter = RayDataStageAdapter(stage)
    return adapter.process_dataset(dataset, self.ignore_head_node)
```

There is no `_process_centralized_stage_dataset()` branch in the active path.

The Ray Data adapter sets the ASR map window:

```python
# nemo_curator/backends/ray_data/adapter.py:185-197
if scheduler_ready_batches or preplanned_batches:
    return 1
if centralized_batches:
    return upstream_prebatching_batch_size(self.stage, self.batch_size)
return self.batch_size
```

The final `take_all()` still exists, but only after all stages:

```python
# nemo_curator/backends/ray_data/executor.py:91-134
output_tasks = self._dataset_to_tasks(current_dataset)
...
items = dataset.take_all()
```

Flow:

```text
Ray Dataset[parent AudioTask rows]
  -> lazy stage transforms
  -> ASR map_batches(batch_size=planner_window)
  -> ASR actor/task plans chunk buckets inside that window
  -> stitched parent rows stay in the Ray Dataset
  -> final take_all() after writer/final stage
```

Overlap:

- Ray Data can pipeline upstream reader/prep work with downstream stage work as
  its streaming executor allows.
- ASR planning is local to a Ray Data map window, so ASR can start before the
  full 100,000-row manifest is read if Ray Data has enough upstream blocks.
- No ASR plan can mix chunks across different Ray Data windows.

RAM:

```text
driver:
  no forced 100k-row ASR-boundary take_all()
  final output collection only if the caller returns tasks

object store / workers:
  active Ray Data blocks
  active ASR map windows
  per-window chunk/bucket plan
  W concurrent ASR model calls
```

Throughput:

- Better than the stale materialized path because the ASR boundary no longer
  drains the entire upstream dataset to the driver.
- Still below a global scheduler ceiling because ready batches are not
  independent Ray work items across all windows.

## Xenna Streaming Bucketing On

Xenna executor path:

```python
# nemo_curator/backends/xenna/executor.py:66-77
def execute(self, stages, initial_tasks=None):
    initial_tasks = initial_tasks if initial_tasks else [EmptyTask]
    return self._run_xenna_pipeline(stages, initial_tasks)
```

All stages enter one Xenna pipeline:

```python
# nemo_curator/backends/xenna/executor.py:90-112
for stage in stages:
    xenna_stage = create_named_xenna_stage_adapter(stage=stage)
    stage_specs.append(StageSpec(stage=xenna_stage, ...))
```

The regular adapter chooses a scheduler-sized ASR candidate window:

```python
# nemo_curator/backends/xenna/adapter.py:82-86
def stage_batch_size(self):
    batch_size = self.processing_stage.batch_size
    return upstream_prebatching_batch_size(self.processing_stage, batch_size)
```

Streaming mode is selected here:

```python
# nemo_curator/backends/xenna/executor.py:114-126
exec_mode = pipelines_v1.ExecutionMode.STREAMING
if self._get_pipeline_config("execution_mode") == "batch":
    exec_mode = pipelines_v1.ExecutionMode.BATCH
```

Flow:

```text
one Xenna streaming pipeline
  -> reader/prep can stream inside Xenna's graph
  -> ASR workers receive planner-sized parent windows
  -> each ASR worker locally chunks/buckets/stitches its window
  -> parent rows continue to writer
```

Overlap:

- The old Curator-side barrier between pre-ASR and ASR is gone.
- Xenna streaming can overlap stages according to Xenna's streaming executor.
- Bucketing is still local to each ASR worker call/window.

RAM:

```text
Xenna/Ray:
  active streaming queues and stage buffers
  active ASR worker windows
  per-window ASR chunk/bucket plans

driver:
  no returned pre-ASR parent list solely for scheduler planning
```

Throughput:

- Better than the stale split path because there is no separate one-stage
  scheduler-ready ASR pipeline and no pre-ASR result list returned just for
  planning.
- Still not a global ready-batch scheduler because one ASR call owns and
  serially drains its local ready batches.

## Xenna Batch Bucketing On

Xenna batch uses the same single Curator pipeline path as streaming. The only
Curator-visible difference is the `ExecutionMode.BATCH` selection at
`xenna/executor.py:114-117`.

Flow:

```text
one Xenna batch pipeline
  -> Xenna finite-stage execution
  -> ASR receives planner-sized parent windows
  -> each ASR worker locally chunks/buckets/stitches its window
  -> parent rows continue to writer
```

Overlap:

- Xenna batch itself is finite-stage oriented, so it has less streaming overlap
  than Xenna streaming.
- The PR no longer adds extra Curator-side segmentation around ASR.
- Bucketing remains local to each ASR batch/window.

RAM:

```text
Xenna batch buffers
+ active ASR worker windows
+ per-window chunk/bucket plans
+ model tensors for W concurrent workers
```

Throughput:

- The fixed path avoids additional Curator pipeline splits.
- Native Xenna batch scheduling still determines how much stage overlap exists.
- It is not a persistent global bucket queue.

## Backend Comparison

| Backend | Current bucket-on planning scope | Removed stale overhead | Remaining ceiling |
|---|---|---|---|
| Ray Data | One `map_batches` ASR window | No driver `take_all()` at ASR boundary | No global scheduling across Ray Data windows |
| Xenna streaming | One ASR worker call/window inside one Xenna streaming pipeline | No pre-ASR / ASR / post-ASR pipeline split | Ready batches are local to one ASR call |
| Xenna batch | One ASR worker call/window inside one Xenna batch pipeline | No extra Curator split around ASR | Batch mode is finite-stage oriented and window-local |

RAM/overlap comparison:

| Backend | Can reader/prep overlap ASR? | Main ASR RAM term | Worker RAM bound |
|---|---|---|---|
| Ray Data | Yes, as Ray Data streaming permits, because no ASR-boundary `take_all()` is forced | Active Ray blocks + active ASR windows + per-window chunk plans | Bucket caps inside each ASR window |
| Xenna streaming | Yes, according to Xenna streaming execution; no Curator ASR split remains | Xenna queues + active ASR windows + per-window chunk plans | Bucket caps inside each ASR window |
| Xenna batch | Limited by Xenna batch execution semantics | Batch buffers + active ASR windows + per-window chunk plans | Bucket caps inside each ASR window |

## Throughput Ceiling Versus Actor-Pool Scheduler Dispatch

The fixed Ray Data/Xenna path is the right low-risk backend-compatible path for
this PR. It removes the two known regressions:

- Ray Data all-parent materialization before ASR
- Xenna pipeline splitting around ASR

It is still not the maximum possible long-tail scheduler design.

Current Ray Data/Xenna:

```text
window -> local chunking/bucketing -> serial ready-batch loop in one worker call
```

Higher-ceiling actor-pool scheduler:

```text
parent tasks -> global bucket queues -> ready batches as independent work items
                                 -> idle GPU actors consume batches
```

The actor-pool shape can rebalance long and short ready batches across all
available GPU workers. The current Ray Data/Xenna shape improves throughput
without changing backend row contracts, but it cannot rebalance across map
window or Xenna-call boundaries.

Live analysis should therefore compare:

- overall pipeline throughput from `merge_info.pipeline_duration_s`
- model inference throughput from `stages.QwenOmni_inference`
- output equivalence across previous approach, current bucket-off, and current
  bucket-on
- scaling from 1 node x 4 GPU to 2 nodes x 4 GPU per node, to identify whether
  remaining limits are local-window or cross-actor scheduling limits
