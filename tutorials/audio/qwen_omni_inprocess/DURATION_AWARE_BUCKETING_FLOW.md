# Duration-Aware Bucketing Execution Flow

This note describes the current PR code on disk for mixed-duration audio,
including long-tail inputs such as 5 seconds, 10 minutes, 50 minutes, 2 hours,
and 4 hours.

The current design is **window-local scheduler planning inside regular backend
stage execution**:

```text
backend row window
  -> BaseStageAdapter.process_batch()
  -> build_scheduled_task_batch_plan()
  -> ASR chunk work units
  -> BatchPolicy bucket queues
  -> ready batches processed inside that worker call
  -> stitched parent rows returned to the backend
```

This replaced the older Ray Data/Xenna scheduler-ready stream path. The default
executors no longer materialize a whole Ray Dataset at the ASR boundary and no
longer split Xenna into separate pre-ASR / scheduler-ready ASR / post-ASR
pipelines.

## Config

The Qwen tutorial keeps chunking and bucketing as separate controls:

```yaml
# qwen_omni_inprocess.yaml:120-131
duration_aware_bucketing:
  enabled: true
  strategy: duration_bucketed
  buckets_sec: [0, 600, 1200, 2400]
  max_items_per_batch_by_bucket: [32, 16, 8, 4]
  max_audio_sec_per_batch: 2400
  prebatching_window_size: null
  flush_interval_ms: 250
```

```yaml
# qwen_omni_inprocess.yaml:149-176
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

`chunking.enabled=true` protects model request size. `duration_aware_bucketing`
only controls whether chunk work units are mixed into duration-coherent batches.
With bucketing off, chunking can still happen in the normal backend batch flow.

## Shared Gate

Every backend reaches the same structural gate:

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

ASR opts in when its `BatchPolicy.enabled` is true because it has:

- `build_prebucketed_tasks()` at `asr/stage.py:548-562`
- `scheduler_task_cost()` at `asr/stage.py:564-568`
- `assemble_prebucketed_task_results()` at `asr/stage.py:570-576`

With `enabled=false`, `_enabled_batch_policy()` returns `None`, the gate is
false, and ASR follows normal backend batching.

## Bucketing On

The core planner is backend-independent:

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
        if sub_tasks
    ]
    return ScheduledTaskBatchPlan(parent_tasks=list(tasks), ready_batches=ready_batches)
```

`BaseStageAdapter.process_batch()` invokes that planner on the backend-visible
row window:

```python
# nemo_curator/backends/base.py:267-283
def process_batch(self, tasks):
    centralized_plan = build_scheduled_task_batch_plan(self.stage, tasks)
    if centralized_plan is not None:
        return self._process_scheduled_task_batch_plan(centralized_plan)
    return self._process_batch_after_worker_planning(tasks)
```

Ready batches are processed serially inside the same worker call:

```python
# nemo_curator/backends/base.py:278-287
def _process_scheduled_task_batch_plan(self, plan):
    processed_tasks = []
    for ready_batch in plan.ready_batches:
        processed_tasks.extend(self.process_scheduler_ready_batch(ready_batch))
    return assemble_scheduled_task_batch_results(self.stage, plan, processed_tasks)
```

ASR builds chunk work units once:

```python
# nemo_curator/stages/audio/inference/asr/stage.py:548-562
def build_prebucketed_tasks(self, tasks):
    policy = self.batch_policy
    if policy is None or not policy.enabled:
        return None
    chunk_plan = self._build_prebucket_chunk_plan(tasks)
    return [chunk.task for chunk in chunk_plan]
```

Chunk descriptors come from `_build_chunk_specs()`:

```python
# nemo_curator/stages/audio/inference/asr/stage.py:655-705
slice_ceiling = float(self.max_inference_duration_s or self.ideal_inference_segment_s)
for parent_idx, task in enumerate(tasks):
    chunks = self._chunk_waveform(waveform, sr, slice_ceiling)
    for chunk_idx, chunk in enumerate(chunks):
        specs.append(_ChunkSpec(parent_idx=parent_idx, chunk_idx=chunk_idx, cost=...))
```

The finite planner computes costs once, disables timer checks, drains at the
end, and sorts heavier batches first:

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

Shape:

```text
backend window of parent rows
  -> ASR chunk work units
  -> buckets:
       [0, 600) seconds     cap 32 items
       [600, 1200) seconds  cap 16 items
       [1200, 2400) seconds cap 8 items
       [2400, +inf) seconds cap 4 items
  -> ready batches processed heavy-first inside the worker call
  -> exact chunk validation and stitch-back
```

The stitch-back validation now rejects:

- parent indices outside the input range
- non-positive chunk counts
- chunk ids outside `0..chunk_count-1`
- duplicate `(parent_idx, chunk_idx)`
- chunk-count disagreement for a parent
- any missing chunk id
- adapter result count mismatches before assembly

Relevant code: `asr/stage.py:538-542` for result-count validation and
`asr/stage.py:596-653` for exact chunk-id validation.

## Ray Data

Ray Data no longer has a special centralized-stage executor branch:

```python
# nemo_curator/backends/ray_data/executor.py:101-104
def _process_stage_dataset(self, stage, dataset):
    adapter = RayDataStageAdapter(stage)
    return adapter.process_dataset(dataset, self.ignore_head_node)
```

For a centralized ASR stage, `RayDataStageAdapter` chooses a larger candidate
window instead of `batch_size=1` scheduler-ready rows:

```python
# nemo_curator/backends/ray_data/adapter.py:185-197
def _map_batch_size(..., centralized_batches):
    if scheduler_ready_batches or preplanned_batches:
        return 1
    if centralized_batches:
        return upstream_prebatching_batch_size(self.stage, self.batch_size)
    return self.batch_size
```

Default `upstream_prebatching_batch_size()` is the sum of bucket item caps,
unless `duration_aware_bucketing.prebatching_window_size` overrides it:

```python
# nemo_curator/backends/base.py:235-253
configured_window = getattr(policy, "prebatching_window_size", None)
if configured_window is not None:
    return int(configured_window)
...
policy_window = sum(int(cap) for cap in caps)
return max(1, policy_window)
```

Ray Data shape:

```text
Ray Dataset[parent rows]
  -> lazy map_batches chain
  -> ASR map_batches window, e.g. sum(bucket caps)
  -> BaseStageAdapter plans and runs bucketed chunk batches in that actor/task
  -> parent rows returned to the Ray Dataset
  -> final dataset.take_all() only after all stages
```

Throughput implication: this removes the previous driver `take_all()` barrier at
the ASR boundary and preserves Ray Data's normal streaming execution model.
However, planning is still local to each Ray Data map window. Ready batches from
different windows do not become independent global work items.

## Xenna Streaming

Xenna no longer detects centralized stages and splits the pipeline. The complete
stage list goes through one pipeline:

```python
# nemo_curator/backends/xenna/executor.py:66-77
def execute(self, stages, initial_tasks=None):
    initial_tasks = initial_tasks if initial_tasks else [EmptyTask]
    return self._run_xenna_pipeline(stages, initial_tasks)
```

Each stage uses the regular Xenna adapter:

```python
# nemo_curator/backends/xenna/executor.py:90-112
for stage in stages:
    xenna_stage = create_named_xenna_stage_adapter(stage=stage)
    stage_specs.append(StageSpec(stage=xenna_stage, ...))
```

The adapter reports a scheduler-sized candidate window:

```python
# nemo_curator/backends/xenna/adapter.py:82-86
def stage_batch_size(self):
    batch_size = self.processing_stage.batch_size
    return upstream_prebatching_batch_size(self.processing_stage, batch_size)
```

Xenna streaming shape:

```text
one Xenna streaming pipeline
  -> reader/prep/writer can stay in the same Xenna graph
  -> ASR worker receives a scheduler-sized parent window
  -> BaseStageAdapter plans chunk buckets inside that worker call
  -> ASR returns stitched parent rows downstream
```

Throughput implication: this avoids the previous pre-ASR segment barrier and
the one-stage scheduler-ready ASR pipeline. It should reduce orchestration and
materialization overhead. It is still not a persistent cross-worker scheduler:
ready batches inside one ASR call are processed serially by that call.

## Xenna Batch

Xenna batch uses the same Curator adapter path as streaming. The only executor
difference is the Xenna execution mode:

```python
# nemo_curator/backends/xenna/executor.py:114-117
exec_mode = pipelines_v1.ExecutionMode.STREAMING
if self._get_pipeline_config("execution_mode") == "batch":
    exec_mode = pipelines_v1.ExecutionMode.BATCH
```

Shape:

```text
one Xenna batch pipeline
  -> Xenna finite-stage execution
  -> ASR receives scheduler-sized parent windows
  -> chunking/bucketing/stitch-back happen inside each ASR worker call
```

This removes the older Curator-side split into multiple `_run_xenna_pipeline()`
segments. Xenna batch itself is still finite-stage oriented, so do not interpret
it as a persistent global GPU work queue.

## Bucketing Off

With `duration_aware_bucketing.enabled=false`, `stage_uses_centralized_batching`
returns false. ASR follows `_process_plain_batch()`:

```python
# nemo_curator/stages/audio/inference/asr/stage.py:439-484
def _process_plain_batch(self, tasks):
    if self.chunking_enabled:
        chunk_specs = self._build_chunk_specs(tasks)
        items = [self._chunk_spec_to_item(spec) for spec in chunk_specs]
    else:
        items = [{... one item per parent task ...} for task in tasks]
    results = self._run_inference_capped(items)
    return self.assemble(tasks, items, parent_of, results)
```

Off mode means no duration-bucket mixing. If chunking is enabled, each normal
backend batch can still be sliced for model safety, then stitched back.

## Throughput Ceiling

Current Ray Data/Xenna bucket-on is better than the stale split/materialization
path, but it is not the same ceiling as actor-pool scheduler-ready dispatch.

Current Ray Data/Xenna:

```text
backend window -> one worker call -> local plan -> serial ready-batch loop
```

Actor-pool scheduler-ready dispatch:

```text
parent tasks -> global scheduler -> ready batches as independent work items
                              -> idle GPU workers pull/receive batches
```

The actor-pool shape can reduce tail time better because each ready batch can be
scheduled independently across workers. The Ray Data/Xenna shape avoids major
driver/materialization overhead and preserves backend pipeline semantics, but it
cannot rebalance ready batches across map-window or Xenna-call boundaries.

Live throughput analysis should therefore report this precisely:

- Ray Data/Xenna fixed path: window-local duration-aware bucketing with no
  centralized driver materialization or Xenna pipeline split.
- Actor-pool ideal/theory path: independent scheduler-ready dispatch, higher
  throughput ceiling under skewed long-tail duration distributions.
