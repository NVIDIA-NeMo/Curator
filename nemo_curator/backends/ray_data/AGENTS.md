# Ray Data Backend — Agent Tuning Guide

Use this guide when diagnosing or tuning a NeMo Curator pipeline running on the
Ray Data backend. Start from measured scheduler behavior; do not apply a setting
only because it helped another pipeline.

**Supported baseline:** Curator requires Ray 2.57.0 or later. The diagnostic shim
in `diagnostics.py` targets exactly Ray 2.57.0 unless the installed Ray version
already provides the diagnostics natively. Ray Data defaults and private APIs can
change between releases, so re-check source before carrying this guidance forward.

## Start here

1. Treat a Curator `Task`, not a document or ordinary Ray row, as the unit of work.
2. Enable diagnostics and inspect `ray-data.log` before changing worker counts,
   concurrency, memory, or backpressure.
3. Distinguish starvation, actor saturation, resource admission, and downstream
   backpressure. Similar GPU utilization can arise from different causes.
4. Change one lever per run and compare scheduler events, operator timing, task
   timing, object-store usage, and GPU telemetry—not wall time alone.

## Curator execution model

Ray Data parallelizes map work over blocks, then forms row batches within those
blocks. Curator constructs its initial dataset with:

```python
ray.data.from_items(tasks, override_num_blocks=len(tasks))
```

This creates one opaque Task row per block. Fanout stages repartition their outputs
back to one row per block. In practice, **one row = one block = one Task**.

The default stage `batch_size=1` passes one Task to each stage call. Ray cannot see
or split records contained inside that Task. If a task is too large, too small, or
too slow, tune Curator task partitioning and per-task payload size rather than Ray
block-size knobs. Pipeline completion calls `take_all()`, so final Task payloads are
collected back to the driver.

### Task stages and actor stages

`RayDataStageAdapter` applies every stage with `map_batches` and chooses the compute
strategy as follows:

| Stage | Curator selection | Worker behavior |
|---|---|---|
| Actor | Overrides `setup()`, or requests both CPU and GPU; can be forced with `IS_ACTOR_STAGE=True` | Persistent state and `ActorPoolStrategy`; fixed or autoscaling pool |
| Task | Stateless and not selected as an actor; can be forced with `IS_ACTOR_STAGE=False` | `TaskPoolStrategy`; no actor startup or actor autoscaler |

For a task stage, `num_workers=N` caps the task pool at N; without it, Ray Data
controls parallelism through resources and backpressure. For an actor stage,
`num_workers=N` creates a fixed pool. Otherwise Curator passes `MIN_WORKERS`,
`MAX_WORKERS`, and `INITIAL_WORKERS` to an autoscaling actor pool.

## Log-first tuning workflow

### 1. Record the pipeline shape

Before tuning, record:

- task count, representative serialized task size, and work contained in each Task;
- stage `batch_size`, task-versus-actor selection, CPU/GPU request, and worker bounds;
- `max_concurrency` and the global `max_tasks_in_flight_per_actor` override;
- cluster CPU/GPU resources, object-store size, and Ray temporary directory;
- per-stage processing time, gaps between tasks, and GPU utilization/power.

Without this baseline, a faster run can be a workload, cache, GPU-power, or startup
artifact rather than a scheduler improvement.

### 2. Enable and find diagnostics

Set the opt-in environment variable before starting the driver:

```bash
export NEMO_CURATOR_RAY_DATA_DIAGNOSTICS=1
```

The values `true`, `yes`, and `on` also enable it.

The shim emits structured logfmt events through the `ray.data` logger to:

```text
$RAY_TEMP_DIR/session_latest/logs/ray-data/ray-data.log
```

Set `RAY_TEMP_DIR` with `RayClient(ray_temp_dir=...)` or
`SlurmRayClient(ray_temp_dir=...)`; the default is `~/.ray`. Ray can also resolve
the directory with `ray.data._internal.logging.get_log_directory()`.

Events are emitted on scheduler **state changes**, not on every tick. Actor counts
and utilization in an event are snapshots, not a time series. Pair them with Ray
operator timing and GPU telemetry.

### 3. Classify the bottleneck

| Symptom | Evidence to inspect | Likely interpretation | First controlled experiment |
|---|---|---|---|
| GPU actors are idle and `queued_input_blocks=0` | Upstream operator timing and admission events | GPU stage is starved | Adjust task partitioning or upstream parallelism; verify the producer is not blocked |
| Inputs are queued but work is not admitted | `scheduling_reason`, remaining budget, actor slots | Resource, capacity, concurrency, or actor-slot limit | Change the indicated limit only |
| Autoscaling pool stays near its minimum despite backlog | `utilization`, `tasks_in_flight`, scheduling reason | In-flight/concurrency ratio cannot reach scale-up threshold, or backpressure suppresses scaling | Fix the ratio or the blocking condition |
| Producer stops while object-store bytes rise | Resource admission reason and internal/output bytes | Pending output or total object-store budget is exhausted | Reduce per-task payload or increase object-store capacity |
| Frequent downstream blocked/allowed transitions | Queue ratio, configured ratio, recovered blocked duration | Producer is repeatedly outrunning consumer capacity | Compare one capacity-ratio change on the same workload |
| `INITIAL_WORKERS=N` starts N actors, then the pool shrinks | Autoscaling decisions after first input | Initial size is not a minimum | Set `MIN_WORKERS`, or use `num_workers` for a fixed pool |
| Concurrency 2 lowers idle gaps but raises processing time or memory | Task timings, queued bytes, object-store and GPU memory | Batching overlap helps, but preparation/contention costs increased | Compare concurrency 1 and 2 with all other settings fixed |

### 4. Compare runs

Use the same input, cluster shape, warmup policy, and instrumentation. Compare:

- total wall time and per-operator time;
- stage processing time versus idle/handoff gaps;
- actor pool size, utilization, and scale transitions;
- count and duration of admission blocks;
- `object_store_internal_bytes` versus `object_store_output_bytes`;
- GPU utilization, power, and memory.

For admission events, count recovered blocked intervals and sum
`blocked_duration_ms`. The sum undercounts an interval still blocked when the
pipeline ends, and a blocked operator does not imply zero progress elsewhere.

## Diagnostic events

Every event includes `operator`, the Curator/Ray Data stage name.

### `ray_data_actor_autoscaling_decision`

Emitted when an actor pool's scaling decision changes.

| Field group | Fields | Interpretation |
|---|---|---|
| Decision | `decision`, `delta`, `scaling_reason`, `scheduling_reason` | Requested scale change and why it did or did not occur |
| Pool | `current_actors`, `min_actors`, `max_actors`, `running_actors`, `pending_actors`, `active_actors`, `idle_actors` | Current actor lifecycle and occupancy snapshot |
| Work | `utilization`, `tasks_in_flight`, `queued_input_blocks`, `queued_input_bytes` | Submitted work relative to actor concurrency and queued input |
| Resources | `allocation_*`, `usage_*`, `remaining_budget_*` | CPU, GPU, heap, and object-store allocation and headroom |
| Object store | `object_store_internal_bytes`, `object_store_output_bytes` | Pending task output versus completed output retained in the pipeline |

`scheduling_reason` can be `runnable`, `ResourceBudget`, `DownstreamCapacity`,
`ConcurrencyCap`, `no_pending_inputs`, `no_actor_slot`,
`operator_cannot_accept_input`, or `completed`.

### `ray_data_resource_budget_admission`

Emitted when an operator changes between resource-budget `allowed` and `blocked`.

| Field group | Fields | Interpretation |
|---|---|---|
| Decision | `state`, `reason` | Whether the next task fits and the failed condition |
| Next task | `requested_*`, `pending_output_estimate` | Incremental resources and worst-case pending output estimate |
| Budget | `remaining_budget_*`, `usage_*`, `allocation_*` | Current headroom, usage, and total allocation |
| Object store | `object_store_internal_bytes`, `object_store_output_bytes` | Where the producing operator's object-store budget is held |
| Duration | `blocked_duration_ms` | Completed blocked interval, populated on recovery to `allowed` |

Resource families ending in `_*` contain `cpu`, `gpu`, `heap_memory`, and
`object_store_memory`. Admission `reason` can be:

| Reason | Meaning |
|---|---|
| `allowed` or `unlimited` | Task can be admitted |
| `incremental_cpu_exceeds_budget` | CPU request exceeds remaining budget |
| `incremental_gpu_exceeds_budget` | GPU request exceeds remaining budget |
| `incremental_heap_memory_exceeds_budget` | Heap-memory request exceeds remaining budget |
| `incremental_object_store_memory_exceeds_budget` | Incremental object-store request exceeds remaining budget |
| `pending_output_exceeds_object_store_budget` | Worst-case task output exceeds remaining object-store budget |

### `ray_data_downstream_capacity_admission`

Emitted when an upstream operator changes between downstream-capacity `allowed` and
`blocked`.

| Field | Interpretation |
|---|---|
| `state` | `allowed` or `blocked` |
| `queue_bytes` | Upstream output queued for downstream consumption |
| `downstream_capacity_bytes` | Bytes held by pending downstream task inputs |
| `queue_ratio`, `configured_ratio` | Current queue/capacity ratio and its threshold |
| `utilized_object_store_budget_fraction` | Object-store budget utilization gate |
| `object_store_internal_bytes`, `object_store_output_bytes` | Producer-attributed object-store categories |
| `blocked_duration_ms` | Completed blocked interval, populated on recovery |

The diagnostics do not directly time metadata fetching. All three event types include
the two object-store byte categories. Resource and downstream events report blocked
duration only after recovery; it is `null` when entering the blocked state.

## Scheduler decisions that matter

### Resource reservation and task admission

Ray Data divides resources between eligible TaskPool and ActorPool operators. With
`op_resource_reservation_ratio=0.5`, each operator starts from:

```text
default_reserved = total_resources * reservation_ratio / eligible_operators
```

The reservation contains task resources and object-store space for operator outputs;
unreserved resources form a shared pool. Usage above an operator's reservation is
deducted from that shared pool before it is divided again.

The resource policy admits the next task only when both are true:

1. its incremental CPU, GPU, heap, and object-store use fit the current budget; and
2. the remaining object-store budget covers the operator's maximum pending output
   estimate for one task.

Ray attributes buffered object-store bytes to the **producing** operator, including
completed blocks already queued at a downstream input. This is why a fast reader can
be blocked on object-store budget while a slower GPU stage consumes its output.

`object_store_internal_bytes` represents pending task outputs. Completed blocks in
the operator output queue or retained downstream appear in
`object_store_output_bytes`. Heap memory is separate process memory for Python
objects, models, and tensors. Standard Curator `Resources` exposes CPU and GPU but
not a heap-memory request, so heap-budget denial is uncommon unless `memory` is
passed through `RAY_REMOTE_ARGS`.

### Downstream-capacity backpressure

When object-store utilization is available, downstream-capacity backpressure blocks
an upstream operator only when both are true:

1. its object-store budget utilization is greater than `0.5`; and
2. `queue_bytes / downstream_capacity_bytes` is greater than the configured ratio,
   which defaults to `2.0`.

If utilization is unavailable, Ray evaluates the queue ratio alone. If downstream
capacity is zero because no downstream tasks are pending, the queue ratio is treated
as zero. When the policy blocks, Ray stops both new task admission and pulling
additional task output. Lower ratios throttle producers sooner but can starve or
suppress scaling of the consumer. Tune the ratio only with controlled measurements.

If every operator is blocked and no task is active, Ray can bypass backpressure to
dispatch pending work and preserve liveness.

### Actor autoscaling

Task stages do not use the actor autoscaler. For an actor pool:

```text
utilization = tasks_in_flight / (max_concurrency * current_actors)
```

Important defaults are `1.75` for scale-up, `0.5` for scale-down, and one actor as
the maximum scale-up delta per decision. In decision order, the autoscaler:

1. drains idle actors after input is exhausted;
2. creates actors up to `MIN_WORKERS`;
3. enforces `MAX_WORKERS` and dynamic resource shrinkage;
4. waits for the first input before utilization-based scaling;
5. scales up above `1.75` only when budgets and backpressure permit; or
6. scales down at or below `0.5` only when no actors are pending and the pool is
   above `MIN_WORKERS`.

`INITIAL_WORKERS` is only the startup size. After the first input, low utilization
can shrink the pool toward `MIN_WORKERS`. Use `num_workers=N` for a fixed pool, or
set `MIN_WORKERS` when a warm pool must be retained. A fixed pool is useful only if
the pipeline can keep all N actors fed.

Scale-up happens in bounded increments, but the scheduling loop can iterate sooner
than its task-completion wait timeout. Do not infer an actor-per-second rate from that
timeout; measure scaling transitions in the log.

### Concurrency and tasks in flight

`max_concurrency` is a per-stage Ray actor setting. The actor's in-flight task limit
defaults to `2 * max_concurrency`, unless the global
`DataContext.max_tasks_in_flight_per_actor` overrides it.

These settings also bound autoscaler utilization. The in-flight limit divided by
`max_concurrency` must be at least the `1.75` scale-up threshold. For example,
`max_concurrency=2` with an in-flight limit of `2` can reach only `1.0` and cannot
scale up; the default limit of `4` can reach `2.0`.

With `enable_true_multi_threading=False` (the default), concurrency above 1 overlaps
input and output batching for multiple actor task envelopes but serializes the stage
UDF. Concurrency 2 can hide handoff latency, but it can also increase CPU preparation,
queued data, object-store pressure, and GPU memory. Compare concurrency 1 and 2 per
stage; do not make 2 a blanket default.

### Metadata fetching

Ray fetches block metadata on a background thread by default, controlled by
`RAY_DATA_METADATA_PREFETCH_ON_THREAD`. This keeps metadata `ray.get()` calls off the
scheduler thread but does not prefetch block contents, run the UDF concurrently, or
remove batching work. It is not a first-line tuning lever, and Curator diagnostics do
not time it. To compare the inline path, set the variable to `0` before Ray Data is
imported, restart the driver, and use a controlled run only when other evidence points
to metadata retrieval.

### Operator fusion

Ray can fuse TaskPool → TaskPool or TaskPool → ActorPool map operators when their
remote arguments are compatible. ActorPool → ActorPool does not fuse. Fusion reduces
the number of scheduled operators and changes how resource reservations are divided.
Curator's `RAY_NUM_CPUS` override can make Ray-only resource arguments compatible,
but changing it also changes resource accounting; do not use it only to force fusion.

## Curator tuning controls

Prefer Curator's stage and client APIs. Use `DataContext` and environment variables
only when the Curator API has no equivalent, and set them before pipeline execution.

### Per-stage controls

| Goal | Curator API | Effect |
|---|---|---|
| Change Task rows per stage call | `stage.with_(batch_size=N)` | Batch size counts Tasks, not records inside a Task |
| Fixed worker pool | `stage.with_(num_workers=N)` | Fixed ActorPool or capped TaskPool, depending on stage type |
| Autoscaling actor bounds | `stage.with_(ray_stage_spec={RayStageSpecKeys.MIN_WORKERS: N, RayStageSpecKeys.MAX_WORKERS: M})` | Maps to Ray `min_size` and `max_size` |
| Actor startup size | `stage.with_(ray_stage_spec={RayStageSpecKeys.INITIAL_WORKERS: N})` | Startup target only; does not retain actors above `MIN_WORKERS` |
| Force actor or task | `stage.with_(ray_stage_spec={RayStageSpecKeys.IS_ACTOR_STAGE: bool})` | Overrides adapter selection |
| Actor task-envelope concurrency | `stage.with_(ray_stage_spec={RayStageSpecKeys.RAY_REMOTE_ARGS: {"max_concurrency": N}})` | Overlaps actor task envelopes; the UDF remains serialized by default |
| CPU/GPU request | `stage.with_(resources=Resources(cpus=N, gpus=M))` | Per-worker resources used in scheduling and budgets |
| Ray-only CPU override | `stage.with_(ray_stage_spec={RayStageSpecKeys.RAY_NUM_CPUS: N})` | Changes Ray Data resource accounting without changing other backends |
| Recycle task workers | `stage.with_(ray_stage_spec={RayStageSpecKeys.MAX_CALLS_PER_WORKER: N})` | Passed as `max_calls` for task stages |

### Client and global controls

| Goal | API or environment variable | Default / scope |
|---|---|---|
| Object-store capacity | `RayClient(object_store_memory=N)` or `SlurmRayClient(...)` | Ray initialization setting, in bytes |
| Ray log directory | `RayClient(ray_temp_dir=PATH)` or `SlurmRayClient(...)` | `~/.ray` |
| Enable Curator diagnostics | `NEMO_CURATOR_RAY_DATA_DIAGNOSTICS=1` | Off; set before driver startup |
| Actor scale-up threshold | `DataContext.get_current().autoscaling_config.actor_pool_util_upscaling_threshold = R` | `1.75` |
| Actor scale-down threshold | `DataContext.get_current().autoscaling_config.actor_pool_util_downscaling_threshold = R` | `0.5` |
| Actors added per decision | `DataContext.get_current().autoscaling_config.actor_pool_max_upscaling_delta = N` | `1` |
| In-flight tasks per actor | `DataContext.get_current().max_tasks_in_flight_per_actor = N` | Global override; otherwise `2 * max_concurrency` |
| Reserved resource fraction | `DataContext.get_current().op_resource_reservation_ratio = R` | `0.5` |
| Downstream queue/capacity ratio | `DataContext.get_current().downstream_capacity_backpressure_ratio = R` | `2.0` |
| Downstream object-store gate | `RAY_DATA_DOWNSTREAM_CAPACITY_OBJECT_STORE_BUDGET_UTIL_THRESHOLD` | `0.5` |
| Metadata fetch path | `RAY_DATA_METADATA_PREFETCH_ON_THREAD` | Background thread enabled |

## Guardrails

- Do not infer a scheduler cause from wall time alone.
- Do not treat `num_workers`, concurrency 2, or a backpressure threshold as a
  universal GPU-utilization fix.
- Do not use Ray block-size knobs to split an opaque Curator Task.
- Keep `max_concurrency`, the in-flight limit, and the scale-up threshold compatible.
- Remember that a state-change log is not continuous utilization telemetry.
- Re-check Ray source and `diagnostics.py` when changing the supported Ray version.
