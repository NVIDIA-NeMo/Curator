# AGENTS.md — Ray Data Scheduler Reference

This document describes Ray Data's scheduling and resource allocation behaviors
as they apply to NeMo Curator pipelines. It is reference material for agents
diagnosing performance or underutilization issues — not a prescriptive tuning
guide, since the right decisions depend on pipeline shape and infrastructure.

**Ray version:** Behaviors, defaults, env var names, and class names are verified
against **Ray 2.56.1**, which is the version pinned by Curator. The NeMo Curator
diagnostics shim (`diagnostics.py`) is version-locked to 2.56.1 and will not
install on other versions. Defaults and internal APIs may differ in later Ray releases.

---

## Resource allocation model

Ray Data splits cluster resources across eligible operators (ActorPool and TaskPool) using a
two-tier reservation system, controlled by `op_resource_reservation_ratio`
(default `0.5`, env `RAY_DATA_OP_RESERVATION_RATIO`).

For each scheduling cycle, `ReservationOpResourceAllocator.update_budgets()`
computes:

1. **Per-op reserved allocation:**
   `default_reserved = total_resources * reservation_ratio / num_eligible_ops`
   This is further split into two sub-allocations:
   - `reserved_for_tasks`: receives all CPU, all GPU, all heap memory, and approximately
     half the object store memory from `default_reserved`
   - `reserved_for_op_outputs`: at least half of `default_reserved.object_store_memory`,
     held exclusively for output blocks already pulled from the operator. Without
     this, task pending outputs can consume all memory and stall block delivery.

2. **Shared pool:** `total_resources - sum(all reserved allocations)`. The pool
   is allocated equally across eligible operators each scheduling cycle, after
   deducting each operator's over-reserved usage from the shared pool first.
   Iteration order is reversed (downstream first) only to allow downstream ops
   to borrow from upstream's share when the downstream budget would fall below
   its minimum scheduling requirement.

3. **Budget per operator:** `remaining_reserved + share_of_shared_pool`. An
   operator that has consumed beyond its reserved floor reduces the remaining
   shared pool for other operators on that cycle.


---

## Autoscaler — actor pool scaling (Actor stages only)

**This section applies to Actor stages only** — stages that override `setup()` or
request both GPU and CPU resources. In Curator these are model inference stages (vLLM, NeMo).
Task stages (stateless, CPU-only readers/filters/writers) use `TaskPoolStrategy`.
If `num_workers()` is set, concurrency is capped at that value; otherwise the pool
is uncapped (bounded only by backpressure policies). They have no ramp-up cost,
no actor pool autoscaling complexity, and
General Rules 1–4 below do not apply to them. Note: Task stages still
participate in the resource budget system (they consume CPU from the shared
pool) but the autoscaler never fires for them.

The `DefaultActorAutoscaler` runs on every scheduling tick for each ActorPool
operator.

**Utilization:**
```
util = num_tasks_in_flight / (max_actor_concurrency * current_actors)
```
`max_actor_concurrency` comes from the `max_concurrency` parameter on the actor
class (default `1`). `max_tasks_in_flight_per_actor` defaults to
`2 × max_concurrency`, so actors can queue 2 tasks while running 1 — meaning
util routinely exceeds 1.0 without every actor being fully saturated.

**Scale-up decision (in order):**
1. If `op.has_completed()` OR (`op._inputs_complete` AND queue is empty) → scale
   down by 1 per tick with `force=True` (bypasses debounce period; removes one idle
   actor per tick via `_remove_inactive_actor()`; no-ops if all actors are busy).
2. If `current_actors < min_size` → scale up to `min_size` in one step.
3. If `current_actors > max_size` → scale down by `current_size - max_size` in one step (all excess removed at once).
4. If net budget is negative for CPU/GPU/memory (`allocation - op_usage < 0`) →
   scale down. Since budgets are clamped to ≥ 0 in normal operation, this
   primarily handles dynamic cluster shrinkage rather than steady-state scheduling.
5. If `op.metrics.num_inputs_received == 0` → no-op (`"no inputs received"`).
   Prevents scale-up before the operator has received any input at all,
   regardless of util. Distinct from step 1 (which fires when all inputs are
   done); this fires when none have arrived yet.
6. If `util >= upscaling_threshold` (default `1.75`):
   - If already at `max_size` → no-op.
   - If any backpressure policy has blocked the operator (not `under_resource_limits`) → no-op.
   - Compute `budget_max_scale_up` from remaining CPU/GPU budget.
   - `delta = min(budget_max_scale_up, actor_pool_max_upscaling_delta, max_size - current)`
   - If `delta == 0` → no-op (`"exceeded resource limits"`).
   - Otherwise scale up by `delta`.
7. If `util <= downscaling_threshold` (default `0.5`) → scale down by 1.

**`actor_pool_max_upscaling_delta` defaults to 1**, so by default the autoscaler
adds at most one actor per tick regardless of how high util is.

---

## Backpressure policies

Three policies are active by default, evaluated in order. All three must return
`can_add_input=True` for a task to be admitted:

1. **`ConcurrencyCapBackpressurePolicy`** — enforces a per-operator concurrency
   cap (`num_tasks_running < cap`). For ActorPool operators, the cap defaults to
   `inf`, making this policy a no-op unless `enable_dynamic_output_queue_size_backpressure=True`
   (deprecated, off by default, emits a warning when enabled). For
   `TaskPoolMapOperator` with an explicit `max_concurrency` limit, the cap is
   enforced. Reports as `"ConcurrencyCap"` in `scheduling_reason`.

2. **`ResourceBudgetBackpressurePolicy`** — calls `can_submit_new_task()` on the
   resource allocator. See Task admission below. Reports as `"ResourceBudget"`.

3. **`DownstreamCapacityBackpressurePolicy`** — limits upstream queuing relative
   to downstream capacity. See Downstream capacity backpressure below. Reports
   as `"DownstreamCapacity"`.

The active policy list can be overridden via
`DataContext.set_config("backpressure_policies.enabled", [...])`.

**Deadlock prevention (liveness guarantee):** If there are no runnable operators
(all are backpressured, have no pending inputs, or have no actor slot) AND no
tasks are currently executing (`num_active_tasks() == 0` across all operators),
Ray Data bypasses backpressure and dispatches to any operator that has pending
inputs and an actor slot, to prevent deadlock.

---

## Task admission

`ResourceBudgetBackpressurePolicy.can_add_input()` calls
`ReservationOpResourceAllocator.can_submit_new_task()`, which requires:

1. `op.incremental_resource_usage().satisfies_limit(budget)` — the incremental
   CPUs/GPUs/memory for the next task fit within the operator's current budget.
2. `budget.object_store_memory >= op.metrics.obj_store_mem_max_pending_output_per_task`
   — the remaining object store budget can cover the worst-case output of the
   next task.

Both must pass. Object store memory is attributed to the **producing** operator;
buffered blocks sitting between stages count against the upstream operator's
budget, not the downstream consumer's.

The `reason` field in diagnostic logs identifies which condition failed:

| `reason` | What it means |
|---|---|
| `allowed` | Task admitted |
| `incremental_cpu_exceeds_budget` | CPU budget for this operator exhausted |
| `incremental_gpu_exceeds_budget` | GPU budget exhausted |
| `incremental_heap_memory_exceeds_budget` | Heap memory budget exhausted |
| `incremental_object_store_memory_exceeds_budget` | Object store memory budget exhausted |
| `pending_output_exceeds_object_store_budget` | Worst-case output estimate exceeds remaining object store budget |

---

## Memory model

Ray Data tracks two separate memory spaces in its resource budget system.

### Object store memory

Ray's **object store** (backed by the plasma shared-memory store) holds the
blocks that flow between pipeline stages as Ray object references. When a stage
finishes processing a batch, the output block sits in the object store until the
downstream stage consumes it. Object store memory is finite and shared across
all Ray workloads on the node.

In Ray Data's budget system, object store bytes are attributed to the
**producing** operator, not the consumer:
- `_mem_op_internal` — bytes of pending task outputs: blocks currently being
  generated by running tasks, still in streaming generator buffers before being
  yielded to the output queue
- `_mem_op_outputs` — bytes of completed output blocks: includes blocks in the
  operator's output queue AND blocks already in downstream's input queue (both
  still attributed to the producing operator)

This attribution is why the upstream CPU reader can exhaust the object store
budget even when the GPU stage is the one that hasn't caught up: the reader's
unread output blocks count against the reader's object store budget, which then
prevents the reader from admitting new tasks (`pending_output_exceeds_object_store_budget`).

**Curator relevance:** GPU inference stages (vLLM, NeMo models) produce large
output blocks — embeddings, captions, transcriptions. Each block can be tens to
hundreds of MB. With many CPU reader actors producing faster than the GPU stage
can consume, the object store fills rapidly. Watch `remaining_budget_object_store_memory`
and `pending_output_estimate` in `ray_data_resource_budget_admission` events.

**Primary levers:**
- `RayClient(object_store_memory=N)` — increase total object store capacity
- `DataContext.get_current().target_max_block_size` — reduce block size to slow
  object store fill rate (default 128 MB)

### Heap memory

Heap memory is each actor or task's process-level RAM — Python objects, loaded
model weights, intermediate tensors. Ray Data tracks heap budget separately from
object store via the `memory` field in `ExecutionResources`.

In practice, **Curator stages rarely hit heap budget limits** because Curator's
`Resources` class does not expose a `memory` field today — stages request
`cpus` and `gpus` only. Heap budget (`incremental_heap_memory_exceeds_budget`)
would only fire if a stage passed `memory=` through `ray_stage_spec[RAY_REMOTE_ARGS]`,
which is uncommon.

The fields `remaining_budget_heap_memory` and `requested_heap_memory` in the
diagnostic logs will typically show `null` or `0` for standard Curator stages.

---

## Downstream capacity backpressure

`DownstreamCapacityBackpressurePolicy` limits upstream output queuing based on
how much the downstream operator can absorb.

**It only activates when both conditions are true:**
1. Object store budget utilization for the operator exceeds
   `OBJECT_STORE_BUDGET_UTIL_THRESHOLD` (default `0.5`,
   env `RAY_DATA_DOWNSTREAM_CAPACITY_OBJECT_STORE_BUDGET_UTIL_THRESHOLD`).
   Below this threshold, downstream capacity backpressure is skipped entirely.
2. `queue_ratio = queue_bytes / downstream_capacity_bytes > backpressure_capacity_ratio`
   (default `10.0`, env `RAY_DATA_DOWNSTREAM_CAPACITY_BACKPRESSURE_RATIO`,
   DataContext field `downstream_capacity_backpressure_ratio`).

`downstream_capacity_bytes` is the downstream operator's `obj_store_mem_pending_task_inputs`
metric — total bytes of input blocks currently held across all of that operator's pending
tasks. If that value is zero (no pending tasks), the ratio is zero and no backpressure
is applied.

When downstream capacity backpressure fires, `max_task_output_bytes_to_read`
returns `0`, which stops the upstream operator from pulling new output blocks in
addition to blocking new task admission.

---

## Log location and diagnostic events

NeMo Curator can install a Ray Data diagnostics shim that emits structured logfmt
events. The shim is **opt-in**: set `NEMO_CURATOR_RAY_DATA_DIAGNOSTICS=1` (or `true`,
`yes`, `on`) before starting the pipeline. Without this env var the shim is silently
disabled and no events are written. See `nemo_curator/backends/ray_data/diagnostics.py`.

When enabled, events are routed through the `ray.data` logger to:

```
$RAY_TEMP_DIR/session_latest/logs/ray-data/ray-data.log
```
where `RAY_TEMP_DIR` is set via `RayClient(ray_temp_dir=...)` / `SlurmRayClient(ray_temp_dir=...)` (default `~/.ray`).
The actual path can be resolved programmatically with `ray.data._internal.logging.get_log_directory()`.

Three event types are emitted on every scheduling **state change** (not every
tick):

### `ray_data_actor_autoscaling_decision`

Emitted when the autoscaler produces a different decision than the previous tick.

| Field | Description |
|---|---|
| `operator` | Stage name |
| `decision` | `scale_up`, `scale_down`, or `no_op` |
| `delta` | Signed actor count change requested (positive=up, negative=down, 0=no-op) |
| `scaling_reason` | Text reason from the autoscaler decision path |
| `scheduling_reason` | `runnable`, `ResourceBudget`, `DownstreamCapacity`, `ConcurrencyCap` (backpressure policy names); or `no_pending_inputs`, `no_actor_slot` (actor pool has no free slot — not a policy), `operator_cannot_accept_input` (non-ActorPool), `completed` |
| `utilization` | `tasks_in_flight / (max_concurrency * current_actors)` |
| `current_actors` | Current pool size (running + pending) |
| `min_actors` / `max_actors` | Pool bounds |
| `running_actors` | Actors that have started running, including those in restart state (`num_running_actors`). Note: `num_alive_actors = running - restarting` but `restarting_actors` is not a logged field, so alive count cannot be derived from the logged fields alone. |
| `pending_actors` | Actors being created, not yet running |
| `active_actors` | Actors currently executing a task |
| `idle_actors` | Running actors not currently executing a task |
| `tasks_in_flight` | Tasks currently submitted to the pool |
| `queued_input_blocks` | Input block count waiting in front of this operator |
| `queued_input_bytes` | Input bytes waiting in front of this operator |
| `remaining_budget_cpu` | CPUs remaining in this operator's current budget |
| `remaining_budget_gpu` | GPUs remaining in this operator's current budget |
| `remaining_budget_heap_memory` | Heap memory remaining in budget |
| `remaining_budget_object_store_memory` | Object store memory remaining in budget |
| `allocation_cpu` | Total CPUs allocated (reserved + shared claimed) |
| `usage_cpu` | CPUs currently in use by this operator |

Each resource group (`allocation`, `usage`, `remaining_budget`) emits four fields:
`{prefix}_cpu`, `{prefix}_gpu`, `{prefix}_heap_memory`, `{prefix}_object_store_memory`.
Only `_cpu` variants are shown above; GPU pipelines should monitor `_gpu` and
`_object_store_memory` variants instead.

### `ray_data_resource_budget_admission`

Emitted when an operator's task admission state changes (allowed ↔ blocked).

| Field | Description |
|---|---|
| `operator` | Stage name |
| `state` | `allowed` or `blocked` |
| `reason` | Denial reason (see table above) |
| `requested_cpu` | CPUs the next task would consume |
| `requested_gpu` | GPUs the next task would consume |
| `requested_heap_memory` | Heap memory the next task would consume |
| `requested_object_store_memory` | Object store memory the next task would consume |
| `remaining_budget_cpu` | CPUs remaining in budget at decision time |
| `remaining_budget_gpu` | GPUs remaining in budget |
| `remaining_budget_heap_memory` | Heap memory remaining in budget |
| `remaining_budget_object_store_memory` | Object store memory remaining in budget |
| `pending_output_estimate` | `obj_store_mem_max_pending_output_per_task` used for condition 2 |
| `usage_cpu` | CPUs currently in use by this operator |
| `usage_gpu` | GPUs currently in use by this operator |
| `allocation_cpu` | Total CPU allocation (budget + usage) |
| `allocation_gpu` | Total GPU allocation (budget + usage) |

### `ray_data_downstream_capacity_admission`

Emitted when an operator's upstream admission state changes due to queue
capacity.

| Field | Description |
|---|---|
| `operator` | The upstream operator being throttled |
| `state` | `blocked` or `allowed` |
| `queue_bytes` | Current output queue size (including ineligible downstream) |
| `downstream_capacity_bytes` | Downstream operator's pending task input bytes |
| `queue_ratio` | `queue_bytes / downstream_capacity_bytes` |
| `configured_ratio` | The `backpressure_capacity_ratio` threshold |
| `utilized_object_store_budget_fraction` | Object store utilization fraction (must exceed 0.5 for this policy to fire) |

---

## General rules (Actor stages — `ActorPoolStrategy`)

These structural invariants apply to **Actor stages** (`ActorPoolStrategy`) and
are derived directly from the scheduler source. Task stages (`TaskPoolStrategy`)
are not subject to actor pool autoscaling and these rules do not apply to them.

### 1. `max_size` is a hard ceiling on shared pool consumption

Setting `max_size=N` on an operator guarantees it can never hold more than N
actors. In the autoscaler: `delta = min(budget_max_scale_up,
actor_pool_max_upscaling_delta, max_size - current_actors)`. When
`current_actors == max_size`, the third term is 0, making delta=0 regardless of
utilization or budget. This is the only mechanism that hard-prevents an operator
from stealing shared pool CPUs.

### 2. `min_size` actors are created before any inputs arrive — `initial_size` alone does not hold them

The min_size check (step 2 in the autoscaler decision chain) fires before the
`num_inputs_received == 0` guard (step 5). Actors are created up to `min_size`
immediately at pipeline startup, not when the first item arrives.

**`initial_size` without a matching `min_size` is a footgun for GPU stages.**
The `num_inputs_received == 0` guard only prevents downscaling *before* any
inputs are received. As soon as the first block arrives and is assigned as a
task, `num_inputs_received` becomes 1 and the guard never fires again. If
all `initial_size` actors are now running but only 1 task is in flight,
`util = 1/initial_size`. With `initial_size=8` and one task: util=0.125 ≤ 0.5
→ step 7 downscales immediately. The pool drains 8→7→6→...→1 and then must
slowly ramp back up one actor per tick.

To actually hold a fixed pool: use `min_size=max_size=N` (equivalently `size=N`),
not `initial_size=N` alone.

**Curator example — single GPU stage on an 8-GPU cluster:**
```python
# Correct: locks pool at 8, autoscaler fully disabled
stage.with_(num_workers=8)

# Wrong: starts at 8 but downscales to 1 on first input
stage.with_(ray_stage_spec={RayStageSpecKeys.INITIAL_WORKERS: 8})
```
The GPU stage should always use `num_workers=N` where N equals the number of
GPUs allocated to it, so every GPU is immediately and permanently reserved.

### 3. `min_size == max_size` eliminates the autoscaler for that stage entirely

With equal bounds, the pool cannot scale up (`max_size - current = 0` →
delta=0) or scale down (`current_size <= min_size` → no-op). During active processing the pool stays exactly at that size. At end-of-pipeline,
step 1 (inputs consumed) uses `force=True` which bypasses the min_size floor
and will drain actors one per tick. Useful for GPU stages where the right actor
count is known in advance.

### 4. Scale-up rate is bounded by `WAIT_FOR_TASK_COMPLETION_TIMEOUT_S = 0.1s`

The scheduling loop blocks on `ray.wait()` for up to 100ms per iteration.
With `actor_pool_max_upscaling_delta=1` (default), at most one actor is added
per iteration. Maximum scale-up rate is therefore 1 actor per 100ms. Ramping
from 1 to 96 actors takes at minimum ~9.5s of scheduling time (in practice
~20–40s with overhead). This is why a fast upstream stage can reach its budget
cap well before a downstream stage receives its first item.

### 5. Stage fusion requires a TaskPool upstream — ActorPool→ActorPool never fuses

From `FuseOperators._can_fuse()`, the upstream operator must be a
`TaskPoolMapOperator`. `ActorPoolMapOperator → ActorPoolMapOperator` is not a
supported fusion pattern. Fusion is possible for:
- `TaskPool → TaskPool` (when remote args match)
- `TaskPool → ActorPool` (upstream task feeds into an actor stage)

When fusion occurs, the two operators collapse into one, reducing
`num_eligible_ops` and increasing `reserved_per_op` for all remaining operators,
which reduces shared pool contention and makes it less likely that a fast upstream
stage steals the shared budget before the bottleneck stage can claim it.

In Curator, the adapter (`RayDataStageAdapter`) decides Task vs Actor automatically.
A stage becomes an Actor if it overrides `setup()` — meaning it has persistent state
such as model weights loaded at init — or if it requests both GPU and CPU resources.
Stateless stages (no `setup()` override, CPU-only) become Tasks. This can be forced
via `stage.with_(ray_stage_spec={RayStageSpecKeys.IS_ACTOR_STAGE: True/False})`.
Task stages (TaskPool) can participate in fusion with an upstream Task; Actor stages
cannot be fused as an upstream.

### 6. `num_cpus=0` operators consume zero CPU budget but still count in `num_eligible_ops`

`min_max_resource_requirements()` returns `max_resource_usage.cpu=0` when
`num_cpus=0`. Both `_update_reservation` and `update_budgets` cap this
operator's reserved and shared CPU allocation to 0. It never consumes CPU
budget.

However, it IS included in `num_eligible_ops`, which shrinks
`default_reserved = total * ratio / num_eligible_ops` for all operators. Because
the 0-cpu op takes nothing from either the reserved or shared pool, the extra
CPU stays available — but the redistribution depends on operator position in the
topology due to the reversed iteration order in `update_budgets`. The net effect
is not simply "invisible": it dilutes the per-op `default_reserved` calculation
even though it returns the CPU to the shared pool.

---

## Tuning knobs

The primary Curator API surface is listed first. Knobs marked *(advanced)* have no
Curator wrapper and require setting `DataContext` directly before pipeline execution.

### Worker sizing (Curator API)

| Knob | Curator API | What it controls |
|---|---|---|
| Fixed actor pool | `stage.with_(num_workers=N)` | Sets `min_size=max_size=N` — autoscaler fully disabled for this stage. Correct pattern for GPU stages. |
| Autoscaling bounds | `stage.with_(ray_stage_spec={RayStageSpecKeys.MIN_WORKERS: N, RayStageSpecKeys.MAX_WORKERS: M})` | `min_size=N`, `max_size=M`. `MAX_WORKERS` is a hard ceiling that prevents shared pool theft. |
| Force actor or task | `stage.with_(ray_stage_spec={RayStageSpecKeys.IS_ACTOR_STAGE: True/False})` | Override the adapter's automatic Task/Actor decision. |

### Resource requirements (Curator API)

| Knob | Curator API | What it controls |
|---|---|---|
| CPU / GPU per actor | `Resources(cpus=N, gpus=N)` on the stage class | Sets `num_cpus` / `num_gpus` in Ray Data — affects budget accounting and utilization denominator |
| CPU override for Ray Data only | `stage.with_(ray_stage_spec={RayStageSpecKeys.RAY_NUM_CPUS: 1.0})` | Overrides `num_cpus` for Ray Data without changing `resources.cpus` used by other executors (e.g., set to `1.0` to enable stage fusion) |
| Object store memory | `RayClient(object_store_memory=N)` / `SlurmRayClient(object_store_memory=N)` | Total object store memory cluster-wide (passed to `ray.init()` internally); affects pending output budgets |
| Block size | *(advanced)* `DataContext.get_current().target_max_block_size` | Max block size in bytes (default 128 MB). Smaller blocks reduce peak object store usage between stages but increase scheduling overhead. Critical for large per-item payloads (video frames, embeddings). |

### Autoscaler thresholds *(advanced — DataContext only)*

| Knob | How to set | What it controls |
|---|---|---|
| Scale-up threshold | `DataContext.get_current().autoscaling_config.actor_pool_util_upscaling_threshold` or `RAY_DATA_DEFAULT_ACTOR_POOL_UTIL_UPSCALING_THRESHOLD` | Utilization threshold to trigger scale-up (default `1.75`) |
| Scale-down threshold | `DataContext.get_current().autoscaling_config.actor_pool_util_downscaling_threshold` or `RAY_DATA_DEFAULT_ACTOR_POOL_UTIL_DOWNSCALING_THRESHOLD` | Utilization threshold to trigger scale-down (default `0.5`) |
| Max actors per tick | `DataContext.get_current().autoscaling_config.actor_pool_max_upscaling_delta` or `RAY_DATA_DEFAULT_ACTOR_POOL_MAX_UPSCALING_DELTA` | Max actors added per scheduling tick (default `1`) |
| Reserved fraction | `DataContext.get_current().op_resource_reservation_ratio` or `RAY_DATA_OP_RESERVATION_RATIO` | Fraction of total resources (CPU, GPU, object store, heap) reserved per operator vs. shared pool (default `0.5`) |
| Downstream capacity ratio | `DataContext.get_current().downstream_capacity_backpressure_ratio` or `RAY_DATA_DOWNSTREAM_CAPACITY_BACKPRESSURE_RATIO` | Queue/capacity ratio threshold before upstream is throttled (default `10.0`) |
| Object store utilization threshold | `RAY_DATA_DOWNSTREAM_CAPACITY_OBJECT_STORE_BUDGET_UTIL_THRESHOLD` | Object store utilization fraction above which downstream capacity backpressure becomes active (default `0.5`). Lower this for pipelines with large blocks to apply backpressure sooner. |
