# PR1967 Local/Windowed Branch: Current-Code Walkthrough

This document describes the current files on disk for
`aaftabv/qwen1967-local-noio-control`. It is a reviewer guide to the code as it
exists now: public contracts, execution flow, task shape, scheduling, memory
ownership, failure behavior, observability, and tests. It intentionally does
not narrate discarded designs or earlier benchmark revisions.

This file is versioned with the implementation it describes; reviewers should
use the PR commit containing the file as the code snapshot instead of relying
on a copied HEAD value that becomes stale after documentation commits. Runtime
benchmark results and development history are deliberately outside this guide.

The branch provides five connected capabilities:

1. a generic pipeline rule that inserts payload materialization and release
   stages around independently scheduled consumers;
2. a Ray-backed payload reference lifecycle that decodes large audio payloads
   once and keeps waveform tensors out of ordinary task rows;
3. model-input segmentation and duration-aware bucketing inside the
   backend-visible ASR stage's finite input window;
4. a pluggable ASR stage and Qwen-Omni adapter with model-call controls and
   detailed performance metrics; and
5. a backend-neutral atomic dispatch-envelope contract that a future
   full-dataset planner can use without changing Ray Data or Xenna. The local
   reader does not emit these envelopes, so the local runtime path remains
   ordinary `AudioTask` rows.

The logical Qwen graph is:

```text
ManifestReader -> ASRStage -> ManifestWriterStage
```

`Pipeline.build()` expands it to:

```text
ManifestReader
  -> AudioPayloadMaterializeStage
  -> ASRStage
  -> PayloadReleaseStage
  -> ManifestWriterStage
```

All five execution stages remain separately visible to Ray Data or Xenna. The
payload lifecycle does not combine the reader and GPU stage, and it does not
change the GPU stage's worker or resource contract.

The local and global branches share every executable implementation file except
`nemo_curator/stages/audio/common.py`. The local file contains ordinary
manifest reading and writing. The global file additionally contains
full-manifest segment planning, parent-row storage, and parent assembly.
Payload handling, ASR, Qwen, backends, task contracts, and performance code are
otherwise byte-identical in the current worktrees.

## 1. Terms Used By The Code

- **Logical stage**: a stage listed by the user in `Pipeline.stages` or Hydra
  `stages:`.
- **Execution stage**: a concrete `ProcessingStage` after graph expansion and
  composite-stage decomposition.
- **Backend-visible stage**: an execution stage independently presented to Ray
  Data or Xenna.
- **Payload**: a large object managed outside ordinary task serialization. The
  audio materializer stores a decoded waveform tensor.
- **Payload binding**: the mapping from a source field such as
  `audio_filepath` to ref, waveform, sample-rate, sample-count, duration, and
  materializer fields.
- **PayloadRef**: the lightweight handle in `task.data` that identifies the
  object store, admission state, producer node, byte count, sample metadata,
  and Ray namespace.
- **Payload consumer**: a backend-visible stage that declares one payload binding
  and resolves refs only while processing a batch.
- **Parent row**: one complete manifest record delivered to the local ASR
  stage.
- **Model-input segment**: one contiguous waveform slice no longer than
  `max_inference_duration_s`.
- **Local/windowed bucketing**: duration grouping over model-input segments
  available in the current backend-provided ASR window.
- **Dispatch batch**: a generic `DispatchBatchTask` row containing the exact
  child tasks intended for one owner adapter call. `ASRStage` can validate and
  consume it, and `DispatchBatchUnpackStage` can fan children back out. The
  local reader never creates this row type.
- **Terminal row**: a row with `_curator_terminal_*` ownership fields that a
  downstream terminal consumer must receive exactly once. The local Qwen graph
  does not create terminal segment rows, but the generic task contract is
  available to other stages.

ASR-internal symbols retain `chunk` in names such as `_ChunkSpec`,
`_build_chunk_specs()`, and `_curator_asr_chunk_*`. Those symbols implement
ASR-local stitch-back. The public planning term is **segment**, the public
model boundary is `max_inference_duration_s`, and the shared planner type is
`AudioSegment`.

## 2. Pipeline Construction And Graph Expansion

### 2.1 Pipeline state

[`nemo_curator/pipeline/pipeline.py`](nemo_curator/pipeline/pipeline.py) keeps:

- `_logical_stages`, the canonical user graph;
- `stages`, the public list and built execution graph.

`Pipeline.__init__()` preserves the caller-visible config mapping, matching
main-branch behavior, and separately creates a private
`_curator_pipeline_run_id`. The id is copied only into the ephemeral graph
expansion config. Payload actor names include it, preventing independent
pipeline objects from attaching to one another without mutating the caller's
configuration.

`Pipeline.add_stage()` updates the logical graph and invalidates the built plan.
`_sync_public_stage_mutations()` also accepts direct mutations to
`pipeline.stages`. `_clear_default_source_sink_roles()` removes roles assigned
automatically by a prior build; explicit roles remain subject to the one-source
and one-sink validation in `_assign_source_sink_roles()`.

`Pipeline.build()`:

1. synchronizes public stage changes;
2. applies pipeline-level graph expansion;
3. decomposes composite stages;
4. assigns source and sink roles;
5. stores the execution graph for idempotent repeated builds.

### 2.2 Payload lifecycle rule

[`nemo_curator/pipeline/payload_lifecycle.py`](nemo_curator/pipeline/payload_lifecycle.py)
implements the backend-neutral `expand_payload_lifecycle_stages()` rule.

The rule validates:

- one `materialize_after` stage and one `release_after` stage;
- all consumers lie within that lifecycle range;
- no materialize/release helper is explicitly listed in the logical graph;
- all consumers are payload-aware;
- consumer ref/waveform bindings match the materialized binding.

Selectors match Hydra stage id, stage name, class name, or fully qualified class
name.

Each row has one payload source configured through singular `source_key`,
`ref_key`, and `waveform_key` fields. Multiple downstream consumers may share
that one ref.

The central rule calls `ManifestReader.build_payload_materialize_stage()` to
construct the audio materializer. The post-release extension path is inactive
because this reader does not enable global planning, so no parent assembler is
inserted. The same expander can insert a generic dispatch unpack stage when a
source declares atomic dispatch output; the local reader makes no such
declaration, so no dispatch stage is added here.

### 2.3 Expanded graph config

```yaml
payload_lifecycle:
  enabled: true
  materialize_after: reader
  source_key: audio_filepath
  ref_key: waveform_ref
  consumers: [qwen_omni]
  release_after: qwen_omni
  target_sample_rate: 16000
  target_nchannels: 1
  node_memory_fraction: 0.80
```

For several consumers, list every payload-aware stage in `consumers` and put
`release_after` on the final one. Each consumer retains its own `Resources`,
`batch_size`, worker count, and backend actor.

## 3. Payload Reference Lifecycle

The handle API is in
[`nemo_curator/pipeline/payload_refs.py`](nemo_curator/pipeline/payload_refs.py).
Audio materialization and consumer support are in
[`nemo_curator/stages/payload_lifecycle.py`](nemo_curator/stages/payload_lifecycle.py).

### 3.1 PayloadRef

`PayloadRef` carries:

| Field | Meaning |
| --- | --- |
| `payload_id` | object id in the payload store |
| `owner_node_id` | node that decoded and stored the object |
| `store_actor_name` | node-local object store actor |
| `admission_actor_name` | cluster admission actor |
| `amount_bytes` | actual stored byte count |
| `sample_rate` / `num_samples` | consumer-visible waveform metadata |
| `actor_namespace` | Ray namespace for actor lookup |

`resolve_payload_refs_batched()` groups handles by actor, issues
one `get_many` RPC per store actor, preserves caller order, and resolves all
handles needed by one already-bounded adapter call in one request wave.
`release_payload_ref()` removes the object and releases admission bytes.
`strip_payload_refs()` recursively removes handles from nested containers.

### 3.2 Admission and stores

`_PayloadAdmissionState` tracks per-node budgets, aggregate cluster usage, and
reservations. The default cluster budget is the sum of registered node budgets;
`max_cluster_payload_bytes` can set an explicit limit. A row larger than either
applicable budget fails immediately. Temporary lack of capacity waits until a
payload is released, bounded by `admission_wait_timeout_s` (four hours by
default). A timeout reports the requested bytes and the actor's node/cluster
usage snapshot rather than polling forever.

`_PayloadStoreState` owns actual objects. Store actors are node-affined. Store
and admission actors use the pipeline run id and active Ray namespace, are
detached across backend worker lifetimes, and are killed by executor cleanup.

There is one logical materializer stage, but Ray Data and Xenna may execute it
concurrently on several workers and nodes. Same-node executions share one
named store actor. All node-local stores share one run-scoped admission actor,
which reserves bytes before decode and enforces the optional cluster cap.

Lifetime is fixed internal policy. Decode reservations use a one-hour lease so
a worker lost during decode cannot reserve admission capacity forever. Once
store insertion succeeds, `publish()` converts admission accounting to
non-expiring state and the store entry is likewise non-expiring. Explicit
`PayloadReleaseStage` release is the normal terminal path; executor cleanup
kills all run-scoped actors. Resolution needs no heartbeat, pin, or renewal RPC,
and queue residence cannot expire a valid published payload.

### 3.3 AudioPayloadMaterializeStage

For each row the materializer:

1. reads the configured metadata duration;
2. estimates waveform bytes and acquires admission capacity with the internal
   reservation lease;
3. decodes the local file through `AudioFileReaderStage`;
4. removes the waveform from normal task data;
5. measures actual tensor bytes and resizes the reservation;
6. stores the tensor in the node-local actor;
7. writes `PayloadRef`, estimated bytes, actual bytes, and producer node id;
8. publishes the completed reservation as non-expiring accounting owned until
   explicit release or executor cleanup.

Duration must be positive and numeric. Byte-limit strings accept integer, `k`,
`m`, and `g` forms and reject invalid values. If actual bytes cannot fit, the
stage releases its reservation and raises. If materialization fails after the
store insert, it also removes the stored object. If a worker dies before the
reservation is committed, its finite materialization lease can be reaped.

When `skip_on_read_error` is enabled, a reader error yields a skipped row with
no payload ref. The reservation and zero-length waveform are removed.
`ASRStage` recognizes that marker before waveform validation, preserves the row
in its original position, writes empty ASR outputs, and excludes it from adapter
calls. Final release safely no-ops on the absent ref.

Metrics include admission wait, poll count, estimated/reserved/stored bytes,
node and cluster budgets, and materialization count.

### 3.4 Payload-aware consumers

`PayloadAwareStageMixin.payload_binding()` declares the one waveform-ref
mapping consumed by a stage.

`resolve_payload_refs_for_batch()` resolves all handles required by its bounded
stage batch with actor-grouped bulk RPCs, restores sample metadata, and records
same-node and cross-node resolution metrics. `drop_resolved_payloads()` removes
temporary waveform fields only. Published payloads are already non-expiring,
so resolution and cleanup perform no heartbeat, pin, or renewal RPC.

The Qwen config opts into `BoundedOneAheadPrefetchIterator` from
`pipeline/prefetch.py`. ASR can plan exact model calls from `PayloadRef`
metadata (`num_samples`, `sample_rate`, and `amount_bytes`) before loading the
waveform. `_PayloadCallMaterializer` then:

1. groups and resolves only the unique parent refs required by one adapter
   call;
2. caches a resolved parent while contiguous calls still need its segments;
3. slices each requested model-input segment locally;
4. allows one byte-bounded successor call to resolve while the current call is
   on the GPU; and
5. drops actor-local waveform references as soon as their call is complete.

The payload-store actor continues to own the original waveform until
`PayloadReleaseStage`; prefetch changes only the ASR actor's temporary working
set. This path is opt-in. Existing payload-aware consumers retain the eager,
batched resolver path.

The lightweight ref remains in task data after one consumer returns. Later
configured consumers can resolve the same waveform without another file read.

### 3.5 Release and exception paths

`PayloadReleaseStage` reads the configured top-level ref directly, releases its
object and byte reservation, removes waveform and payload bookkeeping keys,
and returns the task. If the expected key is absent, recursive discovery is a
defensive cleanup fallback for malformed rows. It supports rows without refs
and keeps the existing task-data mapping object intact.

`BaseStageAdapter` reads only the configured top-level ref key on stages marked
by the lifecycle expander. After a successful invocation, it releases refs only
for rows that the stage actually filtered. It deliberately leaves input refs
owned when an invocation raises so Ray Data or Xenna can retry the same logical
rows with the same handles. ASR removes only temporary actor-local waveforms in
`finally`; explicit `PayloadReleaseStage` release handles successful completion,
and executor cleanup owns terminal run failure. Stages in ordinary pipelines pay
no payload-tracking cost, and payload stages do not recursively scan unrelated
structured model outputs on the normal path.

## 4. Local Audio Decode

[`nemo_curator/stages/audio/io/audio_file_reader.py`](nemo_curator/stages/audio/io/audio_file_reader.py)
defines the single raw audio-byte I/O implementation used by the materializer.

The reader:

- accepts local paths and rejects URI-style paths;
- checks for ffmpeg in per-node setup;
- decodes to float32 PCM at configured sample rate and channel count;
- supports `segment_start_s` and `segment_duration_s` when supplied by another
  stage;
- returns channels-first contiguous tensors;
- writes waveform, sample rate, sample count, duration, mono status, and
  `audio_item_id`;
- converts decode errors into skipped rows when configured.

The local manifest reader does not create segment offsets. It emits complete
parent rows, so the materializer decodes each full source row once. ASR slices
that in-memory waveform only when the model input ceiling requires it.

## 5. Local Manifest Reading

`ManifestReader` in
[`nemo_curator/stages/audio/common.py`](nemo_curator/stages/audio/common.py) is a
`CompositeStage` that decomposes into:

```text
FilePartitioningStage -> ManifestReaderStage
```

`FilePartitioningStage` discovers manifest files. `ManifestReaderStage` streams
each JSONL file line by line with fsspec and emits one `AudioTask` per non-empty
line. It copies task metadata/performance and derives child task ids from the
partition task. It is a one-worker fanout stage.

Rows keep the complete original manifest dictionary and input order. No
full-manifest duration plan, parent-data actor, segment terminal fields, or
assembler exists in this branch.

## 6. Model-Input Segmentation

[`nemo_curator/stages/audio/model_input_segmentation.py`](nemo_curator/stages/audio/model_input_segmentation.py)
contains the shared safety primitive.

`resolve_max_model_input_duration()` validates the positive model input
ceiling. `plan_audio_segments()` converts actual sample count, sample rate, and
the ceiling into contiguous `AudioSegment` records. Each record contains index,
count, start sample, stop sample, and duration.

Properties enforced by the implementation:

- an input at the ceiling produces one segment;
- an input just over the ceiling produces one full segment and one tail;
- no overlap or padding is introduced;
- zero samples remain representable as one empty segment;
- invalid sample rates raise;
- metadata duration conversion uses ceiling sample math.

The local branch applies this helper inside `ASRStage` to the decoded waveform.
With bucketing enabled, the bounded segments are bucketed. With bucketing
disabled, the same segmentation remains the model/OOM safety boundary.

## 7. ASR Stage And Windowed Bucketing

[`nemo_curator/stages/audio/inference/asr/stage.py`](nemo_curator/stages/audio/inference/asr/stage.py)
defines `ASRStage`. Model adapters conform to
[`ASRAdapter`](nemo_curator/models/asr/base.py).

### 7.1 Stage contract

The stage accepts waveform or `waveform_ref`, sample rate, optional source
language, and optional reference text. It writes configured primary text,
optional secondary text, and a skip key.

`setup_on_node()` prefetches model weights with one CPU and zero GPUs.
Worker `setup()` constructs the adapter under the stage's GPU resource
allocation. `teardown()` releases adapter state.

### 7.2 Independent batch controls

| Control | Scope |
| --- | --- |
| `ASRStage.batch_size` | parent-row candidate window passed by Ray Data/Xenna |
| `BatchPolicy.max_items_per_batch_by_bucket` | per-duration-bucket adapter-call item cap |
| `adapter_batch_size` | fallback items per adapter call |
| `BatchPolicy.max_audio_sec_per_batch` | aggregate cost cap for one bucketed batch |

When duration-aware bucketing is enabled, each `BatchPolicy` output batch is one
adapter/model call. `max_items_per_batch_by_bucket` limits item count, while
`max_audio_sec_per_batch` limits aggregate estimated cost for that call. When
the policy is disabled, `adapter_batch_size` is the fallback model-call item
cap. These controls do not govern payload RAM. Payload memory is admitted in
bytes by the materializer.

### 7.3 Process flow

`ASRStage.process_batch()`:

1. recognizes rows already marked skipped before requiring a waveform or ref;
2. validates waveform/ref and sample-rate inputs only for non-skipped rows;
3. uses eager bulk payload resolution unless prefetch is explicitly enabled;
4. in prefetch mode, plans segments and exact adapter calls from payload
   metadata before waveform resolution;
5. creates aligned skipped results for pre-skipped rows and excludes them from
   adapter calls;
6. applies duration-aware policy when enabled; each policy output is already
   the final adapter call, bounded by its bucket item and aggregate-cost caps;
7. when policy is disabled, splits eligible items only by the fallback
   `adapter_batch_size`/stage-batch cap;
8. resolves the current call and overlaps one bounded next call with current
   GPU inference;
9. invokes the adapter and realigns results;
10. joins per-parent text in segment order; and
11. drops current-call waveform references; the payload actor retains the
   original until `PayloadReleaseStage`.

The finite candidate set is the current backend-provided `process_batch()`
window. Local bucketing cannot inspect rows outside that window.

The shared ASR implementation also accepts `DispatchBatchTask` envelopes from
an external planner. That path verifies owner identity, policy signature,
bucket membership, item and aggregate costs, and one-segment-per-child before
making at most one adapter call per envelope. Eligible children stay together
without rebucketing; pre-skipped or unsupported children retain aligned skipped
results and bypass the adapter, so an all-skipped envelope makes no call. This
path is intentionally inactive in this branch because local `ManifestReader`
emits ordinary rows only.

The stage emits `adapter_inference_calls` and `adapter_inference_items`, plus
input, processed, skipped, generated-segment, audio-duration, waveform-byte,
output-character, token, and inference-time metrics.

### 7.4 Code-derived 5h example

Using the same real benchmark manifest as the global guide:

```text
/home/aaftabv/grananary-v2/realdata_5h_yt_alm_part2_20260613/manifest_5h_stratified_duration_tails.jsonl
```

the local reader emits 89 complete parent rows in source order. It does not
create segment rows. For the first two records:

| Source index | Parent duration | Materialized object at 16 kHz mono float32 | ASR model-input segments |
| ---: | ---: | ---: | --- |
| 0 | 7513.3335 s | about 480,853,344 bytes | 2400, 2400, 2400, 313.3335 s |
| 1 | 2756.4135 s | about 176,410,464 bytes | 2400, 356.4135 s |

The difference from global is where segmentation happens. Local materializes
the complete parent waveform once, stores one parent `PayloadRef`, and sends
that row into the backend-provided ASR window. `_build_chunk_specs()` calls the
shared segment planner against the resolved parent sample count. In prefetch
mode, ASR plans those descriptors from ref metadata, resolves a parent once,
and reuses the cached tensor for its contiguous segment calls.

For all 89 rows, model safety is the same as global: no adapter input exceeds
2,400 seconds. Packing scope is different. Local duration-aware bucketing can
combine only segments whose parent rows are present in the current
`process_batch()` window. It cannot reorder the complete 89-row manifest before
materialization, and payload admission accounts for complete parent tensors
rather than globally planned segment tensors.

### 7.5 BatchPolicy

[`nemo_curator/stages/audio/inference/batch_policy.py`](nemo_curator/stages/audio/inference/batch_policy.py)
defines a generic cost-bucket policy.

`BatchPolicy` validates strictly increasing edges starting at zero, per-bucket
item caps, total-cost cap, the advisory `prebatching_window_size`, and
`flush_interval_ms`. `BucketQueueScheduler` can flush on item capacity, cost
capacity, timer, or drain. The finite `bucketize_with_costs()` path used by ASR
constructs it with timers disabled, drains at the end of the supplied item
window, orders ready batches by descending total cost, and returns original
indices for result alignment. Therefore `flush_interval_ms` does not affect the
current local Qwen path. `prebatching_window_size` is currently validated and
serialized but does not resize backend batches; the actual candidate window is
`ASRStage.batch_size`.

`run_bucketed()` exposes the same bucket-dispatch-and-realign loop to other
inference stages through caller-supplied cost and execution functions.

## 8. Qwen-Omni Adapter

[`nemo_curator/models/asr/qwen_omni.py`](nemo_curator/models/asr/qwen_omni.py)
implements `QwenOmniASRAdapter` using `VLLMBase` from
[`nemo_curator/models/vllm_model.py`](nemo_curator/models/vllm_model.py).

Install this adapter with `uv sync --extra audio_qwen`. The Qwen-only extra
composes the unchanged `audio_cuda12` and `vllm` extras with
`qwen-omni-utils`, so existing audio installations keep their main-branch
dependency selection.

It supports:

- inline and file-backed default, English, follow-up, and system prompts;
- per-item `{language}` and `{transcript}` interpolation;
- waveform normalization and 16 kHz resampling;
- threaded multimodal request preparation;
- one-turn or two-turn Qwen inference;
- stable one-result-per-input ordering with skipped placeholders;
- tensor parallelism, model/token sequence limits, GPU memory utilization,
  prefix caching, multimodal limits, sampling, seed, and output-token settings;
- preparation, generation, valid/skipped input, and output-token metrics.

The adapter currently treats every preprocessing exception as an item-local
skip. `_prepare_single()` catches any `Exception`, logs it, and returns `None`;
Turn 1 then emits empty skipped output. A systemic processor, prompt-template,
or dependency failure can therefore produce a successful batch in which all
rows are skipped. `_prepare_turn2_single()` likewise catches every exception and
omits that item's refinement. vLLM generation failures instead propagate from
`VLLMBase._generate()` as `RuntimeError`.

`QwenOmniASRAdapter.setup()` passes `trust_remote_code=True` directly to vLLM;
this snapshot does not expose a configuration field to disable it.

`ASRStage` passes `waveform`, `sample_rate`, `language`, `language_code`,
`reference_text`, `task_id`, `audio_seconds`, and stitch-back indices to the
adapter. Adapter-specific conversion and Qwen request construction remain out
of the stage.

## 9. Backend Scheduling And Autoscaling

The backend implementation is modality-neutral and shared with the global
branch.

### 9.1 Base adapter

[`nemo_curator/backends/base.py`](nemo_curator/backends/base.py) wraps one
`stage.process_batch()` invocation with timing, task-id postprocessing, custom
metrics, and the existing validation flow. Payload-ref scans and terminal
tombstones are enabled only on stages marked by payload lifecycle expansion.
Worker identity and invocation-window GPU/VRAM sampling are enabled only when
`extended_performance_metrics` is explicitly set; the raw Qwen configs opt in
through each stage's `stage_with`, while existing pipelines retain the compact
main-compatible record shape.

The backend does not own duration bucketing or payload prefetch. Ray Data and
Xenna deliver their normal finite `process_batch()` windows, and ASR performs
segmentation, bucketing, bulk resolution, and optional one-call lookahead
inside its existing actor. Ray Actor Pool remains outside this Qwen execution
path; payload lifecycle does not add a second scheduling policy to it.

### 9.2 Ray Data

[`nemo_curator/backends/ray_data/adapter.py`](nemo_curator/backends/ray_data/adapter.py)
maps every backend-visible stage with `Dataset.map_batches()`.

- `stage.batch_size` sets the process-batch window.
- Stages with setup/GPU/actor requirements use actors; stateless CPU helpers can
  use tasks.
- `stage.num_workers()` fixes an actor pool when set.
- Otherwise optional `min_workers`, `max_workers`, and `initial_workers` values
  come from `ray_stage_spec()`; without those bounds Ray Data controls
  actor-pool scaling subject to each actor's declared resources.
- Fanout stages repartition output blocks to one row each.

For this graph, materialize and release are CPU task stages, ASR is a GPU actor
stage, and writer is a one-worker actor stage.

### 9.3 Xenna

[`nemo_curator/backends/xenna/executor.py`](nemo_curator/backends/xenna/executor.py)
creates one Xenna `StageSpec` per execution stage. It forwards stage resources,
batch size, runtime environment, retry/lifetime settings, and worker sizing.

Cluster-wide worker count comes from `stage.num_workers()`.
`xenna_stage_spec()["num_workers_per_node"]` is the Xenna-specific alternative;
setting both is rejected. `xenna_stage_spec()["num_workers"]` is rejected with
a message directing stage authors to `num_workers()`. A stage without either
fixed sizing remains under Xenna allocation/autoscaling. Streaming and batch
modes use the same stage graph.

### 9.4 Setup resources

`ProcessingStage.setup_on_node_resources()` defaults to the stage's processing
resources. `execute_setup_on_node()` submits setup for every stage on every
alive Ray node with those resources. `ASRStage` explicitly requests CPU-only
prefetch setup; model construction remains a GPU-worker operation.

## 10. Tasks And Terminal Rows

[`nemo_curator/tasks/tasks.py`](nemo_curator/tasks/tasks.py) keeps
`Task.task_id` as a public constructor field with default `""`, preserving
existing `Task(task_id=...)` call sites. During pipeline execution the framework
owns it: `BaseStageAdapter` overwrites it at every derivable stage boundary.
One-to-many outputs use parent plus output index/content id, positional
many-to-many uses each matching parent, and an ambiguous
many-to-different-count fanout receives an `r<uuid>` fallback.

`EmptyTask` is a payload-less class rooted at task id `"0"`; its inherited
constructor shape still accepts `task_id=`, but `__post_init__()` normalizes the
value back to `"0"`. Source execution constructs it with `EmptyTask()`.

[`nemo_curator/tasks/task_terminals.py`](nemo_curator/tasks/task_terminals.py)
defines generic `_curator_terminal_*` ownership and tombstone fields. Normal
local Qwen rows have no terminal ownership metadata, so ordinary filtering
still removes them. The helper activates only for rows that explicitly carry a
terminal contract. Tombstones shallow-copy the original task and task-data
mapping, preserving optional subclass fields, task id, metadata, and prior
performance records. Drop attribution prefers `_curator_stage_id`, then stage
name, then class name.

Terminal identity is `(group_id, index, count)`. A stage that constructs a
replacement task must copy those fields. A non-terminal replacement is retained
and a tombstone is also appended for the missing identity; an unmatched
conflicting terminal identity can be discarded when reconstructing an
all-terminal output list. The local Qwen path does not create terminal rows,
but this is the exact shared contract available to another planner.

## 11. Performance And Resource Observability

### 11.1 Stage metrics

`BaseStageAdapter` attaches one `StagePerfStats` record per stage invocation.
Its public `to_dict()` and numeric `items()` schema remains main-compatible.
When `extended_performance_metrics` is enabled, the record also carries an
invocation id, node/worker/actor identity, and per-GPU utilization/VRAM
observations. Actor samplers target only UUIDs assigned to that actor; if
assignment cannot be resolved, actor sampling is skipped rather than
attributing neighboring devices. Audio aggregation deduplicates records by
invocation id because one invocation record may be attached to several output
tasks.

[`nemo_curator/backends/perf_identity.py`](nemo_curator/backends/perf_identity.py)
normalizes Ray and Xenna identity into common node, worker, actor, hostname, GPU
index, GPU UUID, and allocation fields.

### 11.2 Pipeline hardware sampler

When `pipeline_hardware_sampler_enabled` is true,
[`nemo_curator/utils/pipeline_hardware_sampler.py`](nemo_curator/utils/pipeline_hardware_sampler.py)
starts one sampler actor per alive node for the executor lifetime. It observes
every GPU independently of stage ownership. Each sampler is pinned by Ray
`NodeAffinitySchedulingStrategy` using the node's `NodeID`; it does not assume
that Ray's `node:*` resource key contains that id. The generic executor default
is off; the raw Qwen runtime YAML opts in. At shutdown, the executor first
offers the resulting `pipeline_hardware_sampler` record to a terminal
performance recorder. An accepting writer persists it once in
`perf_summary.json`; only when no recorder accepts it does the executor attach
the record to returned tasks as a fallback. Neither path changes stage
placement.

### 11.3 Audio performance summary

[`nemo_curator/stages/audio/metrics/performance.py`](nemo_curator/stages/audio/metrics/performance.py)
aggregates stage totals/percentiles, per-actor and per-GPU views, payload wait
and locality metrics, adapter calls/items, audio throughput, writer timing, and
hardware samples. Shared invocation ids prevent duplicated counts when the same
perf record is attached to several output rows.

## 12. Manifest Output

`ManifestWriterStage` in
[`nemo_curator/stages/audio/common.py`](nemo_curator/stages/audio/common.py) is a
single-worker actor stage. Driver setup truncates the output once; per-node
setup only creates directories.

[`manifest_writer_utils.py`](nemo_curator/stages/audio/io/manifest_writer_utils.py)
applies an explicit serialization policy. By default it writes task data as-is,
matching the existing writer contract. `drop_manifest_keys` and
`drop_array_like_values` opt into omission, and non-JSON values otherwise fail
with the offending key. In the Qwen graph, `PayloadReleaseStage` removes refs
and waveform bookkeeping before the writer; the Qwen writer config also opts
into array/key filtering. `write_perf_stats` defaults off for compatibility and
is enabled by the benchmark config to refresh `perf_summary.json` and merge the
executor's external pipeline-hardware record.

`ShardedManifestWriterStage` and `NemoTarredAudioReader` provide separate
sharded-output and tarred-input APIs. They do not alter the raw Qwen lifecycle
graph.

## 13. Extending The Primitives

### 13.1 Another payload modality

A source stage implements `build_payload_materialize_stage()`. Its materializer
creates the configured handle, and consumers implement
`resolve_payload_refs_for_batch()`. Central graph insertion remains independent
of the payload's modality.

A source with full-dataset planning may emit generic `DispatchBatchTask`
envelopes. Its owner validates one envelope as at most one call, keeping all
eligible children together while skipped children retain aligned placeholders;
`DispatchBatchUnpackStage` restores child rows. This extension uses ordinary
task rows and does not require a Ray Data or Xenna scheduling change.

### 13.2 Multiple consumers

List every stage that needs the one payload ref under
`payload_lifecycle.consumers`, and set `release_after` to the final consumer.
Each consumer remains independently scheduled. Multiple payload sources per
row are intentionally out of scope.

### 13.3 Another inference model

An ASR model implements `ASRAdapter`. Another modality can use `BatchPolicy`
and `run_bucketed()` without adopting ASR. Each model stage retains its own
backend resources and worker count. A model adapter can expose
`estimate_item_cost()` for encoder-token or VRAM-aware cost in place of raw
duration.

## 14. Test Map

- [`tests/pipelines/test_pipelines.py`](tests/pipelines/test_pipelines.py):
  logical/execution graph state, rebuilds, and source/sink roles;
- [`tests/pipelines/audio/test_qwen_omni_raw_inprocess.py`](tests/pipelines/audio/test_qwen_omni_raw_inprocess.py):
  lifecycle expansion, multiple consumers, binding validation, and helper-stage
  rejection;
- [`tests/stages/test_payload_lifecycle.py`](tests/stages/test_payload_lifecycle.py):
  byte admission, stores, explicit-release lifetime, namespaces, batched actor
  methods, internal lease expiry, direct-key release, fallback cleanup,
  read-error rows, and actor cleanup;
- [`tests/tasks/test_dispatch_batch.py`](tests/tasks/test_dispatch_batch.py):
  envelope validation, cardinality preservation, and generic fan-out;
- [`tests/pipeline/test_payload_refs.py`](tests/pipeline/test_payload_refs.py)
  and [`tests/pipeline/test_prefetch.py`](tests/pipeline/test_prefetch.py):
  actor-grouped resolution, stable ref order, cache behavior, and
  one-successor overlap;
- [`tests/stages/audio/test_model_input_segmentation.py`](tests/stages/audio/test_model_input_segmentation.py):
  validation, exact 2400-second boundary, zero samples, and tail segments;
- [`tests/stages/audio/inference/test_asr_stage.py`](tests/stages/audio/inference/test_asr_stage.py):
  payload-backed inputs, exact envelope dispatch, policy/segmentation rejection,
  segmentation, language/reference fields, result ordering, skip behavior,
  adapter calls, and metrics;
- [`tests/stages/audio/inference/test_batch_policy.py`](tests/stages/audio/inference/test_batch_policy.py):
  bucket edges, caps, cost scheduling, adapter batches, ordering, and generic
  scheduler hooks;
- [`tests/backends/test_task_id_postprocess.py`](tests/backends/test_task_id_postprocess.py):
  task-id derivation, shorter/reordered/replacement terminal outputs,
  conflicting identities, stage-id drop markers, and subclass-field-preserving
  tombstones;
- [`tests/backends/ray_data/test_utils.py`](tests/backends/ray_data/test_utils.py):
  actor sizing and backend batch delivery;
- [`tests/backends/test_xenna_executor.py`](tests/backends/test_xenna_executor.py):
  StageSpec construction, `num_workers()`/per-node sizing conflicts, and
  verbosity;
- [`tests/stages/audio/metrics/test_perf_summary.py`](tests/stages/audio/metrics/test_perf_summary.py)
  and [`tests/utils/test_gpu_sampler.py`](tests/utils/test_gpu_sampler.py):
  summary and GPU metrics.

The focused suite also covers same-ref retry for plain and dispatch inputs and
reader-error pass-through without an adapter call. It does not simulate hard
driver loss before executor cleanup or fail-loud behavior for a systemic Qwen
preprocessing error; those remain reviewer-visible properties outside these
focused tests.

## 15. Reviewer File Map

| Concern | Primary files |
| --- | --- |
| pipeline planning | `nemo_curator/pipeline/pipeline.py`, `pipeline/payload_lifecycle.py`, `stages/dispatch_batch.py` |
| payload handles/prefetch | `pipeline/payload_refs.py`, `pipeline/prefetch.py`, `stages/payload_lifecycle.py` |
| local manifest reader/writer | `stages/audio/common.py` |
| local audio decode | `stages/audio/io/audio_file_reader.py` |
| model-input segmentation | `stages/audio/model_input_segmentation.py` |
| ASR and batching | `stages/audio/inference/asr/stage.py`, `batch_policy.py` |
| Qwen adapter | `models/asr/base.py`, `models/asr/qwen_omni.py`, `models/vllm_model.py` |
| backend execution | `backends/base.py`, `backends/ray_data/*`, `backends/xenna/*`; `backends/ray_actor_pool/*` remains current main |
| task contracts | `tasks/tasks.py`, `tasks/dispatch_batch.py`, `tasks/task_terminals.py`, `tasks/sentinels.py` |
| performance | `backends/perf_identity.py`, `utils/gpu_sampler.py`, `utils/pipeline_hardware_sampler.py`, `stages/audio/metrics/*` |
| output safety | `stages/audio/io/manifest_writer_utils.py`, `stages/audio/common.py` |
| Hydra/YAML construction and execution | `config/run.py` |

## 16. Current Contracts And Reviewer Checks

1. On a normal successful attempt, each parent row is decoded once by the
   materializer; ASR slices that stored waveform in memory and downstream
   consumers do not reread the file.
2. The waveform tensor lives in a payload actor between consumers.
3. Every configured consumer can resolve the same ref without another file read.
4. Explicit release removes the stored tensor and its byte reservation;
   published payloads do not expire while queued.
5. Every GPU consumer remains a separate backend-visible stage.
6. Ray Data and Xenna use each stage's normal resources and worker contract.
7. Model inputs never exceed `max_inference_duration_s`.
8. Bucket-on groups only model-safe segments from the current backend window.
9. Bucket-off retains segmentation as its long-row model safety boundary.
10. ASR results are restored to original parent order.
11. Output manifests contain neither waveform tensors nor `PayloadRef` objects.
12. Performance summaries contain adapter-level calls/items and both
    invocation-window and pipeline-wide GPU/VRAM observations.
13. A failed payload-aware invocation leaves its input refs owned by the store,
    so a backend retry can resolve the same handles; successful filtering and the
    explicit release stage remain the only per-row release points.
14. Qwen preprocessing exceptions are converted to skipped/empty item output;
    only vLLM generation failures are guaranteed to fail the batch.
15. A row already marked skipped, including a materialization read error, keeps
    its output position and terminal metadata, receives empty ASR output, and
    does not enter an adapter call.
