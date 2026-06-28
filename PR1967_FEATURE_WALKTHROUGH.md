# PR1967 Local/Windowed Branch: Current-Code Walkthrough

This document describes the current files on disk for
`aaftabv/qwen1967-local-noio-control`. It is a reviewer guide to the code as it
exists now: public contracts, execution flow, task shape, scheduling, memory
ownership, failure behavior, observability, and tests. It intentionally does
not narrate discarded designs or earlier benchmark revisions.

The branch provides four connected capabilities:

1. a generic pipeline rule that inserts payload materialization and release
   stages around independently scheduled consumers;
2. a Ray-backed payload reference lifecycle that decodes large audio payloads
   once and keeps waveform tensors out of ordinary task rows;
3. model-input segmentation and duration-aware bucketing inside the
   backend-visible ASR stage's finite input window;
4. a pluggable ASR stage and Qwen-Omni adapter with model-call controls and
   detailed performance metrics.

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
  lease settings, and Ray namespace.
- **Payload consumer**: a backend-visible stage that declares payload bindings
  and resolves refs only while processing a batch.
- **Parent row**: one complete manifest record delivered to the local ASR
  stage.
- **Model-input segment**: one contiguous waveform slice no longer than
  `max_inference_duration_s`.
- **Local/windowed bucketing**: duration grouping over model-input segments
  available in the current backend-provided ASR window.
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
- consumer ref/waveform bindings match the materialized bindings;
- source, ref, and waveform keys are unique.

Selectors match Hydra stage id, stage name, class name, or fully qualified class
name.

The preferred multiple-input form is `payload_lifecycle.payloads`, with one
mapping per payload. The single-input shorthand derives fields from
`payload_keys` and consumer attributes.

The central rule calls `ManifestReader.build_payload_materialize_stage()` to
construct the audio materializer. The post-release extension path is inactive
because this reader does not enable global planning, so no parent assembler is
inserted.

### 2.3 Expanded graph config

```yaml
payload_lifecycle:
  enabled: true
  materialize_after: reader
  payload_keys: [audio_filepath]
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
| `lease_ttl_s` | heartbeat setting |
| `actor_namespace` | Ray namespace for actor lookup |

`resolve_payload_ref()` refreshes actor state and returns one payload.
`resolve_payload_refs_batched()` groups handles by actor, issues
`heartbeat_many`, `pin_many`, and `get_many` RPCs, preserves caller order, and
splits work by an optional byte bound. Each actor-side bulk method performs one
expiry-reap pass for the whole request, so actor work stays linear in the
number of handles.
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

Materialized payloads use a longer finite `materialized_lease_ttl_s` while they
wait between stages (four hours by default). `_PayloadLeaseKeeper` switches to
active `lease_ttl_s` renewals while a consumer performs long model work.
Explicit release is the normal fast path; finite expiry lets admission and
store actors reclaim a payload whose row is lost before release.

### 3.3 AudioPayloadMaterializeStage

For each row the materializer:

1. reads the configured metadata duration;
2. estimates waveform bytes and acquires admission capacity with a finite
   `lease_ttl_s` lease;
3. decodes the local file through `AudioFileReaderStage`;
4. removes the waveform from normal task data;
5. measures actual tensor bytes and resizes the reservation;
6. stores the tensor in the node-local actor;
7. writes `PayloadRef`, estimated bytes, actual bytes, and producer node id;
8. converts the completed reservation to the finite
   `materialized_lease_ttl_s`, giving queued rows a long handoff window while
   bounding orphan retention.

Duration must be positive and numeric. Byte-limit strings accept integer, `k`,
`m`, and `g` forms and reject invalid values. If actual bytes cannot fit, the
stage releases its reservation and raises. If materialization fails after the
store insert, it also removes the stored object. If a worker dies before the
reservation is committed, its finite materialization lease can be reaped.

When `skip_on_read_error` is enabled, a reader error yields a skipped row with
no payload ref. The reservation and zero-length waveform are removed.

Metrics include admission wait, poll count, estimated/reserved/stored bytes,
node and cluster budgets, and materialization count.

### 3.4 Payload-aware consumers

`PayloadAwareStageMixin.payload_bindings()` provides a single-waveform default.
A multi-input stage overrides it with one binding per payload.

`resolve_payload_refs_for_batch()` resolves handles with actor-grouped,
byte-bounded bulk RPCs, restores sample metadata, records same-node and
cross-node resolution metrics, and starts batched heartbeats.
`drop_resolved_payloads()` stops the heartbeat thread and removes temporary
waveform fields.

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

`PayloadReleaseStage` finds all nested refs, deduplicates payload ids, releases
objects and byte reservations, strips handles, removes waveform and payload
bookkeeping keys, and returns the task. It supports rows without refs and keeps
the existing task-data mapping object intact.

`BaseStageAdapter` performs payload scanning only for stages marked by the
lifecycle expander. On those stages it releases all input refs on an exception
and releases refs that disappear because a stage filtered a row. Stages in
ordinary pipelines do not pay that recursive scan cost.

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
| `BatchPolicy.max_items_per_batch_by_bucket` | model-work grouping per duration bucket |
| `adapter_batch_size` | fallback items per adapter call |
| `BatchPolicy.bucketed_inference_batch_size` | per-duration-bucket adapter-call cap |
| `BatchPolicy.max_audio_sec_per_batch` | aggregate cost cap for one bucketed batch |

These controls do not govern payload RAM. Payload memory is admitted in bytes
by the materializer.

### 7.3 Process flow

`ASRStage.process_batch()`:

1. uses eager bulk payload resolution unless prefetch is explicitly enabled;
2. in prefetch mode, plans segments and exact adapter calls from payload
   metadata before waveform resolution;
3. resolves the current call and overlaps one bounded next call with current
   GPU inference;
4. builds adapter items with language, reference text, task id, duration, and
   stitch-back indices;
5. applies duration-aware policy when enabled;
6. splits each bucket by its adapter-call cap;
7. invokes the adapter and realigns results;
8. joins per-parent text in segment order;
9. drops current-call waveform references; the payload actor retains the
   original until `PayloadReleaseStage`.

The finite candidate set is the current backend-provided `process_batch()`
window. Local bucketing cannot inspect rows outside that window.

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
item caps, optional adapter caps, total-cost cap, candidate window, and flush
interval. `BucketQueueScheduler` flushes on item capacity, cost capacity, timer,
or drain. Finite planning orders ready batches by descending total cost and
returns original indices for result alignment.

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
`extended_performance_metrics` is explicitly set; the Qwen entrypoint opts in,
while existing pipelines retain the compact main-compatible record shape.

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

[`nemo_curator/tasks/tasks.py`](nemo_curator/tasks/tasks.py) makes the base
`Task.task_id` framework-owned (`init=False`). `BaseStageAdapter` overwrites it
at every derivable stage boundary: one-to-many outputs use parent plus output
index/content id, positional many-to-many uses each matching parent, and an
ambiguous many-to-different-count fanout receives an `r<uuid>` fallback.
`AudioTask` retains its audio-specific constructor field, but normal backend
postprocessing still derives the stage-boundary id.

`EmptyTask` is a payload-less class rooted at task id `"0"`; source execution
constructs it with `EmptyTask()`.

[`nemo_curator/tasks/task_terminals.py`](nemo_curator/tasks/task_terminals.py)
defines generic `_curator_terminal_*` ownership and tombstone fields. Normal
local Qwen rows have no terminal ownership metadata, so ordinary filtering
still removes them. The helper activates only for rows that explicitly carry a
terminal contract.

## 11. Performance And Resource Observability

### 11.1 Stage metrics

`BaseStageAdapter` attaches one `StagePerfStats` record per stage invocation.
Its public `to_dict()` and numeric `items()` schema remains main-compatible.
When `extended_performance_metrics` is enabled, the record also carries an
invocation id, expected resources, node/worker/actor identity, and per-GPU
utilization/VRAM observations. Audio aggregation deduplicates records by
invocation id because one invocation record may be attached to several output
tasks.

[`nemo_curator/backends/perf_identity.py`](nemo_curator/backends/perf_identity.py)
normalizes Ray and Xenna identity into common node, worker, actor, hostname, GPU
index, GPU UUID, and allocation fields.

### 11.2 Pipeline hardware sampler

When `pipeline_hardware_sampler_enabled` is true,
[`nemo_curator/utils/pipeline_hardware_sampler.py`](nemo_curator/utils/pipeline_hardware_sampler.py)
starts one sampler actor per alive node for the executor lifetime. It observes
every GPU independently of stage ownership. The generic executor default is
off; the Qwen entrypoint opts in by default. Executors attach the resulting
`pipeline_hardware_sampler` record without using it for placement decisions.

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

### 13.2 Multiple payloads or consumers

Use one `payload_lifecycle.payloads` mapping per source and override
`payload_bindings()` in consumers. All materializers are inserted after the
source. One release stage recursively frees every nested ref after the final
consumer.

### 13.3 Another inference model

An ASR model implements `ASRAdapter`. Another modality can use `BatchPolicy`
and `run_bucketed()` without adopting ASR. Each model stage retains its own
backend resources and worker count. A model adapter can expose
`estimate_item_cost()` for encoder-token or VRAM-aware cost in place of raw
duration.

## 14. Test Map

- [`tests/pipelines/test_pipelines.py`](tests/pipelines/test_pipelines.py):
  logical/execution graph state, rebuilds, and source/sink roles;
- [`tests/pipelines/audio/test_qwen_omni_inprocess.py`](tests/pipelines/audio/test_qwen_omni_inprocess.py):
  lifecycle expansion, multiple consumers, multiple payloads, and helper-stage
  rejection;
- [`tests/stages/test_payload_lifecycle.py`](tests/stages/test_payload_lifecycle.py):
  byte admission, stores, explicit-release lifetime, namespaces, batched actor
  methods, heartbeat, nested release, read-error rows, and actor cleanup;
- [`tests/pipeline/test_payload_refs.py`](tests/pipeline/test_payload_refs.py)
  and [`tests/pipeline/test_prefetch.py`](tests/pipeline/test_prefetch.py):
  actor-grouped resolution, stable ref order, byte bounds, cache behavior, and
  one-successor overlap;
- [`tests/stages/audio/test_model_input_segmentation.py`](tests/stages/audio/test_model_input_segmentation.py):
  validation, exact 2400-second boundary, zero samples, and tail segments;
- [`tests/stages/audio/inference/test_asr_stage.py`](tests/stages/audio/inference/test_asr_stage.py):
  payload-backed inputs, segmentation, language/reference fields, result
  ordering, skip behavior, adapter calls, and metrics;
- [`tests/stages/audio/inference/test_batch_policy.py`](tests/stages/audio/inference/test_batch_policy.py):
  bucket edges, caps, cost scheduling, adapter batches, ordering, and generic
  scheduler hooks;
- [`tests/backends/ray_data/test_utils.py`](tests/backends/ray_data/test_utils.py):
  actor sizing and backend batch delivery;
- [`tests/backends/xenna/test_executor.py`](tests/backends/xenna/test_executor.py):
  StageSpec construction, `num_workers()`/per-node sizing conflicts, and
  verbosity;
- [`tests/stages/audio/metrics/test_perf_summary.py`](tests/stages/audio/metrics/test_perf_summary.py)
  and [`tests/utils/test_gpu_sampler.py`](tests/utils/test_gpu_sampler.py):
  summary and GPU metrics.

## 15. Reviewer File Map

| Concern | Primary files |
| --- | --- |
| pipeline planning | `nemo_curator/pipeline/pipeline.py`, `pipeline/payload_lifecycle.py` |
| payload handles/prefetch | `pipeline/payload_refs.py`, `pipeline/prefetch.py`, `stages/payload_lifecycle.py` |
| local manifest reader/writer | `stages/audio/common.py` |
| local audio decode | `stages/audio/io/audio_file_reader.py` |
| model-input segmentation | `stages/audio/model_input_segmentation.py` |
| ASR and batching | `stages/audio/inference/asr/stage.py`, `batch_policy.py`, `bucketed_stage.py` |
| Qwen adapter | `models/asr/base.py`, `models/asr/qwen_omni.py`, `models/vllm_model.py` |
| backend execution | `backends/base.py`, `backends/ray_data/*`, `backends/xenna/*`; `backends/ray_actor_pool/*` remains current main |
| task contracts | `tasks/tasks.py`, `tasks/task_terminals.py`, `tasks/sentinels.py` |
| performance | `backends/perf_identity.py`, `utils/gpu_sampler.py`, `utils/pipeline_hardware_sampler.py`, `stages/audio/metrics/*` |
| output safety | `stages/audio/io/manifest_writer_utils.py`, `stages/audio/common.py` |
| Hydra entry point | `pipelines/audio/qwen_omni_inprocess.py` |

## 16. Core Invariants To Verify

1. Each input audio file is decoded once by the materializer.
2. The waveform tensor lives in a payload actor between consumers.
3. Every configured consumer can resolve the same ref without another file read.
4. Release removes the stored tensor and its byte reservation.
5. Every GPU consumer remains a separate backend-visible stage.
6. Ray Data and Xenna use each stage's normal resources and worker contract.
7. Model inputs never exceed `max_inference_duration_s`.
8. Bucket-on groups only model-safe segments from the current backend window.
9. Bucket-off retains segmentation as its long-row model safety boundary.
10. ASR results are restored to original parent order.
11. Output manifests contain neither waveform tensors nor `PayloadRef` objects.
12. Performance summaries contain adapter-level calls/items and both
    invocation-window and pipeline-wide GPU/VRAM observations.
