# PR1967 Global Branch: Current-Code Walkthrough

This document describes the current files on disk for the global
metadata-planned bucketing branch. It follows the runtime path a reviewer sees
in code: graph expansion, metadata planning, payload ownership, GPU execution,
release, parent assembly, output, and observability. It intentionally omits
discarded designs and old run history.

The branch provides five connected capabilities:

1. a generic pipeline rule that inserts payload materialization and release
   around independently scheduled consumers;
2. a Ray-backed payload lifecycle that decodes a large input once and passes a
   lightweight `PayloadRef` through ordinary task data;
3. a full-manifest, metadata-only audio planner that emits bounded segment
   rows before waveform decode;
4. a strict parent assembler that restores original rows and merges declared
   segment outputs in deterministic segment and parent order; and
5. a pluggable ASR/Qwen execution path with duration-aware model calls,
   byte-bounded payload prefetch, and detailed performance metrics.

The user-facing Qwen graph is:

```text
ManifestReader -> ASRStage -> ManifestWriterStage
```

With payload lifecycle and global bucketing enabled, `Pipeline.build()` makes
the backend-visible graph:

```text
ManifestReader(global metadata planner)
  -> AudioPayloadMaterializeStage
  -> ASRStage
  -> PayloadReleaseStage
  -> GlobalSegmentAssemblerStage
  -> ManifestWriterStage
```

Every arrow is an ordinary backend edge. Ray Data or Xenna still sees each CPU,
GPU, release, assembly, and writer stage separately. The PR changes graph shape
and task shape; it does not fuse work or add another backend autoscaler.

The global and local branches share every executable implementation file except
`nemo_curator/stages/audio/common.py`. The global version of that file adds the
metadata planner, parent-data store, terminal segment rows, and assembler.
Payload refs, ASR, Qwen, backends, tasks, writer policy, and observability are
otherwise byte-identical in the current worktrees.

## 1. Terms Used By The Code

- **Logical stage**: a stage listed by the user in `Pipeline.stages` or Hydra
  `stages:`.
- **Execution stage**: a concrete `ProcessingStage` after graph expansion and
  composite decomposition.
- **Backend-visible stage**: an execution stage independently presented to Ray
  Data or Xenna.
- **Payload**: a large object managed outside ordinary task serialization. The
  audio materializer stores a decoded waveform tensor.
- **Payload binding**: source, ref, waveform, sample-rate, sample-count,
  duration, and materializer fields describing one payload.
- **PayloadRef**: a lightweight handle carrying actor names, payload id, owner
  node, byte size, sample metadata, lease settings, and namespace.
- **Parent row**: one complete source manifest record.
- **Model-input segment**: one contiguous input interval no longer than
  `max_inference_duration_s`.
- **Segment row**: the `AudioTask` emitted for one globally planned interval.
- **Global owner**: the single consumer whose maximum input duration and
  duration policy define the global plan.
- **Terminal row**: one member of an ordered group that a terminal consumer
  must receive exactly once. Global segment rows use this generic contract.
- **Tombstone**: a lightweight terminal row emitted when an intermediate stage
  drops a segment, allowing assembly to finish and mark configured parent
  outputs with an explicit drop reason.
- **Merge strategy**: an explicit rule for reducing one field across ordered
  segment rows into one parent value.

ASR-internal symbols retain `chunk` in names such as `_ChunkSpec`,
`_build_chunk_specs()`, and `_curator_asr_chunk_*`. Those names describe
ASR-local result stitch-back. The planner's public term is **segment**, its
boundary is `max_inference_duration_s`, and its shared record is
`AudioSegment`.

## 2. Pipeline Construction And Graph Expansion

### 2.1 Logical and execution graphs

[`nemo_curator/pipeline/pipeline.py`](nemo_curator/pipeline/pipeline.py) keeps:

- `_logical_stages`, the canonical user graph;
- `stages`, the public list and, after build, the execution graph; and
- `_planned_stage_snapshot`, used to distinguish a reused plan from direct
  public-list mutation.

`Pipeline.__init__()` preserves the caller-visible config mapping, matching
main-branch behavior, and creates a private `_curator_pipeline_run_id`. The id
is copied only into the ephemeral expansion config. It therefore scopes named
actors without injecting framework state into caller configuration.

`Pipeline.add_stage()` updates the logical graph and invalidates the plan.
Direct `pipeline.stages` mutations remain supported: `_sync_public_stage_mutations()`
treats a changed public list as the next logical graph. Before a rebuild,
`_clear_default_source_sink_roles()` removes only roles previously assigned by
the framework. `_assign_source_sink_roles()` then applies the normal one-source,
one-sink rules to the final execution graph.

`Pipeline.build()` performs this order:

1. synchronize public stage changes;
2. apply pipeline-level graph expansion;
3. decompose composite stages;
4. assign source and sink roles; and
5. retain the plan for idempotent repeated builds.

### 2.2 Generic payload expansion

[`nemo_curator/pipeline/payload_lifecycle.py`](nemo_curator/pipeline/payload_lifecycle.py)
contains `expand_payload_lifecycle_stages()`. It is loaded only when
`payload_lifecycle.enabled` is true, so a base non-audio pipeline does not
import optional audio dependencies.

The expander validates:

- one source selected by `materialize_after`;
- one final consumer selected by `release_after`;
- every configured consumer lies in that range and supports payload bindings;
- ref, waveform, and source keys do not collide; and
- users did not manually insert lifecycle helper stages.

The preferred form is `payload_lifecycle.payloads`, one binding per large
input. `payload_keys` remains a compact single-input form. Stage selectors may
match Hydra id, stage name, class name, or fully qualified class name.

The central expander does not construct an audio stage itself. It calls the
source stage's `build_payload_materialize_stage()` hook. For global audio it
also asks the source for `build_payload_lifecycle_post_release_stage()`, which
returns the assembler. That division keeps graph ordering generic and keeps
audio decode/assembly in the audio module.

The source logical `ManifestReader` is replaced by its physical global planner
before helper insertion. The result is planner -> materializers -> consumers ->
release -> assembler -> writer.

### 2.3 Multi-consumer behavior

`consumers` may list several CPU or GPU stages. Each remains a separate
backend-visible stage with its own resources, batch size, and worker count. A
payload ref remains valid through all consumers and is released only after the
configured final consumer. Global planning still has exactly one owner: other
consumers must accept the owner's segment rows. Configuration fails if the
owner is absent, is not a payload consumer, does not have the largest model
input window, or disagrees with the reader's maximum duration.

## 3. Payload Reference Lifecycle

The handle API is in
[`nemo_curator/pipeline/payload_refs.py`](nemo_curator/pipeline/payload_refs.py).
Actor state, materialization, consumer support, and release are in
[`nemo_curator/stages/payload_lifecycle.py`](nemo_curator/stages/payload_lifecycle.py).

### 3.1 Admission and ownership

`_PayloadAdmissionState` accounts for reserved bytes per node and across the
cluster. The default cluster budget is the sum of registered node budgets;
`max_cluster_payload_bytes` may impose a smaller explicit ceiling. A single
row larger than an applicable budget fails immediately. Temporary pressure
waits for release up to `admission_wait_timeout_s`, then raises with a current
usage snapshot.

`_PayloadStoreState` owns the actual objects. Store actors are node-affined;
admission is shared. Actor names include the private pipeline run id and active
Ray namespace. They are detached so backend worker replacement does not erase
payloads, and executor cleanup kills them after the run.

Materialization reservations initially use `lease_ttl_s`. Once a row has been
stored and published, both its store entry and admission reservation use the
longer finite `materialized_lease_ttl_s` (four hours by default). This protects
normal backend queueing without allowing an orphaned post-decode payload to
hold admission bytes forever. When ASR claims the ref, `_PayloadLeaseKeeper`
renews the active `lease_ttl_s` while model work is in progress. Explicit
release remains the normal fast path; TTL reap is the failure-recovery path.

### 3.2 Materialization

For every segment row, `AudioPayloadMaterializeStage`:

1. validates duration and estimates float waveform bytes;
2. acquires admission capacity;
3. invokes `AudioFileReaderStage` with `segment_start_s` and
   `segment_duration_s`;
4. decodes local audio to contiguous float32 PCM at configured rate/channels;
5. removes the waveform from ordinary task data;
6. resizes the reservation to measured tensor bytes;
7. stores the tensor in the node-local payload actor; and
8. writes a `PayloadRef` plus payload metrics into the row.

If decode or store fails, the stage rolls back the reservation and any inserted
object. With `skip_on_read_error`, it emits a skipped row with no waveform or
ref; release later safely no-ops on that row.

The global planner emits one row per model-safe segment, so materialization
decodes only that segment. A 7,513-second source split into four segments never
requires a 7,513-second tensor in one materializer task.

### 3.3 Batched resolution and one-call prefetch

`resolve_payload_refs_batched()` groups refs by admission/store actor,
deduplicates handles, calls `heartbeat_many`, `pin_many`, and `get_many`, and
restores the caller's order. `get_many` work can be split by a byte ceiling.
This replaces three serial actor round trips per ref with actor-grouped bulk
operations without increasing the payload bytes transferred.

The Qwen ASR config also opts into
[`BoundedOneAheadPrefetchIterator`](nemo_curator/pipeline/prefetch.py).
`ASRStage` plans adapter-call boundaries from ref metadata before loading
waveforms. `_PayloadCallMaterializer` resolves only refs needed by one call,
slices its segment inputs, and may load one byte-bounded successor while the
GPU runs the current call. It keeps a parent cached only while contiguous calls
still need it, then drops actor-local references. The payload store remains the
owner until release.

This mechanism is generic at the pipeline layer: the iterator accepts a
loader, size function, and work iterable. ASR supplies audio descriptors; a
future image or video stage can supply different descriptors without changing
Ray Data or Xenna.

### 3.4 Release and failure paths

`PayloadReleaseStage` recursively finds and deduplicates refs, removes objects,
releases admission bytes, strips refs and lifecycle bookkeeping, and mutates
the existing task-data mapping in place. Preserving that mapping type keeps
`AudioTask.data` attribute access valid downstream.

`BaseStageAdapter` scans payloads only on stages marked by the expander. On
those stages it releases input refs after an exception and releases refs lost
when a stage filters a row. Ordinary pipelines do not pay the recursive scan
cost.

## 4. Global Metadata Planning

### 4.1 Reader decomposition

Global behavior is contained in
[`nemo_curator/stages/audio/common.py`](nemo_curator/stages/audio/common.py).
With `enable_global_bucketing=true`, `ManifestReader.decompose()` returns one
`_ManifestReaderGlobalBucketingStage`. Bucket-off keeps the normal
`FilePartitioningStage -> ManifestReaderStage` decomposition.

The global planner:

1. streams every configured JSONL manifest and records source order;
2. reads duration from `duration_key` or configured fallbacks;
3. derives sample count at the row or target sample rate;
4. calls the shared segment planner with `max_inference_duration_s`;
5. assigns each segment to a duration bucket;
6. asks `BatchPolicy.bucketize_with_costs()` for full-manifest ready batches;
7. flattens those ready batches into ordinary `AudioTask` rows in plan order;
8. stores every full parent row once in a run-scoped parent-data actor; and
9. emits only configured `segment_input_keys` plus planner/terminal fields.

No waveform is decoded during these steps. `estimated_waveform_bytes` is a
planning/metric sum, not allocated RAM.

### 4.2 Row shape and input columns

`segment_input_keys` controls which original columns travel through every
segment consumer. The source audio key must be included; prompt, language, or
other stage inputs must be listed when a consumer needs them. Copying only
these fields prevents multiplying a large parent dictionary by its segment
count. The parent-data actor preserves the complete original row once, and the
assembler restores it after segment processing.

Each emitted segment row adds:

- `segment_start_s`, `segment_duration_s`, segment duration, and sample count;
- generic `_curator_terminal_*` group/index/count/source-order fields;
- `_curator_segment_*` audio/debug aliases; and
- optional plan annotations in task metadata.

The generic terminal fields are the correctness contract. Audio-specific
aliases make logs and inspection easier but are not what backend preservation
depends on.

### 4.3 Code-derived 5h example

The real benchmark manifest is:

```text
/home/aaftabv/grananary-v2/realdata_5h_yt_alm_part2_20260613/manifest_5h_stratified_duration_tails.jsonl
```

Applying the current planner logic with a 2,400-second ceiling, bucket edges
`[0, 600, 1200, 2400]`, item caps `[4, 2, 1, 1]`, and a 2,400-second aggregate
call cap yields:

| Quantity | Current-code result |
| --- | ---: |
| Parent rows | 89 |
| Parent audio | 18,022.1876 s / 5.0062 h |
| Parents longer than 2,400 s | 2 |
| Emitted segment rows | 93 |
| Parents with 1 / 2 / 4 segments | 87 / 1 / 1 |
| Segments in `[0,600)` / `[600,1200)` / `[1200,2400)` / `[2400,+inf)` | 88 / 1 / 0 / 4 |
| Full-manifest ready batches | 27 |
| Estimated decoded bytes summed over segments | 1,153,420,047 bytes |

The first two source rows are transformed as follows:

| Source index | Parent duration | Segment starts | Segment durations |
| ---: | ---: | --- | --- |
| 0 | 7513.3335 s | 0, 2400, 4800, 7200 | 2400, 2400, 2400, 313.3335 |
| 1 | 2756.4135 s | 0, 2400 | 2400, 356.4135 |

For source index 0, the first task data is logically:

```json
{
  "audio_filepath": "harvested_data/youtube/audios/I5ZlmdiKvRE.opus",
  "source_lang": "en",
  "segment_start_s": 0.0,
  "segment_duration_s": 2400.0,
  "duration": 2400.0,
  "num_samples": 38400000,
  "_curator_terminal_group_id": "0:0:0",
  "_curator_terminal_idx": 0,
  "_curator_terminal_count": 4,
  "_curator_terminal_source_index": 0,
  "_curator_segment_bucket": 3,
  "_curator_segment_parent_duration_s": 7513.3335
}
```

The fourth task has index 3, starts at 7,200 seconds, lasts 313.3335
seconds, and belongs to bucket 0. The original parent can contain many other
fields; only configured segment inputs are copied. The complete parent is
restored from the parent-data actor.

The plan order is intentionally not parent order. Ready batches are ordered by
duration cost across the whole manifest, then flattened into rows. The
assembler, not the GPU stage, restores parent order.

### 4.4 Single-owner validation

One global segmentation can optimize for one model-input ceiling. The owner
stage is therefore a scalar selector and must be one of the payload consumers.
The owner must have the largest configured `max_inference_duration_s`; a
consumer requiring a larger parent interval would otherwise receive segments
that were unnecessarily cut for a smaller model. Reader and owner ceilings
must match. These conditions are validated while expanding the graph, before a
backend starts.

## 5. Model-Input Segmentation

[`nemo_curator/stages/audio/model_input_segmentation.py`](nemo_curator/stages/audio/model_input_segmentation.py)
is the shared safety primitive for global and local paths.

`resolve_max_model_input_duration()` validates the positive ceiling.
`plan_audio_segments()` converts sample count, sample rate, and that ceiling
into contiguous `AudioSegment` records containing index/count, sample bounds,
and duration.

The helper guarantees:

- an input exactly at the ceiling remains one segment;
- an input just over it becomes one full segment plus a tail;
- no overlap, gap, or padding is introduced;
- zero samples are representable as one empty segment; and
- invalid sample rates fail immediately.

Global bucket-on invokes it before decode. Bucket-off and the local branch
invoke it inside ASR after full-parent materialization, preserving the same
model/OOM boundary. Bucketing groups bounded inputs; it never replaces the
maximum model-input invariant.

## 6. ASR Stage And Model Calls

[`nemo_curator/stages/audio/inference/asr/stage.py`](nemo_curator/stages/audio/inference/asr/stage.py)
defines the backend-visible GPU consumer. Model implementations conform to
[`ASRAdapter`](nemo_curator/models/asr/base.py).

### 6.1 Inputs, outputs, and setup

The stage accepts waveform or `waveform_ref`, sample rate, optional language,
and optional reference text. It writes configured primary/secondary text and a
skip field. Global segment rows look like ordinary ASR tasks; terminal metadata
passes through untouched.

`setup_on_node()` prefetches model files using the stage's explicit CPU-only
setup resource override. Worker `setup()` constructs the model under the
stage's GPU resources. No generic setup task is forced to CPU: other stages
retain their own processing resources by default.

### 6.2 Batch controls

| Control | Scope |
| --- | --- |
| `ASRStage.batch_size` | backend-provided candidate parent/segment window |
| `max_items_per_batch_by_bucket` | item grouping per duration bucket |
| `adapter_batch_size` | fallback adapter-call item cap |
| `bucketed_inference_batch_size` | per-bucket adapter-call item cap |
| `max_audio_sec_per_batch` | aggregate estimated cost cap per bucketed call |
| `payload_resolve_max_batch_bytes` | maximum bytes in one bulk resolve request |
| `payload_prefetch_max_bytes` | current-plus-one-lookahead payload working bound |

These controls govern model-call packing and ASR-local transfers. Payload-store
RAM is governed separately by byte admission.

### 6.3 Process flow

`ASRStage.process_batch()`:

1. validates every row;
2. creates model-input descriptors and parent stitch-back indices;
3. uses the duration policy to produce exact adapter-call boundaries;
4. in prefetch mode, does this from ref metadata before waveform resolution;
5. resolves/slices only the current call and overlaps one bounded successor;
6. invokes the adapter and realigns one result per item;
7. joins ASR-local pieces in increasing segment index for each input row; and
8. removes temporary waveforms while retaining the ref for later consumers or
   release.

The global planner's ready-batch order influences which rows reach the GPU, but
backends still provide ordinary finite ASR windows. ASR validates and buckets
the rows in its window; no hidden coordinator owns GPU scheduling.

Metrics include adapter calls/items, generated segments, audio duration,
waveform bytes, output characters/tokens, inference time, resolution latency,
resolution bytes, and same-node/cross-node counts.

### 6.4 BatchPolicy

[`batch_policy.py`](nemo_curator/stages/audio/inference/batch_policy.py) validates
strictly increasing bucket edges beginning at zero, per-bucket item caps,
optional adapter caps, aggregate cost, candidate windows, and flush timing.
`bucketize_with_costs()` returns original indices so dispatch order can change
without corrupting result alignment. `run_bucketed()` exposes the same
cost/dispatch/realign pattern to other inference stages.

## 7. Qwen-Omni Adapter

[`nemo_curator/models/asr/qwen_omni.py`](nemo_curator/models/asr/qwen_omni.py)
implements `QwenOmniASRAdapter` on
[`VLLMBase`](nemo_curator/models/vllm_model.py).

The adapter owns model-specific work:

- prompt templates from inline text or files;
- per-item language/transcript interpolation;
- waveform normalization and 16 kHz resampling;
- threaded multimodal request preparation;
- one-turn or two-turn Qwen generation;
- stable one-result-per-input output with skipped placeholders;
- tensor parallelism, model/token/sequence limits, GPU memory utilization,
  multimodal limits, sampling, seed, and output-token configuration; and
- preparation, generation, valid/skipped, and token metrics.

`ASRStage` passes waveform, sample rate, language, reference text, task id,
audio duration, and stitch-back fields. It does not construct Qwen prompts or
vLLM requests. This separation lets another adapter reuse segmentation,
payload resolution, and batch policy without inheriting Qwen details.

Install the adapter with `uv sync --extra audio_qwen`. The Qwen extra composes
audio CUDA and vLLM dependencies rather than placing Qwen packages in the
general `audio_common` extra.

## 8. Global Parent Assembly

`GlobalSegmentAssemblerStage` is inserted after payload release. It is not
ASR-specific: generic terminal fields identify parent ownership, and configured
merge strategies define output reduction.

### 8.1 Collection and strict ordering

The stage uses a run-scoped named actor so all wrapper calls share assembly
state. For every segment it submits parent id, terminal index/count, task data,
metadata, stage perf, and the original parent row fetched from the parent-data
store. `_GlobalSegmentAssemblyState` rejects invalid indices/counts and stores
one entry per index.

A parent is assembled only after all expected indices are present. Segment
values are always read in numeric index order, regardless of completion order.
Completed parents enter `_ready_by_source_index`; `_drain_ready()` emits only
the next source index. The output manifest therefore preserves original parent
order even though planning and GPU completion do not.

To bound memory behind a slow early parent, ready parents above
`max_ready_parents_in_memory` spill to files under `spill_dir`. The actor keeps
small source-index/path metadata and drains spilled rows when their turn
arrives. This preserves ordering without keeping every completed parent object
in actor RAM.

### 8.2 Parent restoration and field rules

Assembly starts from the original parent dictionary, never whichever segment
arrived first. Planner fields, refs, waveform fields, and lifecycle bookkeeping
are removed. Parent duration and sample count are restored.

Per-segment generated fields require an explicit rule when their values differ.
Supported strategies include ordered text joining, list/structured
concatenation, first/last and non-null variants, sum/min/max, any/all,
dictionary merge, overwrite, and explicit drop. `text_keys_to_join` is a
shortcut for `join_text`.

Strict mode prevents silent data loss:

- an unchanged passthrough input may remain the original parent value;
- an explicitly allowlisted field is ignored as a segment output (the stored
  parent value, if any, remains authoritative) without raising;
- a parent-key collision with a changed value requires merge/overwrite; and
- an unknown varying segment output raises instead of disappearing.

The `overwrite` list is an explicit acknowledgement that a generated field may
replace a same-named parent field. It is intentionally strict: all segment
values for that field must agree, otherwise assembly raises instead of choosing
an arrival-order-dependent value. Use `first`, `last`, or another explicit
strategy when differing values have meaningful ordered semantics.

### 8.3 Dropped segments

Intermediate stages can filter rows. A missing segment would otherwise leave
the parent buffered forever. `BaseStageAdapter` calls the generic
`preserve_dropped_terminal_tasks()` only for stages marked by lifecycle
expansion. A dropped terminal row becomes a lightweight tombstone that retains
group/index/count, removes payloads and stage-declared large fields, records the
dropping stage, and sets skip metadata.

The assembler counts tombstones as terminal arrivals. If any segment was
dropped, every configured merged output (except an explicit `drop`) is set to
`"one or more intermediate segments dropped by <stage ids>"`. It does not
fabricate a complete transcript or structured result from partial segments.

### 8.4 Cleanup and metrics

Executor teardown calls stage `cleanup_run_resources()`. The planner kills the
parent-data actor; the assembler kills its state actor and clears local caches;
the payload lifecycle kills admission/store actors. Cleanup runs in `finally`
paths for Ray Data and Xenna.

Assembler metrics include segments and tombstones seen, parents assembled,
buffered/ready parents, spill count/bytes, maximum ready gap, and output rate.

## 9. Backend Scheduling And Autoscaling

Shared backend changes are modality-neutral and opt-in where they add work.

### 9.1 Base adapter

[`nemo_curator/backends/base.py`](nemo_curator/backends/base.py) retains the
normal validation -> `process_batch()` -> task-id/perf flow. Payload scans and
terminal preservation run only on stages carrying lifecycle markers. Extended
identity and GPU sampling run only when `extended_performance_metrics` is true.
The Qwen entrypoint opts in; ordinary pipelines keep the compact main-compatible
perf schema and avoid those scans.

Payload bucketing and prefetch are not backend schedulers. They execute inside
the existing ASR actor after the backend has chosen that actor and delivered a
batch.

### 9.2 Ray Data

Ray Data still calls `Dataset.map_batches()` once per backend-visible stage.
`stage.batch_size` controls the delivered window. Setup/GPU stages use actors;
stateless CPU helpers may use tasks. `stage.num_workers()` or optional
`ray_stage_spec()` actor-pool bounds retain their normal meaning; absent fixed
bounds, Ray Data scales subject to actor resources. Fanout repartitioning keeps
one task row per output block where required.

The Qwen graph presents CPU planner/materializer/release/assembler/writer and
GPU ASR as separate operators. No global planner actor owns GPU placement.

### 9.3 Xenna

Xenna creates one `StageSpec` for each execution stage. Cluster-wide worker
count comes from `stage.num_workers()`; per-node sizing may come from
`xenna_stage_spec()["num_workers_per_node"]`. Setting both is rejected.
`xenna_stage_spec()["num_workers"]` is rejected with guidance to use the
stage method. Stages without fixed sizing remain under Xenna allocation.
Streaming and batch modes use the same expanded graph.

### 9.4 Setup resources

`ProcessingStage.setup_on_node_resources()` defaults to processing resources,
preserving existing GPU-stage behavior. `execute_setup_on_node()` uses that
per-stage contract. ASR alone overrides setup to CPU-only because its node
setup downloads files; actual model construction remains in the GPU worker.

Ray Actor Pool is not used by this Qwen benchmark path and receives no
payload-prefetch or duration-bucketing policy.

## 10. Task Identity And Terminal Contracts

Base `Task.task_id` is framework-owned (`init=False`). `BaseStageAdapter`
overwrites it at every derivable boundary: one-to-many outputs use the parent
plus output index/content id, positional many-to-many uses each corresponding
parent, and ambiguous many-to-different-count fanout receives an `r<uuid>` id.
`AudioTask` retains its audio-specific constructor field, but normal backend
postprocessing still derives the boundary id. `EmptyTask` is a payload-less
class with fixed root id `"0"` and is constructed as `EmptyTask()`.

[`nemo_curator/tasks/task_terminals.py`](nemo_curator/tasks/task_terminals.py)
defines modality-neutral `_curator_terminal_*` fields. It activates only for
rows carrying terminal ownership and only on marked stages. Tombstone payload
cleanup uses generic recursive ref stripping plus stage-provided data keys; it
does not hardcode waveform cleanup in the backend.

## 11. Performance And Resource Observability

`StagePerfStats.to_dict()` and numeric `items()` preserve the public
main-compatible schema. With extended metrics enabled, records also carry an
invocation id, expected resources, actor/node/host identity, GPU allocation,
and invocation-window GPU/VRAM samples. Audio aggregation deduplicates one
invocation record attached to several output rows.

The Qwen pipeline records:

- stage process/items/throughput;
- adapter calls, items, generation/preparation time, tokens, and audio hours;
- payload admission wait, reserved/stored/resolved/released bytes, and locality;
- global planner parent/segment counts and estimated bytes;
- assembler buffering, tombstones, spills, and output rate; and
- writer timing and end-to-end wall time.

When `pipeline_hardware_sampler_enabled` is true, one observational sampler per
alive node records all GPUs for the executor lifetime. The generic executor
default is off; the Qwen entrypoint opts in. Samples never influence placement,
worker counts, or autoscaling.

## 12. Manifest Output

`ManifestWriterStage` is a one-worker actor stage. Driver setup truncates the
output once; per-node setup creates directories without truncation.

[`manifest_writer_utils.py`](nemo_curator/stages/audio/io/manifest_writer_utils.py)
uses an explicit serialization policy. Defaults write task data as-is.
`drop_manifest_keys` and `drop_array_like_values` opt into omission; an
otherwise non-JSON value raises with its key. The Qwen graph relies first on
release/assembly to remove refs and planner internals, then enables writer
filtering for configured transient fields.

`write_perf_stats` defaults off for compatibility. The benchmark config enables
it, causing the writer to aggregate attached records into `perf_summary.json`
and merge the executor's external hardware record.

For global bucket-on, output cardinality is parent cardinality, not segment
cardinality. Parent columns are restored from the parent-data store, configured
outputs are merged, and rows are written in original input order.

## 13. Reusing The Primitives

### 13.1 Another payload modality

A source stage implements `build_payload_materialize_stage()`. Its materializer
creates refs, consumers declare/resolve payload bindings, and the central
expander inserts materialize/release without knowing whether the object is
audio, image, video, or another large payload.

### 13.2 Several payloads or consumers

Use one `payload_lifecycle.payloads` entry per source. A consumer overrides
`payload_bindings()` for multiple handles. Materializers are inserted after the
source and one release stage recursively frees every nested ref after the final
consumer. Each consumer remains independently scheduled.

### 13.3 Another inference model

An audio model implements `ASRAdapter`. A non-ASR inference stage can use
`BatchPolicy`, `run_bucketed()`, batched payload resolution, and the generic
prefetch iterator directly. An adapter may expose `estimate_item_cost()` to
replace duration with encoder-token or VRAM-aware cost.

### 13.4 Another global segmented workflow

The current global planner is audio-specific because it understands duration,
sample rate, and file intervals. The terminal-row and merge contracts are
generic. A new planner can emit `_curator_terminal_*` groups and provide a
post-release assembler through the source-stage hook without adding
modality-specific logic to the backend.

## 14. Reviewer Test Map

- [`tests/pipelines/test_pipelines.py`](tests/pipelines/test_pipelines.py):
  logical/execution graph state, direct mutation, rebuilds, and source/sink
  roles;
- [`tests/pipelines/audio/test_qwen_omni_inprocess.py`](tests/pipelines/audio/test_qwen_omni_inprocess.py):
  lifecycle expansion, owner validation, several consumers/payloads, and
  global assembler insertion;
- [`tests/stages/test_payload_lifecycle.py`](tests/stages/test_payload_lifecycle.py):
  admission, leases, stores, namespaces, bulk actor methods, nested release,
  read errors, and cleanup;
- [`tests/pipeline/test_payload_refs.py`](tests/pipeline/test_payload_refs.py)
  and [`tests/pipeline/test_prefetch.py`](tests/pipeline/test_prefetch.py):
  actor-grouped resolution, order, byte bounds, and one-ahead behavior;
- [`tests/stages/audio/test_model_input_segmentation.py`](tests/stages/audio/test_model_input_segmentation.py):
  validation and exact/over-boundary production durations;
- [`tests/stages/audio/test_global_segment_assembler.py`](tests/stages/audio/test_global_segment_assembler.py):
  parent restoration, strict collisions, ordered merges, overwrite/drop,
  tombstones, spill, and source order;
- [`tests/stages/audio/inference/test_asr_stage.py`](tests/stages/audio/inference/test_asr_stage.py):
  refs, metadata-first planning, parent-cache reuse, one-call prefetch,
  segmentation, ordering, and metrics;
- [`tests/stages/audio/inference/test_batch_policy.py`](tests/stages/audio/inference/test_batch_policy.py):
  bucket/cost caps, adapter calls, and index realignment;
- [`tests/backends/ray_data/test_utils.py`](tests/backends/ray_data/test_utils.py)
  and [`tests/backends/xenna/test_executor.py`](tests/backends/xenna/test_executor.py):
  unchanged sizing semantics and stage-spec construction.

## 15. Reviewer File Map

| Concern | Primary files |
| --- | --- |
| pipeline graph | `pipeline/pipeline.py`, `pipeline/payload_lifecycle.py` |
| payload handles/prefetch | `pipeline/payload_refs.py`, `pipeline/prefetch.py` |
| actor state/materialize/release | `stages/payload_lifecycle.py` |
| global planner/assembler/writer | `stages/audio/common.py` |
| local segment decode | `stages/audio/io/audio_file_reader.py` |
| model-input segmentation | `stages/audio/model_input_segmentation.py` |
| ASR and batch policy | `stages/audio/inference/asr/stage.py`, `batch_policy.py`, `bucketed_stage.py` |
| Qwen model path | `models/asr/base.py`, `models/asr/qwen_omni.py`, `models/vllm_model.py` |
| backend execution | `backends/base.py`, `backends/ray_data/*`, `backends/xenna/*` |
| task contracts | `tasks/tasks.py`, `tasks/task_terminals.py`, `tasks/sentinels.py` |
| observability | `backends/perf_identity.py`, `utils/gpu_sampler.py`, `utils/pipeline_hardware_sampler.py`, `stages/audio/metrics/*` |
| output policy | `stages/audio/io/manifest_writer_utils.py`, `stages/audio/common.py` |
| Hydra entrypoint | `pipelines/audio/qwen_omni_inprocess.py` |

## 16. Core Invariants To Verify

1. Global planning reads metadata only and never decodes waveform bytes.
2. Each emitted segment is no longer than `max_inference_duration_s`.
3. Segment rows carry every configured consumer input and only those parent
   fields plus planner metadata.
4. Each segment file interval is decoded once by the materializer.
5. Payload actor RAM and admission bytes remain owned until explicit release.
6. Bulk resolution preserves ref order and one-ahead prefetch remains
   byte-bounded.
7. Every consumer remains a separate backend-visible stage.
8. Ray Data and Xenna retain their stage resource and worker contracts.
9. Segment outputs are merged strictly in index order.
10. A dropped segment produces explicit drop markers in configured merged
    outputs, never an incomplete apparently successful result.
11. Parent rows leave assembly in original source order.
12. Output rows contain neither waveform tensors, payload refs, nor planner
    internals.
13. Extended metrics and pipeline hardware sampling are opt-in outside the Qwen
    entrypoint and observational only.
