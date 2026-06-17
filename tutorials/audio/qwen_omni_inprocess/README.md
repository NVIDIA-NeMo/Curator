# Qwen-Omni In-Process ASR Pipeline

This tutorial runs the first Granary v2 inference stage with Qwen3-Omni inside a NeMo Curator audio pipeline.

## Overview

The pipeline reads a Granary-style NeMo tarred audio data config, decodes audio in memory, runs Qwen3-Omni text generation with vLLM, and writes sharded JSONL outputs for downstream audio processors.

- **Input**: Granary YAML data config with `nemo_tarred` corpora
- **Reader**: Streams local NeMo tar shards and decodes waveforms in memory
- **Inference**: Runs Qwen3-Omni thinker-only audio-to-text generation through in-process vLLM. The Curator-side glue lives in the generic `ASRStage` (input validation, ISO-code -> language-name lookup, backend duration-bucketing hook, stage-side pre-slicing for clips longer than `max_inference_duration_s`, stitch-back, metrics); the Qwen-Omni-specific vLLM setup, prompt formatting, and two-turn generation live in `QwenOmniASRAdapter`. Swap models by changing the single `adapter_target:` line in the YAML; tweak the bucket shape via `batch_policy:`. `ASRStage` itself subclasses `BucketedInferenceStage` (`nemo_curator/stages/audio/inference/`), so the dispatch -> reassemble loop is shared with any other audio GPU inference stage rather than re-coded per model.
- **Output**: Writes per-shard manifests, `.done` markers, an optional final manifest, and an aggregate `perf_summary.json` (see [Performance Summary](#performance-summary))

### Pipeline Flow

```
+----------------------+    +----------------------+    +----------------------+    +----------------------+
| Granary data config  | -> | NeMo tarred reader   | -> | Qwen-Omni inference  | -> | Sharded manifest     |
|  + local tar shards  |    |  waveform in memory  |    |  vLLM thinker path   |    | writer + perf stats  |
+----------------------+    +----------------------+    +----------------------+    +----------------------+
  data_config.yaml             AudioTask batches            qwen3_prediction_s1        output JSONL shards
```

Curator owns the reader, model invocation, stage metrics, and output manifests. External launchers are responsible for staging remote data, creating GPU resources, selecting job sizes, and collecting artifacts.

## Installation

From the Curator repository root:

```bash
uv sync --extra audio_cuda12
source .venv/bin/activate
```

The `audio_cuda12` extra includes the audio stack, vLLM, and `qwen-omni-utils`. Qwen3-Omni is a large GPU model; use a node with enough GPU memory for the configured tensor parallel size.

## Data Requirements

The tutorial expects a Granary-style YAML file whose corpora use NeMo tarred audio paths:

```yaml
- input_cfg:
    - corpus: mmlpc
      type: nemo_tarred
      manifest_filepath: /data/manifests/manifest__OP_0..7_CL_.json
      tarred_audio_filepaths: /data/audio/audio__OP_0..7_CL_.tar
```

The `__OP_0..N_CL_` pattern expands into one manifest/tar pair per shard. Paths should be local by the time Curator runs. Remote object-store download, credentials, and node-local staging should happen before invoking this tutorial.

Each manifest line should contain at least:

```json
{
  "audio_filepath": "audio_0.wav",
  "duration": 12.4,
  "source_lang": "en"
}
```

The `audio_filepath` value may be an exact tar member name, a relative path, or a path whose basename matches the tar member. If basenames are duplicated within one manifest, exact or relative matches are used and ambiguous basename matches are ignored.

## Quick Start

Run from the Curator repository root:

```bash
python tutorials/audio/qwen_omni_inprocess/main.py \
  --config-path tutorials/audio/qwen_omni_inprocess \
  --config-name qwen_omni_inprocess \
  input_manifest=/path/to/data_config.yaml \
  workspace_dir=./qwen_omni_output
```

For a small smoke run, cap the number of utterances per shard:

```bash
python tutorials/audio/qwen_omni_inprocess/main.py \
  --config-path tutorials/audio/qwen_omni_inprocess \
  --config-name qwen_omni_inprocess \
  input_manifest=/path/to/data_config.yaml \
  workspace_dir=./qwen_omni_output \
  max_utterances_per_shard=8
```

The full Hydra config is logged with secret-like values redacted. Model weights
are downloaded on the Ray workers (once per node via `ASRStage.setup_on_node` ->
adapter `prefetch_weights`). For gated checkpoints, set `HF_TOKEN`/`HF_HOME` in
the worker environment (cluster env or the executor `runtime_env`); the driver
process does not handle Hugging Face credentials.

## Backend

The pipeline defaults to the **Ray Data** executor. To run on Xenna instead,
override `backend=xenna` (Xenna additionally honors `execution_mode` and the
autoscale knobs).

> The GPU-stage worker pin is set in the YAML (`asr_num_workers`, cluster-wide)
> and is honored by every backend as-is — no extra override is needed. See
> [Worker allocation](#worker-allocation-pin-the-gpu-stage-autoscale-the-cheap-stages)
> for how the default (`4`) is derived and when to switch to the per-node knob.

## Configuration

All parameters are defined in `qwen_omni_inprocess.yaml`. Override values from the command line:

```bash
python tutorials/audio/qwen_omni_inprocess/main.py \
  --config-path tutorials/audio/qwen_omni_inprocess \
  --config-name qwen_omni_inprocess \
  input_manifest=/path/to/data_config.yaml \
  workspace_dir=/work/qwen \
  language_short=en \
  max_segment_length=40 \
  candidate_batch_size=32 \
  tensor_parallel_size=2 \
  omni_resource_gpus=2.0
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_manifest` | Granary YAML data config path | Required |
| `data_config` | Data config consumed by the reader | `${input_manifest}` |
| `workspace_dir` | Output workspace root | `/local/qwen_omni_workspace` |
| `output_dir` | Directory for sharded outputs and `perf_summary.json` | `${workspace_dir}` |
| `final_manifest` | Optional combined JSONL written by the writer | `${workspace_dir}/output.jsonl` |
| `language_short` | Default language code/name when an input row lacks `source_lang` | `en` |
| `max_segment_length` | Maximum manifest duration to emit from the reader, in seconds | `40` |
| `backend` | Execution backend (`ray_data` or `xenna`) | `ray_data` |
| `execution_mode` | Xenna execution mode (ignored by `ray_data`) | `streaming` |
| `autoscale_interval_s` | Xenna autoscale interval | `30` |
| `prefetch_fail_on_error` | Fail before pipeline execution if model prefetch fails | `true` |
| `stages_to_run` | Stage selector; `all`, comma-separated names, or list | `all` |
| `stages_to_skip` | Stage selector to skip | `[]` |
| `corpus` | Optional corpus filter for the data config | `null` |
| `max_utterances_per_shard` | Debug/smoke cap per tar shard | `null` |
| `source_lang_key` | Manifest key for per-row language | `source_lang` |
| `duration_key` | Manifest key for utterance duration | `duration` |
| `adapter_target` | Tier-1 swap line; fully-qualified ASR adapter class path | `nemo_curator.models.asr.QwenOmniASRAdapter` |
| `pred_text_key` | Output key for the Turn-1 transcription | `qwen3_prediction_s1` |
| `disfluency_text_key` | Output key for the optional Turn-2 (disfluency) transcription; set to `null` to disable | `qwen3_prediction_s2` |
| `model_id` | Hugging Face model identifier | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| `revision` | Optional Hugging Face revision to pin | `null` |
| `ideal_inference_segment_s` | Model per-turn audio cap; also anchors the bucket shape under `batch_policy`. | `2400` (40 min) |
| `max_inference_duration_s` | Stage-level chunking ceiling. Clips longer than this are pre-sliced into contiguous sub-chunks before inference and stitched back per parent task. | `2400` (= `ideal_inference_segment_s`) |
| `ml_prompt` | Default prompt sent with audio | `Transcribe the audio.` |
| `ml_prompt_file` | File containing the default prompt | `null` |
| `en_prompt_file` | File containing an English-specific prompt | `null` |
| `followup_prompt` | Optional second-turn prompt | `null` |
| `followup_prompt_file` | File containing the optional second-turn prompt | `null` |
| `system_prompt` | Optional system prompt | `null` |
| `tensor_parallel_size` | vLLM tensor parallel size | `2` |
| `omni_resource_gpus` | GPU resources reserved for the Qwen stage | `2.0` |
| `asr_num_workers` | Tier-1, adapter-agnostic. Hard-pin the ASR GPU stage to N workers cluster-wide (honored by **all** backends: Xenna streaming/batch and Ray Data). `= floor(total_gpus / resources.gpus)`; with `omni_resource_gpus=2.0` on 8 GPUs that is `4`. Mutually exclusive with `asr_num_workers_per_node`. `null` = autoscale. | `4` |
| `asr_num_workers_per_node` | Tier-1, adapter-agnostic. Hard-pin per node (Xenna only; Ray Data has no per-node primitive). Use instead of `asr_num_workers` on Xenna to scale with node count. `null` = use the cluster-wide knob / autoscale. | `null` |
| `candidate_batch_size` | Candidate/fallback row window for the Qwen stage. With duration-aware bucketing enabled, `batch_policy` caps define GPU dispatch width. | `32` |
| `max_output_tokens` | Maximum generated tokens per request | `256` |
| `max_model_len` | vLLM maximum model length | `4096` |
| `max_num_batched_tokens` | Optional vLLM scheduler / multimodal encoder-cache budget. Leave `null` for normal short chunks; raise with `max_model_len` for 40-50 minute single-audio probes. | `null` |
| `max_num_seqs` | vLLM maximum concurrent sequences | `16` |
| `gpu_memory_utilization` | vLLM GPU memory utilization fraction | `0.90` |
| `prep_workers` | Thread workers for audio preprocessing | `16` |
| `enable_prefix_caching` | vLLM prefix-cache toggle. Repeating prompts (system + user + follow-up) benefit when enabled. | `true` |
| `prefix_caching_hash_algo` | Hash algorithm backing the prefix cache (`xxhash` or `sha256`). | `xxhash` |
| `limit_mm_per_prompt_audio` | vLLM multi-modal audio-token cap per prompt. Two-turn flow needs `≥ 2`. | `2` |
| `seed` | vLLM scheduler / sampling seed. | `1234` |
| `keep_waveform` | Keep waveform arrays after inference so downstream stages can reuse the in-memory buffer; the writer still drops arrays from JSONL. | `true` |
| `cpu_batch_size` | Writer batch size | `64` |
| `write_perf_stats` | When `true`, the writer aggregates per-task stage perf and maintains `perf_summary.json`. Set `false` to disable perf output (manifests only). | `true` (writer stage default; add under `sharded_manifest_writer` in YAML if overriding) |

#### Duration-aware bucketed batching

The user-facing switch and policy shape live under
`duration_aware_bucketing`:

```yaml
duration_aware_bucketing:
  enabled: true
  strategy: duration_bucketed
  buckets_sec: [0, 600, 1200, 2400]
  max_items_per_batch_by_bucket: [32, 16, 8, 4]
  max_audio_sec_per_batch: 2400
  prebatching_window_size: null
  flush_interval_ms: 250
```

The `qwen_omni` stage converts that block into its `BatchPolicy`. `enabled:
true` lets supporting executors form cost-aware candidate batches before workers
receive them. `prebatching_window_size: null` preserves the default planner
window of `sum(max_items_per_batch_by_bucket)`; set an integer to bound the
candidate window explicitly for larger future policies. ASR uses a chunk-aware
work-unit planner: it materializes chunk tasks once, then the shared
`BatchPolicy` queue scheduler buckets those chunks by duration. Generic planned
batches are submitted longest-first to reduce multi-worker tail time. If a
backend hands a mixed candidate window to a worker, the base adapter runs the
same scheduler path and sends each planned chunk bucket directly to one model
dispatch without re-bucketing.

Bucket edges are anchored on `ideal_inference_segment_s`: short clips fire in
larger batches, long clips fire alone or near-alone. Set
`duration_aware_bucketing.enabled: false` to disable the scheduler and local
bucket partitioning so backend batches pass through like current main. Legacy
configs can still set `batch_policy: null` on the stage for the same
no-bucketing behavior. `flush_interval_ms` is consumed by the shared bucket
queue scheduler; finite candidate windows drain all remaining buckets at window
end.

### Stage Selectors

The explicit `stages` list is the source of truth for the graph:

- `reader`: `NemoTarredAudioReader`
- `qwen_omni`: `ASRStage` with `adapter_target: nemo_curator.models.asr.QwenOmniASRAdapter`
- `sharded_manifest_writer`: `ShardedManifestWriterStage`

Run only selected stages when debugging:

```bash
python tutorials/audio/qwen_omni_inprocess/main.py \
  --config-path tutorials/audio/qwen_omni_inprocess \
  --config-name qwen_omni_inprocess \
  input_manifest=/path/to/data_config.yaml \
  stages_to_run=reader
```

Unknown stage names fail fast with the available selectors.

## Worker allocation: pin the GPU stage, autoscale the cheap stages

**Autoscale optimizes steady-state throughput, not cold-start latency.** Xenna's
streaming autoscaler (and Ray Data's actor autoscaler) start every unpinned stage
at **1 worker** and only grow it once the stage has produced enough speed
measurements to be judged the bottleneck. For a cheap CPU stage that finishes
thousands of items per second, that ramp is instant and harmless. For an
**expensive GPU stage**, it is the opposite:

- The stage sits at 1 GPU worker until it completes enough batches to register a
  speed estimate (Xenna needs `autoscale_speed_estimation_min_data_points` items
  within the estimation window), so most of your GPUs idle during warm-up.
- When the autoscaler finally scales out, every newly spawned worker pays the
  full **model-load tax** (weights → VRAM, vLLM engine init) before it does any
  useful work — exactly when you wanted them already hot.

### Rule for this pipeline (and any GPU stage)

> **Hard-pin the expensive GPU stage's worker count; let autoscale handle only
> the cheap, fast-to-measure stages.**

These pin knobs are **Tier-1 and adapter-agnostic**: they live on the generic
`ASRStage` (as `xenna_num_workers_per_node` / `xenna_num_workers`), not in the
Qwen-Omni `adapter_kwargs`, so they keep working unchanged when you swap
`adapter_target` to any other ASR model. Only the *value* moves with the model,
because it derives from the adapter's per-worker GPU footprint
(`resources.gpus`): the pin is `floor(gpus_per_node / resources.gpus)`. The
Qwen-Omni adapter uses `omni_resource_gpus=2.0` (tensor-parallel 2), so on a
4-GPU node the pin is `4 / 2.0 = 2`; swap in a 1-GPU ASR adapter and the same
formula gives `4`.

The `qwen_omni` stage is pinned out of the box via `asr_num_workers` (cluster-wide,
default `4` for the reference 2-node × 4-GPU shape). With the stage pinned, the
scheduler brings up all GPU workers on the first scheduling pass — no measurement
gate, models load in parallel up front. Set it to `null` only if you deliberately
want autoscaling back.

How the pin reaches each backend (verified in the stage code):

| Knob | Xenna | Ray Data |
|------|-------|----------|
| `asr_num_workers_per_node` → `ASRStage.xenna_stage_spec()["num_workers_per_node"]` | N workers **per node** | not applicable (no per-node primitive) → autoscales |
| `asr_num_workers` → `ASRStage.xenna_stage_spec()["num_workers"]` **and** `ASRStage.num_workers()` | N workers **cluster-wide** | N workers **cluster-wide** (actor path) |

The default uses `asr_num_workers` (cluster-wide) because it is the only knob
honored by **all** backends — including Ray Data, which has no per-node primitive
— so one YAML value pins every backend with no per-backend launcher override.
Switch to `asr_num_workers_per_node` (and set `asr_num_workers: null`) only on
Xenna when you want the pin to scale with node count automatically.

The cheap stages follow the same principle from the opposite side: the tar
reader and discovery are pinned to 1 worker per node (bounded memory with
`keep_waveform=true`), and the writer is a single actor for serialized I/O — none
of them rely on the autoscale ramp.

> **WARNING:** a pin the cluster cannot satisfy makes Xenna's scheduler panic
> (it treats manual worker requests as hard constraints). Keep
> `asr_num_workers_per_node ≤ floor(gpus_per_node / resources.gpus)`.

### Swapping the ASR model (adapter) changes the GPU/worker math

The `ASRStage` mechanism is adapter-agnostic, but **the right values are
model-dependent — you must re-tune them when you change `adapter_target`.** A
different checkpoint has a different size and tensor-parallel degree, which
changes how many GPUs each actor needs and therefore how many actors fit:

| When you swap to… | Per-actor GPU footprint (`resources.gpus`) | Worker pin `floor(gpus_per_node / resources.gpus)` |
|---|---|---|
| **A larger model** (more tensor parallelism) | **more** GPUs per actor (e.g. `tp=4` → `gpus≈4`) | **fewer** actors per node (4-GPU node → `1`) |
| **A smaller model** (less tensor parallelism) | **fewer** GPUs per actor (e.g. `tp=1` → `gpus≈1`) | **more** actors per node (4-GPU node → `4`) |

So the params that move together on an adapter swap are:

1. **Tier-2 `adapter_kwargs`** — rewrite the whole block for the new model
   (prompts, `tensor_parallel_size`, vLLM knobs, etc.).
2. **Per-actor GPU footprint** — set the stage's `resources.gpus` (this tutorial
   exposes it as `omni_resource_gpus`) to match the new model's parallelism. A
   **smaller model needs fewer GPUs per actor**; a larger one needs more.
3. **Worker pin** — recompute `asr_num_workers_per_node =
   floor(gpus_per_node / resources.gpus)`. Because step 2 changed the
   denominator, the pin changes even though the knob itself did not.

Worked example on a 4-GPU node: Qwen3-Omni runs `tensor_parallel_size=2` →
`omni_resource_gpus=2.0` → pin `4 / 2.0 = 2`. Swap in a single-GPU ASR model →
set `resources.gpus=1.0` → pin `4 / 1.0 = 4` (four actors, each its own model
copy). The Tier-1 stage knobs (`pred_text_key`, chunking, `batch_policy`, the pin
knobs) keep working unchanged; only the values above are re-derived.

## Output Format

The writer mirrors input shard keys under `output_dir`:

```
qwen_omni_output/
+-- output.jsonl
+-- perf_summary.json
+-- <corpus>/.../manifest_0.jsonl
    <corpus>/.../manifest_0.jsonl.done
```

Each output row keeps the original manifest metadata and adds:

| Field | Description |
|-------|-------------|
| `sample_rate` | Native decoded audio sample rate |
| `sampling_rate` | Alias for compatibility with existing Granary scripts |
| `num_channels` | Number of decoded channels before mono conversion |
| `corpus` | Corpus name from the data config |
| `qwen3_prediction_s1` | Qwen-Omni first-turn transcription |
| `qwen3_prediction_s2` | Optional second-turn output when follow-up prompt is enabled |
| `_skip_me` | Set to `empty_audio` when preprocessing skips an utterance |

Waveform arrays are not written to JSONL output, even when `keep_waveform=true`.

## Performance Summary

Metrics are **already implemented** in this pipeline — you do not need a
separate “metrics worker” or per-GPU file writers. Every stage (CPU and GPU)
is timed by the backend; a **single CPU writer actor** (`num_workers=1`)
serializes the results to disk.

### How metrics are collected (in memory)

For **every** stage in the graph — tar discovery, tar reader, Qwen inference,
and the manifest writer — the execution backend wraps each batch through
`BaseStageAdapter.process_batch` (`nemo_curator/backends/base.py`):

1. Times the batch (`StageTimer`: `process_time`, `actor_idle_time`,
   `num_items_processed`).
2. Merges any stage-specific counters the stage recorded via
   `_log_metrics({...})` / `_log_metric(...)` / `_time_metric(...)` into
   `StagePerfStats.custom_metrics`.
3. Stamps **stage identity** via `build_xenna_perf_identity()` /
   `build_ray_perf_identity()` (returns `WorkerPerfIdentity`, stamped on
   `WorkerMetadata` at backend setup):
   - **Scheduling join key:** `actor_id`, `node_id`, `gpu_id` (e.g. `node-0:0`).
     `gpu_id` is the within-run bucket for `per_gpu` aggregation — keep it stable
     even when pod IPs change between jobs.
   - **Cluster location (additive):** `physical_address` (`<pod_or_node_ip>:<gpu_indices>`),
     `pod_ip`, `hostname`, `gpu_indices` (all GPUs in the actor’s allocation, so
     `tp=2` shows `[0, 1]` not just the first slot), and optional `gpu_uuids`
     when CUDA is up at setup.
   CPU stages get full timing metrics; `gpu_id` stays empty and no `per_gpu`
   block is emitted for them.
4. Appends one `StagePerfStats` record onto each output task’s
   **`task._stage_perf`** list (a growing per-task chain as the task moves
   through the pipeline).

**CPU stages are not excluded.** Reader and discovery stages contribute
`stages[<name>]` blocks in `perf_summary.json` with invocation counts,
process/idle times, and any custom metrics they emit (e.g. `bytes_loaded`,
`total_items_emitted`). Only GPU-specific breakdown fields (`gpu_ids`,
`per_gpu`) are omitted for CPU stages.

**GPU stages never write metric files directly.** They only extend
`task._stage_perf`. File I/O is centralized in the writer (below).

### How metrics are written (to disk)

`ShardedManifestWriterStage` is the **only** stage that writes perf files.
It runs as a **single worker** (`num_workers=1` / `xenna_stage_spec`) so all
aggregate I/O is serialized — no cross-worker file races.

With `write_perf_stats=true` (the default):

| File | Granularity | When written |
|------|-------------|--------------|
| `perf_summary.json` | **Aggregate** — rolled-up totals, percentiles, per-actor breakdown, `pipeline_throughput` | Refreshed when a shard completes (`.done` marker) **and** again at writer `teardown()` |

Per-task `StagePerfStats` are aggregated in memory only; there is no per-shard
sidecar file.

After the pipeline finishes, `main.py` reads `perf_summary.json` (if present),
adds top-level `pipeline_duration_s` (whole-run wall clock), and rewrites the
file.

Downstream upload tooling may copy **one** rank’s `perf_summary.json` verbatim
for cluster-wide reporting (no re-aggregation in Curator).

### Adding custom metrics in a stage

In any stage’s `process` / `process_batch` implementation:

```python
self._log_metrics({"inference_time_s": elapsed, "output_tokens": float(n_tokens)})
```

Counters flow: `StagePerfStats.custom_metrics` →
summed under `stages[<name>].custom_metrics_sum` in `perf_summary.json`.
To add a **new top-level summary field** consumed by multiple stages, extend
`AudioStageMetrics` in `nemo_curator/stages/audio/metrics/performance.py`.

Disable all perf file output (manifests only):

```yaml
- stage_id: sharded_manifest_writer
  _target_: nemo_curator.stages.audio.io.sharded_manifest_writer.ShardedManifestWriterStage
  write_perf_stats: false
  ...
```

Full contract (dedup, identity fields, per-GPU breakdown, backend call chains):
`nemo_curator/stages/audio/README.md` § “Performance metrics”.

### Fields in `perf_summary.json`

**Pipeline totals:** `total_utterances`, `total_audio_hours`, `writer_wall_time_s`,
`pipeline_audio_s_per_wall_s`, `pipeline_utterances_per_wall_s`, plus top-level
`pipeline_throughput` (`audio_hours_per_wallclock_hour`, union of GPU IDs).

**Per-stage blocks** (e.g. `QwenOmni_inference`, tar reader, discovery, writer):
invocation counts, process/idle time percentiles, throughput ratios,
`custom_metrics_sum`, and for GPU stages — `gpu_ids`, `gpu_count`, `actor_count`,
`per_gpu` (per-actor items processed, audio hours, batch-size / queue-wait p50/p95,
plus cluster location: `physical_address`, `pod_ip`, `hostname`, `gpu_indices`,
optional `gpu_uuids`).
Scheduling identity (`actor_id`, `node_id`, `gpu_id`) is carried on each
`StagePerfStats` and drives dedup; each backend resolves `WorkerPerfIdentity` once
at worker setup via `backends/perf_identity.py` (Xenna:
`WorkerMetadata.allocation.gpus[0].index` for `gpu_id`, all allocation indices
for `gpu_indices`, `POD_IP` for `physical_address`; Ray Data / Actor Pool:
`ray.get_gpu_ids()` — no cross-backend fallback chain).

**Validation:** compare `perf_summary.json` across runs on shared throughput
fields and verify `manifest_*.jsonl` rows keyed on `audio_filepath` stay aligned
(vLLM nondeterminism may differ on a small fraction of prediction text).

Cluster bring-up, data staging, and remote upload timings are outside Curator’s
writer wall clock (`writer_wall_time_s`).

## Troubleshooting

### Model download fails before the pipeline starts

By default, `prefetch_fail_on_error=true` fails fast if Hugging Face model access or credentials are missing. Set it to `false` for local development if you want worker setup to retry later:

```bash
prefetch_fail_on_error=false
```

### No utterances are emitted

Check that the manifest `audio_filepath` values match tar members by exact name, relative path, or basename. Also check `max_segment_length`; rows with `duration` greater than that value are filtered before decoding.

### Ray worker SIGSEGV during model loading

See the parent [audio tutorial README](../README.md#sigsegv-in-ray-stageworker-during-model-loading) for the OpenTelemetry workaround used by other audio GPU pipelines.
