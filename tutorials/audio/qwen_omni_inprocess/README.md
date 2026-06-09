# Qwen-Omni In-Process ASR Pipeline

This tutorial runs the first Granary v2 inference stage with Qwen3-Omni inside a NeMo Curator audio pipeline.

## Overview

The pipeline reads a Granary-style NeMo tarred audio data config, decodes audio in memory, runs Qwen3-Omni text generation with vLLM, and writes sharded JSONL outputs for downstream audio processors.

- **Input**: Granary YAML data config with `nemo_tarred` corpora
- **Reader**: Streams local NeMo tar shards and decodes waveforms in memory
- **Inference**: Runs Qwen3-Omni thinker-only audio-to-text generation through in-process vLLM. The Curator-side glue lives in the generic `ASRStage` (input validation, ISO-code -> language-name lookup, stage-side pre-slicing for clips longer than `max_inference_duration_s`, optional duration-bucketed batching, stitch-back, metrics); the Qwen-Omni-specific vLLM setup, prompt formatting, and two-turn generation live in `QwenOmniASRAdapter`. Swap models by changing the single `adapter_target:` line in the YAML; tweak the bucket shape via `batch_policy:`. `ASRStage` itself subclasses the modality-agnostic `BucketedInferenceStage` (`nemo_curator/stages/inference/`), so the bucketize -> dispatch -> reassemble loop is shared with any other GPU inference stage rather than re-coded per model.
- **Output**: Writes per-shard manifests, `.done` markers, an optional final manifest, per-task `{shard}_perf.jsonl`, and aggregate `perf_summary.json` (see [Performance Summary](#performance-summary))

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
  workspace_dir=./qwen_omni_output \
  hf_token=$HF_TOKEN
```

For a small smoke run, cap the number of utterances per shard:

```bash
python tutorials/audio/qwen_omni_inprocess/main.py \
  --config-path tutorials/audio/qwen_omni_inprocess \
  --config-name qwen_omni_inprocess \
  input_manifest=/path/to/data_config.yaml \
  workspace_dir=./qwen_omni_output \
  max_utterances_per_shard=8 \
  hf_token=$HF_TOKEN
```

The full Hydra config is logged with secret-like values redacted. The `hf_token` override is copied to `HF_TOKEN` before model prefetch.

## Choosing a Backend

The pipeline supports two execution backends. Override with `backend=`:

| Backend | Description | When to use |
|---------|-------------|-------------|
| `xenna` | Default streaming executor. Supports explicit stage resource settings and autoscaling controls. | Multi-GPU throughput runs and launcher-driven jobs. |
| `ray_data` | Ray Data executor. | Local development or environments where Ray Data is preferred. |

Example with Ray Data:

```bash
python tutorials/audio/qwen_omni_inprocess/main.py \
  --config-path tutorials/audio/qwen_omni_inprocess \
  --config-name qwen_omni_inprocess \
  input_manifest=/path/to/data_config.yaml \
  workspace_dir=./qwen_omni_output \
  backend=ray_data
```

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
  batch_size=32 \
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
| `hf_token` | Hugging Face token copied to `HF_TOKEN`; redacted in logs | `""` |
| `backend` | Execution backend | `xenna` |
| `execution_mode` | Xenna execution mode | `streaming` |
| `autoscale_interval_s` | Xenna autoscale interval | `30` |
| `prefetch_fail_on_error` | Fail before pipeline execution if model prefetch fails | `true` |
| `stages_to_run` | Stage selector; `all`, comma-separated names, or list | `all` |
| `stages_to_skip` | Stage selector to skip | `[]` |
| `corpus` | Optional corpus filter for the data config | `null` |
| `max_utterances_per_shard` | Debug/smoke cap per tar shard | `null` |
| `source_lang_key` | Manifest key for per-row language | `source_lang` |
| `duration_key` | Manifest key for utterance duration | `duration` |
| `adapter_target` | Tier-1 swap line; fully-qualified ASR adapter class path | `nemo_curator.adapters.asr.QwenOmniASRAdapter` |
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
| `batch_size` | Qwen stage batch size | `32` |
| `max_output_tokens` | Maximum generated tokens per request | `256` |
| `max_model_len` | vLLM maximum model length | `4096` |
| `max_num_seqs` | vLLM maximum concurrent sequences | `16` |
| `gpu_memory_utilization` | vLLM GPU memory utilization fraction | `0.90` |
| `prep_workers` | Thread workers for audio preprocessing | `16` |
| `enable_prefix_caching` | vLLM prefix-cache toggle. Repeating prompts (system + user + follow-up) benefit when enabled. | `true` |
| `prefix_caching_hash_algo` | Hash algorithm backing the prefix cache (`xxhash` or `sha256`). | `xxhash` |
| `limit_mm_per_prompt_audio` | vLLM multi-modal audio-token cap per prompt. Two-turn flow needs `≥ 2`. | `2` |
| `seed` | vLLM scheduler / sampling seed. | `1234` |
| `keep_waveform` | Keep waveform arrays after inference so downstream stages can reuse the in-memory buffer; the writer still drops arrays from JSONL. | `true` |
| `cpu_batch_size` | Writer batch size | `64` |
| `write_perf_stats` | When `true`, the writer appends per-task `{shard}_perf.jsonl` lines and maintains `perf_summary.json`. Set `false` to disable all perf file output (manifests only). | `true` (writer stage default; add under `sharded_manifest_writer` in YAML if overriding) |

#### Duration-aware bucketed batching (`batch_policy`)

The `qwen_omni` stage declares an optional `batch_policy` block:

```yaml
batch_policy:
  _target_: nemo_curator.stages.inference.batch_policy.BatchPolicy
  strategy: duration_bucketed
  buckets_sec: [0, 600, 1200, 2400]
  max_items_per_batch_by_bucket: [32, 16, 8, 4]
  max_audio_sec_per_batch: 2400
  flush_interval_ms: 250
```

`BatchPolicy` and the `run_bucketed` helper it drives live in the
modality-agnostic `nemo_curator.stages.inference.batch_policy` module so
any GPU inference stage can reuse them.

When set, every `process_batch` invocation internally re-partitions its
items into bucket-respecting sub-batches before dispatching the adapter,
so a single vLLM call never mixes a 40-minute sub-chunk with 5-second
sub-chunks. Bucket edges are anchored on `ideal_inference_segment_s`:
short clips fire in larger batches, long clips fire alone or near-alone.
Set `batch_policy: null` to disable bucketing (single adapter call per
`process_batch` over all items).

Cross-`process_batch` queueing with per-bucket queues and a flush timer
requires a Curator-framework scheduler hook and is a follow-up PR.
`flush_interval_ms` is recorded on the dataclass for forward-compat but
is not consumed by the stage today.

### Stage Selectors

The explicit `stages` list is the source of truth for the graph:

- `reader`: `NemoTarredAudioReader`
- `qwen_omni`: `ASRStage` with `adapter_target: nemo_curator.adapters.asr.QwenOmniASRAdapter`
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

## Output Format

The writer mirrors input shard keys under `output_dir`:

```
qwen_omni_output/
+-- output.jsonl
+-- perf_summary.json
+-- <corpus>/.../manifest_0.jsonl
    <corpus>/.../manifest_0.jsonl.done
    <corpus>/.../manifest_0_perf.jsonl
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
3. Stamps **stage identity** (`actor_id`, `node_id`, `gpu_id`) via
   `build_xenna_perf_identity()` / `build_ray_perf_identity()` (stamped on
   `WorkerMetadata` at backend setup). CPU stages get full timing metrics; `gpu_id`
   stays empty and no `per_gpu` block is emitted for them.
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
append and aggregate I/O is serialized — no cross-worker file races.

With `write_perf_stats=true` (the default):

| File | Granularity | When written |
|------|-------------|--------------|
| `{shard_key}_perf.jsonl` | **Per task** — one JSON line per utterance, containing the full `task._stage_perf` chain | Appended on every task that passes through the writer |
| `perf_summary.json` | **Aggregate** — rolled-up totals, percentiles, `per_gpu`, `pipeline_throughput` | Refreshed when a shard completes (`.done` marker) **and** again at writer `teardown()` |

After the pipeline finishes, `main.py` reads `perf_summary.json` (if present),
adds top-level `pipeline_duration_s` (whole-run wall clock), and rewrites the
file.

Downstream upload tooling may copy **one** rank’s `perf_summary.json` verbatim
for cluster-wide reporting; per-shard `*_perf.jsonl` detail files are
transported as-is (no re-aggregation in Curator).

### Adding custom metrics in a stage

In any stage’s `process` / `process_batch` implementation:

```python
self._log_metrics({"inference_time_s": elapsed, "output_tokens": float(n_tokens)})
```

Counters flow: `StagePerfStats.custom_metrics` → per-task `_perf.jsonl` →
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
`per_gpu` (per-actor items processed, audio hours, batch-size / queue-wait p50/p95).
Identity fields (`actor_id`, `node_id`, `gpu_id`) appear in per-task `_perf.jsonl`
and drive dedup; each backend resolves them once at worker setup via
`backends/perf_identity.py` (Xenna: `WorkerMetadata.allocation.gpus[0].index`;
Ray Data / Actor Pool: `ray.get_gpu_ids()` — no cross-backend fallback chain).

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
