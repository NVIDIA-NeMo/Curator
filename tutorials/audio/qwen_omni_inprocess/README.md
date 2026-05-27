# Qwen-Omni In-Process ASR Pipeline

This tutorial runs the first Granary v2 inference stage with Qwen3-Omni inside a NeMo Curator audio pipeline.

## Overview

The pipeline reads a Granary-style NeMo tarred audio data config, decodes audio in memory, runs Qwen3-Omni text generation with vLLM, and writes sharded JSONL outputs for downstream audio processors.

- **Input**: Granary YAML data config with `nemo_tarred` corpora
- **Reader**: Streams local NeMo tar shards and decodes waveforms in memory
- **Inference**: Runs Qwen3-Omni thinker-only audio-to-text generation through in-process vLLM
- **Output**: Writes per-shard manifests, `.done` markers, an optional final manifest, and `perf_summary.json`

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
| `model_id` | Hugging Face model identifier | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
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
| `keep_waveform` | Keep waveform arrays after inference; writer still drops arrays from JSONL | `false` |
| `reader_num_workers` | Optional total reader worker count | `null` |
| `reader_num_workers_per_node` | Optional reader worker count per node | `null` |
| `omni_num_workers` | Optional Qwen stage worker count | `null` |
| `cpu_batch_size` | Writer batch size | `64` |

### Stage Selectors

The explicit `stages` list is the source of truth for the graph:

- `reader`: `NemoTarredAudioReader`
- `qwen_omni`: `InferenceQwenOmniStage`
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

`perf_summary.json` contains pipeline totals and per-stage metric aggregates, including:

- utterance and audio-hour totals
- reader shard counts, decode time, filtered duration counts
- Qwen inference time, output token counts, skipped utterances
- writer process/write timing

These metrics are intended for Curator-level throughput analysis. Cluster creation, data staging, and remote upload timings belong to the external launcher.

## Troubleshooting

### Model download fails before the pipeline starts

By default, `prefetch_fail_on_error=true` fails fast if Hugging Face model access or credentials are missing. Set it to `false` for local development if you want worker setup to retry later:

```bash
prefetch_fail_on_error=false
```

### vLLM or Qwen utilities are missing

Install the CUDA audio extra:

```bash
uv sync --extra audio_cuda12
```

or:

```bash
pip install -e ".[audio_cuda12]"
```

### No utterances are emitted

Check that the manifest `audio_filepath` values match tar members by exact name, relative path, or basename. Also check `max_segment_length`; rows with `duration` greater than that value are filtered before decoding.

### Ray worker SIGSEGV during model loading

See the parent [audio tutorial README](../README.md#sigsegv-in-ray-stageworker-during-model-loading) for the OpenTelemetry workaround used by other audio GPU pipelines.
