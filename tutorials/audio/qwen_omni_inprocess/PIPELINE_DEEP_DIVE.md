# Granary v2 Qwen-Omni In-Process Pipeline Deep Dive

This document describes the current Curator-side implementation of the
Granary v2 Qwen-Omni in-process audio pipeline. It focuses on the code as
it exists now, the runtime contract expected by NvLLMOps/Kratos, and the
knobs that matter for correctness, scaling, and throughput analysis.

The important boundary is simple:

- Curator owns data processing, model inference, text filtering, output
  writing, and per-stage performance accounting.
- NvLLMOps owns infrastructure: Swift/S3 staging, Kratos submission,
  Docker image selection, multi-node MPI/Ray wiring, output upload, and
  rank-level perf aggregation.

## File Map

```text
tutorials/audio/qwen_omni_inprocess/
|-- main.py
|-- qwen_omni_inprocess.yaml
`-- PIPELINE_DEEP_DIVE.md

nemo_curator/stages/audio/io/
|-- nemo_tarred_reader.py
`-- sharded_manifest_writer.py

nemo_curator/stages/audio/inference/
|-- qwen_omni.py
|-- qwen_asr.py
`-- sed.py

nemo_curator/stages/audio/postprocessing/
`-- sed_postprocessing.py

nemo_curator/stages/audio/text_filtering/
|-- initialize_fields.py
|-- disfluency_wer_guard.py
|-- whisper_hallucination.py
|-- select_best_prediction.py
|-- fasttext_lid.py
|-- regex_substitution.py
|-- abbreviation_concat.py
|-- pnc_restoration.py
|-- pnc_content_guard.py
`-- itn_restoration.py

nemo_curator/models/
|-- qwen_omni.py
|-- qwen_asr.py
`-- qwen_text_llm.py
```

## Invocation Contract

`main.py` is a Hydra entry point. NvLLMOps invokes it as:

```bash
python tutorials/audio/qwen_omni_inprocess/main.py \
  --config-path=<runtime_config_dir> \
  --config-name=<runtime_config_name> \
  workspace_dir=<local_output_dir> \
  output_dir=<local_output_dir> \
  input_manifest=<local_or_staged_data_config.yaml> \
  data_config=<local_or_staged_data_config.yaml> \
  language_short=<lang> \
  max_segment_length=<seconds>
```

The `input_manifest` name is retained for compatibility with older
Curator/NvLLMOps flows. For this native Granary v2 pipeline it points to a
NeMo-tarred YAML data config, not to a flat JSONL utterance manifest.
`data_config` is set to the same value so the pipeline can be run from
either the tutorial config or from a direct Python caller.

At startup `main.py` sets:

```text
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_LOGGING_LEVEL=ERROR
```

These defaults are set before vLLM is imported. `setdefault()` is used so
Kratos or a developer shell can intentionally override them.

If `hf_token` is present in the Hydra config, `main.py` exports it as
`HF_TOKEN` and defaults `HF_HOME` to `/tmp/hf_home`. The token is not part
of the pipeline logic; it is only used for HuggingFace downloads.

## Pipeline Assembly

`build_granary_v2_pipeline(cfg)` constructs a single Curator `Pipeline`
named `qwen_omni_inference`. The final stage order is:

```text
NemoTarredAudioReader
InitializeFieldsStage
[SEDInferenceStage]
[SEDPostprocessingStage]
InferenceQwenOmniStage
[DisfluencyWerGuardStage]
WhisperHallucinationStage for Qwen-Omni
[InferenceQwenASRStage]
[WhisperHallucinationStage for Qwen-ASR recovery]
SelectBestPredictionStage
FastTextLIDStage
RegexSubstitutionStage
AbbreviationConcatStage
[PnCRestorationStage]
[PnCContentGuardStage]
[ITNRestorationStage]
ShardedManifestWriterStage
```

Square brackets mean the stage is optional and controlled by config.

## Helper Functions

### `_read_file_or_str`

Prompt fields can be either literal text or a path to a prompt file.
`_read_file_or_str(value)` checks whether `value` is a readable local
file. If yes, it returns the stripped file contents. Otherwise it returns
the value unchanged. This is used for Omni, PnC, and ITN prompts.

### `_with_optional_gpu_resources`

This helper applies scheduler resource overrides such as
`omni_resource_gpus`, `asr_resource_gpus`, `pnc_resource_gpus`, and
`itn_resource_gpus`.

These values affect Xenna scheduling/accounting only. They do not change
how many CUDA devices vLLM uses. vLLM GPU topology is still controlled by
the model stage's tensor-parallel or model-specific settings, for example:

- `tensor_parallel_size` for Qwen-Omni
- `pnc_tensor_parallel_size` for PnC
- `itn_tensor_parallel_size` for ITN
- Qwen-ASR's vLLM cache knobs such as `asr_max_model_len`,
  `asr_gpu_memory_utilization`, and `asr_batch_size`

This distinction matters for single-node streaming experiments. A
fractional scheduler value can allow Xenna to co-schedule stages, but it
does not isolate real GPU memory if multiple vLLM engines initialize on
the same physical GPU.

## Data Reader

`NemoTarredAudioReader` is a composite stage made of:

1. `NemoTarShardDiscoveryStage`
2. `NemoTarShardReaderStage`

### Shard Discovery

`NemoTarShardDiscoveryStage` parses the Granary data config YAML and emits
one `FileGroupTask` per manifest/tar pair. It supports the NeMo
`_OP_<start>..<end>_CL_` pattern and expands it before pairing manifests
with tar files.

The shard key is derived from the manifest path starting at the corpus
directory component. For example:

```text
/tmp/work/mmlpc/aaftabv/granary_data_sample/manifests/manifest_0.json
```

with `corpus: mmlpc` becomes:

```text
mmlpc/aaftabv/granary_data_sample/manifests/manifest_0
```

This key is later used by the writer, so output paths mirror the input
manifest layout.

The discovery stage also implements restart behavior:

- It scans `output_dir` for `.jsonl.done` marker files.
- Completed shards are skipped.
- If a partial `.jsonl` exists without a `.done`, it is removed before
  reprocessing the shard.

It emits custom metrics such as `corpora_seen`, `shards_seen`,
`shards_emitted`, `shards_skipped_completed`, and `discovery_time_s`.

### Shard Reader

`NemoTarShardReaderStage` consumes one shard task and emits one
`AudioTask` per utterance. It can read local paths or `s3://` paths via
AIS using `lhotse.open_best`. S3 paths require `AIS_ENDPOINT` or an
explicit `s3_endpoint_url`. If `AIS_AUTHN_TOKEN` is set, it is passed as a
bearer token to the curl pipe.

For each tar member that has a manifest row:

1. The tar member bytes are read.
2. `soundfile` decodes them to a float32 waveform.
3. Multi-channel audio is averaged to mono.
4. The manifest row is copied into `task.data`.
5. `waveform`, `sample_rate`, `sampling_rate`, `num_channels`, and
   `corpus` are added.
6. `_shard_key` and `_shard_total` metadata are attached.

`max_utterances_per_shard` can cap the number of utterances emitted from
each shard. This is useful for debugging and throughput sweeps on small
subsets.

Reader scaling knobs:

- `reader_num_workers`: exact Xenna worker count for the shard reader.
- `reader_num_workers_per_node`: per-node worker count for the shard reader.

Reader metrics include manifest entries, tar members seen, decoded
utterances, corrupt audio count, decoded audio seconds, waveform bytes,
manifest read time, tar open time, decode time, and total reader time.

## Initialize Fields

`InitializeFieldsStage` normalizes the row shape before model and filtering
stages run. It ensures downstream keys such as skip flags and prediction
fields exist consistently, which lets later stages update rather than
reconstruct row state.

## Optional Sound Event Detection

SED is enabled when `sed_checkpoint` is set.

`SEDInferenceStage` runs a sound event model over the waveform. It uses
`sed_model_type`, `sed_batch_size`, `sed_num_workers`, and
`sed_gpu_memory_gb`. `SEDPostprocessingStage` then converts model output
into speech segments and event metadata using `sed_speech_threshold`,
`sed_min_duration`, `sed_merge_gap`, and `sed_subcategories`.

SED is intentionally optional so the default Granary v2 ASR cleanup path
does not require additional model checkpoints.

## Qwen-Omni Inference

`InferenceQwenOmniStage` wraps `nemo_curator.models.qwen_omni.QwenOmni`.
It expects each task to contain `waveform` and `sample_rate`.

Important config:

- `model_id`
- `tensor_parallel_size`
- `batch_size`
- `max_output_tokens`
- `max_model_len`
- `max_num_seqs`
- `gpu_memory_utilization`
- `prep_workers`
- `omni_num_workers`
- `omni_resource_gpus`

`tensor_parallel_size` sets the stage's default `Resources(gpus=tp)` and
is also passed to vLLM. `omni_resource_gpus`, when set, overrides only the
Xenna scheduler resource value. It does not change vLLM tensor
parallelism.

The model wrapper:

- loads vLLM with `trust_remote_code=True`
- enables prefix caching with `xxhash`
- sets `limit_mm_per_prompt` for audio/image/video support
- loads `Qwen3OmniMoeProcessor`
- creates a `ThreadPoolExecutor` for audio preprocessing

Audio stays in memory. `QwenOmni` resamples to 16 kHz and uses
`qwen_omni_utils.process_mm_info` with numpy arrays, so no temporary audio
files are created.

Two-turn behavior is controlled by `followup_prompt`. Without a follow-up
prompt, only `qwen3_prediction_s1` is written. With a follow-up prompt,
the model first transcribes, then runs a second prompt using the full
conversation history and writes `qwen3_prediction_s2`. Downstream stages
use `qwen3_prediction_s2` when it exists.

If ASR recovery is enabled, `keep_waveform=True` so Qwen-ASR can reuse the
waveform later. Otherwise Qwen-Omni removes `waveform` after inference to
reduce memory pressure.

Qwen-Omni logs and perf metrics include input utterances, processed and
skipped utterances, audio duration, waveform bytes, output characters,
output tokens, turn 1 and turn 2 preparation/generation timings, and total
inference time.

## Disfluency WER Guard

`DisfluencyWerGuardStage` is active only when `followup_prompt` is set. It
compares the first-turn and second-turn predictions. If the second turn
diverges too far, controlled by `max_wer_pct`, it can flag the sample so
the pipeline does not silently accept a corrupted disfluency refinement.

## Hallucination Detection

`WhisperHallucinationStage` always runs on the active Omni prediction key.
It uses `hall_phrases`, `unique_words_threshold`, `long_word_threshold`,
`long_word_rel_threshold`, and `max_char_rate`.

The stage updates `_skip_me` when it finds known hallucination patterns,
degenerate output, extreme character rate, or suspicious long-word
patterns.

## Optional Qwen-ASR Recovery

Qwen-ASR recovery is enabled when `asr_model_id` is set.

`InferenceQwenASRStage` is configured with:

- `asr_model_id`
- `asr_batch_size`
- `asr_gpu_memory_utilization`
- `asr_max_model_len`
- `asr_max_new_tokens`
- `asr_num_workers`
- `asr_resource_gpus`

It runs only on rows whose `_skip_me` begins with `Hallucination`. Rows
that do not match pass through with empty `qwen3_asr_prediction` and
`qwen3_asr_language`. This keeps the expensive recovery path focused on
samples where Qwen-Omni likely failed.

`asr_max_model_len` is passed to the `qwen_asr` vLLM backend when set. It
is important for constrained GPU experiments because vLLM reserves KV
cache from model length, batch size, and memory utilization. If
`max_inference_batch_size` is not set directly in the stage, it follows
`asr_batch_size`.

The Qwen-ASR model wrapper applies a compatibility patch for
`transformers.check_model_inputs` before importing `qwen_asr`. This keeps
the existing qwen-asr package working with newer Transformers versions
where the decorator API changed.

Qwen-ASR always removes `waveform` from task data after it has either
processed or skipped the batch. This prevents the writer and upload path
from serializing large arrays.

Qwen-ASR metrics include utterances input, selected, skipped, unsupported
language, selected and skipped audio duration, selected and skipped
waveform bytes, output characters, and inference time.

## Prediction Selection

`SelectBestPredictionStage` writes `best_prediction`.

The intended behavior is:

- prefer the active Qwen-Omni text when it is usable
- use Qwen-ASR recovery when Omni was flagged and ASR produced usable text
- preserve useful skip/recovery annotations for downstream filters

This stage is the bridge between the model inference section and the text
cleanup section.

## Text Cleanup Chain

The deterministic cleanup chain is:

```text
FastTextLIDStage -> RegexSubstitutionStage -> AbbreviationConcatStage
```

`FastTextLIDStage` checks `best_prediction` against `source_lang_key` and
uses `min_lang_prob` to flag likely wrong-language rows.

`RegexSubstitutionStage` reads `regex_yaml` and writes `cleaned_text`.
This stage is the main rule-based normalization point.

`AbbreviationConcatStage` reads `cleaned_text` and writes
`abbreviated_text`, joining abbreviation-like token sequences.

Each stage logs per-batch counters into Curator's stage perf stream. These
metrics later appear in `_perf.jsonl`, `perf_summary.json`, and, for
multi-node Kratos runs, `perf_summary_merged.json`.

## Optional PnC Restoration

PnC runs unless `skip_pnc` is true.

`PnCRestorationStage` uses the text-only Qwen model wrapper to restore
punctuation and capitalization:

- input key: `abbreviated_text`
- output key: `pnc_text`
- model: `pnc_model_id`
- prompt: `pnc_prompt_file` or `pnc_prompt`

Important runtime knobs:

- `pnc_tensor_parallel_size`
- `pnc_batch_size`
- `pnc_max_model_len`
- `pnc_max_num_seqs`
- `pnc_gpu_memory_utilization`
- `pnc_prep_workers`
- `pnc_num_workers`
- `pnc_resource_gpus`

`PnCContentGuardStage` then compares `abbreviated_text` and `pnc_text`. If
the LLM changes content rather than only punctuation/casing, it records
the rejected output in `rejected_pnc_text` and keeps the safer text.

## Optional ITN

ITN runs when `enable_itn` is true.

By default ITN reads from:

- `pnc_text` when PnC is enabled
- `abbreviated_text` when PnC is skipped

The source can be overridden with `itn_text_key`, and the output key
defaults to `itn_text`.

Important runtime knobs:

- `itn_model_id`
- `itn_prompt_file`
- `itn_tensor_parallel_size`
- `itn_batch_size`
- `itn_max_output_tokens`
- `itn_max_model_len`
- `itn_max_num_seqs`
- `itn_gpu_memory_utilization`
- `itn_no_validation`
- `itn_num_workers`
- `itn_resource_gpus`

## Writer And Output Layout

`ShardedManifestWriterStage` is always the final stage.

It writes JSONL outputs under `output_dir`, preserving the shard key
derived by `NemoTarShardDiscoveryStage`:

```text
output_dir/
  <corpus>/.../manifest_0.jsonl
  <corpus>/.../manifest_0_perf.jsonl
  <corpus>/.../manifest_0.jsonl.done
  perf_summary.json
```

For every task, it appends:

- the final `task.data` row to `<shard>.jsonl`
- the task's stage perf chain to `<shard>_perf.jsonl`

When a shard reaches `_shard_total`, it writes `<shard>.jsonl.done` and
refreshes `perf_summary.json`. It also writes `perf_summary.json` during
teardown. Refresh-on-shard-completion is important on Kratos because a
failed or interrupted teardown should not be the only chance to persist
throughput stats.

The writer is intentionally single-worker:

```python
num_workers() -> 1
xenna_stage_spec() -> {"num_workers": 1}
```

Within one Curator driver this prevents multiple writer actors from racing
on the same root `perf_summary.json`. In multi-node Kratos runs, NvLLMOps
uploads each rank's local output under a rank-scoped prefix and merges
summaries explicitly.

## Performance Summary Semantics

`perf_summary.json` contains:

- `total_utterances`
- `total_audio_seconds`
- `total_audio_hours`
- `writer_wall_time_s`
- pipeline-level audio and utterance throughput against writer wall time
- `perf_invocations_counted`
- per-shard utterance/audio totals
- per-stage aggregate process time, idle time, input size, items processed,
  invocation count, and custom metric sums

Derived per-stage metrics include:

- `avg_invocation_time_s`
- `throughput_items_per_s`
- `throughput_audio_s_per_process_s`
- `throughput_audio_s_per_inference_s`
- token/character throughput where output token/char metrics exist
- waveform MB throughput where waveform byte metrics exist
- per-input utterance rates for filtering/recovery counters

In single-node runs this file is uploaded directly by NvLLMOps. In
multi-node runs, NvLLMOps uploads each rank under `_rank_outputs/rank=N/`
and writes `perf_summary_merged.json` after all ranks upload.

## Model Prefetching

`_prefetch_models(cfg)` downloads all relevant HuggingFace artifacts in
parallel before `pipeline.run()`:

- Qwen-Omni model
- PnC model unless PnC is skipped
- ITN model when enabled and distinct from PnC
- Qwen-ASR model when enabled
- FastText language ID model when configured as a HuggingFace repo

Each stage also has `setup_on_node()` model prefetching. The top-level
prefetch reduces first-run latency, while `setup_on_node()` protects
multi-worker scheduling from triggering duplicated downloads on the same
node.

Prefetch failures are warnings. The real setup path still attempts the
download/load and surfaces hard failures there.

## Execution Mode

The executor is:

```python
XennaExecutor(config={
    "execution_mode": cfg.get("execution_mode", "streaming"),
    "autoscale_interval_s": cfg.get("autoscale_interval_s", 180),
})
```

`streaming` allows stages to overlap and is the right mode for measuring
end-to-end pipelined throughput once resource placement is stable.

`batch` is more conservative. It can reduce peak co-location pressure
because fewer downstream stages need to be live at the same time, but it
usually gives lower end-to-end throughput.

For single-node 4-GPU experiments, remember that a pipeline with
Qwen-Omni, Qwen-ASR, PnC, and ITN can contain more GPU-using stages than
physical GPUs. Fractional `*_resource_gpus` controls Xenna accounting
only; actual vLLM memory pressure is still controlled by model lengths,
batch sizes, tensor parallel sizes, and GPU memory utilization.

## Config Reference

Core runtime keys:

| Key | Purpose |
| --- | --- |
| `input_manifest` | NvLLMOps override; for this pipeline it is the data config path |
| `data_config` | Data config path consumed by `NemoTarredAudioReader` |
| `workspace_dir` / `output_dir` | Local output root |
| `final_manifest` | Legacy compatibility field |
| `language_short` | Runtime language code passed by NvLLMOps |
| `execution_mode` | Xenna mode, usually `streaming` or `batch` |
| `autoscale_interval_s` | Xenna autoscaling interval |
| `hf_token` | HuggingFace token forwarded by infra |

Reader/scaling keys:

| Key | Purpose |
| --- | --- |
| `corpus` | Optional corpus filter |
| `s3_endpoint_url` | AIS endpoint for direct S3 reads |
| `max_utterances_per_shard` | Debug/throughput subset cap |
| `reader_num_workers` | Exact reader worker count |
| `reader_num_workers_per_node` | Reader workers per node |
| `omni_num_workers`, `asr_num_workers`, `pnc_num_workers`, `itn_num_workers` | Stage worker overrides |
| `omni_resource_gpus`, `asr_resource_gpus`, `pnc_resource_gpus`, `itn_resource_gpus` | Xenna GPU scheduler resource overrides |

Model keys are grouped in `qwen_omni_inprocess.yaml` for Qwen-Omni,
Qwen-ASR, PnC, ITN, SED, hallucination filters, regex cleanup, and
language ID. That YAML is the source of truth for defaults.

## Practical Debugging Notes

If Qwen-ASR fails with:

```text
ValueError: No available memory for the cache blocks.
```

then the ASR vLLM engine could not reserve KV cache. The relevant knobs
are `asr_max_model_len`, `asr_batch_size`, `asr_gpu_memory_utilization`,
and stage co-location. A fractional `asr_resource_gpus` alone will not fix
real CUDA memory pressure.

If output JSONL exists but throughput is zero or missing, inspect:

- `<shard>_perf.jsonl`
- root `perf_summary.json`
- Kratos `perf_summary_merged.json` for multi-node runs

NvLLMOps intentionally fails the merge when per-shard perf JSONL exists
but no rank-level `perf_summary.json` exists. That prevents silently
accepting zero-throughput summaries.

If a shard is skipped unexpectedly, check for an existing matching
`.jsonl.done` marker under `output_dir`.

If no ASR predictions appear, confirm:

- `asr_model_id` is set
- Qwen-Omni hallucination detection is flagging rows with `_skip_me`
  beginning with `Hallucination`
- the row language is supported by Qwen-ASR

## What To Keep Out Of Curator

Curator should not contain Kratos, Swift, NGC, Docker tag, or cluster auth
logic. The reusable contract is:

- input paths are readable by the process
- model IDs or local paths are supplied by config
- output directory is writable
- runtime secrets are provided as environment variables or config values

Everything else belongs in NvLLMOps or the deployment environment.
