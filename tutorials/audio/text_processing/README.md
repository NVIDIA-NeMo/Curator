# Process audio transcripts with Dynamo

This tutorial reads a JSONL audio manifest, applies the complete text-processing
chain, and writes an output manifest. Every pass uses `AudioTextLLMStage` and
keeps records as `AudioTask` objects. Audio paths, segments, and unrelated fields
are preserved; audio is never loaded or decoded.

The default configuration follows the active text graph and serving settings in
[`text_dynamo_1node_gemma4_indic_packer.sh`](https://gitlab-master.nvidia.com/nkoluguri/granary-v2-asr/-/blob/d1d341d7184d7c5c90beb012cc79bf4bcfb2cb33/slurm_pipeline/text_pipeline/text_dynamo_1node_gemma4_indic_packer.sh),
using the Curator implementation at
[`1a260a3`](https://github.com/nithinraok/Curator/tree/1a260a35472229a8a75a122b9a2bb9d92d5855f9/examples/audio/text_processing).
The launcher enables entity recovery and eight text operations. Its filename
includes “packer,” but instruction packing is not enabled by that command.
This tutorial uses the current manifest reader/writer and serving APIs; it does
not reproduce Slurm submission, shard scheduling, or resume markers.

## Install and configure

From the Curator repository root:

```bash
uv sync --extra audio_cuda12 --extra inference_server
source .venv/bin/activate
```

Running the provided serving configuration requires eight compatible NVIDIA GPUs with enough memory for
`RedHatAI/gemma-4-31B-it-FP8-dynamic`, model access, and the `etcd` and
`nats-server` executables on `PATH`. Four Dynamo replicas each use tensor
parallelism of two. The Gemma model's isolated runtime requests
`transformers>=5`; the repository's global dependency constraints are unchanged.
Use a Gemma-4-capable Dynamo/vLLM stack; the reference launcher requires vLLM
0.19 or later, Dynamo 1.2 or later, and Transformers 5.x.

The tutorial uses Curator's unchanged shared YAML runner, `nemo_curator/config/run.py`,
with an existing OpenAI-compatible `/v1` endpoint. The runner does not start or
stop an inference server. The text workers use CPUs and the
shared `AsyncOpenAIClient`. The configuration specifies 12 workers per stage,
batches of 128, and at most 256 simultaneous requests per worker. These are
reference settings, not measured throughput guarantees. The server uses an
8,192-token context, 256 concurrent sequences, 8,192 batched tokens, automatic
KV-cache dtype, 95% GPU memory utilization, and text-only model loading.

For throughput and fallback metrics, use the
[audio text benchmark](../../../benchmarking/README.md#audio-text-pipeline-benchmark),
which reuses the graph and excludes manifest I/O and model startup from its
processing timer.

## Run the full chain

Create a UTF-8 JSONL manifest with normalized text, its reference transcription,
and the expected language:

```json
{"audio_filepath":"/data/sample.wav","abbreviated_text":"आज हम दिल्ली जाएंगे","granary_v1_prediction":"आज हम दिल्ली जाएंगे","source_lang":"hi","duration":2.4}
```

Run from the repository root with absolute paths available to all workers:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/text_processing/config \
  --config-name text \
  manifest_path=/data/input.jsonl \
  output_path=/data/text_output.jsonl \
  endpoint=http://inference-host:8000/v1
```

`--config-path` is relative to the shared runner; manifest and output paths are
relative to the current working directory. Every worker must be able to reach
the endpoint. Set `model_name` to the name exposed by that server.

The configured data flow is:

| Pass | Input field(s) | Output field(s) |
| --- | --- | --- |
| Entity recovery | `granary_v1_prediction`, `abbreviated_text` | `entity_recovered_text` |
| Punctuation and capitalization | `entity_recovered_text` | `pnc_text` |
| Language prediction and verification | `pnc_text`, `source_lang` | `llm_language_prediction`, `_skipme`, verification note |
| Text normalization | `pnc_text` | `tn_raw` |
| Parallel captioning, code switching, speech QA | `tn_raw` | `captioning_text`, `code_switched_text`, `speech_qa_text` |
| Inverse text normalization | `tn_raw` | `itn_raw` |
| Disfluency removal | `itn_raw` | `itn_clean` |

The three independent operations share one stage, client, and concurrency limit.
The other passes execute in the order above. Their prompts remain separate
requests; they are not combined into one model instruction. All nine operations
preserve the original input fields and write notes under `additional_notes`.

Use `text_key=pred_text`, `ground_truth_key=reference_text`, or
`language_key=language` to map differently named input fields. The default PnC
prompt is the reference's standard template. `pnc_indic`, with explicit rules
for 22 Indic languages, remains an optional single-operation preset.

Language verification compares the text prediction with `source_lang`. A
single-language mismatch or an unparseable prediction flags `_skipme`, preventing
later model requests. Parsed multilingual predictions are kept, including when
the expected language is absent. Verification records its decision under
`additional_notes.LLMLanguageVerification`; rows are retained in the output.
This is text-based language prediction, not acoustic language identification.

The tutorial accepts one input manifest file. `ManifestWriterStage` replaces
existing output, and the audio manifest writer rejects an output path resolving to the input
file. Use shared filesystem paths for multi-node execution. Manifest values must
be JSON-serializable.

The tutorial's Ray client requests no GPUs. For a differently named served model:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/text_processing/config \
  --config-name text \
  manifest_path=/data/input.jsonl \
  output_path=/data/text_output.jsonl \
  endpoint=http://inference-host:8000/v1 \
  model_name=served-text-model
```

To own a Dynamo server separately, use the existing `InferenceServer` API on
the serving host. For example, from the repository root in a Python session:

```python
from pathlib import Path

import hydra

from nemo_curator.core.client import RayClient

config_dir = str(Path("tutorials/audio/text_processing/config").resolve())
with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = hydra.compose(config_name="text")

with RayClient(num_gpus=8), hydra.utils.instantiate(cfg.inference_server) as server:
    print(server.endpoint)
    input("Run the tutorial against this endpoint; press Enter to stop the server. ")
```

The `inference_server` YAML section is used by this explicit server owner and
the audio benchmark; the shared runner ignores it. The server owner keeps the
model alive while the tutorial runs and releases its resources afterward.

## English workflow and serving options

`english.yaml` uses the same nine operations, seven text stages, prompts, and
generation settings. It disables the TN guard for English number expansion,
matching the native reference's `--disable_tn_validation` option; the other
guards are unchanged. Its reference transcription field is `text`:

```json
{"audio_filepath":"/data/sample.flac","abbreviated_text":"she paid 100 dollars","text":"she paid one hundred dollars","source_lang":"en","duration":2.4}
```

Supply English rows and an existing endpoint. This configuration does not detect
or filter the manifest's language.

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/text_processing/config \
  --config-name english \
  manifest_path=/data/english_input.jsonl \
  output_path=/data/english_text_output.jsonl \
  endpoint=http://inference-host:8000/v1
```

When starting the server through the API above or the audio benchmark, select
`english` and supply `cache_root=/data/curator_cache` during composition. The
cache directory must be writable and visible on every serving node. Serving
settings in YAML do not reconfigure an already running endpoint.

The English serving profile retains four TP2 replicas, the eight-GPU
budget, the memory/token limits above, and the 128/12/256 client settings. It
explicitly enables prefix caching and chunked prefill, disables async
scheduling, and leaves eager execution disabled. The target revision is pinned
in `serving/explicit.yaml`. These are controlled tuning settings, not a measured
speedup. `text.yaml` retains the reference engine defaults. Both use the existing
Dynamo actor dependency installer and request `transformers>=5` through the
model's existing `runtime_env`. The repository's vLLM dependency remains
`0.22.0+cu129`; this tutorial does not change shared runtime installation.

The explicit profile uses `runtime_env.env_vars` to direct vLLM, TorchInductor,
and Triton caches under `cache_root`; compilation also uses an explicit directory. Keep that
directory persistent between compatible runs. Change it when changing the
runtime or GPU architecture. Model weights keep their existing Hugging Face
cache location.

Choose one serving profile with `serving=NAME`; all retain eight GPUs:

| Profile | Change from the explicit TP2 settings | Constraint |
| --- | --- | --- |
| `reference` | Leave engine switches at the original defaults | Default for `text.yaml` |
| `explicit` | Explicit caching/scheduling switches and model revision | English default |
| `tp1_uni` | Eight TP1 replicas, `distributed_executor_backend=uni` | Each GPU must fit the whole model and KV cache |
| `kv` | Dynamo KV-aware routing with approximate cache tracking | Cache reuse depends on actual request prefixes |
| `disaggregated` | Two TP2 prefill and two TP2 decode workers, KV routing | Requires compatible NIXL transfer; each TP group stays on one node |

TP1 memory fit, KV routing, disaggregated transfer, and the Gemma configuration
still require GPU validation on the selected runtime. Configuration composition
tests do not establish runtime compatibility or throughput.

Individual switches remain ordinary Hydra overrides, for example
`serving.model.engine_kwargs.enforce_eager=true` or
`serving.model.engine_kwargs.async_scheduling=true`. Change one setting at a time
and compare both throughput and outputs; eager execution disables the graph
optimization being evaluated, and async scheduling compatibility depends on
the serving mode. Revision pins belong to these Gemma checkpoints: replace
them together when choosing another model.

Queue admission, the preinstalled-runtime bypass, and the Gemma assistant MTP
profile are not included. They depended on shared serving changes or a newer
runtime than this repository supports. An externally managed endpoint may use
its own serving optimizations; the audio stage only requires the compatible API.

## Run one operation

`single_operation.yaml` retains the smaller workflow and its one-replica Qwen
serving configuration. For example, apply ITN to an existing transcript:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/text_processing/config \
  --config-name single_operation \
  manifest_path=/data/input.jsonl \
  output_path=/data/itn_output.jsonl \
  endpoint=http://inference-host:8000/v1 \
  operation=itn text_key=pnc_text
```

Available presets are `recover_entities`, `pnc`, `pnc_indic`, `tn`, `itn`,
`disfluency`, `captioning`, `code_switching`, `speech_qa`, and `language_id`.
`recover_entities` also reads `ground_truth_key`; `language_id` includes the
same verification used by the full chain. A generated speech-QA `SKIP` is stored
as text and does not remove the record.

## Sampling and validation

The reference presets use temperature 0, top-p 1, seed 0, and
`chat_template_kwargs.enable_thinking=false`. Entity recovery allows 1,024
output tokens; each other operation allows 512. For another model, adjust the
operation's generation parameters and the server configuration together.
For example, `tn.generation_config.max_tokens=1024` updates the full chain,
while `operation.generation_config.max_tokens=1024` updates a single operation.

Prompts support `{text}`, `{language}`, and `{language_rules}`. Entity recovery
uses separate system/user templates with `{ground_truth}` and `{normalized}`.
The operation YAMLs preserve the reference's resolved prompt text.

Rewrite validation is a safeguard, not a correctness score. ITN validation
checks word additions, novel Latin tokens, and excessive deletion. TN permits
expansion while checking English leakage and script drift. Its inherited
heuristic targets non-English script input and can reject valid English number
expansions such as `100` to `one hundred`. The English profile sets
`tn.validation_mode=none`; use `--disable_tn_validation` in the native reference
to compare the same policy. `text.yaml` retains the guard. Set `tn.validation_mode=none` in the
full chain, or `operation.validation_mode=none` for a single operation, when
those checks are inappropriate for a separate workflow. Disfluency removal requires 50% word retention;
other transcript rewrites require 30%, once the input has at least eight words.

TN/ITN validation failures fall back to the input with a reason in the operation
note. Unvalidated operations retain the response text, including empty or
length-limited responses, matching the reference. Entity recovery removes reasoning
blocks, code fences, and enclosing quotes; an empty cleaned response falls back
to the normalized input. Empty inputs and already-flagged rows skip inference.
Exhausted request retries fail the batch. Ray Data is the default executor;
`backend=xenna execution_mode=streaming` or `execution_mode=batch` selects Xenna.
