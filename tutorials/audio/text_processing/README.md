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

Managed serving requires eight compatible NVIDIA GPUs with enough memory for
`RedHatAI/gemma-4-31B-it-FP8-dynamic`, model access, and the `etcd` and
`nats-server` executables on `PATH`. Four Dynamo replicas each use tensor
parallelism of two. The Gemma model's isolated runtime requests
`transformers>=5`; the repository's global dependency constraints are unchanged.
Use a Gemma-4-capable Dynamo/vLLM stack; the reference launcher requires vLLM
0.19 or later, Dynamo 1.2 or later, and Transformers 5.x.

The tutorial uses Curator's shared YAML runner, `nemo_curator/config/run.py`.
One `InferenceServer` owns the model and GPUs. The text workers use CPUs and the
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
  output_path=/data/text_output.jsonl
```

`--config-path` is relative to the shared runner; manifest and output paths are
relative to the current working directory. The runner starts the configured
server and stops it before Ray, including when pipeline execution fails.

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
existing output, and the runner rejects an output path resolving to the input
file. Use shared filesystem paths for multi-node execution. Manifest values must
be JSON-serializable.

To use an existing OpenAI-compatible server, pass its `/v1` endpoint and model
name. Every worker must be able to reach the endpoint. The runner leaves that
server's lifecycle to its owner; a CPU-only client cluster can use:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/text_processing/config \
  --config-name text \
  manifest_path=/data/input.jsonl \
  output_path=/data/text_output.jsonl \
  endpoint=http://inference-host:8000/v1 \
  model_name=served-text-model ray_client.num_gpus=0
```

## Run one operation

`single_operation.yaml` retains the smaller workflow and its one-replica Qwen
serving configuration. For example, apply ITN to an existing transcript:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/text_processing/config \
  --config-name single_operation \
  manifest_path=/data/input.jsonl \
  output_path=/data/itn_output.jsonl \
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
expansions such as `100` to `one hundred`. Set `tn.validation_mode=none` in the
full chain, or `operation.validation_mode=none` for a single operation, when
those checks are inappropriate. Disfluency removal requires 50% word retention;
other transcript rewrites require 30%, once the input has at least eight words.

TN/ITN validation failures fall back to the input with a reason in the operation
note. Unvalidated operations retain the response text, including empty or
length-limited responses, matching the reference. Entity recovery removes reasoning
blocks, code fences, and enclosing quotes; an empty cleaned response falls back
to the normalized input. Empty inputs and already-flagged rows skip inference.
Exhausted request retries fail the batch. Ray Data is the default executor;
`backend=xenna execution_mode=streaming` or `execution_mode=batch` selects Xenna.
