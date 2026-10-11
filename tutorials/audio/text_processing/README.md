# Process audio transcripts with Dynamo

This tutorial reads a JSONL audio manifest, applies `AudioTextLLMStage`, and
writes an output manifest. Records remain `AudioTask` objects throughout the
processing chain. Audio paths, segments, and other record fields are preserved;
the tutorial does not load or decode audio.

`run.py` uses Curator's existing YAML pipeline and executor builders. The stage
calls the shared `AsyncOpenAIClient`; one `InferenceServer` owns the Dynamo
deployment and its GPUs. The text-processing workers request CPUs only.

## Install and configure

From the Curator repository root, install the audio and serving dependencies:

```bash
uv sync --extra audio_cuda12 --extra inference_server
source .venv/bin/activate
```

Local serving requires compatible NVIDIA GPUs, model access, and the `etcd`
and `nats-server` executables on `PATH`. The default model is
`Qwen/Qwen3.5-35B-A3B-FP8`, with one replica and tensor parallelism of one.
Choose a model and tensor-parallel size that fit your hardware. The configuration
is a functional starting point; throughput and model quality are not benchmarked
by this tutorial.

For stage throughput and fallback metrics, use the
[audio text benchmark](../../../benchmarking/README.md#audio-text-stage-benchmark),
which reuses this configuration and measures only the text-stage pipeline.

To use an existing OpenAI-compatible server, install the audio stack and pass
`endpoint` and `model_name`. The runner then manages only its Ray client and
pipeline; it does not start or stop the external inference server. The endpoint
must be reachable by every pipeline worker and include `/v1`.

## Run punctuation and capitalization

Create a UTF-8 JSONL manifest whose records contain a transcript and language:

```json
{"audio_filepath":"/data/sample.wav","text":"hello how are you","source_lang":"en","duration":2.4}
```

Run from the repository root, using absolute paths available to all workers:

```bash
python tutorials/audio/text_processing/run.py \
  manifest_path=/data/input.jsonl \
  output_path=/data/pnc_output.jsonl
```

The default `operation=pnc` adds `pnc_text`; it leaves `text` unchanged. To
process a manifest whose ASR output is called `pred_text`, add
`text_key=pred_text`. Use `language_key=language` if the language field has
that name.

Existing output is replaced by `ManifestWriterStage`. Use a distinct output
file; the runner rejects an output path that resolves to the input file. This
tutorial accepts a single input manifest path. For multi-node execution, use
a shared filesystem for both paths. All manifest values must be JSON-serializable.

An existing server can be used as follows:

```bash
python tutorials/audio/text_processing/run.py \
  manifest_path=/data/input.jsonl \
  output_path=/data/pnc_output.jsonl \
  endpoint=http://inference-host:8000/v1 \
  model_name=served-text-model
```

## Select the operation

Each operation is configuration for the same `AudioTextLLMStage`:

| `operation` | Output field | Purpose | Validation mode |
| --- | --- | --- | --- |
| `pnc` | `pnc_text` | Punctuation and capitalization | `itn` |
| `pnc_indic` | `pnc_text` | Punctuation with explicit rules for 22 Indic languages | `itn` |
| `tn` | `tn_raw` | Written forms to spoken forms | `tn` |
| `itn` | `itn_raw` | Spoken forms to written forms | `itn` |
| `disfluency` | `itn_clean` | Remove repetitions and speech disfluencies | `itn` |
| `captioning` | `captioning_text` | Summarize transcript content | `none` |
| `code_switching` | `code_switched_text` | Restore English words written phonetically in another script | `none` |
| `speech_qa` | `speech_qa_text` | Produce a transcript-grounded Q/A pair or `SKIP` | `none` |
| `language_id` | `llm_language_prediction` | Predict primary and mixed languages from transcript text | `none` |

For example, apply ITN to a punctuation-restored transcript:

```bash
python tutorials/audio/text_processing/run.py \
  manifest_path=/data/pnc_output.jsonl \
  output_path=/data/itn_output.jsonl \
  operation=itn text_key=pnc_text
```

The operation YAML files preserve the prompt templates from
[`nkoluguri/text-dynamo-inference`](https://github.com/nithinraok/Curator/tree/nkoluguri/text-dynamo-inference/nemo_curator/stages/audio/text_filtering/prompts).
`pnc_indic` also includes its 22-language punctuation rules. These prompts are
model instructions; they do not establish model quality or guarantee immutable
character preservation. `language_id` predicts from text, not acoustic evidence.
Speech QA and language predictions remain text fields; the stage does not parse
them into new record schemas. A generated `SKIP` is stored in `speech_qa_text`
and does not remove the record.

Prompts support `{text}`, `{language}`, and `{language_rules}`. Language-aware
templates read `language_key`; the Indic template additionally requires a
matching entry in `operation.language_rules`. Keep the language codes in the
manifest consistent with that mapping.

Validation checks are transcript safeguards, not correctness scores. `itn`
checks for excessive changes and newly introduced Latin words; `tn` allows
number expansion and checks for language/script drift. Generative operations
use `none` because their output is intentionally different from the transcript.
The inherited TN heuristic targets non-English script input and can reject
valid English number expansions, such as `100` to `one hundred`. When those
checks do not suit your data, set `operation.validation_mode=none` explicitly.
The disfluency preset requires at least 50% word retention, compared with 30%
for the other transcript presets, once the input has at least eight words.
Empty or truncated model output falls back to the input text regardless of
validation mode. Rejected rewrites also fall back to the input, with the reason
stored under `additional_notes[stage_name]`. Already-flagged records and empty
transcripts do not invoke the model. Exhausted request retries fail the batch;
they are not written as successful rewrites.

## Chain PnC, TN, and ITN

For a chain, save the following as `config/chain.yaml` beside `text.yaml`, then
run with `--config-name chain`. Hydra loads the existing operation prompts;
there is no second pipeline implementation.

```yaml
defaults:
  - text
  - operation@tn: tn
  - operation@itn: itn
  - _self_

stages:
  - _target_: nemo_curator.stages.audio.ManifestReader
    manifest_path: ${manifest_path}

  - &text_stage
    _target_: nemo_curator.stages.audio.text_filtering.AudioTextLLMStage
    _convert_: all
    name: PnCRestoration
    endpoint: ${endpoint}
    model_name: ${model_name}
    prompt: ${operation.prompt}
    language_key: ${language_key}
    language_rules: ${operation.language_rules}
    text_key: ${text_key}
    output_text_key: pnc_text
    validation_mode: itn
    generation_config: ${operation.generation_config}
    batch_size: ${batch_size}
    max_concurrent_requests: ${max_concurrent_requests}
    request_timeout_s: ${request_timeout_s}
    max_retries: ${max_retries}

  - <<: *text_stage
    name: TextNormalization
    prompt: ${tn.prompt}
    language_rules: null
    text_key: pnc_text
    output_text_key: tn_raw
    validation_mode: tn
    generation_config: ${tn.generation_config}

  - <<: *text_stage
    name: ITNRestoration
    prompt: ${itn.prompt}
    language_rules: null
    text_key: tn_raw
    output_text_key: itn_raw
    generation_config: ${itn.generation_config}

  - _target_: nemo_curator.stages.audio.ManifestWriterStage
    output_path: ${output_path}
```

Each pass has a distinct stage name and output field. Downstream passes use the
preceding output explicitly. All passes share the same Dynamo deployment.

## Execution and sampling

Ray Data is the default executor. Use `backend=xenna execution_mode=streaming`
or `backend=xenna execution_mode=batch` to select Xenna. `batch_size` limits
the candidate records per stage call; `max_concurrent_requests` limits in-flight
requests within each worker. Total server load also depends on worker count and
the number of active stages.

The presets use `temperature=0`, `top_p=1`, `seed=0`, `max_tokens=512`, and
`chat_template_kwargs.enable_thinking=false` for the default Qwen model. For
another model, adjust `operation.generation_config.extra_kwargs` to match its
supported request parameters. For example, set
`operation.generation_config.max_tokens=1024` for longer transcripts, while
keeping the full prompt plus output within the server's model context limit.

Server deployment settings remain under `inference_server`, for example
`inference_server.models.0.engine_kwargs.tensor_parallel_size=2`. Request
sampling stays in the operation configuration. Neither setting allocates GPUs
to the HTTP text stage.

This tutorial does not include contextual ASR, entity recovery, acoustic
language identification, task fusion, or instruction-dataset packing.
