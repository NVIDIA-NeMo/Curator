# Process English audio transcripts

[text_pipeline.yaml](text_pipeline.yaml) contains the complete pipeline: an audio
manifest reader, seven `AudioTextLLMStage` instances performing nine operations,
and a manifest writer. Records stay as `AudioTask` objects; audio files are not
loaded. Prompts and operation order follow the
[reference text pipeline](https://github.com/nithinraok/Curator/tree/1a260a35472229a8a75a122b9a2bb9d92d5855f9/examples/audio/text_processing).

Use an activated Curator environment or project image with the audio and
inference-server dependencies installed; see the
[installation guide](https://docs.nvidia.com/nemo/curator/latest/get-started/installation).
Serving the model also requires compatible GPUs, model access, and `etcd` and
`nats-server` on the serving host's `PATH`.

Use a UTF-8 JSONL manifest with one English record per line:

```json
{"audio_filepath":"/data/sample.flac","abbreviated_text":"she paid 100 dollars","text":"she paid one hundred dollars","source_lang":"en","duration":2.4}
```

`abbreviated_text` is the transcript to process; `text` is the reference
transcription used for entity recovery. Change the top-level `text_key`,
`ground_truth_key`, and `language_key` to use different field names. Supply
English input; the configuration does not filter records by language.

From the repository root, run the standard YAML runner against an existing
OpenAI-compatible endpoint reachable by every worker:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/text_processing \
  --config-name text_pipeline \
  manifest_path=/data/english_input.jsonl \
  output_path=/data/english_output.jsonl \
  endpoint=http://inference-host:8000/v1
```

The runner uses CPU workers and leaves the endpoint's lifecycle to its owner.
If the server exposes a different model name, add `model_name=YOUR_SERVED_NAME`.
Use absolute manifest paths available to all workers. Existing output is
replaced; the writer rejects an output path identifying the input manifest.
Audio paths, segments, and unrelated JSON fields are preserved.

| Stage | Reads | Writes |
| --- | --- | --- |
| Entity recovery | `text`, `abbreviated_text` | `entity_recovered_text` |
| Punctuation and capitalization | `entity_recovered_text` | `pnc_text` |
| Language prediction and verification | `pnc_text`, `source_lang` | `llm_language_prediction`, `_skipme` |
| Text normalization | `pnc_text` | `tn_raw` |
| Fused captioning, code switching, speech QA | `tn_raw` | `captioning_text`, `code_switched_text`, `speech_qa_text` |
| Inverse text normalization | `tn_raw` | `itn_raw` |
| Disfluency removal | `itn_raw` | `itn_clean` |

The fused stage sends three separate prompts concurrently through one shared
client and concurrency limit. Entity recovery allows 1,024 output tokens;
other operations allow 512. Temperature is zero, seed is zero, and model
thinking is disabled. Defaults are batch size 128, 12 workers, and 256
concurrent requests per worker; change the top-level settings to tune them.

Operation decisions are recorded in `additional_notes`. A single-language
mismatch or unparseable prediction flags `_skipme` and skips later requests
while retaining the row; parsed multilingual predictions are retained. Rewrite
guards may fall back to the input. English TN validation is
disabled to allow number expansion, matching the reference. Exhausted request
retries fail the batch. These checks are safeguards, not model-quality scores.

The [audio text benchmark](../../../benchmarking/README.md#audio-text-pipeline-benchmark)
loads this same file. It can use an existing endpoint or own an `InferenceServer`
through Curator's existing API. Its managed-server path uses the inline
`inference_server` block: four TP2 Gemma FP8 replicas on eight GPUs, an
8,192-token context and batch-token limit, 256 sequences, prefix caching,
chunked prefill, and 95% GPU memory utilization. Supply `cache_root` when
starting that server and keep it specific to the runtime and GPU architecture.
The normal YAML runner ignores this server block; editing it does not
reconfigure an existing endpoint.

The model's `runtime_env` pins Transformers `5.8.0` for Gemma configuration
compatibility, disables DeepGEMM, and sets persistent cache paths. Curator's
base environment uses vLLM `0.22.0+cu129`; Dynamo selects its own compatible
vLLM version in the model's actor environment. Local checks use a stub endpoint;
GPU compatibility, model quality, and throughput require separate validation.
