# Improve human transcripts with Qwen3-Omni

Use an existing human transcript and its audio to restore spoken disfluencies,
write numbers as spoken words, and correct mismatches. The original transcript
is preserved in `text`; Qwen's proposed revision and each post-processing result
are written to separate fields. This is a reference-conditioned pipeline, so
every input row must contain a non-empty reference transcript.

The [notebook](qwen_omni_reference_improvement_tutorial.ipynb) walks through
configuration, execution, and comparing original and revised text. Both the
notebook and the [paired benchmark](../../../benchmarking/scripts/audio_qwen_omni_reference_improvement_benchmark.py)
load the same [pipeline.yaml](pipeline.yaml).

## Pipeline

```text
ManifestReader
  → ResampleAudioStage (16 kHz mono, measured duration)
  → ASRStage + QwenOmniASRAdapter (audio + reference transcript)
  → WhisperHallucinationStage
  → SelectBestPredictionStage (original-reference fallback)
  → RegexSubstitutionStage
  → AbbreviationConcatStage
  → ManifestWriterStage
```

English rows use [the English improvement prompt](prompts/english.md). Other
supported languages use [the multilingual improvement prompt](prompts/multilingual.md).
`{transcript}` is replaced with that row's reference, and `{language}` is also
supported in custom prompts. English uses the reference as a starting point,
preserves named entities, and follows the audio when wording differs. The
multilingual prompt limits changes to spoken-number normalization and adding
audible disfluencies. Neither prompt requests explanations or structured output.

There is one ASR model and one inference turn. Sound-event detection is
disabled. Hallucination detection uses the existing phrase list, repeated-word,
long-word, and character-rate checks. When flagged, `best_prediction` becomes
the original reference, `best_prediction_source` becomes `reference`, and the
hallucination skip flag is cleared. The detection reason remains in
`additional_notes`. Unsupported languages use the reference with source
`ground_truth`. Audio decoding/preparation errors and empty model output remain
flagged; they are not treated as successful reference recovery.

The selected text then passes through the supplied multilingual regex rules
and abbreviation joining. The rules normalize punctuation, remove bracketed
annotations, and remove characters outside the configured scripts. Consequently,
`abbreviated_text` may differ from the original even after reference recovery;
`text` and `best_prediction` remain available for inspection. No rows are
dropped. Inspect `_skipme` before consuming the final text for training.

## Requirements

Use x86_64 Linux, FFmpeg on every worker's `PATH`, and an environment with the
`audio_cuda12` and `vllm` extras:

```bash
uv sync --extra audio_cuda12 --extra vllm
source .venv/bin/activate
```

The default model is `Qwen/Qwen3-Omni-30B-A3B-Instruct`. Each ASR actor reserves
two GPUs for tensor parallelism; model loading and inference require adequate
GPU memory. Download/cache the weights before an offline run, or set `model_id`
to a local checkpoint and `model_revision=null`. The default Hugging Face
revision is `26291f793822fb6be9555850f06dfe95f2d7e695`; keep the same revision
when staging weights and comparing runs. On a shared cluster, put audio, prompts, and the workspace on
storage accessible from every worker.

## Input

Provide a JSONL manifest with one utterance per row:

```json
{"audio_filepath": "/data/audio/utterance.wav", "text": "The existing human transcript.", "source_lang": "en"}
```

- `audio_filepath` must resolve to a local/shared audio file on every worker.
- `text` must be a non-empty string aligned to the entire clip. Set
  `reference_text_key` if your manifest uses another field name.
- `source_lang` is an ISO language code. Missing language defaults to `en`.
  Supply `source_lang` on every non-English row so inference, language-specific
  hallucination checks, and abbreviation joining use the same language.
- `duration` is measured after resampling, so an input duration is optional.
- Optional `audio_item_id` values must uniquely identify the audio paths.
  Use a fresh workspace if the files behind those paths change.

The default 30-second limit is a conservative tutorial choice for the 4096-token
context. Reference-conditioned ASR rejects clips that would need automatic
splitting: repeating the whole transcript beside every audio chunk would break
alignment. For longer recordings, prepare utterance clips and their aligned
human labels first. Increase the duration and context settings only after
validating the model's capacity for your inputs.

This portable pipeline reads unpacked audio and writes one JSONL manifest. It
preserves the input field `text` instead of renaming it to
`granary_v1_prediction`; both inference and recovery read the same field. It
does not implement tar-shard ingestion, per-shard completion markers, or Slurm
dependency-chain resumption. Existing non-empty `_skipme` values remain visible
and prevent post-processing; start with an unflagged human-labelled manifest.

## Run

For public labelled audio, use the existing LibriSpeech staging helper. This
example downloads a small prefix (at least 0.02 hours, keeping the last clip
whole) from a pinned dataset revision:

```bash
python benchmarking/data_prep/prepare_librispeech_data.py \
  --output-path /data/qwen_reference_example \
  --cache-dir /data/librispeech_cache \
  --hf-repo-id openslr/librispeech_asr \
  --hf-revision 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1 \
  --hf-config all --hf-split train.clean.100 \
  --target-audio-hours 0.02
```

Use `/data/qwen_reference_example/manifest.jsonl` as `manifest_path` below.
The [paired benchmark setup](../../../benchmarking/README.md#qwen-omni-human-reference-improvement-draft)
stages a five-hour prefix and pinned model weights, with exact commands for
`audio_qwen_omni_reference_improvement_xenna` and
`audio_qwen_omni_reference_improvement_raydata`. Both entries are disabled until
their hardware/runtime and throughput requirements are calibrated.

Run from the Curator repository root. All relative data and asset paths are
resolved from that directory. Choose a fresh output path and workspace:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_omni_reference_improvement \
  --config-name pipeline \
  manifest_path=/data/human_labels.jsonl \
  output_path=/data/improved_labels.jsonl \
  workspace_dir=/data/qwen_improvement_workspace
```

The default backend is Ray Data. For Xenna streaming, add
`backend=xenna execution_mode=streaming`; for Xenna batch use
`backend=xenna execution_mode=batch`. These modes use identical prompts and
stages. To connect to an existing Ray cluster, set `RAY_ADDRESS` before running.

For a multi-node Slurm allocation, run one process per node and select the
existing `SlurmRayClient`. For example, inside an allocation with at least two
GPUs per node, from a shared checkout with the environment active on every node:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_PORT_BROADCAST_DIR=/shared/qwen_improvement_run
mkdir -p "$RAY_PORT_BROADCAST_DIR"
srun --ntasks-per-node=1 python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_omni_reference_improvement \
  --config-name pipeline \
  manifest_path=/shared/human_labels.jsonl \
  output_path=/shared/qwen_improvement_run/improved_labels.jsonl \
  workspace_dir=/shared/qwen_improvement_run/workspace \
  +ray_client._target_=nemo_curator.core.client.SlurmRayClient \
  +ray_client.ray_temp_dir=/tmp/ray_qwen_improvement
```

The port-broadcast directory must be shared, while the Ray temporary directory
is local to each node. Only the head runs the pipeline; the other processes
join the cluster and wait. The scheduler allocation, account, and container are
site-specific and are not part of this tutorial.

| Setting | Default |
|---|---|
| GPUs per ASR actor / tensor parallel size | 2 |
| Stage batch size | 32 |
| Concurrent vLLM sequences | 16 |
| Context / output token limits | 4096 / 256 |
| GPU memory utilization | 0.90 |
| Temperature / top-k / repetition penalty | 0.0 / 1 / 1.0 |
| Seed / prefix caching | 1234 / enabled |
| Prompt content order | Text, then audio |
| Maximum clip duration | 30 seconds |
| Padded-audio budget per adapter call | 960 seconds |
| Local duration bucketing | Disabled |

`max_audio_sec_per_actor` controls inference batching; it is not a GPU-memory
guarantee. Prompt length and audio duration both consume context. Sub-second
clips are not automatically replaced with their reference; recovery occurs
through the configured hallucination checks or unsupported-language policy.

## Output and evaluation

| Field | Meaning |
|---|---|
| `text` (or configured reference key) | Unchanged input human label |
| `primary_model_prediction` | Qwen's proposed revision |
| `best_prediction` | Selected revision or reference fallback |
| `best_prediction_source` | `primary`, `reference`, or `ground_truth` |
| `cleaned_text` | Selected text after regex normalization |
| `abbreviated_text` | Final text after joining spelled abbreviations |
| `_skipme` | Empty/missing for accepted rows; reason for flagged rows |
| `additional_notes` | Hallucination and selection decisions |

Output order may differ between backends. Compare rows by audio path or a
stable unique ID. Review original, generated, and final text together with the
audio: agreement with the input label measures edit size, not transcription
quality, because improvement intentionally changes that label. Benchmark
throughput and fallback counts do not establish transcription accuracy.
