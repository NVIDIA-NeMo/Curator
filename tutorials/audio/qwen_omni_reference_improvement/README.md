# Improve existing transcripts with Qwen3-Omni

[pipeline.yaml](pipeline.yaml) contains the complete pipeline, English and
multilingual prompts, model settings, and text-cleanup rules. Run it with the
existing `nemo_curator/config/run.py` entry point. The benchmark loads this same
YAML with a larger input manifest; model and processing settings stay the same.

```text
ManifestReader → ResampleAudioStage → ASRStage (Qwen3-Omni + reference text)
  → WhisperHallucinationStage → SelectBestPredictionStage
  → RegexSubstitutionStage → AbbreviationConcatStage → ManifestWriterStage
```

Qwen receives the audio and its existing transcript. The prompt asks it to
restore audible disfluencies and normalize spoken numbers while preserving
names. Hallucination checks can select the original transcript as a fallback.
The pipeline uses one ASR model, with no recovery model or sound-event detection.

## Run

Use an environment with the `audio_cuda12` and `vllm` extras, FFmpeg, and at least
two GPUs with enough memory for `Qwen/Qwen3-Omni-30B-A3B-Instruct`:

```bash
uv sync --extra audio_cuda12 --extra vllm
source .venv/bin/activate
```

From the repository root, stage a small labelled LibriSpeech sample with the
existing data-preparation helper:

```bash
python benchmarking/data_prep/prepare_librispeech_data.py \
  --output-path /data/qwen_reference_example \
  --cache-dir /data/librispeech_cache \
  --hf-repo-id openslr/librispeech_asr \
  --hf-revision 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1 \
  --hf-config all --hf-split train.clean.100 \
  --target-audio-hours 0.02

export VLLM_WORKER_MULTIPROC_METHOD=spawn
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_omni_reference_improvement \
  --config-name pipeline \
  manifest_path=/data/qwen_reference_example/manifest.jsonl \
  output_path=/data/qwen_reference_example/improved.jsonl \
  workspace_dir=/data/qwen_reference_example/workspace
```

The default executor is Xenna streaming. Add `backend=ray_data` for Ray Data.
Set `RAY_ADDRESS` to use an existing cluster. Audio and workspace paths must be
accessible to its workers; choose a fresh output path and workspace for each run.
For offline use, set `model_id` to a local copy of the pinned model revision in
the YAML.

## Input and output

Your own JSONL manifest can use the same format:

```json
{"audio_filepath": "/data/clip.wav", "text": "The existing transcript.", "source_lang": "en"}
```

Each reference must be a nonempty string aligned to the entire clip. Set
`reference_text_key` if your label column has another name. Supply `source_lang`
for non-English audio; missing language defaults to English. The configured
30-second clip limit requires longer recordings to be segmented with aligned
transcripts before running.

The original `text` is preserved. Outputs include `primary_model_prediction`,
`best_prediction`, `best_prediction_source`, `cleaned_text`, and
`abbreviated_text`. Reference recovery happens before cleanup, so the final
cleaned text can differ from the original label. Inspect `_skipme` and
`additional_notes` for flagged rows; no rows are dropped.

## Benchmark

The [paired Xenna and Ray Data benchmarks](../../../benchmarking/README.md#qwen-omni-human-reference-improvement-draft)
load this exact YAML. Their default input is a five-hour prefix of the same
pinned LibriSpeech split, compared with the tutorial's 0.02-hour sample. Only
input/output locations, the equivalent staged model location, and the selected
executor change. The benchmark also records timing and validates output coverage
and reference preservation.

The entries remain disabled pending benchmark acceptance. Earlier EOS results
and their limits are recorded in the benchmark README. Agreement with a label
supplied in the prompt does not independently establish transcription quality.
