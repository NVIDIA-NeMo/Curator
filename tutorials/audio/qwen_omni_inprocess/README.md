# Qwen3-Omni In-Process ASR

This tutorial reads a NeMo-style audio manifest, loads each audio file as a
mono waveform, transcribes it with Qwen3-Omni through vLLM, and writes the
results to JSONL. It uses the generic YAML runner in
`nemo_curator/config/run.py`; no tutorial-specific Python runner is needed.

## Requirements

- x86_64 Linux with CUDA
- About 40 GB of available GPU memory for the default model
- About 15 GB of disk space for the first model download
- Audio files accessible from the machine running the pipeline

From the Curator repository root, install the optional Qwen stack:

```bash
uv sync --extra audio_qwen
source .venv/bin/activate
```

## Run the bundled smoke input

The bundled manifest contains two short OPUS files:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_omni_inprocess \
  --config-name pipeline \
  manifest_path=tests/fixtures/audio/tagging/sample_input.jsonl \
  output_path=/tmp/qwen_omni_output.jsonl
```

`--config-path` is relative to `nemo_curator/config/run.py`, while manifest,
audio, prompt, and output paths are resolved from the current working
directory. Run the command from the repository root as shown.

The first run downloads `Qwen/Qwen3-Omni-30B-A3B-Instruct`. The pipeline uses
one GPU and a batch size of one by default. To use two GPUs with tensor
parallelism:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_omni_inprocess \
  --config-name pipeline \
  manifest_path=tests/fixtures/audio/tagging/sample_input.jsonl \
  output_path=/tmp/qwen_omni_output.jsonl \
  tensor_parallel_size=2 \
  stages.2.resources.gpus=2.0
```

## Input and output

The input is a JSONL manifest with one object per audio file. Each object must
contain `audio_filepath`; `source_lang` is optional because the tutorial
defaults to English:

```json
{"audio_filepath": "/data/sample.wav", "source_lang": "en"}
```

The output preserves the input fields and adds `pred_text`, `duration`,
`num_samples`, `sample_rate`, and `is_mono`. The temporary in-memory waveform
is removed before the manifest is written.

## Prompts and two-turn inference

The default is single-turn, disfluency-preserving English transcription using
the existing prompt at
`examples/audio/qwen_omni_inprocess/prompts/en_qwen3_omni_disfluency_asr.md`.
Override `prompt_file` to use a different UTF-8 prompt.

For reference-guided correction followed by a second disfluency pass, each
input row must also contain a `text` reference transcript:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_omni_inprocess \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  output_path=/tmp/qwen_omni_output.jsonl \
  reference_text_key=text \
  disfluency_text_key=pred_text_disfluency \
  prompt_file=examples/audio/qwen_omni_inprocess/prompts/en_qwen3_omni_reference_improvement.md \
  followup_prompt_file=examples/audio/qwen_omni_inprocess/prompts/en_qwen3_omni_disfluency_asr.md \
  limit_mm_per_prompt_audio=2
```

The first-turn result is written to `pred_text`; the second-turn result is
written to `pred_text_disfluency`.

## Scope

This is a functional, local manifest-to-transcript example. It does not
calculate WER, tune throughput, or provide distributed long-audio bucketing.
For larger runs, tune the adapter batch and vLLM settings for the available
GPU memory and validate output quality on representative audio.
