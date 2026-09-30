# Hindi Indic ASR Benchmark Pipeline

This tutorial runs the same Hindi Indic Canary + Parakeet recovery benchmark
used by the `audio_indic_asr_xenna` and `audio_indic_asr_raydata` entries. Like
the [ALM tutorial](../alm/), it uses a Hydra YAML configuration, manages the Ray
cluster lifecycle, supports Xenna and Ray Data, and keeps backend runs in
separate output directories.

The tutorial does not contain a second pipeline implementation. `main.py`
loads `benchmarking/scripts/audio_indic_asr_benchmark.py` and calls its
canonical benchmark function. Input normalization and sharding, the complete
processor graph, timing boundary, and output validation therefore stay exact
as the benchmark evolves.

For an ALM-style walkthrough that runs and compares both backends, open
`indic_asr_tutorial.ipynb`. The command-line sections below run the same code.

## Pipeline

The canonical graph is:

```text
ManifestReader
  -> PrepareIndicASRInputStage
  -> InferenceIndicCanaryStage (primary)
  -> WhisperHallucinationStage
  -> InferenceParakeetStage (TensorRT recovery)
  -> WhisperHallucinationStage
  -> SelectBestPredictionStage
  -> RegexSubstitutionStage
  -> AbbreviationConcatStage
  -> GetPairwiseWerStage
  -> ManifestWriterStage
```

Each run uses eight Indic Canary actors and eight Parakeet actors. Their GPU
memory reservations are 50 GB and 24 GB per actor, respectively, so the two
model pools share the eight GPUs instead of requesting 16 physical GPUs.

## Prerequisites

- Python 3.11+
- NeMo Curator with the `audio_cuda12` dependencies
- One x86_64 node with 128 CPUs and eight H100 GPUs for benchmark-equivalent
  results
- A TensorRT-LLM-compatible environment containing `tensorrt` and
  `tensorrt_llm`
- The production-compatible Indic Canary and Parakeet TensorRT engine bundles

```bash
uv sync --extra audio_cuda12
source .venv/bin/activate
```

The engine directories must contain these files:

```text
indic_canary_engine_dir/
├── encoder/encoder.plan
├── encoder/config.json
├── decoder/config.json
├── decoder/rank0.engine
├── decoder/vocab.json
├── preprocessor/config.json
└── preprocessor/mel_basis.pt

parakeet_tensorrt_engine_dir/
├── encoder.plan
├── metadata.json
└── model.nemo
```

## Prepare the pinned public data

From the repository root, stage the open Hugging Face Hindi corpus:

```bash
python benchmarking/data_prep/prepare_audio_indic_asr_data.py \
  --output-path /data/audio_indic_asr_parakeet_hindi_531h_35376a11 \
  --cache-dir /data/_hf_cache/audio_indic_asr
```

The setup pins `ketav/parakeet-hindi-asr` revision
`35376a112c4b79318eeaba0c0dd1b6f1a9bf0ea0` and validates its train split:
216,169 unique mono 16 kHz clips totaling 531.7738 audio hours. The generated
`manifest.jsonl` contains `audio_item_id`, `audio_filepath`, `duration`,
`source_lang`, and `text` for every clip.

## Run with Xenna

Run from the repository root and use a new output directory:

```bash
python tutorials/audio/indic_asr/main.py \
  --config-path . \
  --config-name pipeline \
  input_manifest=/data/audio_indic_asr_parakeet_hindi_531h_35376a11/manifest.jsonl \
  indic_canary_engine_dir=/models/audio_indic_asr/indic_canary_trtllm/engine_bfloat16_64_new \
  parakeet_tensorrt_engine_dir=/models/audio_indic_asr/parakeet_indic_trt/encoder_fp16_b1_o8_m16_f8_o800_m4001 \
  benchmark_results_path=./indic_asr_benchmark_output/xenna \
  backend=xenna
```

## Run with Ray Data

Use the same manifest and engine directories, but a different output path:

```bash
python tutorials/audio/indic_asr/main.py \
  --config-path . \
  --config-name pipeline \
  input_manifest=/data/audio_indic_asr_parakeet_hindi_531h_35376a11/manifest.jsonl \
  indic_canary_engine_dir=/models/audio_indic_asr/indic_canary_trtllm/engine_bfloat16_64_new \
  parakeet_tensorrt_engine_dir=/models/audio_indic_asr/parakeet_indic_trt/encoder_fp16_b1_o8_m16_f8_o800_m4001 \
  benchmark_results_path=./indic_asr_benchmark_output/ray_data \
  backend=ray_data
```

The paired methodology holds the data, engines, processor settings, actor
counts, and output checks constant. Only the executor changes. Run the two
backends independently; do not reuse an output directory.

## Configuration

All user-overridable values are in `pipeline.yaml`:

| Parameter | Benchmark value | Description |
| --- | ---: | --- |
| `backend` | `xenna` | `xenna` or `ray_data` |
| `expected_num_rows` | 216169 | Exact input and output cardinality |
| `read_concurrency` | 4 | Number of normalized input shards and reader workers |
| `prep_workers` | 24 | Audio loading and validation workers |
| `primary_workers` | 8 | Indic Canary actors |
| `fallback_workers` | 8 | Parakeet TensorRT actors |
| `ray.num_cpus` | 128 | Local Ray CPU resources |
| `ray.num_gpus` | 8 | Local Ray GPU resources |

The model settings are intentionally owned by the canonical benchmark script:
Indic Canary uses batch size 64, four beams, 347 maximum new tokens, and 0.1
fractions for both KV-cache pools; Parakeet uses TensorRT engine chunking and
batch size 64. Keeping those values in one implementation prevents tutorial
and benchmark drift.

## Outputs and validation

Each result directory contains:

```text
params.json
metrics.json
tasks.pkl
results/audio_indic_asr_output.jsonl
scratch/audio_indic_asr_input/manifest-00.jsonl ... manifest-03.jsonl
```

Before execution, the benchmark resolves audio paths, rejects missing or
duplicate clips, verifies Hindi text and positive durations, and creates four
absolute-path manifests. After execution, it requires exactly one output per
input identity, unchanged paths and durations, finite WER, recognized primary
or fallback provenance, and complete selected prediction text.

The accepted EOS benchmark gate applies only to Xenna: the full process wall
time must be 600–900 seconds on the standard eight-H100 runner. Ray Data uses
the identical cohort as an ungated comparison. Local tutorial timing is useful
for diagnosis but is not a replacement for the standard EOS benchmark result.

## Troubleshooting

- **Output already exists**: choose a fresh `benchmark_results_path`; the
  benchmark fails closed instead of overwriting evidence.
- **Missing `tensorrt_llm`**: use the TensorRT-LLM-compatible benchmark image.
- **Missing engine file**: stage the complete engine bundle listed above.
- **Wrong row count**: rerun the pinned data preparation and do not edit or
  subset its manifest.
- **Ray sees fewer than eight GPUs**: verify `CUDA_VISIBLE_DEVICES` and the
  cluster resources before comparing runtime.
