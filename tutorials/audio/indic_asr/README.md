# Hindi Indic ASR Pipeline

This tutorial executes the full processor graph used by the
`audio_indic_asr_xenna` and `audio_indic_asr_raydata` benchmarks directly from
`pipeline.yaml`. It follows the same YAML-first method as the ALM and Qwen ASR
tutorials: the shared `nemo_curator/config/run.py` runner instantiates every
stage, then selects Xenna or Ray Data from one configuration.

There is no tutorial-specific Python wrapper.

## Pipeline

```text
ManifestReader
  -> PrepareIndicASRInputStage
  -> InferenceIndicCanaryStage (primary; isolated TensorRT-LLM runtime)
  -> WhisperHallucinationStage
  -> InferenceParakeetStage (TensorRT recovery)
  -> WhisperHallucinationStage
  -> SelectBestPredictionStage
  -> RegexSubstitutionStage
  -> AbbreviationConcatStage
  -> GetPairwiseWerStage
  -> ManifestWriterStage
```

The YAML matches the benchmark's model parameters, batch sizes, text fields,
worker counts, GPU-memory reservations, cleanup stages, and writer count.
The benchmark harness remains responsible for four-way input sharding, GPU
recording, harness wall-clock timing, and its row/identity/coverage acceptance
gates; a tutorial run is functional pipeline evidence, not EOS performance
evidence.

## Install the audio environment

Indic Canary uses a TensorRT encoder and a TensorRT-LLM decoder. TensorRT-LLM
1.2.1 requires a CPython 3.12/CUDA 13/Torch 2.9 native stack that conflicts
with Curator's parent audio environment. The `audio_canary_trtllm` extra keeps
the parent on `audio_tensorrt` and ships a second, independently locked runtime
specification for Canary.

From the repository root:

```bash
uv sync --frozen --extra audio_canary_trtllm --no-default-groups
source .venv/bin/activate
```

That one audio extra installs the parent dependencies and the runtime manager.
At first adapter setup, the runtime manager synchronizes the packaged child
`pyproject.toml` and `uv.lock` into a lock-keyed cache. Concurrent workers share
a file lock, and later runs reuse the validated environment. The benchmark
harness performs the same setup in preflight, before its timer starts. To use a
pre-provisioned shared runtime instead, set:

```bash
export NEMO_CURATOR_INDIC_CANARY_RUNTIME_PYTHON=/shared/indic-canary-runtime/bin/python
```

The supported Canary runtime is Linux x86_64. Consumer GPUs require a native
CUDA-13-capable NVIDIA driver. Supported data-center GPUs can instead use
NVIDIA CUDA 13 forward-compatibility libraries supplied by the execution
environment. Those driver libraries are a system prerequisite, not a Python
dependency. Parakeet stays in the Curator parent and uses plain TensorRT.

## Prepare the pinned public Hindi dataset

The benchmark data setup uses the open Hugging Face dataset
`ketav/parakeet-hindi-asr` at revision
`35376a112c4b79318eeaba0c0dd1b6f1a9bf0ea0`.

```bash
python benchmarking/data_prep/prepare_audio_indic_asr_data.py \
  --output-path DATASETS_ROOT/audio_indic_asr_parakeet_hindi_531h_35376a11 \
  --cache-dir DATASETS_ROOT/_hf_cache/audio_indic_asr \
  --subset-hours 1 \
  --subset-manifest DATASETS_ROOT/audio_indic_asr_parakeet_hindi_531h_35376a11/manifest-1h.jsonl
```

The setup validates 216,169 unique mono 16 kHz clips totaling 531.7738 audio
hours. Its manifest stores paths such as `audio/<clip>.flac`; the tutorial passes
the dataset directory to `PrepareIndicASRInputStage`, which resolves those paths
before loading audio. The canonical `manifest.jsonl` remains unchanged and is
used by both benchmark entries. `manifest-1h.jsonl` is a deterministic
at-least-one-hour prefix of that same pinned cohort for the local functional
run below.

## Stage the engines

The tutorial and benchmark expect these production-compatible bundles:

```text
MODEL_WEIGHTS_ROOT/audio_indic_asr/
├── indic_canary_trtllm/engine_bfloat16_64_new/
│   ├── encoder/encoder.plan
│   ├── encoder/config.json
│   ├── decoder/config.json
│   ├── decoder/rank0.engine
│   ├── decoder/vocab.json
│   ├── preprocessor/config.json
│   └── preprocessor/mel_basis.pt
└── parakeet_indic_trt/encoder_fp16_b1_o8_m16_f8_o800_m4001/
    ├── encoder.plan
    ├── metadata.json
    └── model.nemo
```

TensorRT plans are hardware-specific. Build or obtain bundles for the GPU on
which the pipeline will run.

## Run with Xenna

Run the shared YAML runner from the repository root:

```bash
RAY_DATA_OP_RESERVATION_RATIO=0 \
RAY_DATA_STREAMING_MAX_BUFFER_SIZE_MB=4096 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/indic_asr \
  --config-name pipeline \
  dataset_dir=DATASETS_ROOT/audio_indic_asr_parakeet_hindi_531h_35376a11 \
  indic_canary_engine_dir=MODEL_WEIGHTS_ROOT/audio_indic_asr/indic_canary_trtllm/engine_bfloat16_64_new \
  parakeet_tensorrt_engine_dir=MODEL_WEIGHTS_ROOT/audio_indic_asr/parakeet_indic_trt/encoder_fp16_b1_o8_m16_f8_o800_m4001 \
  output_path=./indic_asr_output/xenna.jsonl \
  backend=xenna \
  execution_mode=streaming
```

## Run with Ray Data

Use the same manifest, audio root, engines, processor settings, and actor
counts. Change only the executor and output path:

```bash
RAY_DATA_OP_RESERVATION_RATIO=0 \
RAY_DATA_STREAMING_MAX_BUFFER_SIZE_MB=4096 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/indic_asr \
  --config-name pipeline \
  dataset_dir=DATASETS_ROOT/audio_indic_asr_parakeet_hindi_531h_35376a11 \
  indic_canary_engine_dir=MODEL_WEIGHTS_ROOT/audio_indic_asr/indic_canary_trtllm/engine_bfloat16_64_new \
  parakeet_tensorrt_engine_dir=MODEL_WEIGHTS_ROOT/audio_indic_asr/parakeet_indic_trt/encoder_fp16_b1_o8_m16_f8_o800_m4001 \
  output_path=./indic_asr_output/ray_data.jsonl \
  backend=ray_data
```

Do not reuse an output path between runs.

## Small one-GPU functional run

For a short local manifest, retain every stage and reduce only concurrency:

```bash
CUDA_VISIBLE_DEVICES=0 \
RAY_DATA_OP_RESERVATION_RATIO=0 \
RAY_DATA_STREAMING_MAX_BUFFER_SIZE_MB=4096 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/indic_asr \
  --config-name pipeline \
  dataset_dir=DATASETS_ROOT/audio_indic_asr_parakeet_hindi_531h_35376a11 \
  manifest_path=DATASETS_ROOT/audio_indic_asr_parakeet_hindi_531h_35376a11/manifest-1h.jsonl \
  indic_canary_engine_dir=/models/indic_canary_engine_for_this_gpu \
  parakeet_tensorrt_engine_dir=/models/parakeet_engine_for_this_gpu \
  output_path=/tmp/indic_asr_1h.jsonl \
  backend=xenna \
  read_concurrency=1 \
  prep_workers=1 \
  primary_workers=1 \
  fallback_workers=1
```

This still executes all eleven stages. The GPU must satisfy both engine
architectures, the CUDA 13 driver requirement, and the configured model-memory
reservations; reducing the row count does not reduce static engine memory.

## Benchmark entries and acceptance

`benchmarking/benchmarks.yaml` contains the canonical performance entries:

- `audio_indic_asr_xenna` uses the full pinned cohort and requires harness
  `exec_time_s` between 600 and 900 seconds on the standard eight-H100 EOS
  runner.
- `audio_indic_asr_raydata` uses the same cohort and correctness requirements,
  but has no runtime gate.

Both entries require one output per input identity, unchanged audio paths and
durations, finite WER, a recognized primary/fallback provenance, and complete
selected prediction text.

## Output fields

Each JSONL row retains the input identity and includes:

- `primary_model_prediction` from Indic Canary;
- `fallback_model_prediction` from Parakeet TensorRT;
- `best_prediction` and `best_prediction_source`;
- `cleaned_text` and `abbreviated_text`;
- `wer_pct`, `_skipme`, and `additional_notes`.

## Troubleshooting

- **No isolated runtime**: rerun the `uv sync` command above. Benchmark
  preflight or first adapter setup creates the child automatically. To prewarm
  it explicitly, run `python -m nemo_curator.stages.audio.inference.scripts.install_indic_canary_trtllm_runtime`,
  or set `NEMO_CURATOR_INDIC_CANARY_RUNTIME_PYTHON` to an existing runtime's
  `bin/python`.
- **CUDA initialization error**: verify the driver supports CUDA 13 and the
  engine was built for the current GPU architecture.
- **Missing audio**: `dataset_dir` must contain both `manifest.jsonl` and its
  referenced `audio/` tree.
- **Missing engine file**: compare the bundle with the complete layout above.
- **Output already exists**: choose a fresh output path.
- **EOS timing comparison**: use the benchmark harness's `exec_time_s`, not a
  shell timer around this tutorial command.
