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
  -> InferenceIndicCanaryStage (primary; TensorRT-LLM)
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
CPU preparation, cleanup, and writer calls group up to 64 rows to amortize
scheduling overhead; each row still receives the same transformations. The
writer remains a single worker appending to one output manifest.
The benchmark harness remains responsible for four-way input sharding, GPU
recording, harness wall-clock timing, and its row/identity/coverage acceptance
gates; a tutorial run is functional pipeline evidence, not EOS performance
evidence.

## Build the audio container

Indic Canary uses a TensorRT encoder and a TensorRT-LLM decoder. Its dedicated
[`Dockerfile`](../../../docker/audio/Dockerfile.trt_llm) installs Python 3.12,
TensorRT-LLM 1.2.1, and Torch 2.9.1 in one `/opt/venv` environment for Curator,
Canary, Parakeet TensorRT, and WER. This native stack is isolated from Curator's
normal dependency profiles; there is no root `trt_llm` extra or separate lockfile.

From the repository root:

```bash
docker build -f docker/audio/Dockerfile.trt_llm -t curator-indic-trtllm .
docker run --rm --gpus all curator-indic-trtllm python docker/audio/check_trt_llm.py
```

The build checks Python and native package versions. The second command also
loads the actual TRT-LLM bindings and checks the GPU; a successful build alone
does not establish driver compatibility or model inference parity.

Run the remaining tutorial commands inside this image, mounting your data,
engine bundles, and output directory (replace the host paths):

```bash
docker run --rm -it --gpus all --ipc=host \
  -v /absolute/path/to/datasets:/data \
  -v /absolute/path/to/models:/models:ro \
  -v /absolute/path/to/outputs:/outputs \
  curator-indic-trtllm bash
```

The image includes Jupyter for `indic_asr_tutorial.ipynb`; use its `/opt/venv`
kernel and direct notebook outputs to your mounted output directory. No child
environment is created at adapter setup, and no packages are installed during
inference. Do not run root `uv sync` inside this image: it would replace the
container's native stack with the normal Curator dependencies.

The supported container is Python 3.12 on Linux x86_64. Consumer GPUs require a native
CUDA-13-capable NVIDIA driver. Supported data-center GPUs can instead use
NVIDIA CUDA 13 forward-compatibility libraries supplied by the execution
environment. Those driver libraries are a system prerequisite, not a Python
dependency. Parakeet uses plain TensorRT in the same Curator environment.

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

The upstream source contract is kept separate from the runnable cohort. The
pinned source manifest has 216,169 rows / 1,914,385,701 ms and SHA-256
`407b58ccb9c74c75a5129e882b1fd000970e082e109adf95a1889592c66964a4`.
The 40,543,034,328-byte archive has SHA-256
`9f481545c1fe183eeab3a80c1a170215299c333f1cd754f4fab221eebf517c20`;
it is a superset, and only members referenced by the source manifest define
this train cohort.

The setup decodes every referenced FLAC to EOF and rejects exactly this pinned
corrupt set:

| Archive member | Payload bytes | Source duration (ms) | Decode result |
| --- | ---: | ---: | --- |
| `audio/hindi_017643.flac` | 0 | 7,457 | Empty; format not recognized |
| `audio/hindi_017655.flac` | 262,144 | 17,362 | Truncated; decoder lost sync |
| `audio/hindi_017656.flac` | 262,144 | 15,329 | Truncated; decoder lost sync |
| `audio/hindi_017666.flac` | 0 | 9,242 | Empty; format not recognized |
| `audio/hindi_018619.flac` | 0 | 8,843 | Empty; format not recognized |

Those five rows total 58,233 ms. The canonical `manifest.jsonl` used by both
benchmark entries therefore contains 216,164 unique mono 16 kHz clips /
1,914,327,468 ms (531.75763 hours), is 116,382,162 bytes, and has SHA-256
`0a8ccc0f3ff8d4ad35b3e7e104e5e093b0de14a92542d71fa6911c8045373727`.
Its paths have the form `audio/<clip>.flac`; the tutorial passes the dataset
directory to `PrepareIndicASRInputStage`, which resolves them before loading
audio. `manifest-1h.jsonl` is a deterministic at-least-one-hour prefix of that
same retained cohort for the local functional run below.

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

Choose a fresh output path for every tutorial run: the manifest writer
overwrites an existing file. The benchmark harness separately rejects an
existing output manifest.

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

- `audio_indic_asr_xenna` uses the full 216,164-row retained cohort, requires
  the exact duration, manifest byte count, and manifest hash above, and gates
  harness `exec_time_s` between 600 and 900 seconds on the standard eight-H100
  EOS runner.
- `audio_indic_asr_raydata` uses that byte-identical cohort and the same exact
  correctness requirements, but has no runtime gate.

Both entries require one output per input identity, unchanged audio paths and
durations, finite WER, a recognized primary/fallback provenance, and complete
selected prediction text.

Selecting those benchmark names does not select a Dockerfile or install
TRT-LLM. The benchmark harness invokes `python` from its existing launch
environment. Run `benchmarking/run.py` directly inside the dedicated image;
NeMo-CI can select a published copy with `--test-image`. Do not wrap this image
with `benchmarking/Dockerfile`, which re-syncs the normal `all` stack. Existing
site-specific launchers that require the removed root extra must be updated
before submission. See the environment requirements in
[`benchmarking/README.md`](../../../benchmarking/README.md).

## Output fields

Each JSONL row retains the input identity and includes:

- `primary_model_prediction` from Indic Canary;
- `fallback_model_prediction` from Parakeet TensorRT;
- `best_prediction` and `best_prediction_source`;
- `cleaned_text` and `abbreviated_text`;
- `wer_pct`, `_skipme`, and `additional_notes`.

## Troubleshooting

- **Missing TensorRT-LLM or native import failure**: rerun the root `uv sync`
  command above and activate its environment. Confirm `python --version` is
  Python 3.12; do not install this profile into an active `all` / Torch 2.11
  environment with `uv pip install`.
- **CUDA initialization error**: verify the driver supports CUDA 13 and the
  engine was built for the current GPU architecture.
- **Missing audio**: `dataset_dir` must contain both `manifest.jsonl` and its
  referenced `audio/` tree.
- **Missing engine file**: compare the bundle with the complete layout above.
- **Output already exists**: choose a fresh path before running the tutorial;
  its manifest writer overwrites existing files.
- **EOS timing comparison**: use the benchmark harness's `exec_time_s`, not a
  shell timer around this tutorial command.
