# Hindi Indic ASR Benchmark Pipeline

This tutorial runs the exact `audio_indic_asr_xenna` or
`audio_indic_asr_raydata` entry from `benchmarking/benchmarks.yaml`. Like the
[ALM tutorial](../alm/), it uses a Hydra YAML configuration, manages the Ray
cluster lifecycle, supports Xenna and Ray Data, and keeps backend runs in
separate output directories.

The tutorial does not contain a second pipeline or benchmark implementation.
`main.py` invokes `benchmarking/run.py` with the canonical config and an exact
entry name. The resource allocation, object store, timeout, environment, input
normalization, processor graph, GPU statistics, harness timing boundary, output
validation, and requirements therefore stay exact as the benchmark evolves.

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
- One x86_64 node with 128 CPUs, at least 500 GB available for the Ray object
  store, and eight H100 GPUs for benchmark-equivalent results
- A TensorRT-LLM-compatible environment containing `tensorrt` and
  `tensorrt_llm`
- The production-compatible Indic Canary and Parakeet TensorRT engine bundles

```bash
uv sync --extra audio_cuda12
source .venv/bin/activate
```

Under a shared model root, stage the exact directory layout expected by the
canonical entries:

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

## Prepare the pinned public data

From the repository root, run the separate data-preparation command:

```bash
python benchmarking/data_prep/prepare_audio_indic_asr_data.py \
  --output-path DATASETS_ROOT/audio_indic_asr_parakeet_hindi_531h_35376a11 \
  --cache-dir DATASETS_ROOT/_hf_cache/audio_indic_asr
```

The setup pins `ketav/parakeet-hindi-asr` revision
`35376a112c4b79318eeaba0c0dd1b6f1a9bf0ea0` and validates its train split:
216,169 unique mono 16 kHz clips totaling 531.7738 audio hours. The generated
`manifest.jsonl` contains `audio_item_id`, `audio_filepath`, `duration`,
`source_lang`, and `text` for every clip.

## Run with Xenna

Run from the repository root and use a new session name:

```bash
python tutorials/audio/indic_asr/main.py \
  --config-path . \
  --config-name pipeline \
  datasets_path=DATASETS_ROOT \
  model_weights_path=MODEL_WEIGHTS_ROOT \
  results_path=./indic_asr_benchmark_output \
  session_name=indic-asr-xenna-001 \
  backend=xenna
```

This selects `audio_indic_asr_xenna`, including its 1,800-second timeout and
its `exec_time_s` requirement of 600–900 seconds.

## Run with Ray Data

Use the same data and model roots, but a different session name:

```bash
python tutorials/audio/indic_asr/main.py \
  --config-path . \
  --config-name pipeline \
  datasets_path=DATASETS_ROOT \
  model_weights_path=MODEL_WEIGHTS_ROOT \
  results_path=./indic_asr_benchmark_output \
  session_name=indic-asr-ray-data-001 \
  backend=ray_data
```

This selects `audio_indic_asr_raydata`, including its 3,600-second timeout.
Ray Data has the same correctness gates but no runtime gate.

The paired methodology holds the data, engines, processor settings, actor
counts, Ray resources, output checks, and runner instrumentation constant.
Only the executor and its configured timeout/runtime requirement differ. Run
the two backends independently; do not reuse a session name.

## Configuration

The tutorial config contains only path and entry-selection values. The
canonical config remains the single owner of benchmark behavior.

| Parameter | Description |
| --- | --- |
| `datasets_path` | Root containing the pinned dataset directory |
| `model_weights_path` | Root containing both engine directories |
| `results_path` | Root for benchmark sessions |
| `session_name` | Fresh session directory name |
| `backend` | `xenna` or `ray_data` |
| `benchmark_config` | Canonical benchmark suite config |

The canonical entry uses four manifest-reader workers, 24 preparation workers,
eight primary actors, and eight recovery actors. Indic Canary uses batch size
64, four beams, 347 maximum new tokens, and 0.1 fractions for both KV-cache
pools; Parakeet uses TensorRT engine chunking and batch size 64. The runner
starts Ray with 128 CPUs, eight GPUs, object spilling disabled, and a 500 GB
object store.

## Outputs and acceptance

Each session writes entry evidence under:

```text
RESULTS_ROOT/SESSION_NAME/ENTRY_NAME/
├── results.json
├── params.json
├── metrics.json
├── tasks.pkl
├── gpustats.csv
├── logs/
└── results/audio_indic_asr_output.jsonl
```

`results.json` contains the harness-level `exec_time_s`, exit status, timeout
state, captured environment/resource evidence, metrics, and any unmet
requirements. The tutorial exits unsuccessfully if the entry or any configured
requirement fails.

Before execution, the benchmark resolves audio paths, rejects missing or
duplicate clips, verifies Hindi text and positive durations, and creates four
absolute-path manifests. After execution, it requires exactly one output per
input identity, unchanged paths and durations, finite WER, recognized primary
or fallback provenance, and complete selected prediction text.

The accepted EOS timing gate applies only to Xenna: harness-level
`exec_time_s` must be 600–900 seconds on the standard eight-H100 runner. Ray
Data uses the identical cohort as an ungated comparison. Runs on other systems
still exercise every correctness requirement, but their performance is not an
EOS-equivalent result.

## Troubleshooting

- **Session already exists**: choose a fresh `session_name`; evidence is never
  overwritten by the tutorial.
- **Missing `tensorrt_llm`**: use the TensorRT-LLM-compatible benchmark image.
- **Missing engine file**: stage the complete engine bundle listed above.
- **Wrong row count**: rerun the pinned data preparation and do not edit or
  subset its manifest.
- **Ray sees fewer than eight GPUs**: verify `CUDA_VISIBLE_DEVICES` and the
  cluster resources before comparing runtime.
- **Xenna is outside 600–900 seconds**: inspect `results.json`, `gpustats.csv`,
  and `logs/stdouterr.log`; do not substitute the script's `time_taken_s` for
  the harness `exec_time_s` gate.
