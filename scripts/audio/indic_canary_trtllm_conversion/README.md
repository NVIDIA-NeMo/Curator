# Indic Canary → TensorRT-LLM engine conversion

Self-contained tooling to build a **prebuilt** (Indic) Canary TensorRT-LLM engine
from a `.nemo` checkpoint. `IndicCanaryLIDAdapter`
(`nemo_curator/models/audio/lid/indic_canary.py`) only *runs* such an engine—it
never builds one. This directory produces the engine it loads.

## Contents

| File | Purpose |
| --- | --- |
| `build_engine.py` | One command that runs all three build steps and validates the output. |
| `convert_checkpoint.py` | Exports the encoder ONNX, preprocessor, and decoder weights/vocab from the `.nemo`. |
| `conformer_onnx_trt.py` | Builds the FastConformer encoder TensorRT engine. |
| `requirements.txt` | The dedicated `tensorrt_llm` build stack (provides `trtllm-build`) for the isolated conversion environment. |

`convert_checkpoint.py` and `conformer_onnx_trt.py` are third-party build scripts
kept **verbatim** (and excluded from the repo's ruff config); do not edit them.
`build_engine.py` is the Curator-owned orchestrator.

### Script provenance

The two retained build scripts entered the reference Curator history in commit
[`c245cd1b6dc97c88c39052d06b67875ad8f537b8`](https://github.com/nithinraok/Curator/commit/c245cd1b6dc97c88c39052d06b67875ad8f537b8)
(`Add canary trtllm build scripts`). This integration was checked against the
final reference snapshot
[`1e2d639946f28dfd289a0a9456480385d4480c28`](https://github.com/nithinraok/Curator/commit/1e2d639946f28dfd289a0a9456480385d4480c28).
Those commits establish the exact repository provenance used here; this README
does not claim an upstream TensorRT-LLM source revision that the reference
history does not record.

## Prerequisites

**A CUDA-13-capable NVIDIA driver.** `tensorrt_llm==1.2.1` is a CUDA-13 build and needs
a driver that supports CUDA 13 (>= 580; check `nvidia-smi`). On an older driver it fails
at import with `CUDA error 35: insufficient driver`. Engines are not portable across
TensorRT versions or GPU architectures — build on the box you run inference on.

You also need the **Indic Canary NeMo fork** checked out (the AI4Bharat fork with
`CanaryMultilingualTokenizer` — stock `nemo_toolkit` cannot restore the `.nemo`), and the
**`.nemo` checkpoint** you want to convert.

## Setup

Build in a **dedicated conda env** — do **not** reuse the Curator runtime venv, since
`tensorrt_llm`'s pins (transformers / numpy / torch / cuda-python) conflict with Curator's
`audio_cuda12` stack. The order below matters: install the NeMo fork first, then layer the
`tensorrt_llm` stack on top.

```bash
# 1. Fresh conda env.
conda create -y -n canary_build python=3.12
conda activate canary_build

# 2. Install the NeMo fork (its ASR extra pulls everything convert_checkpoint.py needs).
#    Use the [asr] extra, NOT `./reinstall.sh` — the fork's installer hard-codes the
#    `.[all]` extra, which pulls a broken `clip`/`store-it` multimodal dep and aborts
#    outside the NGC PyTorch container. `[asr]` is the relevant subset and installs cleanly.
cd /path/to/NeMo            # the AI4Bharat Indic Canary fork
pip install -e ".[asr]"

# 3. MPI runtime — tensorrt_llm imports mpi4py, which needs libmpi.so at runtime.
conda install -y -c conda-forge openmpi

# 4. Layer the tensorrt_llm build stack on top (use uv — pip's resolver backtracks for
#    hours reconciling the trtllm/NeMo pins; uv resolves it in seconds). requirements.txt
#    also pins numpy<2 and pulls torchtune (needed by the NeMo ASR import chain).
cd /path/to/Curator/scripts/audio/indic_canary_trtllm_conversion
uv pip install -r requirements.txt          # downloads the ~2.5 GB tensorrt_llm wheel

# 5. Make sure numpy stayed <2 (see the note below). On a fresh env the resolve keeps
#    numpy 1.26.4, but uv will NOT downgrade an already-present numpy 2.x, so force it:
python -c "import numpy, sys; sys.exit(0 if numpy.__version__.startswith('1.26') else 1)" \
  || pip install numpy==1.26.4
```

### Runtime environment

`tensorrt_llm` needs its CUDA-13 shared libs, cuDNN, and `libmpi.so` on `LD_LIBRARY_PATH`,
and `CUDA_HOME` set (`tensorrt_llm.deep_gemm` asserts it). Export these in the same shell
you run the build from:

```bash
ENV=$(python -c "import sys; print(sys.prefix)")     # active conda env prefix
SP=$ENV/lib/python3.12/site-packages
export LD_LIBRARY_PATH="$SP/nvidia/cu13/lib:$SP/nvidia/cudnn/lib:$ENV/lib:$LD_LIBRARY_PATH"
export CUDA_HOME=/usr/local/cuda-12.6                # any installed CUDA toolkit dir
```

### Why the NeMo fork first, then trtllm (and why the numpy pin)?

`tensorrt_llm 1.2.1` and the NeMo fork pin overlapping packages to different versions
(`protobuf`, `onnx`, `numpy`, `torch`, `transformers`). Installing the fork first, then
`tensorrt_llm` with **uv**, lets uv reconcile them in one resolve — trtllm's newer pins win
where they overlap, and the fork still imports and exports ONNX at runtime.

The one pin that must be forced is **`numpy<2`** (pinned to `numpy==1.26.4` in
`requirements.txt`): `tensorrt_llm 1.2.1` requires `numpy<2`, and on numpy 2.x the decoder
`trtllm-build` step dies with `set_weights_name(): incompatible function arguments … got
array(...)` when naming bias weights (e.g. `lm_head.bias`). `numpy 1.26.4` also satisfies
the NeMo fork (`<2.0`) and `numba` (`<=2.1`), so it is the common working version.

> **Verified:** this exact flow builds a complete engine end-to-end on an RTX A5000
> (driver 595 / CUDA 13) with `tensorrt_llm==1.2.1`, Python 3.12, and the AI4Bharat NeMo
> fork — resolving to `numpy 1.26.4`, `torch 2.9.1`, `transformers 4.57.3`.

## Build (single command)

```bash
python build_engine.py \
  --nemo_model_path /models/indic-canary.nemo \
  --nemo_repo_path /path/to/NeMo \
  --engine_dir /models/indic_canary/engine_bfloat16 \
  --dtype bfloat16 \
  --max_batch_size 64 \
  --max_beam_width 4 \
  --max_audio_seconds 40 \
  --max_prompt_tokens 10
```

`build_engine.py` runs the three steps in order, then verifies the engine dir has
everything the runtime needs. On success it prints
`CANARY_TRTLLM_ENGINE_BUILD_PASSED engine_dir=...`.

- Specify exactly one checkpoint source: pass `--nemo_model_path` for a local
  `.nemo` file, or omit it and explicitly pass `--model_name <hf_id>` for a
  Hugging Face checkpoint. The builder does not silently select a stock model.
- The intermediate TRT-LLM checkpoint defaults to `<engine_dir>/tllm_checkpoint`
  (override with `--checkpoint_dir`).

### Steps it runs

1. `convert_checkpoint.py --dtype <dtype> --model_path <nemo> --output_dir <checkpoint_dir> <engine_dir>`
2. `conformer_onnx_trt.py --max_BS <bs> --max_feat_len <feat> <checkpoint_dir> <engine_dir>`
3. `trtllm-build --checkpoint_dir <checkpoint_dir>/decoder --output_dir <engine_dir>/decoder ...`

Everything below is derived from the single `--max_audio_seconds` knob so the
encoder and decoder budgets stay consistent:

- `max_feat_len = round(seconds * 100) + 1` (10 ms window shift → 100 frames/s),
  i.e. `4001` for the default `40`s (`3001` for `30`s).
- `max_output_tokens`: `~8 tokens/s + prompt`, rounded up to the next multiple of
  `128`, minus the prompt. This reproduces the shipped engines — `30`s → `246`,
  `40`s → `374` — and avoids the truncation a 128-token engine causes on long Hindi.
- `max_seq_len = max_prompt_tokens + max_output_tokens` (default `384` at `40`s).
- `max_encoder_input_len = 1 + max_feat_len / 8` (8 = Conformer subsampling), i.e.
  `501` at `40`s.

## Output layout

The resulting `--engine_dir` contains what `IndicCanaryLIDAdapter` loads (the
build fails loudly if any required artifact is missing):

```text
engine_dir/
  encoder/encoder.plan        # FastConformer encoder engine (step 2)
  encoder/config.json         # Encoder runtime/profile settings (step 2)
  decoder/rank0.engine        # TensorRT-LLM decoder engine     (step 3)
  decoder/config.json         # Transformer decoder engine   (step 3)
  decoder/vocab.json          # Canary vocabulary             (step 1)
  preprocessor/config.json    # mel front-end config          (step 1)
  preprocessor/mel_basis.pt   # mel filterbank weights        (step 1)
```

## Key parameters

| Flag | Default | Meaning |
| --- | --- | --- |
| `--nemo_model_path` | no default | Local `.nemo` checkpoint to convert; mutually exclusive with `--model_name`. |
| `--model_name` | no default | Explicit Hugging Face id; mutually exclusive with `--nemo_model_path`. One source flag is required. |
| `--nemo_repo_path` | – | Indic Canary NeMo fork checkout (prepended to `PYTHONPATH`). |
| `--dtype` | `bfloat16` | Engine precision (`float16` or `bfloat16`). |
| `--max_batch_size` | `64` | Engine max batch size. |
| `--max_beam_width` | `4` | Decoder beam width; must be **>=** inference `--num_beams`. |
| `--max_audio_seconds` | `40` | Longest audio window (s); `max_feat_len` and `max_output_tokens` are derived from it. |
| `--max_prompt_tokens` | `10` | Canary2 control-prompt length. |

## Using the engine in Curator

```python
from nemo_curator.models.audio.lid.indic_canary import IndicCanaryLIDAdapter

adapter = IndicCanaryLIDAdapter(
    engine_dir="/models/indic_canary/engine_bfloat16",
    num_beams=4,             # must be <= --max_beam_width used at build time
    max_new_tokens=1,        # LID needs only the first generated language token
    max_duration_sec=40.0,   # match the encoder window the engine was built for (--max_feat_len 4001)
)
```

In a Curator pipeline, configure the generic stage with this adapter target and
pass the engine path through `adapter_kwargs`:

```yaml
- _target_: nemo_curator.stages.audio.inference.lid.stage.AudioLIDInferenceStage
  adapter_target: nemo_curator.models.audio.lid.indic_canary.IndicCanaryLIDAdapter
  model_id: IndicCanaryLangID
  tag: secondary
  adapter_kwargs:
    engine_dir: /models/indic_canary/engine_bfloat16
```

Do not install TensorRT-LLM over Curator's locked runtime venv. To run this
engine, use a tested immutable container or isolated Python 3.12 x86_64 Linux
environment that already combines the exact Curator source with compatible
CUDA libraries and `tensorrt_llm==1.2.1`. Keep that inference environment
separate from both the standard Curator `audio_cuda12` environment and this
conversion environment.

Indic Canary uses the 10-token Canary2 control prompt, e.g.:

```text
<|startofcontext|> <|startoftranscript|> <|emo:undefined|> <|hi|> <|hi|> <|nopnc|> <|noitn|> <|noromanized|> <|notimestamp|> <|nodiarize|>
```
