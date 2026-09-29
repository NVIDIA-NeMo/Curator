#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build an (Indic) Canary TensorRT-LLM engine from a ``.nemo`` checkpoint.

Self-contained end-to-end wrapper around the two vendored build scripts in this
directory plus ``trtllm-build``:

  1. ``convert_checkpoint.py`` — export encoder ONNX + preprocessor + decoder
     weights/vocab from the ``.nemo`` checkpoint.
  2. ``conformer_onnx_trt.py`` — build the FastConformer encoder TensorRT engine.
  3. ``trtllm-build`` — build the Transformer decoder engine.

It derives the dependent sequence lengths, runs the three steps in order, and
validates that the engine directory contains the artifacts
:class:`nemo_curator.models.audio.lid.indic_canary.IndicCanaryLIDAdapter`
loads at runtime.

Prerequisites (dedicated conda env; keep it separate from the Curator venv). See
README.md "Setup" for the full recipe — in short:

  * Install the Indic Canary NeMo fork (``pip install -e ".[asr]"``) so its
    multi-softmax model classes / ``CanaryMultilingualTokenizer`` are importable.
  * ``uv pip install -r requirements.txt`` (installs tensorrt_llm -> ``trtllm-build``);
    make sure numpy stays ``<2`` (1.26.4) or the decoder build fails.
  * Set ``LD_LIBRARY_PATH`` (CUDA-13 libs, cuDNN, libmpi.so) and ``CUDA_HOME``.

``--nemo_repo_path`` is optional once the fork is installed editable, but pass it to
prepend the fork to PYTHONPATH for the conversion subprocesses.

Example
-------
    python build_engine.py \\
        --nemo_model_path /models/indic-canary.nemo \\
        --nemo_repo_path /path/to/NeMo \\
        --engine_dir /models/indic_canary/engine_bfloat16 \\
        --dtype bfloat16 --max_batch_size 8 --max_beam_width 4
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from pathlib import Path

# The vendored build scripts live next to this file.
_HERE = Path(__file__).resolve().parent
_CONVERT_CHECKPOINT = _HERE / "convert_checkpoint.py"
_CONFORMER_ONNX_TRT = _HERE / "conformer_onnx_trt.py"

# Files IndicCanaryLIDAdapter.download_weights_on_node() requires inside the engine dir.
# Kept in sync with indic_canary.py so a green build here means a loadable engine.
_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "encoder/encoder.plan",
    "encoder/config.json",
    "decoder/rank0.engine",
    "decoder/config.json",
    "decoder/vocab.json",
    "preprocessor/config.json",
    "preprocessor/mel_basis.pt",
)

# Conformer subsampling factor: encoder output frames = 1 + feat_len / SUBSAMPLING.
_SUBSAMPLING_FACTOR = 8

# Length derivation from the max audio window (see _derive_engine_lengths):
#   * 10 ms window shift -> 100 feature frames per second.
#   * ~8 decoder output tokens per second is a safe ASR upper bound.
#   * The decoder sequence budget is rounded up to a multiple of 128, matching
#     the known-good engines (30s -> 246 output tokens, 40s -> 374).
_FRAMES_PER_SECOND = 100
_OUTPUT_TOKENS_PER_SECOND = 8
_SEQ_LEN_MULTIPLE = 128


def _derive_engine_lengths(max_audio_seconds: float, max_prompt_tokens: int) -> tuple[int, int]:
    """Derive (max_feat_len, max_output_tokens) from the max audio window.

    ``max_feat_len`` is the encoder input length in feature frames; the decoder
    budget is ``~8 tokens/s + prompt`` rounded up to the next 128, minus the
    prompt. Reproduces the shipped engines: 30s -> (3001, 246), 40s -> (4001, 374).
    """
    max_feat_len = round(max_audio_seconds * _FRAMES_PER_SECOND) + 1
    raw_seq_len = round(max_audio_seconds * _OUTPUT_TOKENS_PER_SECOND) + max_prompt_tokens
    max_seq_len = -(-raw_seq_len // _SEQ_LEN_MULTIPLE) * _SEQ_LEN_MULTIPLE
    max_output_tokens = max_seq_len - max_prompt_tokens
    return max_feat_len, max_output_tokens


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Echo and run a subprocess, raising on non-zero exit."""
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}\n", flush=True)
    result = subprocess.run(cmd, cwd=cwd, env=env, check=False)  # noqa: S603 - fixed argv, no shell
    if result.returncode != 0:
        message = f"Build step failed (exit {result.returncode}): {printable}"
        raise RuntimeError(message)


def _build_env(args: argparse.Namespace) -> dict[str, str]:
    """Prepend the NeMo fork (and this dir) to PYTHONPATH for the subprocesses."""
    env = os.environ.copy()
    extra_paths: list[str] = []
    if args.nemo_repo_path:
        repo = Path(args.nemo_repo_path).resolve()
        if not repo.is_dir():
            message = f"--nemo_repo_path does not exist: {repo}"
            raise FileNotFoundError(message)
        extra_paths.append(str(repo))
    extra_paths.append(str(_HERE))
    existing = env.get("PYTHONPATH", "")
    if existing:
        extra_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)
    return env


def _convert_checkpoint(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    engine_dir: Path,
    env: dict[str, str],
) -> None:
    """Step 1: export encoder ONNX + preprocessor + decoder weights/vocab."""
    cmd = ["python", str(_CONVERT_CHECKPOINT), f"--dtype={args.dtype}", "--output_dir", str(checkpoint_dir)]
    if args.nemo_model_path:
        cmd += ["--model_path", args.nemo_model_path]
    else:
        cmd += ["--model_name", args.model_name]
    cmd.append(str(engine_dir))  # positional engine_dir (receives preprocessor/ + decoder/vocab.json)
    _run(cmd, cwd=_HERE, env=env)


def _build_encoder(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    engine_dir: Path,
    env: dict[str, str],
) -> None:
    """Step 2: build the FastConformer encoder TensorRT engine (encoder/encoder.plan)."""
    cmd = [
        "python",
        str(_CONFORMER_ONNX_TRT),
        "--max_BS",
        str(args.max_batch_size),
        "--max_feat_len",
        str(args.max_feat_len),
        str(checkpoint_dir),
        str(engine_dir),
    ]
    _run(cmd, cwd=_HERE, env=env)


def _build_decoder(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    engine_dir: Path,
    env: dict[str, str],
) -> None:
    """Step 3: build the Transformer decoder engine with trtllm-build."""
    trtllm_build = shutil.which("trtllm-build")
    if trtllm_build is None:
        message = "trtllm-build not found on PATH. Install tensorrt_llm (pip install -r requirements.txt)."
        raise FileNotFoundError(message)
    max_seq_len = args.max_prompt_tokens + args.max_output_tokens
    max_encoder_output_len = 1 + args.max_feat_len // _SUBSAMPLING_FACTOR
    cmd = [
        trtllm_build,
        "--checkpoint_dir",
        str(checkpoint_dir / "decoder"),
        "--output_dir",
        str(engine_dir / "decoder"),
        "--moe_plugin",
        "disable",
        "--max_beam_width",
        str(args.max_beam_width),
        "--max_batch_size",
        str(args.max_batch_size),
        "--max_seq_len",
        str(max_seq_len),
        "--max_input_len",
        str(args.max_prompt_tokens),
        "--max_encoder_input_len",
        str(max_encoder_output_len),
        "--gemm_plugin",
        args.dtype,
        "--bert_attention_plugin",
        "disable",
        "--gpt_attention_plugin",
        args.dtype,
        "--remove_input_padding",
        "enable",
    ]
    _run(cmd, env=env)


def _validate_engine(engine_dir: Path) -> None:
    missing = [rel for rel in _REQUIRED_ARTIFACTS if not (engine_dir / rel).is_file()]
    if missing:
        message = (
            f"Engine dir '{engine_dir}' is missing required file(s): {missing}. "
            "The build did not produce a complete IndicCanaryLIDAdapter engine."
        )
        raise FileNotFoundError(message)
    print("CANARY_TRTLLM_ENGINE_ARTIFACTS_PRESENT " + " ".join(_REQUIRED_ARTIFACTS))


def build(args: argparse.Namespace) -> None:
    for script in (_CONVERT_CHECKPOINT, _CONFORMER_ONNX_TRT):
        if not script.is_file():
            message = f"Vendored build script missing: {script}"
            raise FileNotFoundError(message)
    if args.nemo_model_path and not Path(args.nemo_model_path).is_file():
        message = f"--nemo_model_path does not exist: {args.nemo_model_path}"
        raise FileNotFoundError(message)

    # Derive the dependent sequence lengths from the single audio-window knob.
    args.max_feat_len, args.max_output_tokens = _derive_engine_lengths(args.max_audio_seconds, args.max_prompt_tokens)

    engine_dir = Path(args.engine_dir).resolve()
    engine_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else engine_dir / "tllm_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env = _build_env(args)

    _convert_checkpoint(args, checkpoint_dir, engine_dir, env)
    _build_encoder(args, checkpoint_dir, engine_dir, env)
    _build_decoder(args, checkpoint_dir, engine_dir, env)

    _validate_engine(engine_dir)
    max_seq_len = args.max_prompt_tokens + args.max_output_tokens
    print(f"\nCANARY_TRTLLM_ENGINE_BUILD_PASSED engine_dir={engine_dir}")
    print(
        f"  dtype={args.dtype} max_batch_size={args.max_batch_size} "
        f"max_beam_width={args.max_beam_width} max_audio_seconds={args.max_audio_seconds} "
        f"max_feat_len={args.max_feat_len} max_output_tokens={args.max_output_tokens} "
        f"max_seq_len={max_seq_len}"
    )
    print(f"Use it with IndicCanaryLIDAdapter(engine_dir=...) and num_beams <= {args.max_beam_width}.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--nemo_model_path",
        default=None,
        help="Local AI4Bharat Indic Canary .nemo checkpoint to convert.",
    )
    source.add_argument(
        "--model_name",
        default=None,
        help="Explicit Indic Canary Hugging Face model id. Stock Canary checkpoints do not provide the required language inventory.",
    )
    parser.add_argument(
        "--nemo_repo_path",
        default=None,
        help="Path to the Indic Canary NeMo fork checkout, prepended to PYTHONPATH so its "
        "multi-softmax model classes import during conversion. Recommended.",
    )
    parser.add_argument(
        "--engine_dir",
        required=True,
        help="Output engine directory, consumed by IndicCanaryLIDAdapter.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help="Intermediate TRT-LLM checkpoint dir (default: <engine_dir>/tllm_checkpoint).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Inference precision for the engine (default: bfloat16).",
    )
    parser.add_argument("--max_batch_size", type=int, default=64, help="Engine max batch size.")
    parser.add_argument(
        "--max_beam_width",
        type=int,
        default=4,
        help="Decoder beam width. Must be >= the --num_beams used at inference.",
    )
    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=40.0,
        help="Longest audio window the engine must handle, in seconds. Both the encoder "
        "feature length and the decoder token budget are derived from this "
        "(e.g. 30s -> feat_len 3001 / 246 tokens, 40s -> 4001 / 374).",
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=10,
        help="Canary2 control-prompt length (fixed 10 tokens for Indic Canary).",
    )
    args = parser.parse_args(argv)
    for field_name in ("max_batch_size", "max_beam_width", "max_prompt_tokens"):
        if getattr(args, field_name) <= 0:
            parser.error(f"--{field_name} must be positive")
    if not math.isfinite(args.max_audio_seconds) or args.max_audio_seconds <= 0:
        parser.error("--max_audio_seconds must be a finite positive number")
    return args


if __name__ == "__main__":
    build(parse_args())
