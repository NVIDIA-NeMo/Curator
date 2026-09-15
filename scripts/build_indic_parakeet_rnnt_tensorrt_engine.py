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

"""Build an FP16 TensorRT encoder bundle for an Indic Parakeet RNN-T model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from tensorrt_encoder_utils import build_encoder_bundle

if TYPE_CHECKING:
    import torch


def _load_model(model_path: Path) -> torch.nn.Module:
    import nemo.collections.asr as nemo_asr
    import torch

    if not torch.cuda.is_available():
        msg = "Building an Indic Parakeet TensorRT engine requires CUDA"
        raise RuntimeError(msg)
    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(model_path),
        map_location=torch.device("cuda"),
    )
    if not hasattr(model, "encoder") or not hasattr(model, "decoder") or not hasattr(model, "joint"):
        msg = "The supplied NeMo checkpoint is not an RNN-T encoder-decoder model"
        raise TypeError(msg)
    model.eval().to(dtype=torch.float16)
    return model


def build_bundle(args: argparse.Namespace) -> None:
    model_path = Path(args.model).resolve()
    if not model_path.is_file() or model_path.suffix != ".nemo":
        msg = f"Local NeMo checkpoint not found: {model_path}"
        raise FileNotFoundError(msg)

    model = _load_model(model_path)
    feature_count = int(getattr(model.encoder, "_feat_in", model.cfg.encoder.feat_in))
    subsampling_factor = int(model.encoder.subsampling_factor)
    sample_rate = int(model.cfg.preprocessor.sample_rate)
    vocabulary_size = int(getattr(model.joint, "_vocab_size", model.cfg.joint.num_classes))

    build_encoder_bundle(
        model,
        model_path,
        Path(args.output_dir).resolve(),
        args=args,
        metadata={
            "model_type": "indic_parakeet_rnnt",
            "sample_rate": sample_rate,
            "feature_count": feature_count,
            "subsampling_factor": subsampling_factor,
            "vocabulary_size": vocabulary_size,
            "max_symbols_per_step": args.max_symbols_per_step,
        },
        temporary_prefix=".indic-parakeet-rnnt-",
        parity_message="INDIC_PARAKEET_RNNT_TENSORRT_ENCODER_PARITY_PASSED",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the Indic Parakeet RNN-T .nemo checkpoint")
    parser.add_argument("--output-dir", required=True, help="Destination engine bundle directory")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--min-frames", type=int, default=8, help="Minimum input feature frames")
    parser.add_argument("--opt-frames", type=int, default=800, help="Optimization input feature frames")
    parser.add_argument("--max-frames", type=int, default=4001, help="Maximum input feature frames")
    parser.add_argument("--max-symbols-per-step", type=int, default=10)
    parser.add_argument("--workspace-gb", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.min_batch <= args.opt_batch <= args.max_batch:
        parser.error("batch profile must satisfy 1 <= min-batch <= opt-batch <= max-batch")
    if not 1 <= args.min_frames <= args.opt_frames <= args.max_frames:
        parser.error("frame profile must satisfy 1 <= min-frames <= opt-frames <= max-frames")
    if args.max_frames < 4001:  # noqa: PLR2004
        parser.error("max-frames must be at least 4001 to support 40-second audio")
    if args.max_symbols_per_step < 1:
        parser.error("max-symbols-per-step must be at least 1")
    if args.workspace_gb < 1:
        parser.error("workspace-gb must be at least 1")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_bundle(parse_args())
