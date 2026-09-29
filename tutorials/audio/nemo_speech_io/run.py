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

"""Read NeMo speech YAML and write per-shard manifests with Opus audio."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.io import (
    NeMoSpeechAudioReader,
    NeMoSpeechWriterStage,
    finalize_nemo_speech_output,
)

_EXECUTOR_FACTORIES = {
    "ray_data": "nemo_curator.backends.ray_data.executor:RayDataExecutor",
    "xenna": "nemo_curator.backends.xenna.executor:XennaExecutor",
}


def _create_executor(backend: str, execution_mode: str) -> object:
    module_path, class_name = _EXECUTOR_FACTORIES[backend].rsplit(":", 1)
    executor_class = getattr(importlib.import_module(module_path), class_name)
    if backend == "xenna":
        return executor_class(config={"execution_mode": execution_mode})
    return executor_class()


def _local_path(value: str, *, label: str) -> Path:
    if "://" in value:
        msg = f"{label} must be a local POSIX-style path, not a URI: {value!r}"
        raise ValueError(msg)
    return Path(value).expanduser().resolve()


def _input_config_path(value: str) -> str:
    if "://" in value:
        return value
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        msg = f"Input config does not exist: {path}"
        raise FileNotFoundError(msg)
    return str(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read NeMo YAML speech data and write Opus plus one manifest per input shard."
    )
    parser.add_argument(
        "--input-config",
        help="Local path or fsspec URI for a YAML file containing input_cfg (not needed with --finalize-only)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Shared POSIX-style output directory, visible at the same path on every worker",
    )
    parser.add_argument(
        "--resume-mode",
        choices=["done_markers", "checkpoint"],
        default="done_markers",
        help="Choose exactly one scheduling authority (default: done_markers)",
    )
    parser.add_argument(
        "--checkpoint-path",
        help="Local checkpoint directory; required only when --resume-mode=checkpoint",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Publish complete receipt sets without running the pipeline (for recovery after an interrupted finalizer)",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(_EXECUTOR_FACTORIES),
        default="xenna",
        help="Execution backend (default: xenna)",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["streaming", "batch"],
        default="streaming",
        help="Xenna execution mode; ignored by Ray Data (default: streaming)",
    )
    parser.add_argument("--corpus", action="append", help="Include this corpus; repeat to include more than one")
    parser.add_argument("--language", action="append", help="Include this language; repeat to include more than one")
    parser.add_argument("--reader-workers", type=int, help="Fixed reader worker count (default: executor-managed)")
    parser.add_argument("--writer-concurrency", type=int, default=1, help="Number of writer actors (default: 1)")
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=16000,
        help="Required input and output sample rate; this pipeline does not resample (default: 16000)",
    )
    parser.add_argument(
        "--max-audio-duration-sec",
        type=float,
        default=12 * 60 * 60,
        help="Decode limit per row; values <= 0 disable the limit (default: 43200)",
    )
    parser.add_argument(
        "--no-cleanup-partial",
        action="store_true",
        help="Keep incomplete manifests and receipts in done-marker mode",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write receipts/manifests preserving original audio paths, without encoding Opus files",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.finalize_only:
        return
    if not args.input_config:
        parser.error("--input-config is required unless --finalize-only is set")
    if args.resume_mode == "done_markers" and args.checkpoint_path:
        parser.error("--checkpoint-path cannot be combined with --resume-mode=done_markers")
    if args.resume_mode == "checkpoint" and not args.checkpoint_path:
        parser.error("--checkpoint-path is required with --resume-mode=checkpoint")
    if args.reader_workers is not None and args.reader_workers < 1:
        parser.error("--reader-workers must be at least 1")
    if args.writer_concurrency < 1:
        parser.error("--writer-concurrency must be at least 1")
    if args.target_sample_rate < 1:
        parser.error("--target-sample-rate must be positive")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    output_dir = _local_path(args.output_dir, label="--output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        finalized = finalize_nemo_speech_output(str(output_dir))
    else:
        checkpoint_path = None
        if args.checkpoint_path:
            checkpoint_path = _local_path(args.checkpoint_path, label="--checkpoint-path")

        pipeline = Pipeline(
            name="nemo_speech_io",
            description="Read NeMo speech shards and atomically stage Opus plus manifest rows",
        )
        pipeline.add_stage(
            NeMoSpeechAudioReader(
                yaml_path=_input_config_path(args.input_config),
                corpus_filter=args.corpus,
                language_filter=args.language,
                output_dir=str(output_dir),
                cleanup_partial=not args.no_cleanup_partial,
                resume_mode=args.resume_mode,
                max_audio_duration_sec=args.max_audio_duration_sec,
                keep_waveform=True,
                reader_workers=args.reader_workers,
            )
        )
        pipeline.add_stage(
            NeMoSpeechWriterStage(
                output_dir=str(output_dir),
                target_sample_rate=args.target_sample_rate,
                writer_concurrency=args.writer_concurrency,
                save_audio=not args.manifest_only,
            )
        )

        executor = _create_executor(args.backend, args.execution_mode)
        pipeline.run(executor=executor, checkpoint_path=checkpoint_path)

        # Finalize only after a successful run. This validates every receipt set,
        # then atomically publishes each manifest followed by its .done marker.
        finalized = finalize_nemo_speech_output(str(output_dir))

    if finalized:
        print("Finalized manifests:")
        for manifest in finalized:
            print(f"  {manifest}")
    else:
        print("No pending NeMo speech shards were found.")


if __name__ == "__main__":
    main()
