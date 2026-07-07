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

"""
Chatterbox TTS Synthesis Pipeline.

Reads a JSONL manifest of conversation turns, synthesises per-turn audio with
ChatterboxTTS (English or multilingual) using reference-voice cloning, and
writes an enriched manifest with the generated ``audio_filepath`` per turn.

Each input manifest line is one turn, e.g.::

    {"conversation_id": "conv001", "speaker": "Alice", "utterance": "Hello there."}

Example:
    python pipeline.py \\
        --input-manifest /data/turns.jsonl \\
        --reference-voices-dataset /data/reference_voices \\
        --output-dir /data/tts_output
"""

import argparse
import importlib
import os
import shutil
import sys
import time

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import ManifestReader
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.audio.tts import ChatterboxTTSStage
from nemo_curator.stages.text.io.writer import JsonlWriter

_EXECUTOR_FACTORIES = {
    "xenna": "nemo_curator.backends.xenna:XennaExecutor",
    "ray_data": "nemo_curator.backends.ray_data:RayDataExecutor",
}


def _create_executor(backend: str, **kwargs) -> object:
    if backend not in _EXECUTOR_FACTORIES:
        msg = f"Unknown backend '{backend}'. Choose from: {list(_EXECUTOR_FACTORIES)}"
        raise ValueError(msg)
    module_path, class_name = _EXECUTOR_FACTORIES[backend].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)(**kwargs)


def create_pipeline(args: argparse.Namespace) -> Pipeline:
    """Create the Chatterbox TTS synthesis pipeline.

    1. ManifestReader        - Reads conversation-turn JSONL into AudioTasks
    2. ChatterboxTTSStage    - Synthesises per-turn audio (GPU)
    3. AudioToDocumentStage  - Converts AudioTask to DocumentBatch
    4. JsonlWriter           - Writes the enriched manifest to JSONL
    """
    pipeline = Pipeline(
        name="chatterbox_tts",
        description="Chatterbox TTS conversation-turn synthesis",
    )

    pipeline.add_stage(ManifestReader(manifest_path=args.input_manifest))

    pipeline.add_stage(
        ChatterboxTTSStage(
            output_audio_dir=args.output_audio_dir,
            reference_voices_dataset=args.reference_voices_dataset,
            language=args.language,
            device=args.device,
            cache_dir=args.cache_dir,
            sample_rate=args.sample_rate,
            cfg_weight=args.cfg_weight,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            max_reference_duration=args.max_reference_duration,
        ).with_(batch_size=1)
    )

    pipeline.add_stage(AudioToDocumentStage())

    result_dir = os.path.join(args.output_dir, "result")
    if args.clean and os.path.isdir(result_dir):
        shutil.rmtree(result_dir)

    pipeline.add_stage(
        JsonlWriter(
            path=result_dir,
            write_kwargs={"force_ascii": False},
        )
    )

    return pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chatterbox TTS conversation-turn synthesis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-manifest",
        required=True,
        help="Path to input JSONL manifest of conversation turns",
    )
    parser.add_argument(
        "--reference-voices-dataset",
        required=True,
        help="Root directory of reference audio (wavs/<dialog>/<speaker>.wav or MLS <spk>/<book>/<seg>.flac)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output directory for the result manifest",
    )
    parser.add_argument(
        "--output-audio-dir",
        default=None,
        help="Directory for generated WAV files (default: <output-dir>/audio)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="ISO 639-1 code for the multilingual model (omit for English-only)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for inference (default: cuda)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace cache directory for Chatterbox model weights",
    )
    parser.add_argument("--sample-rate", type=int, default=24000, help="Output WAV sample rate")
    parser.add_argument("--cfg-weight", type=float, default=0.5, help="Classifier-free guidance weight")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion exaggeration")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument(
        "--max-reference-duration",
        type=float,
        default=60.0,
        help="Maximum seconds of reference speech to use",
    )
    parser.add_argument("--clean", action="store_true", help="Remove existing result directory before running")
    parser.add_argument(
        "--backend",
        choices=["xenna", "ray_data"],
        default="xenna",
        help="Execution backend: 'xenna' (default) or 'ray_data'",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose (DEBUG) logging")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.output_audio_dir is None:
        args.output_audio_dir = os.path.join(args.output_dir, "audio")

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_audio_dir, exist_ok=True)

    pipeline = create_pipeline(args)
    logger.info(pipeline.describe())

    executor = _create_executor(args.backend)

    logger.info(f"Starting Chatterbox TTS pipeline (backend: {args.backend})...")
    t0 = time.monotonic()
    try:
        pipeline.run(executor)
    except Exception as e:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        logger.exception(f"Pipeline failed after {elapsed:.2f}s: {e}")
        sys.exit(1)
    elapsed = time.monotonic() - t0
    logger.info(f"Pipeline completed in {elapsed:.2f}s ({elapsed / 60:.2f} min)")
    logger.info(f"Results written to {os.path.join(args.output_dir, 'result')}/*.jsonl")


if __name__ == "__main__":
    main()
