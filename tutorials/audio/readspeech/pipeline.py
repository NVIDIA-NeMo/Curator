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
DNS Challenge Read Speech Audio Data Filtration Pipeline.

This script processes the DNS Challenge Read Speech dataset through
the AudioDataFilterStage for quality filtering and analysis.

Dataset: Microsoft DNS Challenge 5 - Read Speech (Track 1 Headset)
Source: https://github.com/microsoft/DNS-Challenge

The pipeline:
1. Creates initial manifest from read_speech WAV files (14,279 files at 48kHz)
2. Applies AudioDataFilterStage (VAD, quality filters, speaker separation)
3. Outputs filtered manifest with quality scores and timestamps

Example:
    python pipeline.py --raw_data_dir /path/to/read_speech --enable-utmos --enable-vad
"""

import argparse
import os
import shutil
import sys

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import AudioDataFilterStage
from nemo_curator.stages.audio.datasets.readspeech import CreateInitialManifestReadSpeechStage
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.text.io.writer import JsonlWriter


def create_readspeech_pipeline(args: argparse.Namespace) -> Pipeline:
    """
    Create the Read Speech audio processing pipeline.

    The pipeline combines:
    1. CreateInitialManifestReadSpeechStage - Scans directory and creates initial manifest
    2. AudioDataFilterStage - Applies quality filters with timestamp tracking
    3. AudioToDocumentStage - Converts AudioTask to DocumentBatch
    4. JsonlWriter - Writes filtered manifest to disk
    """
    pipeline = Pipeline(
        name="readspeech_audio_filter",
        description="DNS Challenge Read Speech dataset curation with AudioDataFilterStage"
    )

    pipeline.add_stage(
        CreateInitialManifestReadSpeechStage(
            raw_data_dir=args.raw_data_dir,
            max_samples=args.max_samples,
            auto_download=args.auto_download,
            batch_size=args.batch_size,
        )
    )

    pipeline.add_stage(AudioDataFilterStage(config={
        "mono_conversion": {
            "output_sample_rate": args.sample_rate,
        },
        "vad": {
            "enable": args.enable_vad,
            "min_duration_sec": args.vad_min_duration,
            "max_duration_sec": args.vad_max_duration,
            "threshold": args.vad_threshold,
            "min_interval_ms": args.vad_min_interval_ms,
            "speech_pad_ms": args.vad_speech_pad_ms,
        },
        "band_filter": {
            "enable": args.enable_band_filter,
            "band_value": args.band_value,
        },
        "utmos": {
            "enable": args.enable_utmos,
            "mos_threshold": args.utmos_mos_threshold,
        },
        "sigmos": {
            "enable": args.enable_sigmos,
            "noise_threshold": args.sigmos_noise_threshold,
            "ovrl_threshold": args.sigmos_ovrl_threshold,
        },
        "speaker_separation": {
            "enable": args.enable_speaker_separation,
            "exclude_overlaps": args.speaker_exclude_overlaps,
            "min_duration": args.speaker_min_duration,
        },
        "timestamp_mapper": {
            "passthrough_keys": [
                "band_prediction", "utmos_mos",
                "sigmos_noise", "sigmos_ovrl",
                "speaker_id", "num_speakers",
            ],
        },
    }))

    pipeline.add_stage(AudioToDocumentStage())
    pipeline.add_stage(JsonlWriter(
        path=args.output_dir,
        write_kwargs={"force_ascii": False},
    ))

    return pipeline


def main():
    parser = argparse.ArgumentParser(
        description="DNS Challenge Read Speech Audio Data Filtration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset:
  DNS Challenge Read Speech (Track 1 Headset)
  https://github.com/microsoft/DNS-Challenge

  Contains 14,279 clean read speech WAV files at 48kHz (19.3 hours, 4.88 GB download).

Examples:
  # Basic usage with UTMOS filter (5000 samples)
  python pipeline.py --raw_data_dir /path/to/read_speech --enable-utmos

  # Full pipeline with all filters
  python pipeline.py --raw_data_dir /path/to/read_speech \\
      --enable-vad --enable-utmos --enable-sigmos

  # Process all samples
  python pipeline.py --raw_data_dir /path/to/read_speech \\
      --max-samples -1 --enable-utmos
        """
    )

    # Required arguments
    parser.add_argument("--raw_data_dir", required=True,
                        help="Directory containing read_speech WAV files")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for results (default: raw_data_dir/result)")

    # Dataset selection
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Maximum samples to process (default: 5000, -1 for all)")

    # Download settings
    parser.add_argument("--auto-download", action="store_true", default=True,
                        help="Automatically download dataset (~4.88 GB) (default: True)")
    parser.add_argument("--no-auto-download", dest="auto_download", action="store_false",
                        help="Disable automatic download (expects data already exists)")

    # Resource settings
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    # General settings
    parser.add_argument("--sample_rate", type=int, default=48000, help="Target sample rate")
    parser.add_argument("--clean", action="store_true", help="Clean output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    # VAD settings
    parser.add_argument("--enable-vad", action="store_true", help="Enable VAD segmentation")
    parser.add_argument("--vad-min-duration", type=float, default=2.0, help="Min VAD segment (sec)")
    parser.add_argument("--vad-max-duration", type=float, default=60.0, help="Max VAD segment (sec)")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="VAD threshold (0-1)")
    parser.add_argument("--vad-min-interval-ms", type=int, default=500, help="Min silence to split (ms)")
    parser.add_argument("--vad-speech-pad-ms", type=int, default=300, help="Padding before/after speech (ms)")

    # Band filter settings
    parser.add_argument("--enable-band-filter", action="store_true", help="Enable band filter")
    parser.add_argument("--band-value", choices=["full_band", "narrow_band"], default="full_band")

    # UTMOS settings
    parser.add_argument("--enable-utmos", action="store_true", help="Enable UTMOS filter")
    parser.add_argument("--utmos-mos-threshold", type=float, default=3.4, help="Min UTMOS MOS (1-5)")

    # SIGMOS settings
    parser.add_argument("--enable-sigmos", action="store_true", help="Enable SIGMOS filter")
    parser.add_argument("--sigmos-noise-threshold", type=float, default=4.0, help="Min SIGMOS noise")
    parser.add_argument("--sigmos-ovrl-threshold", type=float, default=3.5, help="Min SIGMOS overall")

    # Speaker separation settings
    parser.add_argument("--enable-speaker-separation", action="store_true", help="Enable speaker sep")
    parser.add_argument("--speaker-exclude-overlaps", action="store_true", default=True,
                        help="Exclude overlapping speech (default: True)")
    parser.add_argument("--no-speaker-exclude-overlaps", dest="speaker_exclude_overlaps", action="store_false",
                        help="Allow overlapping speaker segments")
    parser.add_argument("--speaker-min-duration", type=float, default=0.8, help="Min speaker segment")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.raw_data_dir, "result")

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    if args.clean and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # Log configuration
    logger.info("=" * 70)
    logger.info("DNS Challenge Read Speech Audio Data Filtration Pipeline")
    logger.info("=" * 70)
    logger.info(f"Dataset: DNS Challenge Read Speech (Track 1 Headset)")
    logger.info(f"Raw Data Dir: {args.raw_data_dir}")
    logger.info(f"Output Dir:   {args.output_dir}")
    logger.info(f"Max Samples:  {args.max_samples}")

    enabled = []
    if args.enable_vad:
        enabled.append("VAD")
    if args.enable_band_filter:
        enabled.append("Band")
    if args.enable_utmos:
        enabled.append("UTMOS")
    if args.enable_sigmos:
        enabled.append("SIGMOS")
    if args.enable_speaker_separation:
        enabled.append("SpeakerSep")

    logger.info(f"Enabled Filters: {enabled or ['none']}")
    logger.info("=" * 70)

    pipeline = create_readspeech_pipeline(args)
    logger.info(pipeline.describe())

    logger.info("Starting pipeline execution...")

    try:
        executor = XennaExecutor(config={"execution_mode": "batch"})
        pipeline.run(executor)

        logger.info(f"Results written to {args.output_dir}/*.jsonl")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

    logger.info("Done!")


if __name__ == "__main__":
    main()
