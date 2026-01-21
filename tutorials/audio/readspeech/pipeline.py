# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
1. Creates initial manifest from read_speech WAV files (~14,000+ files at 48kHz)
2. Applies AudioDataFilterStage (VAD, quality filters, speaker separation)
3. Outputs filtered manifest with quality scores and timestamps

Example:
    python pipeline.py --raw_data_dir /path/to/read_speech --enable-nisqa --enable-vad
"""

import argparse
import json
import os
import shutil
import sys

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import AudioDataFilterConfig, AudioDataFilterStage
from nemo_curator.stages.audio.datasets.readspeech import CreateInitialManifestReadSpeechStage


def create_readspeech_pipeline(args: argparse.Namespace) -> Pipeline:
    """
    Create the Read Speech audio processing pipeline.

    The pipeline combines:
    1. CreateInitialManifestReadSpeechStage - Scans directory and creates initial manifest
    2. AudioDataFilterStage - Applies quality filters with timestamp tracking
    """
    pipeline = Pipeline(
        name="readspeech_audio_filter",
        description="DNS Challenge Read Speech dataset curation with AudioDataFilterStage"
    )

    # Stage 1: Create initial manifest from read_speech directory
    # Auto-downloads dataset if enabled (default=True)
    pipeline.add_stage(
        CreateInitialManifestReadSpeechStage(
            raw_data_dir=args.raw_data_dir,
            max_samples=args.max_samples,
            auto_download=args.auto_download,
            download_parts=args.download_parts,
        ).with_(batch_size=args.batch_size)
    )

    # Stage 2: AudioDataFilterStage for quality filtering
    config = AudioDataFilterConfig(
        # Resource allocation
        cpus=args.cpus,
        gpus=args.gpus,

        # General
        sample_rate=args.sample_rate,

        # VAD
        enable_vad=args.enable_vad,
        vad_min_duration_sec=args.vad_min_duration,
        vad_max_duration_sec=args.vad_max_duration,
        vad_threshold=args.vad_threshold,

        # Quality Filters
        enable_band_filter=args.enable_band_filter,
        band_value=args.band_value,

        enable_nisqa=args.enable_nisqa,
        nisqa_mos_threshold=args.nisqa_mos_threshold,
        nisqa_noi_threshold=args.nisqa_noi_threshold,

        enable_sigmos=args.enable_sigmos,
        sigmos_noise_threshold=args.sigmos_noise_threshold,
        sigmos_ovrl_threshold=args.sigmos_ovrl_threshold,

        # Speaker Separation
        enable_speaker_separation=args.enable_speaker_separation,
        speaker_exclude_overlaps=args.speaker_exclude_overlaps,
        speaker_min_duration=args.speaker_min_duration,
    )

    audio_filter_stage = AudioDataFilterStage(config=config).with_(
        batch_size=args.batch_size,
    )

    pipeline.add_stage(audio_filter_stage)

    return pipeline


def save_results(results: list, output_dir: str, clean_intermediate: bool = True) -> str:
    """
    Save pipeline results to a single manifest.jsonl file.
    
    Args:
        results: List of result dictionaries
        output_dir: Output directory
        clean_intermediate: If True, remove intermediate UUID-based JSONL files
        
    Returns:
        Path to the manifest.jsonl file
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.jsonl")

    with open(manifest_path, "w") as f:
        for entry in results:
            clean_entry = {}
            for key, value in entry.items():
                if hasattr(value, "item"):
                    clean_entry[key] = value.item()
                elif isinstance(value, (int, float, str, bool, type(None))):
                    clean_entry[key] = value
                else:
                    clean_entry[key] = str(value)
            # Use ensure_ascii=False to avoid escaping forward slashes
            f.write(json.dumps(clean_entry, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(results)} results to {manifest_path}")
    
    # Clean up intermediate UUID-based JSONL files
    if clean_intermediate:
        intermediate_count = 0
        for filename in os.listdir(output_dir):
            if filename.endswith(".jsonl") and filename != "manifest.jsonl":
                filepath = os.path.join(output_dir, filename)
                try:
                    os.remove(filepath)
                    intermediate_count += 1
                except OSError:
                    pass
        if intermediate_count > 0:
            logger.info(f"Cleaned up {intermediate_count} intermediate JSONL files")
    
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="DNS Challenge Read Speech Audio Data Filtration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset:
  DNS Challenge Read Speech (Track 1 Headset)
  https://github.com/microsoft/DNS-Challenge

  Contains ~14,000+ clean read speech WAV files at 48kHz.

Examples:
  # Basic usage with NISQA filter (5000 samples)
  python pipeline.py --raw_data_dir /path/to/read_speech --enable-nisqa

  # Full pipeline with all filters
  python pipeline.py --raw_data_dir /path/to/read_speech \\
      --enable-vad --enable-nisqa --enable-sigmos

  # Process all samples
  python pipeline.py --raw_data_dir /path/to/read_speech \\
      --max-samples -1 --enable-nisqa
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
                        help="Automatically download dataset (default: True)")
    parser.add_argument("--no-auto-download", dest="auto_download", action="store_false",
                        help="Disable automatic download (expects data already exists)")
    parser.add_argument("--download-parts", type=int, default=1, choices=range(1, 7),
                        help="Number of parts to download (1-6). 1=~30GB partial, 6=~182GB full (default: 1)")

    # Resource settings
    parser.add_argument("--gpus", type=float, default=1.0, help="GPU allocation per worker")
    parser.add_argument("--cpus", type=float, default=1.0, help="CPU allocation per worker")
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

    # Band filter settings
    parser.add_argument("--enable-band-filter", action="store_true", help="Enable band filter")
    parser.add_argument("--band-value", choices=["full_band", "narrow_band"], default="full_band")

    # NISQA settings
    parser.add_argument("--enable-nisqa", action="store_true", help="Enable NISQA filter")
    parser.add_argument("--nisqa-mos-threshold", type=float, default=4.5, help="Min NISQA MOS")
    parser.add_argument("--nisqa-noi-threshold", type=float, default=4.3, help="Min NISQA noisiness")

    # SIGMOS settings
    parser.add_argument("--enable-sigmos", action="store_true", help="Enable SIGMOS filter")
    parser.add_argument("--sigmos-noise-threshold", type=float, default=4.0, help="Min SIGMOS noise")
    parser.add_argument("--sigmos-ovrl-threshold", type=float, default=3.5, help="Min SIGMOS overall")

    # Speaker separation settings
    parser.add_argument("--enable-speaker-separation", action="store_true", help="Enable speaker sep")
    parser.add_argument("--speaker-exclude-overlaps", action="store_true", default=True)
    parser.add_argument("--speaker-min-duration", type=float, default=0.8, help="Min speaker segment")

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.raw_data_dir, "result")

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    # Clean output if requested
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
    logger.info(f"GPUs: {args.gpus}")

    enabled = []
    if args.enable_vad:
        enabled.append("VAD")
    if args.enable_band_filter:
        enabled.append("Band")
    if args.enable_nisqa:
        enabled.append("NISQA")
    if args.enable_sigmos:
        enabled.append("SIGMOS")
    if args.enable_speaker_separation:
        enabled.append("SpeakerSep")

    logger.info(f"Enabled Filters: {enabled or ['none']}")
    logger.info("=" * 70)

    # Create and run pipeline
    pipeline = create_readspeech_pipeline(args)
    logger.info(pipeline.describe())

    # Execute with XennaExecutor
    logger.info("Starting pipeline execution...")

    try:
        executor = XennaExecutor()
        results = pipeline.run(executor)

        # Collect and save results
        all_results = []
        for result in results:
            if result is not None:
                if isinstance(result.data, list):
                    all_results.extend(result.data)
                else:
                    all_results.append(result.data)

        logger.info(f"Pipeline completed with {len(all_results)} output segments")

        if all_results:
            save_results(all_results, args.output_dir)

            # Log sample output
            sample = all_results[0]
            logger.info("Sample output:")
            logger.info(f"  audio_filepath: {sample.get('audio_filepath', 'N/A')}")
            logger.info(f"  sample_rate: {sample.get('sample_rate', 'N/A')}")
            logger.info(f"  original_start_ms: {sample.get('original_start_ms', 'N/A')}")
            logger.info(f"  original_end_ms: {sample.get('original_end_ms', 'N/A')}")
        else:
            logger.warning("No segments passed filters")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logger.info("Done!")


if __name__ == "__main__":
    main()

