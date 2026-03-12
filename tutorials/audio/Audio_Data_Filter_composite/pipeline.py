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
Audio Data Filtration Pipeline using NeMo Curator Pipeline API.

AudioDataFilterStage is a CompositeStage that decomposes into independent
pipeline stages. The executor schedules them with cross-file parallelism.

Decomposed pipeline (when all filters + speaker separation enabled):
    1. MonoConversion (1:1)
    2. VAD batch mode (1:1, 1 item -> N segment items)
    3. BandFilter (1:1, filter items)
    4. NISQA (1:1, filter items)
    5. SIGMOS (1:1, filter items)
    6. UTMOS (1:1, filter items)
    7. SegmentConcatenation (1:1, M items -> 1 item + timestamp mappings)
    8. SpeakerSeparation (1:N fan-out, 1 task per speaker)
    9-13. Per-speaker: VAD + Band + NISQA + SIGMOS + UTMOS
    14. TimestampMapper (1:1, resolve to original file positions)

Timestamp Mapping:
    SegmentConcatenationStage stores segment-to-original mappings in
    task._metadata["segment_mappings"]. TimestampMapperStage at the end
    resolves each segment's position back to the original file.

Example:
    python pipeline.py --raw_data_dir ./audio_data --output_dir ./output --enable-nisqa
"""

import argparse
import glob
import os
import shutil
import sys

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import (
    AudioDataFilterStage,
    AudioDataFilterConfig,
)
from nemo_curator.stages.audio.advance_pipelines.Audio_data_filter.config import (
    SUPPORTED_AUDIO_FORMATS,
    DEFAULT_OUTPUT_FORMAT,
)
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.tasks import AudioBatch


def create_pipeline(args: argparse.Namespace) -> Pipeline:
    """
    Create an audio data filtration pipeline using AudioDataFilterStage.
    
    AudioDataFilterStage is a CompositeStage -- the pipeline framework
    automatically decomposes it into independent stages at build time.
    Each stage gets its own resource allocation via gpu_resources.
    """
    pipeline = Pipeline(
        name="audio_data_filter_pipeline",
        description="Audio curation pipeline with consistent timestamp mapping"
    )
    
    config = AudioDataFilterConfig(
        sample_rate=args.sample_rate,
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
        
        enable_utmos=args.enable_utmos,
        utmos_mos_threshold=args.utmos_mos_threshold,
        
        enable_speaker_separation=args.enable_speaker_separation,
        speaker_exclude_overlaps=args.speaker_exclude_overlaps,
        speaker_min_duration=args.speaker_min_duration,
    )
    
    audio_filter_stage = AudioDataFilterStage(
        config=config,
        gpu_resources=Resources(cpus=args.cpus, gpus=args.gpus),
    )

    pipeline.add_stage(audio_filter_stage)

    pipeline.add_stage(AudioToDocumentStage().with_(batch_size=1))
    pipeline.add_stage(JsonlWriter(
        path=args.output_dir,
        write_kwargs={"force_ascii": False},
    ))

    return pipeline


def load_audio_tasks(input_dir: str, recursive: bool = False, 
                     formats: tuple = SUPPORTED_AUDIO_FORMATS) -> list:
    """
    Load audio files as AudioBatch tasks.
    
    Args:
        input_dir: Directory containing audio files
        recursive: Whether to search recursively in subdirectories
        formats: Tuple of supported audio file extensions (e.g., (".wav", ".mp3", ".flac"))
    
    Returns:
        List of AudioBatch tasks for processing
    
    Supported formats: wav, mp3, flac, ogg, m4a, aac, wma, opus, webm
    Note: Non-wav formats require ffmpeg to be installed on the system.
    """
    audio_files = []
    
    for ext in formats:
        ext = ext if ext.startswith('.') else f'.{ext}'
        pattern = f"*{ext}"
        
        if recursive:
            found = glob.glob(os.path.join(input_dir, "**", pattern), recursive=True)
        else:
            found = glob.glob(os.path.join(input_dir, pattern))
        
        audio_files.extend(found)
    
    audio_files = sorted(set(audio_files))
    
    if not audio_files:
        format_str = ", ".join(formats)
        logger.warning(f"No audio files found in {input_dir} (searched for: {format_str})")
        return []
    
    format_counts = {}
    for f in audio_files:
        ext = os.path.splitext(f)[1].lower()
        format_counts[ext] = format_counts.get(ext, 0) + 1
    format_summary = ", ".join(f"{ext}: {count}" for ext, count in sorted(format_counts.items()))
    logger.info(f"Found {len(audio_files)} audio files ({format_summary})")
    
    tasks = []
    for i, audio_file in enumerate(audio_files):
        task = AudioBatch(
            data={"audio_filepath": audio_file},
            task_id=f"audio_{i:05d}",
            dataset_name="audio_filter"
        )
        tasks.append(task)
    
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Audio Data Filtration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Timestamp Mapping:
  All output segments maintain accurate timestamps back to the original file:
  - original_file: Path to source audio
  - original_start_ms: Start position in original file (milliseconds)
  - original_end_ms: End position in original file (milliseconds)

Examples:
  # Basic usage with NISQA filter
  python pipeline.py --raw_data_dir ./audio --output_dir ./output --enable-nisqa
  
  # Full pipeline with all filters and speaker separation
  python pipeline.py --raw_data_dir ./audio --output_dir ./output \\
      --enable-vad --enable-nisqa --enable-sigmos --enable-utmos --enable-speaker-separation
        """
    )
    
    # Required arguments
    parser.add_argument("--raw_data_dir", required=True, help="Input directory with audio files")
    parser.add_argument("--output_dir", default=None, help="Output directory for results")
    
    # Resource settings
    parser.add_argument("--gpus", type=float, default=1.0, help="GPU allocation per worker")
    parser.add_argument("--cpus", type=float, default=1.0, help="CPU allocation per worker")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    
    # General settings
    parser.add_argument("--sample_rate", type=int, default=48000, help="Sample rate")
    parser.add_argument("--recursive", action="store_true", help="Search recursively")
    parser.add_argument("--clean", action="store_true", help="Clean output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    # Audio format settings
    parser.add_argument(
        "--input-formats", 
        type=str, 
        nargs="+",
        default=None,
        help="Audio formats to process (e.g., wav mp3 flac). Default: all supported formats"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default=DEFAULT_OUTPUT_FORMAT,
        choices=["wav", "mp3", "flac", "ogg", "m4a"],
        help="Output format for extracted segments (default: wav)"
    )
    
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
    
    # UTMOS settings
    parser.add_argument("--enable-utmos", action="store_true", help="Enable UTMOS filter")
    parser.add_argument("--utmos-mos-threshold", type=float, default=3.5, help="Min UTMOS MOS")
    
    # Speaker separation settings
    parser.add_argument("--enable-speaker-separation", action="store_true", help="Enable speaker sep")
    parser.add_argument("--speaker-exclude-overlaps", action="store_true", default=True)
    parser.add_argument("--speaker-min-duration", type=float, default=0.8, help="Min speaker segment")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.raw_data_dir, "result")
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    if not os.path.isdir(args.raw_data_dir):
        logger.error(f"Input directory not found: {args.raw_data_dir}")
        sys.exit(1)
    
    if args.clean and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    
    # Log configuration
    logger.info("=" * 70)
    logger.info("Audio Data Filtration Pipeline")
    logger.info("=" * 70)
    logger.info(f"Input:  {args.raw_data_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"GPUs:   {args.gpus}")
    
    enabled = []
    if args.enable_vad:
        enabled.append("VAD")
    if args.enable_band_filter:
        enabled.append("Band")
    if args.enable_nisqa:
        enabled.append("NISQA")
    if args.enable_sigmos:
        enabled.append("SIGMOS")
    if args.enable_utmos:
        enabled.append("UTMOS")
    if args.enable_speaker_separation:
        enabled.append("SpeakerSep")
    
    logger.info(f"Enabled: {enabled or ['none']}")
    logger.info(f"Timestamp Mapping: ENABLED (always)")
    logger.info("=" * 70)
    
    pipeline = create_pipeline(args)
    logger.info(pipeline.describe())
    
    if args.input_formats:
        input_formats = tuple(
            f".{fmt.lstrip('.')}" for fmt in args.input_formats
        )
        logger.info(f"Input formats: {', '.join(input_formats)}")
    else:
        input_formats = SUPPORTED_AUDIO_FORMATS
        logger.info(f"Input formats: all supported ({', '.join(input_formats)})")
    
    # Load input tasks
    tasks = load_audio_tasks(args.raw_data_dir, args.recursive, input_formats)
    if not tasks:
        logger.error("No audio files to process")
        sys.exit(1)
    
    logger.info("Starting pipeline execution...")
    
    try:
        from datetime import datetime

        from nemo_curator.backends.xenna import XennaExecutor

        start_time = datetime.now()
        executor = XennaExecutor()
        pipeline.run(executor, initial_tasks=tasks)
        duration = (datetime.now() - start_time).total_seconds()

        logger.info(f"Completed in {duration:.2f}s")
        logger.info(f"Results written to {args.output_dir}/*.jsonl")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

    logger.info("Done!")


if __name__ == "__main__":
    main()

