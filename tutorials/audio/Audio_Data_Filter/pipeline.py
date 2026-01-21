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
Audio Data Filtration Pipeline using NeMo Curator Pipeline API.

This script builds an audio curation pipeline using the AudioDataFilterStage
module which maintains consistent timestamp mapping throughout all transformations.

The pipeline performs:
1. Mono conversion
2. VAD segmentation (optional)
3. Quality filtering (NISQA, SIGMOS, Band)
4. Concatenation with timestamp tracking
5. Speaker separation (optional)
6. Per-speaker VAD and quality filtering
7. Timestamp mapping back to original file positions

Timestamp Mapping:
    The AudioDataFilterStage uses TimestampTracker to ensure that all
    output segments contain accurate original_start_ms and original_end_ms
    values that point back to the source audio file, even after:
    - VAD segmentation
    - Quality filtering
    - Concatenation
    - Speaker separation

Example:
    python pipeline.py --raw_data_dir ./audio_data --output_dir ./output --enable-nisqa
"""

import argparse
import glob
import json
import os
import shutil
import sys

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import (
    AudioDataFilterStage,
    AudioDataFilterConfig,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch


def create_pipeline(args: argparse.Namespace) -> Pipeline:
    """
    Create an audio data filtration pipeline using AudioDataFilterStage.
    
    This uses the composite AudioDataFilterStage which maintains timestamp
    mapping through all transformations via TimestampTracker. The output
    segments always contain:
    - original_file: Path to source audio
    - original_start_ms: Start position in original file
    - original_end_ms: End position in original file
    - duration_ms, duration_sec: Segment duration
    - speaker_id: Speaker identifier (if speaker separation enabled)
    - Quality scores (nisqa_*, sigmos_*, band_*)
    """
    pipeline = Pipeline(
        name="audio_data_filter_pipeline",
        description="Audio curation pipeline with consistent timestamp mapping"
    )
    
    # Create AudioDataFilterConfig from args
    # Resource allocation (cpus, gpus) is now part of the config
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
    
    # Create the AudioDataFilterStage
    # Resources are applied from config automatically
    audio_filter_stage = AudioDataFilterStage(config=config).with_(
        batch_size=args.batch_size,
    )
    
    pipeline.add_stage(audio_filter_stage)
    return pipeline


def load_audio_tasks(input_dir: str, recursive: bool = False) -> list:
    """Load audio files as AudioBatch tasks."""
    if recursive:
        audio_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True))
    else:
        audio_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    
    if not audio_files:
        logger.warning(f"No .wav files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    tasks = []
    for i, audio_file in enumerate(audio_files):
        task = AudioBatch(
            data={"audio_filepath": audio_file},
            task_id=f"audio_{i:05d}",
            dataset_name="audio_filter"
        )
        tasks.append(task)
    
    return tasks


def save_results(results: list, output_dir: str) -> str:
    """
    Save pipeline results to JSONL manifest.
    
    Each result contains timestamp-mapped fields:
    - original_file: Source audio file path
    - original_start_ms: Start position in source file
    - original_end_ms: End position in source file
    - duration_ms, duration_sec: Segment duration
    - speaker_id: Speaker identifier (if enabled)
    - Quality scores
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
            f.write(json.dumps(clean_entry) + "\n")
    
    logger.info(f"Saved {len(results)} results to {manifest_path}")
    return manifest_path


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
      --enable-vad --enable-nisqa --enable-sigmos --enable-speaker-separation
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
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Validate input
    if not os.path.isdir(args.raw_data_dir):
        logger.error(f"Input directory not found: {args.raw_data_dir}")
        sys.exit(1)
    
    # Clean output if requested
    if args.clean and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    if args.enable_speaker_separation:
        enabled.append("SpeakerSep")
    
    logger.info(f"Enabled: {enabled or ['none']}")
    logger.info(f"Timestamp Mapping: ENABLED (always)")
    logger.info("=" * 70)
    
    # Create pipeline
    pipeline = create_pipeline(args)
    logger.info(pipeline.describe())
    
    # Load input tasks
    tasks = load_audio_tasks(args.raw_data_dir, args.recursive)
    if not tasks:
        logger.error("No audio files to process")
        sys.exit(1)
    
    # Execute pipeline
    logger.info("Starting pipeline execution...")
    
    try:
        from datetime import datetime
        start_time = datetime.now()
        
        pipeline.build()
        stages = pipeline.stages
        
        all_results = []
        for stage in stages:
            stage.setup()
        
        try:
            for i, task in enumerate(tasks):
                logger.info(f"Processing {i+1}/{len(tasks)}: {task.task_id}")
                result = task
                for stage in stages:
                    if result is None:
                        break
                    result = stage.process(result)
                
                if result is not None:
                    if isinstance(result.data, list):
                        all_results.extend(result.data)
                    else:
                        all_results.append(result.data)
        finally:
            for stage in stages:
                stage.teardown()
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Completed in {duration:.2f}s, {len(all_results)} segments")
        
        if all_results:
            save_results(all_results, args.output_dir)
            
            # Log sample output showing timestamp mapping
            if all_results:
                sample = all_results[0]
                logger.info("Sample output (showing timestamp mapping):")
                logger.info(f"  original_file: {sample.get('original_file', 'N/A')}")
                logger.info(f"  original_start_ms: {sample.get('original_start_ms', 'N/A')}")
                logger.info(f"  original_end_ms: {sample.get('original_end_ms', 'N/A')}")
                logger.info(f"  duration_sec: {sample.get('duration_sec', 'N/A')}")
                if 'speaker_id' in sample:
                    logger.info(f"  speaker_id: {sample.get('speaker_id')}")
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

