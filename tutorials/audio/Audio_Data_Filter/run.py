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
Hydra-based runner for Audio Data Filtration Pipeline.

This script loads pipeline configuration from YAML and executes stages
using the NeMo Curator Pipeline and Executor pattern. It uses the
AudioDataFilterStage which maintains consistent timestamp mapping
throughout all transformations via TimestampTracker.

Timestamp Mapping:
    All output segments contain:
    - original_file: Path to source audio
    - original_start_ms: Start position in source file
    - original_end_ms: End position in source file

Usage:
    SCRIPT_DIR=nemo_curator/stages/audio/advance_pipelines/Audio_data_filter
    python ${SCRIPT_DIR}/run.py --config-path . --config-name pipeline.yaml \
        raw_data_dir=/path/to/audio/files
"""

import glob
import json
import os

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.tasks import AudioBatch


def save_results(results: list, output_dir: str) -> str:
    """
    Save pipeline results to JSONL manifest.
    
    Args:
        results: List of AudioBatch results from pipeline
        output_dir: Directory to save manifest
        
    Returns:
        Path to saved manifest file
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    
    all_entries = []
    for result in results:
        if result is None:
            continue
        # Handle AudioBatch data (list of dicts)
        if hasattr(result, "data") and result.data:
            if isinstance(result.data, list):
                all_entries.extend(result.data)
            elif isinstance(result.data, dict):
                all_entries.append(result.data)
    
    with open(manifest_path, "w") as f:
        for entry in all_entries:
            # Clean up non-serializable values
            clean_entry = {}
            for key, value in entry.items():
                if hasattr(value, "item"):
                    clean_entry[key] = value.item()
                elif isinstance(value, (int, float, str, bool, type(None))):
                    clean_entry[key] = value
                else:
                    clean_entry[key] = str(value)
            f.write(json.dumps(clean_entry) + "\n")
    
    logger.info(f"Saved {len(all_entries)} segments to {manifest_path}")
    return manifest_path


def load_audio_tasks(raw_data_dir: str) -> list[AudioBatch]:
    """
    Load audio files from directory and create AudioBatch tasks.
    
    Args:
        raw_data_dir: Directory containing .wav audio files
        
    Returns:
        List of AudioBatch tasks with audio_filepath set
    """
    # Find all .wav files
    audio_files = sorted(glob.glob(os.path.join(raw_data_dir, "*.wav")))
    
    # Also search recursively
    if not audio_files:
        audio_files = sorted(glob.glob(os.path.join(raw_data_dir, "**", "*.wav"), recursive=True))
    
    if not audio_files:
        logger.warning(f"No .wav files found in {raw_data_dir}")
        return []
    
    logger.info(f"Found {len(audio_files)} audio files in {raw_data_dir}")
    
    tasks = []
    for i, audio_file in enumerate(audio_files):
        task = AudioBatch(
            data={"audio_filepath": audio_file},
            task_id=f"audio_{i:05d}",
            dataset_name="audio_filter"
        )
        tasks.append(task)
    
    return tasks


def create_pipeline_from_yaml(cfg: DictConfig) -> Pipeline:
    """Create pipeline by instantiating stages from YAML config."""
    pipeline = Pipeline(
        name="audio_filter_yaml_pipeline",
        description="Audio filtration pipeline created from YAML config"
    )
    
    for p in cfg.processors:
        stage = hydra.utils.instantiate(p)
        pipeline.add_stage(stage)
    
    return pipeline


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Load YAML config and run the audio filtration pipeline.
    """
    logger.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Load audio tasks from raw_data_dir
    raw_data_dir = cfg.get("raw_data_dir")
    if not raw_data_dir:
        logger.error("raw_data_dir is required!")
        return
    
    if not os.path.isdir(raw_data_dir):
        logger.error(f"raw_data_dir does not exist: {raw_data_dir}")
        return
    
    initial_tasks = load_audio_tasks(raw_data_dir)
    if not initial_tasks:
        logger.error("No audio files to process!")
        return
    
    # Create pipeline from YAML
    pipeline = create_pipeline_from_yaml(cfg)
    
    # Print pipeline description
    logger.info(pipeline.describe())
    logger.info("\n" + "=" * 50 + "\n")
    
    # Create executor
    executor = XennaExecutor()
    
    # Execute pipeline with initial tasks
    logger.info(f"Starting pipeline execution with {len(initial_tasks)} audio files...")
    results = pipeline.run(executor, initial_tasks=initial_tasks)
    
    # Save results to JSONL
    output_dir = cfg.get("output_dir", os.path.join(raw_data_dir, "result"))
    if results:
        save_results(results, output_dir)
    else:
        logger.warning("No results returned from pipeline")
    
    logger.info("\nPipeline completed!")


if __name__ == "__main__":
    main()

