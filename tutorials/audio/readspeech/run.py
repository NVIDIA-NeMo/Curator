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
Hydra-based runner for DNS Challenge Read Speech pipeline.

This script loads the pipeline configuration from YAML and executes it.

Usage:
    # Run with default config
    python run.py --config-path . --config-name pipeline.yaml \
        raw_data_dir=/path/to/read_speech

    # Override settings
    python run.py --config-path . --config-name pipeline.yaml \
        raw_data_dir=/path/to/read_speech \
        max_samples=3000 \
        processors.1.config.enable_vad=true
"""

import glob
import json
import os

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline


def create_pipeline_from_yaml(cfg: DictConfig) -> Pipeline:
    """Create pipeline from Hydra config."""
    pipeline = Pipeline(
        name="readspeech_yaml_pipeline",
        description="DNS Challenge Read Speech pipeline created from YAML config"
    )

    for processor_cfg in cfg.processors:
        stage = hydra.utils.instantiate(processor_cfg)
        pipeline.add_stage(stage)

    return pipeline


def merge_jsonl_files(output_dir: str) -> str:
    """
    Merge all JSONL files in output_dir into a single manifest.jsonl.
    Fixes escaped paths and removes intermediate files.
    
    Returns:
        Path to the merged manifest.jsonl file
    """
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    jsonl_files = glob.glob(os.path.join(output_dir, "*.jsonl"))
    
    # Exclude manifest.jsonl itself if it exists
    jsonl_files = [f for f in jsonl_files if os.path.basename(f) != "manifest.jsonl"]
    
    if not jsonl_files:
        logger.warning("No JSONL files found to merge")
        return manifest_path
    
    logger.info(f"Merging {len(jsonl_files)} JSONL files into manifest.jsonl...")
    
    all_records = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        all_records.append(record)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in {jsonl_file}")
    
    # Write merged manifest with proper paths (no escaped slashes)
    with open(manifest_path, "w") as f:
        for record in all_records:
            # Use ensure_ascii=False to avoid escaping forward slashes
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"✓ Merged {len(all_records)} records into {manifest_path}")
    
    # Remove intermediate files
    for jsonl_file in jsonl_files:
        os.remove(jsonl_file)
        logger.debug(f"  Removed: {os.path.basename(jsonl_file)}")
    
    logger.info(f"✓ Cleaned up {len(jsonl_files)} intermediate JSONL files")
    
    return manifest_path


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Run DNS Challenge Read Speech pipeline from YAML configuration.
    """
    logger.info("DNS Challenge Read Speech Audio Data Filtration Pipeline (YAML)")
    logger.info("=" * 60)
    logger.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    # Create pipeline
    pipeline = create_pipeline_from_yaml(cfg)

    # Print pipeline description
    logger.info(pipeline.describe())
    logger.info("=" * 60)

    # Create executor
    executor = XennaExecutor()

    # Execute pipeline
    logger.info("Starting pipeline execution...")
    pipeline.run(executor)

    # Merge JSONL files into single manifest
    logger.info("\nPost-processing results...")
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = merge_jsonl_files(output_dir)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed!")
    logger.info(f"Results saved to: {manifest_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

