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
Hydra-based runner for the Chatterbox TTS synthesis pipeline.

This script loads the pipeline configuration from YAML and executes it.

Usage:
    python run.py --config-path . --config-name pipeline \
        input_manifest=/data/turns.jsonl \
        reference_voices_dataset=/data/reference_voices \
        output_dir=/data/tts_output

    # Multilingual (French)
    python run.py --config-path . --config-name pipeline \
        input_manifest=/data/turns.jsonl \
        reference_voices_dataset=/data/mls_french \
        output_dir=/data/tts_output_fr \
        language=fr
"""

import importlib
import os
import time

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.tts import ChatterboxTTSStage

_EXECUTOR_FACTORIES = {
    "xenna": "nemo_curator.backends.xenna:XennaExecutor",
    "ray_data": "nemo_curator.backends.ray_data:RayDataExecutor",
}


def _create_executor(backend: str) -> object:
    if backend not in _EXECUTOR_FACTORIES:
        msg = f"Unknown backend '{backend}'. Choose from: {list(_EXECUTOR_FACTORIES)}"
        raise ValueError(msg)
    module_path, class_name = _EXECUTOR_FACTORIES[backend].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


def create_pipeline_from_yaml(cfg: DictConfig) -> Pipeline:
    """Create pipeline from Hydra config."""
    pipeline = Pipeline(
        name="chatterbox_tts",
        description="Chatterbox TTS conversation-turn synthesis (YAML config)",
    )
    for processor_cfg in cfg.processors:
        stage = hydra.utils.instantiate(processor_cfg)
        # TTS turns are synthesised serially, so keep one task per batch.
        if isinstance(stage, ChatterboxTTSStage):
            stage = stage.with_(batch_size=1)
        pipeline.add_stage(stage)
    return pipeline


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "audio"), exist_ok=True)

    pipeline = create_pipeline_from_yaml(cfg)
    logger.info(pipeline.describe())

    backend = cfg.get("backend", "xenna")
    executor = _create_executor(backend)

    logger.info(f"Starting Chatterbox TTS pipeline (backend: {backend})...")
    t0 = time.monotonic()
    pipeline.run(executor)
    elapsed = time.monotonic() - t0
    logger.info(f"Pipeline completed in {elapsed:.2f}s ({elapsed / 60:.2f} min)")
    logger.info(f"Results written to {os.path.join(cfg.output_dir, 'result')}/*.jsonl")


if __name__ == "__main__":
    main()
