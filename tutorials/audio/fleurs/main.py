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
FLEURS audio curation pipeline for NeMo Curator.

Downloads a FLEURS split, runs ASR, scores WER, filters by threshold, and
writes a JSONL manifest — all driven by ``pipeline.yaml`` via Hydra.

Usage (from Curator repo root)::

    python tutorials/audio/fleurs/main.py \\
        --config-path . \\
        --config-name pipeline \\
        raw_data_dir=./example_audio/fleurs

    python tutorials/audio/fleurs/main.py \\
        --config-path . \\
        --config-name pipeline \\
        raw_data_dir=./example_audio/fleurs \\
        lang=en_us \\
        stages.1.model_name=nvidia/parakeet-tdt-0.6b-v2 \\
        wer_threshold=25.0 \\
        backend=ray_data
"""

import importlib

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.core.client import RayClient

_EXECUTOR_FACTORIES = {
    "xenna": "nemo_curator.backends.xenna:XennaExecutor",
    "ray_data": "nemo_curator.backends.ray_data:RayDataExecutor",
}


def _create_executor(backend: str) -> object:
    module_path, class_name = _EXECUTOR_FACTORIES[backend].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the FLEURS pipeline using Hydra configuration."""
    ray_client = RayClient()
    try:
        ray_client.start()
        logger.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")
        pipeline = create_pipeline_from_yaml(cfg, log_config=False)

        logger.info(pipeline.describe())
        logger.info("\n" + "=" * 50 + "\n")

        backend = cfg.get("backend", "xenna")
        if backend not in _EXECUTOR_FACTORIES:
            msg = f"Unknown backend '{backend}'. Choose from: {list(_EXECUTOR_FACTORIES)}"
            raise ValueError(msg)
        logger.info(f"Using backend: {backend}")
        executor = _create_executor(backend)

        logger.info("Starting pipeline execution...")
        pipeline.run(executor)

        logger.info("\nPipeline completed!")
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
