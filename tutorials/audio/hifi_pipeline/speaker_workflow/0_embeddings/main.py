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

from __future__ import annotations

import importlib

import hydra
from loguru import logger
from omegaconf import DictConfig

from nemo_curator.checkpointing.audio import AudioCheckpointRunner
from nemo_curator.config.run import create_pipeline_from_yaml

_EXECUTOR_FACTORIES = {
    "xenna": "nemo_curator.backends.xenna:XennaExecutor",
    "ray_data": "nemo_curator.backends.ray_data:RayDataExecutor",
}


def _create_executor(backend: str) -> object:
    module_path, class_name = _EXECUTOR_FACTORIES[backend].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the tarred speaker-embedding workflow from Hydra YAML."""
    pipeline = create_pipeline_from_yaml(cfg)
    pipeline.name = cfg.get("pipeline_name", "speaker_embeddings_workflow")
    pipeline.description = "Tarred diarized speaker-embedding workflow"
    pipeline.config["link_stages_via_io"] = bool(cfg.get("link_stages_via_io", False))

    logger.info(pipeline.describe())
    logger.info("\n" + "=" * 50 + "\n")

    backend = cfg.get("backend", "xenna")
    if backend not in _EXECUTOR_FACTORIES:
        msg = f"Unknown backend '{backend}'. Choose from: {list(_EXECUTOR_FACTORIES)}"
        raise ValueError(msg)
    logger.info(f"Using backend: {backend}")
    executor = _create_executor(backend)

    logger.info("Starting speaker workflow pipeline...")
    checkpoint_dir = cfg.get("checkpoint_dir")
    if checkpoint_dir:
        runner = AudioCheckpointRunner(
            pipeline=pipeline,
            checkpoint_dir=checkpoint_dir,
            executor=executor,
            ignore_failed=bool(cfg.get("ignore_failed", False)),
        )
        results = runner.run()
    else:
        results = pipeline.run(executor)

    uploaded_paths = sorted(
        {
            str(task.data.get("uploaded_output_filepath"))
            for task in results or []
            if task.data.get("uploaded_output_filepath")
        }
    )
    logger.info("\n" + "=" * 50)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 50)
    logger.info(f"  Output tasks: {len(results or [])}")
    logger.info(f"  Uploaded NPZ files: {len(uploaded_paths)}")
    for path in uploaded_paths:
        logger.info(f"    - {path}")


if __name__ == "__main__":
    main()
