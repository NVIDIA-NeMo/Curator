# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the combined audio/text tutorial on Ray Data with managed serving."""

from contextlib import ExitStack
from pathlib import Path

import hydra
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf

from nemo_curator.config.run import (
    create_executor_from_yaml,
    create_pipeline_from_yaml,
    create_ray_client_from_yaml,
)


@hydra.main(version_base=None, config_path=".", config_name="pipeline")
def main(cfg: DictConfig) -> None:
    if cfg.backend != "ray_data":
        msg = "The full English pipeline supports only backend=ray_data; see README.md."
        raise ValueError(msg)
    if Path(cfg.output_dir).exists():
        msg = f"Choose a new output_dir: {cfg.output_dir}"
        raise FileExistsError(msg)
    with ExitStack() as stack:
        stack.enter_context(create_ray_client_from_yaml(cfg))
        if OmegaConf.is_missing(cfg, "endpoint") or not cfg.endpoint:
            cfg.model_name = str(cfg.model_name)
            if not Path(cfg.model_identifier).is_dir():
                cfg.model_identifier = snapshot_download(
                    repo_id=cfg.model_identifier, revision=cfg.inference_server.models[0].engine_kwargs.revision
                )
            server = stack.enter_context(hydra.utils.instantiate(cfg.inference_server))
            cfg.endpoint = server.endpoint
        pipeline = create_pipeline_from_yaml(cfg)
        pipeline.run(executor=create_executor_from_yaml(cfg))


if __name__ == "__main__":
    main()
