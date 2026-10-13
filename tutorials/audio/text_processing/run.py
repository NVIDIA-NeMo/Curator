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

"""Process audio transcripts with a YAML-configured Dynamo inference server."""

from contextlib import ExitStack
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from nemo_curator.config.run import (
    create_executor_from_yaml,
    create_pipeline_from_yaml,
    create_ray_client_from_yaml,
)


@hydra.main(version_base=None, config_path="config", config_name="text")
def main(cfg: DictConfig) -> None:
    manifest_path = Path(cfg.manifest_path).resolve()
    output_path = Path(cfg.output_path).resolve()
    if manifest_path == output_path or (
        manifest_path.exists() and output_path.exists() and manifest_path.samefile(output_path)
    ):
        msg = "output_path must differ from manifest_path: the writer replaces existing output."
        raise ValueError(msg)

    with ExitStack() as stack:
        stack.enter_context(create_ray_client_from_yaml(cfg))
        if not cfg.endpoint:
            server = stack.enter_context(hydra.utils.instantiate(cfg.inference_server))
            cfg.endpoint = server.endpoint

        pipeline = create_pipeline_from_yaml(cfg, log_config=False)
        logger.info(pipeline.describe())
        pipeline.run(executor=create_executor_from_yaml(cfg))

    logger.info("Processed transcripts written to {}", cfg.output_path)


if __name__ == "__main__":
    main()
