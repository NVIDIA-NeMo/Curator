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

"""Run a YAML-defined audio ASR adapter pipeline."""

import hydra
from omegaconf import DictConfig

from nemo_curator.config.run import run_pipeline_from_yaml


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Build and run the complete pipeline declared by the selected YAML."""
    run_pipeline_from_yaml(cfg)


if __name__ == "__main__":
    main()
