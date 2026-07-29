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

"""Run the Granary-v2 pipeline with current Curator configuration APIs."""

from __future__ import annotations

import hydra
from nemo_curator.stages.audio.pipeline_utils import resolve_model_route
from omegaconf import DictConfig, OmegaConf

from nemo_curator.config.run import (
    create_executor_from_yaml,
    create_pipeline_from_yaml,
    create_ray_client_from_yaml,
)
from tutorials.audio.granary_v2.pipeline_config import (
    GranaryV2PipelineSettings,
    build_stage_configs,
)


@hydra.main(version_base=None, config_path=".", config_name="pipeline")
def main(cfg: DictConfig) -> None:
    route = resolve_model_route(
        str(cfg.language),
        primary=cfg.get("primary_model"),
        recovery=cfg.get("recovery_model"),
    )
    settings = GranaryV2PipelineSettings(
        input_config=str(cfg.input_config),
        output_dir=str(cfg.output_dir),
        language=route.language,
        primary_model=route.primary,
        recovery_model=route.recovery,
        regex_yaml=str(cfg.regex_yaml),
        hallucination_phrases=str(cfg.hallucination_phrases),
        qwen_omni_model_id=str(cfg.models.qwen_omni),
        qwen_asr_model_id=str(cfg.models.qwen_asr),
        parakeet_v3_model_id=str(cfg.models.parakeet_v3),
        parakeet_riva_model_id=str(cfg.models.parakeet_riva),
        indic_model_id=str(cfg.models.indic_monolingual),
        whisper_model_size_or_path=str(cfg.models.whisper),
        sed_checkpoint=cfg.get("sed_checkpoint"),
        reader_concurrency=int(cfg.reader_concurrency),
        primary_batch_size=int(cfg.primary_batch_size),
        recovery_batch_size=int(cfg.recovery_batch_size),
    )
    runtime_cfg = OmegaConf.create(
        {
            "stages": build_stage_configs(settings),
            "backend": "ray_data",
        }
    )
    pipeline = create_pipeline_from_yaml(runtime_cfg)
    ray_client = create_ray_client_from_yaml(runtime_cfg)
    ray_client.start()
    try:
        pipeline.run(executor=create_executor_from_yaml(runtime_cfg))
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
