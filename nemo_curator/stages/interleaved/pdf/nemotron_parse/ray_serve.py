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

"""Ray Serve worker customization for Nemotron-Parse."""

from ray.serve.llm import LLMConfig, LLMServer

from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import set_nemotron_parse_attention_backend


class NemotronParseRayServeServer(LLMServer):
    """Resolve the attention backend on the GPU assigned to this replica."""

    async def __init__(self, llm_config: LLMConfig) -> None:
        set_nemotron_parse_attention_backend(llm_config.engine_kwargs)
        await super().__init__(llm_config)
