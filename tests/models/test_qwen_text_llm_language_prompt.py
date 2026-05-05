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

from nemo_curator.models.qwen_text_llm import QwenTextLLM


def test_prepare_single_formats_language_placeholder() -> None:
    model = QwenTextLLM(model_id="mock/model")
    prompt = "Restore punctuation in {language}: {text}"

    prepared = model._prepare_single("bonjour le monde", prompt, language="French")

    assert prepared == {"prompt": "Restore punctuation in French: bonjour le monde"}
