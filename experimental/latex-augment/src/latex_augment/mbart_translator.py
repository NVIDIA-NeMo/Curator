# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration


class MBartTranslatorFromEnglish:
    def __init__(
        self,
        lang: str = None,
        *,
        model_id: str = "facebook/mbart-large-50-one-to-many-mmt",
        repetition_penalty: float = 1.0,
        device=None,
    ):
        self.lang = lang
        self._model = None
        self._tokenizer = None
        self.model_id = model_id
        self.repetition_penalty = repetition_penalty
        self.device = device

    @property
    def model(self):
        if self._model is None:
            if self.device and "cuda" in self.device:
                torch_dtype = torch.bfloat16
                attn_implementation = find_attn_implementation()
            else:
                torch_dtype = attn_implementation = None
            self._model = MBartForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                # device_map="auto",
            )
            self._model = self._model.to(self.device)
            self._model = torch.compile(
                self._model, mode="max-autotune", dynamic=False, fullgraph=True
            )
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = MBart50TokenizerFast.from_pretrained(
                self.model_id, src_lang="en_XX", clean_up_tokenization_spaces=True
            )
            # remove funny suffixes
            self._tokenizer.lang_code_to_id = {
                k[:2]: v for k, v in self._tokenizer.lang_code_to_id.items()
            }
            # https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
            # self._tokenizer.padding_side = "left"
            assert self._tokenizer.pad_token is not None, "pad_token is missing"
        return self._tokenizer

    def __call__(self, sentence: str, lang=None, **kwargs) -> str:
        """Translate sentence(s) from English to target language."""
        if not sentence:  # empty string
            return sentence
        max_context_length = 512  # tokens
        max_input_length = max_context_length // 2
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        max_new_tokens = max_context_length - inputs.input_ids.shape[1]
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        lang_token_id = self.tokenizer.lang_code_to_id[lang or self.lang]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=lang_token_id,
                max_new_tokens=max_new_tokens,
                repetition_penalty=self.repetition_penalty,
                **kwargs,
            )
        translated = self.tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
        return translated


def find_attn_implementation():
    try:
        import flash_attn

        if flash_attn.__version__.startswith("2."):
            return "flash_attention_2"
        else:
            return "flash_attention"
    except ImportError:
        if torch.cuda.is_available():
            return "sdpa"
        else:
            return None
