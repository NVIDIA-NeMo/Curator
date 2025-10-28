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

import time
from functools import lru_cache
import numpy as np
from transformers import MBart50TokenizerFast
from .mbart_heuristics import mbart_heuristics


class TritonTranslator:
    def __init__(
        self,
        lang: str,
        *,
        tokenizer_name: str = "facebook/mbart-large-50-one-to-many-mmt",
        server_address: str = "localhost:8001",
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        request_output_len: int = 512,
        model_name: str = "tensorrt_llm",
    ):
        self.lang = lang
        self.server_address = server_address
        self.repetition_penalty = repetition_penalty
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.temperature = temperature
        self.request_output_len = request_output_len
        self._triton_client = None

    def __enter__(self):
        # tritonclient is an optional dependency
        from tritonclient.grpc import InferenceServerClient
        from tritonclient.utils import InferenceServerException

        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            self.tokenizer_name, src_lang="en_XX", clean_up_tokenization_spaces=True
        )
        # remove funny suffixes
        self.tokenizer.lang_code_to_id = {
            k[:2]: v for k, v in self.tokenizer.lang_code_to_id.items()
        }
        # https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
        # self.tokenizer.padding_side = "left"
        assert self.tokenizer.pad_token is not None, "pad_token is missing"
        self.end_id_data = np.array([[self.tokenizer.eos_token_id]], dtype=np.int32)
        self.pad_id_data = np.array([[self.tokenizer.pad_token_id]], dtype=np.int32)
        self.request_output_len_data = np.array(
            [[self.request_output_len]], dtype=np.int32
        )
        self.temperature_data = np.array([[self.temperature]], dtype=np.float32)
        self.repetition_penalty_data = np.array(
            [[self.repetition_penalty]], dtype=np.float32
        )
        # streaming_data = np.array([[False]], dtype=bool)
        # mBART-50 decoder has to be prompted with [eos] [lang]
        target_lang_id = self.tokenizer.lang_code_to_id[self.lang]
        self.decoder_input_ids_data = np.array(
            [[self.tokenizer.eos_token_id, target_lang_id]], dtype=np.int32
        )
        assert self._triton_client is None, "cannot nest TritonTranslators"
        self._triton_client = InferenceServerClient(
            self.server_address, verbose=False
        ).__enter__()

        # poll server until ready
        for retry in range(30):
            try:
                if not self._triton_client.is_server_ready():
                    raise RuntimeError("triton server not ready")
                if not self._triton_client.is_model_ready(self.model_name):
                    raise RuntimeError(f"triton model {self.model_name} not ready")
                return TritonClient(self)
            except (InferenceServerException, RuntimeError):
                if retry == 29:
                    raise
                time.sleep(1.0)

    def __exit__(self, exc_type, exc_value, traceback):
        assert self._triton_client
        self._triton_client.__exit__(exc_type, exc_value, traceback)
        self._triton_client = None

    def _translate(self, sentence: str) -> str:
        """Translate sentence from English to target language."""
        # tritonclient is an optional dependency
        from tritonclient.grpc import InferInput, InferRequestedOutput
        from tritonclient.utils import np_to_triton_dtype

        if not sentence:  # empty string
            return sentence
        max_context_length = 512  # tokens
        max_input_length = max_context_length // 2
        inputs = self.tokenizer(
            sentence,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )

        input_ids_data = inputs.input_ids.astype(np.int32)
        attention_mask_data = inputs.attention_mask.astype(np.bool_)
        input_lengths_data = inputs.attention_mask.sum(1)[:, None].astype(np.int32)

        def infer_input_from(name, input):
            t = InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
            t.set_data_from_numpy(input)
            return t

        inputs = [
            infer_input_from("input_ids", input_ids_data),
            infer_input_from("input_lengths", input_lengths_data),
            # infer_input_from("cross_attention_mask", ???),
            infer_input_from("decoder_input_ids", self.decoder_input_ids_data),
            infer_input_from("request_output_len", self.request_output_len_data),
            infer_input_from("temperature", self.temperature_data),
            # infer_input_from("streaming", self.streaming_data),
            infer_input_from("end_id", self.end_id_data),
            infer_input_from("pad_id", self.pad_id_data),
            infer_input_from("repetition_penalty", self.repetition_penalty_data),
        ]

        outputs = [
            InferRequestedOutput("output_ids"),
            # InferRequestedOutput("sequence_index"),
            InferRequestedOutput("sequence_length"),
        ]

        result = self._triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
            # client_timeout=FLAGS.client_timeout,
        )

        output_ids = result.as_numpy("output_ids")
        assert output_ids is not None
        seq_len = result.as_numpy("sequence_length")[0, 0]
        seq = output_ids[0, 0][:seq_len]
        translated = self.tokenizer.decode(seq, skip_special_tokens=True)
        return translated


class TritonClient:
    def __init__(self, translator: TritonTranslator):
        self._translate = translator._translate
        self.lang = translator.lang

    @lru_cache(maxsize=100000)
    def __call__(self, sentence: str) -> str:
        """Translate sentence from English to target language."""
        return mbart_heuristics(self._translate, sentence)
