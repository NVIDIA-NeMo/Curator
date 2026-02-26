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

import huggingface_hub
from transformers import AutoTokenizer

from nemo_curator.stages.text.filters.doc_filter import DocumentFilter


class TokenCountFilter(DocumentFilter):
    """
    If the document contains more or less than a specified number of tokens, then discard.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer | None = None,
        hf_model_name: str | None = None,
        hf_token: str | None = None,
        min_tokens: int = 0,
        max_tokens: int = float("inf"),
    ):
        """
        Args:
            tokenizer (AutoTokenizer | None): The pre-loaded tokenizer to use to count the tokens.
                If None, the tokenizer will be initialized from the hf_model_name.
            hf_model_name (str | None): The name of the Hugging Face model to use to count the tokens.
                If None, the pre-loaded tokenizer must be provided via the tokenizer argument.
            hf_token (str | None): The token to use to access the Hugging Face model, if needed.
            min_tokens (int): The minimum number of tokens the document must contain.
                Set to 0 to disable the minimum token count filter.
            max_tokens (int): The maximum number of tokens the document can contain.
                Set to infinity to disable the maximum token count filter.
        """
        super().__init__()

        if tokenizer is None and hf_model_name is None:
            msg = "Either tokenizer or hf_model_name must be provided"
            raise ValueError(msg)
        if tokenizer is not None and hf_model_name is not None:
            msg = "Either tokenizer or hf_model_name must be provided, not both"
            raise ValueError(msg)

        self._token_count_filter_tokenizer = tokenizer
        self._hf_model_name = hf_model_name
        self._hf_token = hf_token
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._name = "token_count"

    def model_check_or_download(self) -> None:
        if self._hf_model_name is not None:
            # Use snapshot_download to download all files without loading the model into memory.
            huggingface_hub.snapshot_download(
                repo_id=self._hf_model_name,
                token=self._hf_token,
                local_files_only=False,  # Download if not cached
                resume_download=True,  # Resume interrupted downloads
            )

    def load_tokenizer(self) -> None:
        if self._hf_model_name is not None:
            self._token_count_filter_tokenizer = AutoTokenizer.from_pretrained(
                self._hf_model_name, local_files_only=True
            )

    def score_document(self, text: str) -> int:
        tokens = self._token_count_filter_tokenizer.encode(text)
        return len(tokens)

    def keep_document(self, score: int) -> bool:
        return self._min_tokens <= score <= self._max_tokens
