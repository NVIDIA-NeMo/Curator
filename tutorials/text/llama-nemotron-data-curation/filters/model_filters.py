# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pandas as pd

from nemo_curator.stages.text.filters import DocumentFilter


# Tokenize and filter out non-English text
class NonEnglishFilter(DocumentFilter):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        lang_id_model_path: str,
        text_fields: list[str] | None = None,
    ):
        self._name = "non_english_filter"
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.lang_id_model_path = lang_id_model_path
        if text_fields is None:
            self.text_fields = ["system_prompt", "input", "output"]
        else:
            self.text_fields = text_fields

    def is_english(self, system: str, inpt: list[dict], outpt: str) -> bool:
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                *inpt,
                {"role": "assistant", "content": outpt},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        text = str(text).replace("\n", " ").strip()
        return self.model.predict(text)[0][0] == "__label__en"

    def score_document(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(
            lambda row: self.is_english(
                row[self.text_fields[0]],
                row[self.text_fields[1]],
                row[self.text_fields[2]],
            ),
            axis=1,
        )

    def keep_document(self, scores: pd.Series) -> pd.Series:
        return scores


# Tokenize system_prompt, input, and output and filter out samples with too many tokens
class TokenCountFilter(DocumentFilter):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_token_count: int = 16384,
        text_fields: list[str] | None = None,
    ):
        super().__init__()
        self._name = "token_count_filter"
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_token_count = max_token_count
        if text_fields is None:
            self.text_fields = ["system_prompt", "input", "output"]
        else:
            self.text_fields = text_fields

    def apply_chat_template(self, system: str, inpt: list[dict], outpt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                *inpt,
                {"role": "assistant", "content": outpt},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    def score_document(self, df: pd.DataFrame) -> pd.Series:
        templates_list = df.apply(
            lambda row: self.apply_chat_template(
                row[self.text_fields[0]],
                row[self.text_fields[1]],
                row[self.text_fields[2]],
            ),
            axis=1,
        ).tolist()
        tokenized = self.tokenizer(templates_list)
        return pd.Series([len(tokens) for tokens in tokenized["input_ids"]], index=df.index)

    def keep_document(self, scores: pd.Series) -> pd.Series:
        return (scores > 0) & (scores <= self.max_token_count)


# Tokenize text and filter out samples with too many tokens
class CompletionTokenCountFilter(DocumentFilter):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_completion_token_count: int = 8192,
        text_fields: list[str] | None = None,
    ):
        super().__init__()
        self._name = "completion_token_count_filter"
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.max_completion_token_count = max_completion_token_count
        if text_fields is None:
            self.text_fields = ["output"]
        else:
            self.text_fields = text_fields

    def score_document(self, df: pd.DataFrame) -> pd.Series:
        outpt = df[self.text_fields[0]]

        outpt_copy = outpt.copy()
        templates_list = outpt_copy.apply(
            lambda text: self.tokenizer.apply_chat_template(
                [{"role": "assistant", "content": text}],
                tokenize=False,
                add_generation_prompt=False,
                truncation=False,
            )
        ).tolist()
        tokenized = self.tokenizer(templates_list)
        return pd.Series([len(tokens) for tokens in tokenized["input_ids"]], index=outpt_copy.index)

    def keep_document(self, scores: pd.Series) -> pd.Series:
        return (scores > 0) & (scores <= self.max_completion_token_count)
