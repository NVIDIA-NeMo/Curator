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
from transformers import AutoTokenizer

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.models.utils import format_name_with_suffix
from nemo_curator.tasks import DocumentBatch


class TokenSplitterStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Token-based text chunking stage that splits long texts into smaller chunks
    while preserving paragraph boundaries.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_name: str,
        max_length_tokens: int = 8000,
        separator: str = "\n\n",
        text_field: str = "text",
        chunk_id_field: str = "chunk_id",
        n_tokens_field: str = "n_tokens",
    ):
        self.model_name = model_name
        self.max_length_tokens = max_length_tokens
        self.separator = separator
        self.text_field = text_field
        self.chunk_id_field = chunk_id_field
        self.n_tokens_field = n_tokens_field
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._name = format_name_with_suffix(self.model_name, suffix="_token_splitter")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field, self.chunk_id_field, self.n_tokens_field]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Process a batch of documents and split them into token-based chunks."""
        df = batch.to_pandas()

        records = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            text = str(row_dict.get(self.text_field, ""))
            if self.text_field in row_dict:
                row_dict.pop(self.text_field)

            raw_paragraphs = text.split(self.separator)
            paragraphs = []
            for para_idx, para in enumerate(raw_paragraphs):
                if para.strip():
                    is_last = para_idx == len(raw_paragraphs) - 1
                    para_to_add = para if is_last else para + self.separator
                    paragraphs.append(para_to_add)

            chunks = []
            current_paragraphs = []
            token_count = 0

            for para_text in paragraphs:
                tokens = self._tokenizer.encode(para_text, add_special_tokens=False)
                n_tokens = len(tokens)

                if token_count + n_tokens > self.max_length_tokens and token_count > 0:
                    chunk_text = "".join(current_paragraphs)
                    chunk_dict = {
                        self.text_field: chunk_text,
                        self.chunk_id_field: len(chunks),
                        self.n_tokens_field: token_count,
                        **row_dict,
                    }
                    chunks.append(chunk_dict)

                    current_paragraphs = []
                    token_count = 0

                current_paragraphs.append(para_text)
                token_count += n_tokens

            if current_paragraphs:
                chunk_text = "".join(current_paragraphs)
                chunk_dict = {
                    self.text_field: chunk_text,
                    self.chunk_id_field: len(chunks),
                    self.n_tokens_field: token_count,
                    **row_dict,
                }
                chunks.append(chunk_dict)

            records.extend(chunks)

        output_df = pd.DataFrame(records)

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=output_df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
