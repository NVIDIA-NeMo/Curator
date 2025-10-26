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

import os

import fasttext
import pandas as pd
from transformers import AutoTokenizer

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


# Tokenize and filter out non-English text
class NonEnglishFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    def __init__(  # noqa: PLR0913
        self,
        tokenizer_identifier: str,
        hf_token: str | None = None,
        lang_id_model_path: str = "./lid.176.ftz",
        input_field: str = "input",
        output_field: str = "output",
        system_prompt_field: str = "system_prompt",
    ):
        self._name = "non_english_filter"
        self.tokenizer_identifier = tokenizer_identifier
        self.hf_token = hf_token
        self.lang_id_model_path = lang_id_model_path
        self.input_field = input_field
        self.output_field = output_field
        self.system_prompt_field = system_prompt_field

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.input_field, self.output_field, self.system_prompt_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata = None) -> None:
        try:
            self._setup(local_files_only=False)
        except Exception as e:
            msg = f"Failed to download {self.tokenizer_identifier}"
            raise RuntimeError(msg) from e

        if not (os.path.exists(self.lang_id_model_path) and os.path.isfile(self.lang_id_model_path)):
            msg = f"FastText model path {self.lang_id_model_path} does not exist"
            raise RuntimeError(msg)

    # We use the _setup function to ensure that everything needed for the tokenizer is downloaded and loaded properly
    def _setup(self, local_files_only: bool = True) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_identifier,
            local_files_only=local_files_only,
            token=self.hf_token,
        )

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self._setup(local_files_only=True)
        self.model = fasttext.load_model(path=self.lang_id_model_path)

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

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        if df.empty:
            return batch

        mask = df.apply(
            lambda row: self.is_english(row[self.system_prompt_field], row[self.input_field], row[self.output_field]),
            axis=1,
        )
        df_filtered = df[mask]

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df_filtered,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


# Tokenize system_prompt, input, and output and filter out samples with too many tokens
class TokenCountFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    def __init__(  # noqa: PLR0913
        self,
        tokenizer_identifier: str,
        hf_token: str | None = None,
        max_token_count: int = 16384,
        input_field: str = "input",
        output_field: str = "output",
        system_prompt_field: str = "system_prompt",
    ):
        self._name = "token_count_filter"
        self.tokenizer_identifier = tokenizer_identifier
        self.hf_token = hf_token
        self.max_token_count = max_token_count
        self.input_field = input_field
        self.output_field = output_field
        self.system_prompt_field = system_prompt_field

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.input_field, self.output_field, self.system_prompt_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata = None) -> None:
        try:
            self._setup(local_files_only=False)
        except Exception as e:
            msg = f"Failed to download {self.tokenizer_identifier}"
            raise RuntimeError(msg) from e

    # We use the _setup function to ensure that everything needed for the tokenizer is downloaded and loaded properly
    def _setup(self, local_files_only: bool = True) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_identifier,
            local_files_only=local_files_only,
            token=self.hf_token,
        )

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self._setup(local_files_only=True)

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

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        if df.empty:
            return batch

        templates_list = df.apply(
            lambda row: self.apply_chat_template(
                row[self.system_prompt_field],
                row[self.input_field],
                row[self.output_field],
            ),
            axis=1,
        ).tolist()
        tokenized = self.tokenizer(templates_list)
        scores = pd.Series([len(tokens) for tokens in tokenized["input_ids"]], index=df.index)
        mask = (scores > 0) & (scores <= self.max_token_count)
        df_filtered = df[mask]

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df_filtered,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


# Tokenize text and filter out samples with too many tokens
class CompletionTokenCountFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    def __init__(
        self,
        tokenizer_identifier: str,
        hf_token: str | None = None,
        max_completion_token_count: int = 8192,
        output_field: str = "output",
    ):
        self._name = "completion_token_count_filter"
        self.tokenizer_identifier = tokenizer_identifier
        self.hf_token = hf_token
        self.max_completion_token_count = max_completion_token_count
        self.output_field = output_field

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.output_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata = None) -> None:
        try:
            self._setup(local_files_only=False)
        except Exception as e:
            msg = f"Failed to download {self.tokenizer_identifier}"
            raise RuntimeError(msg) from e

    # We use the _setup function to ensure that everything needed for the tokenizer is downloaded and loaded properly
    def _setup(self, local_files_only: bool = True) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_identifier,
            local_files_only=local_files_only,
            token=self.hf_token,
        )

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self._setup(local_files_only=True)

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        if df.empty:
            batch.data["completion_token_count"] = []
            return batch

        outpt = df[self.output_field]
        outpt_copy = outpt.copy()

        templates_list = outpt_copy.apply(
            lambda text: self.tokenizer.apply_chat_template(
                [{"role": "assistant", "content": text}],
                # For maximum accuracy it should be tokenize=True here,
                # but for speedups we just use the length of the base string instead of tokenizing it
                # (we consider this to be acceptable since 1 token is approx 4 characters for English
                # and we are only using the CompletionTokenCountFilter as a proxy for text complexity)
                # Please keep this in mind when setting the max_completion_token_count
                tokenize=False,
                add_generation_prompt=False,
                truncation=False,
            )
        ).tolist()
        tokenized = self.tokenizer(templates_list)
        scores = pd.Series([len(tokens) for tokens in tokenized["input_ids"]], index=outpt_copy.index)
        df["completion_token_count"] = scores
        mask = (scores > 0) & (scores <= self.max_completion_token_count)
        df_filtered = df[mask]

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df_filtered,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class ApplyChatTemplate(ProcessingStage[DocumentBatch, DocumentBatch]):
    def __init__(
        self,
        tokenizer_identifier: str,
        hf_token: str | None = None,
        input_field: str = "input",
        output_field: str = "output",
        system_prompt_field: str = "system_prompt",
    ):
        self._name = "apply_chat_template"

        self.tokenizer_identifier = tokenizer_identifier
        self.hf_token = hf_token
        self.input_field = input_field
        self.output_field = output_field
        self.system_prompt_field = system_prompt_field

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.input_field, self.output_field, self.system_prompt_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.input_field, self.output_field]

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata = None) -> None:
        try:
            self._setup(local_files_only=False)
        except Exception as e:
            msg = f"Failed to download {self.tokenizer_identifier}"
            raise RuntimeError(msg) from e

    # We use the _setup function to ensure that everything needed for the tokenizer is downloaded and loaded properly
    def _setup(self, local_files_only: bool = True) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_identifier,
            local_files_only=local_files_only,
            token=self.hf_token,
        )

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self._setup(local_files_only=True)

    # Modifier for input and output chat templates
    def format_input_output(self, system_prompt: str, inpt: list[dict], outpt: str) -> tuple[str, str]:
        prompt_and_completion = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                *inpt,
                {"role": "assistant", "content": outpt},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                *inpt,
            ],
            tokenize=False,
            # We expect the model to start predicting tokens after it sees the start of the assistant response turn
            add_generation_prompt=True,
        )

        # Remove the prompt from prompt_and_completion via string manipulation to extract the completion part
        completion = prompt_and_completion[len(prompt) :]

        # input, output
        return prompt, completion

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        if df.empty:
            return batch

        df[self.input_field], df[self.output_field] = zip(
            *df.apply(
                lambda row: self.format_input_output(
                    row[self.system_prompt_field], row[self.input_field], row[self.output_field]
                ),
                axis=1,
            ),
            strict=True,
        )

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
