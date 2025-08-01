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

import gc
import os
from collections.abc import Generator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

os.environ["RAPIDS_NO_INITIALIZE"] = "1"

import numpy as np
import pandas as pd
import torch
from huggingface_hub import PyTorchModelHubMixin, snapshot_download
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ray_curator.backends.base import NodeInfo, WorkerMetadata
from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.stages.modules.score_filter import Filter
from ray_curator.stages.resources import Resources
from ray_curator.tasks import DocumentBatch

from .constants import ATTENTION_MASK_COLUMN, INPUT_ID_COLUMN, format_name_with_suffix

SEQ_ORDER_COLUMN = "_curator_seq_order"
TOKEN_LENGTH_COLUMN = "_curator_token_length"  # noqa: S105


class HFTokenizerStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Tokenizer stage for Hugging Face models.

    Args:
        model_identifier: The identifier of the Hugging Face model.
        hf_token: Hugging Face token for downloading the model, if needed. Defaults to None.
        text_field: The name of the text field in the input data. Defaults to "text".
        max_chars: Limits the total number of characters that can be fed to the tokenizer.
            If None, text will not be truncated. Defaults to None.
        max_seq_length: Limits the total sequence returned by the tokenizer so that it has a maximum length.
            If None, the tokenizer's model_max_length is used. Defaults to None.
        padding_side: The side to pad the input tokens. Defaults to "right".
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        unk_token: If True, set the pad_token to the tokenizer's unk_token. Defaults to False.

    """

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        hf_token: str | None = None,
        text_field: str = "text",
        max_chars: int | None = None,
        max_seq_length: int | None = None,
        padding_side: Literal["left", "right"] = "right",
        sort_by_length: bool = True,
        unk_token: bool = False,
    ):
        self._name = format_name_with_suffix(model_identifier, suffix="_tokenizer")

        self.model_identifier = model_identifier
        self.hf_token = hf_token
        self.text_field = text_field
        self.max_chars = max_chars
        self.max_seq_length = max_seq_length
        self.padding_side = padding_side
        self.sort_by_length = sort_by_length
        self.unk_token = unk_token

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field, INPUT_ID_COLUMN, ATTENTION_MASK_COLUMN] + (
            [SEQ_ORDER_COLUMN] if self.sort_by_length else []
        )

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_actor_stage": True}

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata = None) -> None:
        try:
            snapshot_download(repo_id=self.model_identifier, token=self.hf_token, local_files_only=False)
        except Exception as e:
            msg = f"Failed to download {self.model_identifier}"
            raise RuntimeError(msg) from e

    @lru_cache(maxsize=1)  # noqa: B019
    def load_cfg(self) -> AutoConfig:
        return AutoConfig.from_pretrained(self.model_identifier, local_files_only=True)

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_identifier, padding_side=self.padding_side, local_files_only=True
        )
        if self.unk_token:
            self.tokenizer.pad_token = self.tokenizer.unk_token

        if self.max_seq_length is None:
            self.max_seq_length = self.tokenizer.model_max_length

            # Guard against the HF bug
            # which sets max_seq_length to max(int) for some models
            if self.max_seq_length > 1e5:  # noqa: PLR2004
                self.max_seq_length = self.load_cfg().max_position_embeddings

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()

        if self.max_chars is not None and self.max_chars > 0:
            df[self.text_field] = df[self.text_field].str.slice(0, self.max_chars)

        with torch.no_grad():
            tokens = self.tokenizer.batch_encode_plus(
                df[self.text_field].tolist(),
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="np",
                truncation=True,
                add_special_tokens=True,
                return_token_type_ids=False,
            )

        output = df.copy()
        output[INPUT_ID_COLUMN] = tokens.input_ids.tolist()
        output[ATTENTION_MASK_COLUMN] = tokens.attention_mask.tolist()

        if self.sort_by_length:
            # Add column to preserve original order
            output[SEQ_ORDER_COLUMN] = np.arange(len(df))
            output[TOKEN_LENGTH_COLUMN] = tokens.attention_mask.sum(axis=1)
            output = output.sort_values(by=TOKEN_LENGTH_COLUMN, kind="stable", ignore_index=True).drop(
                columns=[TOKEN_LENGTH_COLUMN]
            )

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=output,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


def clip_tokens(token_o: dict, padding_side: Literal["left", "right"] = "right") -> dict[str, torch.Tensor]:
    """
    Clip the tokens to the smallest size possible.

    Args:
        token_o: The dictionary containing the input tokens (input_ids, attention_mask).
        padding_side: The side to pad the input tokens. Defaults to "right".

    Returns:
        The clipped tokens (input_ids, attention_mask).

    """
    clip_len = token_o[ATTENTION_MASK_COLUMN].sum(axis=1).max()

    if padding_side == "right":
        token_o[INPUT_ID_COLUMN] = token_o[INPUT_ID_COLUMN][:, :clip_len]
        token_o[ATTENTION_MASK_COLUMN] = token_o[ATTENTION_MASK_COLUMN][:, :clip_len]
    else:
        token_o[INPUT_ID_COLUMN] = token_o[INPUT_ID_COLUMN][:, -clip_len:]
        token_o[ATTENTION_MASK_COLUMN] = token_o[ATTENTION_MASK_COLUMN][:, -clip_len:]

    token_o.pop("metadata", None)

    return token_o


class HFModel(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Base class for Hugging Face model inference.

    Args:
        model_identifier: The identifier of the Hugging Face model.
        hf_token: Hugging Face token for downloading the model, if needed. Defaults to None.
        pred_column: The name of the prediction column.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 256.
        has_seq_order: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        padding_side: The side to pad the input tokens. Defaults to "right".

    """

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        hf_token: str | None = None,
        model_inference_batch_size: int = 256,
        has_seq_order: bool = True,
        padding_side: Literal["left", "right"] = "right",
        unpack_inference_batch: bool = False,
    ):
        self._name = format_name_with_suffix(model_identifier, suffix="_model")
        # Assume that the model can fit on a single GPU
        self._resources = Resources(cpus=1, gpus=1)

        self.model_identifier = model_identifier
        self.hf_token = hf_token
        self.model_inference_batch_size = model_inference_batch_size
        self.has_seq_order = has_seq_order
        self.padding_side = padding_side
        self.unpack_inference_batch = unpack_inference_batch

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [INPUT_ID_COLUMN, ATTENTION_MASK_COLUMN] + ([SEQ_ORDER_COLUMN] if self.has_seq_order else [])

    def outputs(self) -> tuple[list[str], list[str]]:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata = None) -> None:
        try:
            snapshot_download(repo_id=self.model_identifier, token=self.hf_token, local_files_only=False)
        except Exception as e:
            msg = f"Failed to download {self.model_identifier}"
            raise RuntimeError(msg) from e

    def setup(self, _: WorkerMetadata | None = None) -> None:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def yield_next_batch(self, df: pd.DataFrame) -> Generator[dict[str, torch.Tensor]]:
        """
        Yields a generator of model inputs for the next batch.
        We only move the batch to the GPU to reduce the memory overhead.

        Args:
            df (pd.DataFrame): The Pandas DataFrame (with input_ids and attention_mask) to process.

        Yields:
            Generator[dict[str, torch.Tensor]]: A generator of model inputs for the next batch.

        """
        for i in range(0, len(df), self.model_inference_batch_size):
            yield clip_tokens(
                {
                    INPUT_ID_COLUMN: torch.tensor(df[INPUT_ID_COLUMN][i : i + self.model_inference_batch_size].tolist()).to(
                        self.model.device
                    ),
                    ATTENTION_MASK_COLUMN: torch.tensor(
                        df[ATTENTION_MASK_COLUMN][i : i + self.model_inference_batch_size].tolist()
                    ).to(self.model.device),
                },
                padding_side=self.padding_side,
            )

    def process_model_output(self, outputs: torch.Tensor) -> dict[str, np.ndarray]:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def collect_outputs(self, processed_outputs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        result = {}
        for key in processed_outputs[0]:
            result[key] = np.concatenate([out[key] for out in processed_outputs], axis=0)
        return result

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]) -> pd.DataFrame:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        processed_outputs = []
        df_cpu = batch.to_pandas()

        for model_input_batch in self.yield_next_batch(df_cpu):
            # Forward pass
            with torch.no_grad():
                if self.unpack_inference_batch:
                    outputs = self.model(**model_input_batch)
                else:
                    outputs = self.model(model_input_batch)

            del model_input_batch

            processed_output = self.process_model_output(outputs)
            processed_outputs.append(processed_output)

        # Collect all outputs
        collected_output = self.collect_outputs(processed_outputs)

        # Create output Pandas DataFrame
        df_cpu = self.create_output_dataframe(df_cpu, collected_output)

        # Sort by seq_order to preserve original order from tokenizer
        if self.has_seq_order:
            df_cpu = df_cpu.sort_values(by=SEQ_ORDER_COLUMN, ignore_index=True).drop(columns=[SEQ_ORDER_COLUMN])

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df_cpu,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def teardown(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()


class HFDeberta(nn.Module, PyTorchModelHubMixin):
    """
    Base PyTorch model where we add a classification head.

    Args:
        config: The configuration of the model.

    """

    def __init__(self, config: dataclass):
        super().__init__()
        self.model = AutoModel.from_pretrained(config["base_model"], local_files_only=True)
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def _forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.model(batch[INPUT_ID_COLUMN], batch[ATTENTION_MASK_COLUMN]).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)

        del batch, features, dropped

        return torch.softmax(outputs[:, 0, :], dim=1)

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.autocast:
            with torch.autocast(device_type="cuda"):
                return self._forward(batch)
        else:
            return self._forward(batch)

    def set_autocast(self, autocast: bool) -> None:
        self.autocast = autocast


class HFModelStage(HFModel):
    """
    Stage for Hugging Face model inference.

    Args:
        model_identifier: The identifier of the Hugging Face model.
        pred_column: The name of the prediction column.
        prob_column: The name of the probability column. Defaults to None.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 256.
        has_seq_order: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        padding_side: The side to pad the input tokens. Defaults to "right".
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        pred_column: str,
        prob_column: str | None = None,
        model_inference_batch_size: int = 256,
        has_seq_order: bool = True,
        padding_side: Literal["left", "right"] = "right",
        autocast: bool = True,
    ):
        super().__init__(
            model_identifier=model_identifier,
            has_seq_order=has_seq_order,
            model_inference_batch_size=model_inference_batch_size,
            padding_side=padding_side,
            unpack_inference_batch=False,
        )

        self.pred_column = pred_column
        if prob_column is not None:
            self.prob_column = prob_column
            self.keep_prob_column = True
        else:
            self.prob_column = "probs"
            self.keep_prob_column = False
        self.autocast = autocast

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.pred_column] + ([self.prob_column] if self.keep_prob_column else [])

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self.model = HFDeberta.from_pretrained(self.model_identifier, local_files_only=True).cuda().eval()
        self.model.set_autocast(self.autocast)

        config = AutoConfig.from_pretrained(self.model_identifier, local_files_only=True)
        self.labels = list(config.label2id.keys())
        self.labels.sort(key=lambda x: config.label2id[x])

    def process_model_output(self, outputs: torch.Tensor) -> dict[str, np.ndarray]:
        probs = outputs.cpu().numpy()
        preds = np.argmax(probs, axis=1)

        pred_labels = [self.labels[idx] for idx in preds]

        return {
            self.prob_column: probs,
            self.pred_column: np.array(pred_labels),
        }

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]) -> pd.DataFrame:
        df_cpu = df_cpu.drop(columns=[INPUT_ID_COLUMN, ATTENTION_MASK_COLUMN])
        df_cpu[self.pred_column] = collected_output[self.pred_column]

        if self.keep_prob_column:
            df_cpu[self.prob_column] = collected_output[self.prob_column].tolist()

        return df_cpu


@dataclass(kw_only=True)
class DistributedDataClassifier(CompositeStage[DocumentBatch, DocumentBatch]):
    """
    Base composite stage for distributed data classification.

    It decomposes into a tokenizer stage and a model stage.

    Args:
        model_identifier: The identifier of the Hugging Face model.
        pred_column: The name of the prediction column.
        prob_column: The name of the probability column. Defaults to None.
        text_field: The name of the text field in the input data. Defaults to "text".
        filter_by: For categorical classifiers, the list of labels to filter the data by. Defaults to None.
        max_chars: Limits the total number of characters that can be fed to the tokenizer.
            If None, text will not be truncated. Defaults to None.
        max_seq_length: Limits the total sequence returned by the tokenizer so that it has a maximum length.
            If None, the tokenizer's model_max_length is used. Defaults to 512.
        padding_side: The side to pad the input tokens. Defaults to "right".
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 256.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    model_identifier: str
    pred_column: str
    prob_column: str | None = None
    text_field: str = "text"
    filter_by: list[str] | None = None
    max_chars: int | None = None
    max_seq_length: int | None = None
    padding_side: Literal["left", "right"] = "right"
    sort_by_length: bool = True
    model_inference_batch_size: int = 256
    autocast: bool = True

    def __post_init__(self) -> None:
        super().__init__()

        self.stages = [
            HFTokenizerStage(
                model_identifier=self.model_identifier,
                text_field=self.text_field,
                max_chars=self.max_chars,
                max_seq_length=self.max_seq_length,
                padding_side=self.padding_side,
                sort_by_length=self.sort_by_length,
            ),
            HFModelStage(
                model_identifier=self.model_identifier,
                pred_column=self.pred_column,
                prob_column=self.prob_column,
                model_inference_batch_size=self.model_inference_batch_size,
                has_seq_order=self.sort_by_length,
                padding_side=self.padding_side,
                autocast=self.autocast,
            ),
        ]

        if self.filter_by is not None and len(self.filter_by) > 0:
            self.stages.append(Filter(filter_fn=self.filter_by_category, filter_field=self.pred_column))

    def inputs(self) -> tuple[list[str], list[str]]:
        return self.stages[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self.stages[1].outputs()

    def filter_by_category(self, value: str) -> bool:
        return value in self.filter_by

    def decompose(self) -> list[ProcessingStage]:
        return self.stages
