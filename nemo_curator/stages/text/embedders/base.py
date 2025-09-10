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

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from transformers import AutoModel

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.models.model import ModelStage
from nemo_curator.stages.text.models.tokenizer import TokenizerStage
from nemo_curator.stages.text.models.utils import ATTENTION_MASK_COLUMN
from nemo_curator.tasks import DocumentBatch


class EmbeddingModelStage(ModelStage):
    """HuggingFace model stage that produces embeddings with pooling."""

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        embedding_field: str = "embeddings",
        pooling: Literal["mean_pooling", "last_token"] = "mean_pooling",
        hf_token: str | None = None,
        model_inference_batch_size: int = 1024,
        has_seq_order: bool = True,
        padding_side: Literal["left", "right"] = "right",
        autocast: bool = True,
    ):
        super().__init__(
            model_identifier=model_identifier,
            hf_token=hf_token,
            model_inference_batch_size=model_inference_batch_size,
            has_seq_order=has_seq_order,
            padding_side=padding_side,
            unpack_inference_batch=True,
            autocast=autocast,
        )
        self.embedding_field = embedding_field
        self.pooling = pooling

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.embedding_field]

    def setup(self, _: WorkerMetadata | None = None) -> None:
        """Load the model for inference."""
        self.model = AutoModel.from_pretrained(self.model_identifier, local_files_only=True)
        self.model.eval().to("cuda")

    def process_model_output(
        self, outputs: torch.Tensor, model_input_batch: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Process model outputs to create embeddings."""
        if self.pooling == "mean_pooling":
            return self._mean_pooling(outputs, model_input_batch[ATTENTION_MASK_COLUMN]).cpu()
        else:
            return self._get_last_token(outputs, model_input_batch[ATTENTION_MASK_COLUMN]).cpu()

    def collect_outputs(self, processed_outputs: list[torch.Tensor]) -> list[list[float]]:
        return torch.cat(processed_outputs, dim=0).numpy().tolist()

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: list[list[float]]) -> pd.DataFrame:
        """Create output dataframe with embeddings."""
        return df_cpu.assign(**{self.embedding_field: collected_output})

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        # Mask out irrelevant tokens directly without expanding the mask
        masked_embeddings = token_embeddings.masked_fill(attention_mask.unsqueeze(-1) == 0, 0.0)
        sum_embeddings = masked_embeddings.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return F.normalize(sum_embeddings / sum_mask, dim=1)

    def _get_last_token(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        # Get indices of last non-padded tokens for each sequence in batch
        last_token_indices = attention_mask.sum(dim=1) - 1  # -1 for 0-based indexing
        last_token_indices = last_token_indices.to(torch.long)  # Ensure indices are of type long
        batch_size = attention_mask.size(0)
        batch_indices = torch.arange(batch_size, device=attention_mask.device)
        # Get embeddings of last non-padded tokens
        last_token_embeddings = token_embeddings[batch_indices, last_token_indices]
        return F.normalize(last_token_embeddings, dim=1)


@dataclass(kw_only=True)
class EmbeddingCreatorStage(CompositeStage[DocumentBatch, DocumentBatch]):
    model_identifier: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_field: str = "text"
    embedding_field: str = "embeddings"
    max_chars: int | None = None
    max_seq_length: int | None = None
    padding_side: Literal["left", "right"] = "right"
    embedding_pooling: Literal["mean_pooling", "last_token"] = "mean_pooling"
    model_inference_batch_size: int = 1024
    autocast: bool = True
    sort_by_length: bool = True
    hf_token: str | None = None

    def __post_init__(self) -> None:
        super().__init__()

        self.stages = [
            TokenizerStage(
                model_identifier=self.model_identifier,
                hf_token=self.hf_token,
                text_field=self.text_field,
                max_chars=self.max_chars,
                max_seq_length=self.max_seq_length,
                padding_side=self.padding_side,
                sort_by_length=self.sort_by_length,
            ),
            EmbeddingModelStage(
                model_identifier=self.model_identifier,
                embedding_field=self.embedding_field,
                pooling=self.embedding_pooling,
                hf_token=self.hf_token,
                model_inference_batch_size=self.model_inference_batch_size,
                has_seq_order=self.sort_by_length,
                padding_side=self.padding_side,
                autocast=self.autocast,
            ),
        ]

    def decompose(self) -> list[ProcessingStage]:
        return self.stages
