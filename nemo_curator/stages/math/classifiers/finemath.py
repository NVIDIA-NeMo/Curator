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

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.classifiers.constants import (
    DEBERTA_TOKENIZER_PADDING_SIDE,
)
from nemo_curator.stages.text.models.model import ModelStage
from nemo_curator.stages.text.models.tokenizer import TokenizerStage
from nemo_curator.stages.text.models.utils import (
    ATTENTION_MASK_COLUMN,
    INPUT_ID_COLUMN,
    format_name_with_suffix,
)
from nemo_curator.tasks import DocumentBatch


FINEMATH_MODEL_ID = "HuggingFaceTB/finemath-classifier"
MAX_SEQ_LENGTH = 512


class FineMathModelStage(ModelStage):
    """
    HF sequence classification model stage for FineMath.

    Outputs columns:
    - finemath_scores (float list)
    - finemath_int_scores (int list)
    """

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        cache_dir: str | None = None,
        float_score_column: str = "finemath_scores",
        int_score_column: str = "finemath_int_scores",
        model_inference_batch_size: int = 256,
        has_seq_order: bool = True,
        autocast: bool = True,
    ):
        super().__init__(
            model_identifier=model_identifier,
            cache_dir=cache_dir,
            has_seq_order=has_seq_order,
            model_inference_batch_size=model_inference_batch_size,
            padding_side=DEBERTA_TOKENIZER_PADDING_SIDE,
            unpack_inference_batch=True,
        )
        self.float_score_column = float_score_column
        self.int_score_column = int_score_column
        self.autocast = autocast

    def outputs(self) -> tuple[list[str], list[str]]:
        return (
            ["data"],
            [self.float_score_column, self.int_score_column],
        )

    @staticmethod
    def _configure_forward(
        model: torch.nn.Module, autocast: bool = True
    ) -> torch.nn.Module:
        original_forward = model.forward

        @torch.no_grad()
        def custom_forward(*args, **kwargs) -> torch.Tensor:
            if autocast:
                with torch.autocast(device_type="cuda"):
                    output = original_forward(*args, **kwargs)
            else:
                output = original_forward(*args, **kwargs)
            return output.logits.squeeze(-1).float()

        model.forward = custom_forward
        return model

    def _setup(self, local_files_only: bool = True) -> None:
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_identifier,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        ).cuda()
        self.model = self._configure_forward(model, self.autocast)

    def process_model_output(
        self, outputs: torch.Tensor, _: dict[str, torch.Tensor] | None = None
    ) -> dict[str, np.ndarray]:
        logits = outputs.cpu().numpy()
        float_scores = [min(5.0, max(0.0, x)) for x in logits]
        int_scores = [round(max(0, min(score, 5))) for score in logits]
        return {
            self.float_score_column: float_scores,
            self.int_score_column: int_scores,
        }

    def create_output_dataframe(
        self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]
    ) -> pd.DataFrame:
        df_cpu = df_cpu.drop(columns=[INPUT_ID_COLUMN, ATTENTION_MASK_COLUMN])
        df_cpu[self.float_score_column] = collected_output[
            self.float_score_column
        ]
        df_cpu[self.int_score_column] = collected_output[
            self.int_score_column
        ]
        return df_cpu


@dataclass(kw_only=True)
class FineMathClassifier(CompositeStage[DocumentBatch, DocumentBatch]):
    """
    FineMath composite: TokenizerStage -> FineMathModelStage.
    """

    cache_dir: str | None = None
    float_score_column: str = "finemath_scores"
    int_score_column: str = "finemath_int_scores"
    text_field: str = "text"
    max_chars: int | None = None
    max_seq_length: int = MAX_SEQ_LENGTH
    sort_by_length: bool = True
    model_inference_batch_size: int = 256
    autocast: bool = True

    def __post_init__(self) -> None:
        super().__init__()
        self.stages = [
            TokenizerStage(
                model_identifier=FINEMATH_MODEL_ID,
                cache_dir=self.cache_dir,
                text_field=self.text_field,
                max_chars=self.max_chars,
                max_seq_length=self.max_seq_length,
                padding_side=DEBERTA_TOKENIZER_PADDING_SIDE,
                sort_by_length=self.sort_by_length,
            ),
            FineMathModelStage(
                model_identifier=FINEMATH_MODEL_ID,
                cache_dir=self.cache_dir,
                float_score_column=self.float_score_column,
                int_score_column=self.int_score_column,
                model_inference_batch_size=self.model_inference_batch_size,
                has_seq_order=self.sort_by_length,
                autocast=self.autocast,
            ),
        ]
        self._name = format_name_with_suffix(FINEMATH_MODEL_ID)

    def inputs(self) -> tuple[list[str], list[str]]:
        return self.stages[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self.stages[1].outputs()

    def decompose(self) -> list[ProcessingStage]:
        return self.stages 