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
from dataclasses import dataclass

os.environ["RAPIDS_NO_INITIALIZE"] = "1"

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.models.model import ModelStage
from nemo_curator.stages.text.models.tokenizer import TokenizerStage
from nemo_curator.stages.text.models.utils import ATTENTION_MASK_COLUMN, INPUT_ID_COLUMN, format_name_with_suffix
from nemo_curator.stages.text.modules.score_filter import Filter
from nemo_curator.tasks import DocumentBatch

from .constants import DEBERTA_TOKENIZER_PADDING_SIDE

FINEWEB_EDU_MODEL_IDENTIFIER = "HuggingFaceFW/fineweb-edu-classifier"
FINEWEB_MIXTRAL_EDU_MODEL_IDENTIFIER = "nvidia/nemocurator-fineweb-mixtral-edu-classifier"
FINEWEB_NEMOTRON_EDU_MODEL_IDENTIFIER = "nvidia/nemocurator-fineweb-nemotron-4-edu-classifier"
MAX_SEQ_LENGTH = 512


class FineWebModelStage(ModelStage):
    """
    Stage for Hugging Face model inference.

    Args:
        model_identifier: The identifier of the Hugging Face model.
        cache_dir: The Hugging Face cache directory. Defaults to None.
        pred_column: The name of the prediction column.
        float_score_column: The name of the float score column.
        int_score_column: The name of the integer score column.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 256.
        has_seq_order: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        cache_dir: str | None = None,
        pred_column: str = "preds",
        float_score_column: str = "float_score",
        int_score_column: str = "int_score",
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

        self.pred_column = pred_column
        self.float_score_column = float_score_column
        self.int_score_column = int_score_column
        self.autocast = autocast

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.pred_column, self.float_score_column, self.int_score_column]

    @staticmethod
    def configure_forward(model: torch.nn.Module) -> torch.nn.Module:
        original_forward = model.forward

        @torch.no_grad()
        def custom_forward(*args, **kwargs) -> torch.Tensor:
            output = original_forward(*args, **kwargs)
            del args, kwargs
            return output.logits.squeeze(-1).float()

        model.forward = custom_forward
        return model

    def _setup(self, local_files_only: bool = True) -> None:
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_identifier,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        ).cuda()
        self.model = self.configure_forward(model)

    def process_model_output(
        self, outputs: torch.Tensor, _: dict[str, torch.Tensor] | None = None
    ) -> dict[str, np.ndarray]:
        logits = outputs.cpu().numpy()

        float_scores = logits.tolist()
        float_scores = [min(5.0, max(0.0, x)) for x in float_scores]
        int_scores = [round(max(0, min(score, 5))) for score in logits]
        pred_labels = ["high_quality" if score >= 2.5 else "low_quality" for score in logits]  # noqa: PLR2004

        return {
            self.float_score_column: float_scores,
            self.int_score_column: int_scores,
            self.pred_column: pred_labels,
        }

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]) -> pd.DataFrame:
        df_cpu = df_cpu.drop(columns=[INPUT_ID_COLUMN, ATTENTION_MASK_COLUMN])

        df_cpu[self.float_score_column] = collected_output[self.float_score_column]
        df_cpu[self.int_score_column] = collected_output[self.int_score_column]
        df_cpu[self.pred_column] = collected_output[self.pred_column]

        return df_cpu


@dataclass(kw_only=True)
class _FineWebBaseClassifier(CompositeStage[DocumentBatch, DocumentBatch]):
    """
    Parent class for FineWebEduClassifier, FineWebMixtralEduClassifier, and FineWebNemotronEduClassifier,
    since their implementations are almost identical.

    Args:
        model_identifier: The identifier of the Hugging Face model.
        cache_dir: The Hugging Face cache directory. Defaults to None.
        pred_column: The name of the prediction column.
        float_score_column: The name of the float score column.
        int_score_column: The name of the integer score column.
        text_field: The name of the text field in the input data. Defaults to "text".
        filter_by: For categorical classifiers, the list of labels to filter the data by. Defaults to None.
        max_chars: Limits the total number of characters that can be fed to the tokenizer.
            If None, text will not be truncated. Defaults to None.
        max_seq_length: Limits the total sequence returned by the tokenizer so that it has a maximum length.
            Defaults to 512.
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 256.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    model_identifier: str
    cache_dir: str | None = None
    pred_column: str = "preds"
    float_score_column: str = "float_score"
    int_score_column: str = "int_score"
    text_field: str = "text"
    filter_by: list[str] | None = None
    max_chars: int | None = None
    max_seq_length: int = MAX_SEQ_LENGTH
    sort_by_length: bool = True
    model_inference_batch_size: int = 256
    autocast: bool = True

    def __post_init__(self) -> None:
        super().__init__()

        self.stages = [
            TokenizerStage(
                model_identifier=self.model_identifier,
                cache_dir=self.cache_dir,
                text_field=self.text_field,
                max_chars=self.max_chars,
                max_seq_length=self.max_seq_length,
                padding_side=DEBERTA_TOKENIZER_PADDING_SIDE,
                sort_by_length=self.sort_by_length,
            ),
            FineWebModelStage(
                model_identifier=self.model_identifier,
                cache_dir=self.cache_dir,
                pred_column=self.pred_column,
                float_score_column=self.float_score_column,
                int_score_column=self.int_score_column,
                model_inference_batch_size=self.model_inference_batch_size,
                has_seq_order=self.sort_by_length,
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


class FineWebEduClassifier(_FineWebBaseClassifier):
    """
    FineWebEduClassifier is a specialized classifier designed for educational content assessment,
    utilizing the Hugging Face FineWeb EDU Classifier model (https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier).
    This classifier is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large text datasets.

    Attributes:
        cache_dir: The Hugging Face cache directory. Defaults to None.
        pred_column: The name of the prediction column. Defaults to "fineweb-edu-score-label".
        float_score_column: The name of the float score column. Defaults to "fineweb-edu-score-float".
        int_score_column: The name of the integer score column. Defaults to "fineweb-edu-score-int".
        text_field: The name of the text field in the input data. Defaults to "text".
        filter_by: For categorical classifiers, the list of labels to filter the data by. Defaults to None.
        max_chars: Limits the total number of characters that can be fed to the tokenizer.
            If None, text will not be truncated. Defaults to None.
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 256.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    def __init__(  # noqa: PLR0913
        self,
        cache_dir: str | None = None,
        pred_column: str = "fineweb-edu-score-label",
        float_score_column: str = "fineweb-edu-score-float",
        int_score_column: str = "fineweb-edu-score-int",
        text_field: str = "text",
        filter_by: list[str] | None = None,
        max_chars: int | None = None,
        sort_by_length: bool = True,
        model_inference_batch_size: int = 256,
        autocast: bool = True,
    ):
        super().__init__(
            model_identifier=FINEWEB_EDU_MODEL_IDENTIFIER,
            cache_dir=cache_dir,
            pred_column=pred_column,
            float_score_column=float_score_column,
            int_score_column=int_score_column,
            text_field=text_field,
            filter_by=filter_by,
            max_chars=max_chars,
            max_seq_length=MAX_SEQ_LENGTH,
            sort_by_length=sort_by_length,
            model_inference_batch_size=model_inference_batch_size,
            autocast=autocast,
        )

        self._name = format_name_with_suffix(FINEWEB_EDU_MODEL_IDENTIFIER)


class FineWebMixtralEduClassifier(_FineWebBaseClassifier):
    """
    FineWebMixtralEduClassifier is a specialized classifier designed for educational content assessment,
    utilizing the NemoCurator FineWeb Mixtral Edu Classifier model (https://huggingface.co/nvidia/nemocurator-fineweb-mixtral-edu-classifier).
    It is similar to the FineWeb-Edu classifier and was trained on the same text samples, but using annotations from Mixtral 8x22B-Instruct.
    This classifier is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large text datasets.

    Attributes:
        cache_dir: The Hugging Face cache directory. Defaults to None.
        pred_column: The name of the prediction column. Defaults to "fineweb-mixtral-edu-score-label".
        float_score_column: The name of the float score column. Defaults to "fineweb-mixtral-edu-score-float".
        int_score_column: The name of the integer score column. Defaults to "fineweb-mixtral-edu-score-int".
        text_field: The name of the text field in the input data. Defaults to "text".
        filter_by: For categorical classifiers, the list of labels to filter the data by. Defaults to None.
        max_chars: Limits the total number of characters that can be fed to the tokenizer.
            If None, text will not be truncated. Defaults to None.
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 1024.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    def __init__(  # noqa: PLR0913
        self,
        cache_dir: str | None = None,
        pred_column: str = "fineweb-mixtral-edu-score-label",
        float_score_column: str = "fineweb-mixtral-edu-score-float",
        int_score_column: str = "fineweb-mixtral-edu-score-int",
        text_field: str = "text",
        filter_by: list[str] | None = None,
        max_chars: int | None = None,
        sort_by_length: bool = True,
        model_inference_batch_size: int = 1024,
        autocast: bool = True,
    ):
        super().__init__(
            model_identifier=FINEWEB_MIXTRAL_EDU_MODEL_IDENTIFIER,
            cache_dir=cache_dir,
            pred_column=pred_column,
            float_score_column=float_score_column,
            int_score_column=int_score_column,
            text_field=text_field,
            filter_by=filter_by,
            max_chars=max_chars,
            max_seq_length=MAX_SEQ_LENGTH,
            sort_by_length=sort_by_length,
            model_inference_batch_size=model_inference_batch_size,
            autocast=autocast,
        )

        self._name = format_name_with_suffix(FINEWEB_MIXTRAL_EDU_MODEL_IDENTIFIER)


class FineWebNemotronEduClassifier(_FineWebBaseClassifier):
    """
    FineWebNemotronEduClassifier is a specialized classifier designed for educational content assessment,
    utilizing the NemoCurator FineWeb Nemotron-4 Edu Classifier model (https://huggingface.co/nvidia/nemocurator-fineweb-nemotron-4-edu-classifier).
    It is similar to the FineWeb-Edu classifier and was trained on the same text samples, but using annotations from Nemotron-4-340B-Instruct.
    This classifier is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large text datasets.

    Attributes:
        cache_dir: The Hugging Face cache directory. Defaults to None.
        pred_column: The name of the prediction column. Defaults to "fineweb-nemotron-edu-score-label".
        float_score_column: The name of the float score column. Defaults to "fineweb-nemotron-edu-score-float".
        int_score_column: The name of the integer score column. Defaults to "fineweb-nemotron-edu-score-int".
        text_field: The name of the text field in the input data. Defaults to "text".
        filter_by: For categorical classifiers, the list of labels to filter the data by. Defaults to None.
        max_chars: Limits the total number of characters that can be fed to the tokenizer.
            If None, text will not be truncated. Defaults to None.
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 1024.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    def __init__(  # noqa: PLR0913
        self,
        cache_dir: str | None = None,
        pred_column: str = "fineweb-nemotron-edu-score-label",
        float_score_column: str = "fineweb-nemotron-edu-score-float",
        int_score_column: str = "fineweb-nemotron-edu-score-int",
        text_field: str = "text",
        filter_by: list[str] | None = None,
        max_chars: int | None = None,
        sort_by_length: bool = True,
        model_inference_batch_size: int = 1024,
        autocast: bool = True,
    ):
        super().__init__(
            model_identifier=FINEWEB_NEMOTRON_EDU_MODEL_IDENTIFIER,
            cache_dir=cache_dir,
            pred_column=pred_column,
            float_score_column=float_score_column,
            int_score_column=int_score_column,
            text_field=text_field,
            filter_by=filter_by,
            max_chars=max_chars,
            max_seq_length=MAX_SEQ_LENGTH,
            sort_by_length=sort_by_length,
            model_inference_batch_size=model_inference_batch_size,
            autocast=autocast,
        )

        self._name = format_name_with_suffix(FINEWEB_NEMOTRON_EDU_MODEL_IDENTIFIER)
