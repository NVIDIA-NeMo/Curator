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
from typing import Literal

os.environ["RAPIDS_NO_INITIALIZE"] = "1"

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.stages.classifiers.base import HFModel, HFTokenizerStage
from ray_curator.stages.modules.score_filter import Filter
from ray_curator.tasks import DocumentBatch


class HFFineWebModelStage(HFModel):
    """
    Stage for Hugging Face model inference.

    Args:
        model_identifier: The identifier of the Hugging Face model.
        pred_column: The name of the prediction column.
        float_score_column: The name of the float score column.
        int_score_column: The name of the integer score column.
        micro_batch_size: The size of the micro-batch. Defaults to 256.
        has_seq_order: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        pred_column: str,
        float_score_column: str,
        int_score_column: str,
        micro_batch_size: int = 256,
        has_seq_order: bool = True,
        autocast: bool = True,
    ):
        super().__init__(
            pred_column=pred_column,
            model_identifier=model_identifier,
            has_seq_order=has_seq_order,
            micro_batch_size=micro_batch_size,
            padding_side="right",
        )

        self.float_score_column = float_score_column
        self.int_score_column = int_score_column
        self.autocast = autocast

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["input_ids", "attention_mask"] + (["_curator_seq_order"] if self.has_seq_order else [])

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.pred_column, self.float_score_column, self.int_score_column]

    @staticmethod
    def configure_forward(model: torch.nn.Module, autocast: bool = True) -> torch.nn.Module:
        original_forward = model.forward

        def custom_forward(*args, **kwargs) -> torch.Tensor:
            if autocast:
                with torch.autocast(device_type="cuda"):
                    output = original_forward(*args, **kwargs)
            else:
                output = original_forward(*args, **kwargs)
            return output.logits.squeeze(-1).float()

        model.forward = custom_forward
        return model

    def setup(self, _: WorkerMetadata | None) -> None:
        model = AutoModelForSequenceClassification.from_pretrained(self.model_identifier).cuda()
        self.model = self.configure_forward(model, self.autocast)

    def process_model_output(self, outputs: torch.Tensor) -> dict[str, np.ndarray]:
        logits = outputs.cpu().numpy()

        float_scores = logits.tolist()
        float_scores = [min(5.0, max(0.0, x)) for x in float_scores]
        int_scores = [round(max(0, min(score, 5))) for score in logits]
        pred_labels = ["high_quality" if score >= 2.5 else "low_quality" for score in logits]  # noqa: PLR2004

        return {
            "floats": float_scores,
            "ints": int_scores,
            "preds": pred_labels,
        }

    def collect_outputs(self, processed_outputs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {
            "floats": np.concatenate([out["floats"] for out in processed_outputs], axis=0),
            "ints": np.concatenate([out["ints"] for out in processed_outputs], axis=0),
            "preds": np.concatenate([out["preds"] for out in processed_outputs], axis=0),
        }

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]) -> pd.DataFrame:
        df_cpu = df_cpu.drop(columns=["input_ids", "attention_mask"])

        df_cpu[self.float_score_column] = collected_output["floats"]
        df_cpu[self.int_score_column] = collected_output["ints"]
        df_cpu[self.pred_column] = collected_output["preds"]

        return df_cpu

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        processed_outputs = []
        df_cpu = batch.to_pandas()

        for model_input_batch in self.yield_next_batch(df_cpu):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**model_input_batch)

            processed_output = self.process_model_output(outputs)
            processed_outputs.append(processed_output)

        # Collect all outputs
        collected_output = self.collect_outputs(processed_outputs)

        # Create output Pandas DataFrame
        df_cpu = self.create_output_dataframe(df_cpu, collected_output)

        # Sort by seq_order to preserve original order from tokenizer
        if self.has_seq_order:
            df_cpu = df_cpu.sort_values(by="_curator_seq_order", ignore_index=True).drop(
                columns=["_curator_seq_order"]
            )

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df_cpu,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass(kw_only=True)
class _FineWebBaseClassifier(CompositeStage[DocumentBatch, DocumentBatch]):
    """
    Parent class for FineWebEduClassifier, FineWebMixtralEduClassifier, and FineWebNemotronEduClassifier,
    since their implementations are almost identical.

    Args:
        model_identifier: The identifier of the Hugging Face model.
        pred_column: The name of the prediction column.
        float_score_column: The name of the float score column.
        int_score_column: The name of the integer score column.
        text_field: The name of the text field in the input data. Defaults to "text".
        filter_by: For categorical classifiers, the list of labels to filter the data by. Defaults to None.
        max_seq_length: Limits the total sequence returned by the tokenizer so that it has a maximum length.
            Defaults to 512.
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        micro_batch_size: The size of the micro-batch. Defaults to 256.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    model_identifier: str
    pred_column: str
    float_score_column: str
    int_score_column: str
    text_field: str = "text"
    filter_by: list[str] | None = None
    max_seq_length: int = 512
    sort_by_length: bool = True
    micro_batch_size: int = 256
    autocast: bool = True

    def __post_init__(self) -> None:
        super().__init__()

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.pred_column, self.float_score_column, self.int_score_column]

    def filter_by_category(self, value: str) -> bool:
        return value in self.filter_by

    def decompose(self) -> list[ProcessingStage]:
        stages = [
            HFTokenizerStage(
                model_identifier=self.model_identifier,
                text_field=self.text_field,
                max_seq_length=self.max_seq_length,
                padding_side="right",
                sort_by_length=self.sort_by_length,
            ),
            HFFineWebModelStage(
                model_identifier=self.model_identifier,
                pred_column=self.pred_column,
                float_score_column=self.float_score_column,
                int_score_column=self.int_score_column,
                micro_batch_size=self.micro_batch_size,
                has_seq_order=self.sort_by_length,
                autocast=self.autocast,
            ),
        ]

        if self.filter_by is not None and len(self.filter_by) > 0:
            stages.append(Filter(filter_fn=self.filter_by_category, filter_field=self.pred_column))

        return stages


class FineWebEduClassifier(_FineWebBaseClassifier):
    """
    FineWebEduClassifier is a specialized classifier designed for educational content assessment,
    utilizing the Hugging Face FineWeb EDU Classifier model (https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier).
    This classifier is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large text datasets.

    Attributes:
        pred_column: The name of the prediction column. Defaults to "fineweb-edu-score-label".
        float_score_column: The name of the float score column. Defaults to "fineweb-edu-score-float".
        int_score_column: The name of the integer score column. Defaults to "fineweb-edu-score-int".
        text_field: The name of the text field in the input data. Defaults to "text".
        filter_by: For categorical classifiers, the list of labels to filter the data by. Defaults to None.
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        micro_batch_size: The size of the micro-batch. Defaults to 256.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    def __init__(  # noqa: PLR0913
        self,
        pred_column: str = "fineweb-edu-score-label",
        float_score_column: str = "fineweb-edu-score-float",
        int_score_column: str = "fineweb-edu-score-int",
        text_field: str = "text",
        filter_by: list[str] | None = None,
        sort_by_length: bool = True,
        micro_batch_size: int = 256,
        autocast: bool = True,
    ):
        self._name = "fineweb_edu_classifier"

        super().__init__(
            model_identifier="HuggingFaceFW/fineweb-edu-classifier",
            pred_column=pred_column,
            float_score_column=float_score_column,
            int_score_column=int_score_column,
            text_field=text_field,
            filter_by=filter_by,
            max_seq_length=512,
            sort_by_length=sort_by_length,
            micro_batch_size=micro_batch_size,
            autocast=autocast,
        )


class FineWebMixtralEduClassifier(_FineWebBaseClassifier):
    """
    FineWebMixtralEduClassifier is a specialized classifier designed for educational content assessment,
    utilizing the NemoCurator FineWeb Mixtral Edu Classifier model (https://huggingface.co/nvidia/nemocurator-fineweb-mixtral-edu-classifier).
    It is similar to the FineWeb-Edu classifier and was trained on the same text samples, but using annotations from Mixtral 8x22B-Instruct.
    This classifier is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large text datasets.

    Attributes:
        pred_column: The name of the prediction column. Defaults to "fineweb-mixtral-edu-score-label".
        float_score_column: The name of the float score column. Defaults to "fineweb-mixtral-edu-score-float".
        int_score_column: The name of the integer score column. Defaults to "fineweb-mixtral-edu-score-int".
        text_field: The name of the text field in the input data. Defaults to "text".
        filter_by: For categorical classifiers, the list of labels to filter the data by. Defaults to None.
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        micro_batch_size: The size of the micro-batch. Defaults to 1024.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    def __init__(  # noqa: PLR0913
        self,
        pred_column: str = "fineweb-mixtral-edu-score-label",
        float_score_column: str = "fineweb-mixtral-edu-score-float",
        int_score_column: str = "fineweb-mixtral-edu-score-int",
        text_field: str = "text",
        filter_by: list[str] | None = None,
        sort_by_length: bool = True,
        micro_batch_size: int = 1024,
        autocast: bool = True,
    ):
        self._name = "fineweb_mixtral_edu_classifier"

        super().__init__(
            model_identifier="nvidia/nemocurator-fineweb-mixtral-edu-classifier",
            pred_column=pred_column,
            float_score_column=float_score_column,
            int_score_column=int_score_column,
            text_field=text_field,
            filter_by=filter_by,
            max_seq_length=512,
            sort_by_length=sort_by_length,
            micro_batch_size=micro_batch_size,
            autocast=autocast,
        )


class FineWebNemotronEduClassifier(_FineWebBaseClassifier):
    """
    FineWebNemotronEduClassifier is a specialized classifier designed for educational content assessment,
    utilizing the NemoCurator FineWeb Nemotron-4 Edu Classifier model (https://huggingface.co/nvidia/nemocurator-fineweb-nemotron-4-edu-classifier).
    It is similar to the FineWeb-Edu classifier and was trained on the same text samples, but using annotations from Nemotron-4-340B-Instruct.
    This classifier is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large text datasets.

    Attributes:
        pred_column: The name of the prediction column. Defaults to "fineweb-nemotron-edu-score-label".
        float_score_column: The name of the float score column. Defaults to "fineweb-nemotron-edu-score-float".
        int_score_column: The name of the integer score column. Defaults to "fineweb-nemotron-edu-score-int".
        text_field: The name of the text field in the input data. Defaults to "text".
        filter_by: For categorical classifiers, the list of labels to filter the data by. Defaults to None.
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        micro_batch_size: The size of the micro-batch. Defaults to 1024.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    def __init__(  # noqa: PLR0913
        self,
        pred_column: str = "fineweb-nemotron-edu-score-label",
        float_score_column: str = "fineweb-nemotron-edu-score-float",
        int_score_column: str = "fineweb-nemotron-edu-score-int",
        text_field: str = "text",
        filter_by: list[str] | None = None,
        sort_by_length: bool = True,
        micro_batch_size: int = 1024,
        autocast: bool = True,
    ):
        self._name = "fineweb_nemotron_edu_classifier"

        super().__init__(
            model_identifier="nvidia/nemocurator-fineweb-nemotron-4-edu-classifier",
            pred_column=pred_column,
            float_score_column=float_score_column,
            int_score_column=int_score_column,
            text_field=text_field,
            filter_by=filter_by,
            max_seq_length=512,
            sort_by_length=sort_by_length,
            micro_batch_size=micro_batch_size,
            autocast=autocast,
        )
