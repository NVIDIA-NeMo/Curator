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
from typing import Literal

os.environ["RAPIDS_NO_INITIALIZE"] = "1"

from .base import DistributedDataClassifier


class QualityClassifier(DistributedDataClassifier):
    """
    QualityClassifier is a specialized classifier designed for quality assessment tasks,
    utilizing the NemoCurator Quality Classifier DeBERTa model (https://huggingface.co/nvidia/quality-classifier-deberta).
    This classifier is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large datasets.

    Attributes:
        pred_column: The name of the prediction column. Defaults to "quality_pred".
        prob_column: The name of the probability column. Defaults to None.
        text_field: The name of the text field in the input data. Defaults to "text".
        filter_by: For categorical classifiers, the list of labels to filter the data by. Defaults to None.
        max_seq_length: The maximum number of characters that can be fed to the tokenizer.
            If None, the tokenizer's model_max_length is used. Defaults to None.
        padding_side: The side to pad the input tokens. Defaults to "right".
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        micro_batch_size: The size of the micro-batch. Defaults to 256.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    def __init__(  # noqa: PLR0913
        self,
        pred_column: str = "quality_pred",
        prob_column: str | None = None,
        text_field: str = "text",
        filter_by: list[str] | None = None,
        max_seq_length: int | None = None,
        padding_side: Literal["left", "right"] = "right",
        sort_by_length: bool = True,
        micro_batch_size: int = 256,
        autocast: bool = True,
    ):
        self._name = "quality_classifier"

        super().__init__(
            model_identifier="nvidia/quality-classifier-deberta",
            pred_column=pred_column,
            prob_column=prob_column,
            text_field=text_field,
            filter_by=filter_by,
            max_seq_length=max_seq_length,
            padding_side=padding_side,
            sort_by_length=sort_by_length,
            micro_batch_size=micro_batch_size,
            autocast=autocast,
        )
