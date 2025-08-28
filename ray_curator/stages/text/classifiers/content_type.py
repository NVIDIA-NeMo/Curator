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

os.environ["RAPIDS_NO_INITIALIZE"] = "1"

from nemo_curator.stages.text.models.utils import format_name_with_suffix

from .base import DistributedDataClassifier
from .constants import DEBERTA_TOKENIZER_PADDING_SIDE

CONTENT_TYPE_MODEL_IDENTIFIER = "nvidia/content-type-classifier-deberta"
MAX_SEQ_LENGTH = 1024


class ContentTypeClassifier(DistributedDataClassifier):
    """
    ContentTypeClassifier is a text classification model designed to categorize documents into one of 11 distinct speech types based on their content.
    It analyzes and understands the nuances of textual information, enabling accurate classification across a diverse range of content types.
    The pretrained model used by this class is called NemoCurator Content Type Classifier DeBERTa.
    It can be found on Hugging Face here: https://huggingface.co/nvidia/content-type-classifier-deberta.
    This classifier is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large datasets.

    Attributes:
        cache_dir: The Hugging Face cache directory. Defaults to None.
        pred_column: The name of the prediction column. Defaults to "quality_pred".
        prob_column: The name of the probability column. Defaults to None.
        text_field: The name of the text field in the input data. Defaults to "text".
        filter_by: For categorical classifiers, the list of labels to filter the data by. Defaults to None.
        max_chars: The maximum number of characters to use from the input text. Defaults to 5000.
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 256.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    def __init__(  # noqa: PLR0913
        self,
        cache_dir: str | None = None,
        pred_column: str = "content_pred",
        prob_column: str | None = None,
        text_field: str = "text",
        filter_by: list[str] | None = None,
        max_chars: int = 5000,
        sort_by_length: bool = True,
        model_inference_batch_size: int = 256,
        autocast: bool = True,
    ):
        super().__init__(
            model_identifier=CONTENT_TYPE_MODEL_IDENTIFIER,
            cache_dir=cache_dir,
            pred_column=pred_column,
            prob_column=prob_column,
            text_field=text_field,
            filter_by=filter_by,
            max_chars=max_chars,
            max_seq_length=MAX_SEQ_LENGTH,
            padding_side=DEBERTA_TOKENIZER_PADDING_SIDE,
            sort_by_length=sort_by_length,
            model_inference_batch_size=model_inference_batch_size,
            autocast=autocast,
        )

        self._name = format_name_with_suffix(CONTENT_TYPE_MODEL_IDENTIFIER)
