# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Optional

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
from crossfit.backend.torch.hf.model import HFModel
from transformers import AutoConfig, AutoTokenizer

from nemo_curator.classifiers.base import (
    DistributedDataClassifier,
    HFDeberta,
    _get_suggest_memory_for_classifier,
    _run_classifier_helper,
)
from nemo_curator.datasets import DocumentDataset

CONTENT_TYPE_IDENTIFIER = "nvidia/content-type-classifier-deberta"


@dataclass
class ContentTypeModelConfig:
    model: str = "microsoft/deberta-v3-base"
    fc_dropout: float = 0.2
    max_len: int = 1024


class ContentTypeModel(HFModel):
    def __init__(
        self,
        config: ContentTypeModelConfig,
        autocast: bool = False,
        max_mem_gb: Optional[int] = None,
    ):
        self.config = config
        self.autocast = autocast
        if max_mem_gb is None:
            max_mem_gb = _get_suggest_memory_for_classifier()

        super().__init__(self.config.model, max_mem_gb=max_mem_gb)

    def load_model(self, device: str = "cuda"):
        model = HFDeberta.from_pretrained(CONTENT_TYPE_IDENTIFIER)
        model.set_autocast(self.autocast)
        model = model.to(device)
        return model.eval()

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(CONTENT_TYPE_IDENTIFIER)

    def load_config(self):
        return AutoConfig.from_pretrained(CONTENT_TYPE_IDENTIFIER)


class ContentTypeClassifier(DistributedDataClassifier):
    """
    ContentTypeClassifier is a text classification model designed to categorize documents into one of 11 distinct speech types based on their content.
    It analyzes and understands the nuances of textual information, enabling accurate classification across a diverse range of content types.
    The pretrained model used by this class can be found on Hugging Face here: https://huggingface.co/nvidia/content-type-classifier-deberta.
    This classifier is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large datasets.

    Attributes:
        filter_by (list[str], optional): The classes to filter the dataset by.
                                         If None, all classes will be included. Defaults to None.
        batch_size (int): The number of samples per batch for inference. Defaults to 256.
        text_field (str): The field in the dataset that should be classified.
        pred_column (str): The column name where predictions will be stored. Defaults to "content_pred".
        prob_column (str, optional): The column name where prediction probabilities will be stored. Defaults to None.
        max_chars (int): The maximum number of characters in each document to consider for classification. Defaults to 5000.
        device_type (str): The type of device to use for inference, either "cuda" or "cpu". Defaults to "cuda".
        autocast (bool): Whether to use mixed precision for faster inference. Defaults to True.
        max_mem_gb (int, optional): The maximum amount of memory in GB to allocate for the model. If None,
                                      it defaults to the available GPU memory minus 4 GB.

    """

    def __init__(
        self,
        filter_by: Optional[List[str]] = None,
        batch_size: int = 256,
        text_field: str = "text",
        pred_column: str = "content_pred",
        prob_column: Optional[str] = None,
        max_chars: int = 5000,
        device_type: str = "cuda",
        autocast: bool = True,
        max_mem_gb: Optional[int] = None,
    ):
        config = AutoConfig.from_pretrained(CONTENT_TYPE_IDENTIFIER)

        self.text_field = text_field
        self.prob_column = prob_column
        self.labels = list(config.label2id.keys())
        self.labels.sort(key=lambda x: config.label2id[x])
        self.out_dim = len(self.labels)

        model = ContentTypeModel(
            config=ContentTypeModelConfig, autocast=autocast, max_mem_gb=max_mem_gb
        )

        super().__init__(
            model=model,
            labels=self.labels,
            filter_by=filter_by,
            batch_size=batch_size,
            out_dim=self.out_dim,
            pred_column=pred_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
        )

    def _run_classifier(self, dataset: DocumentDataset) -> DocumentDataset:
        print("Starting content type classifier inference", flush=True)
        df = dataset.df
        df = _run_classifier_helper(
            df=df,
            model=self.model,
            labels=self.labels,
            max_chars=self.max_chars,
            batch_size=self.batch_size,
            label_col=self.pred_column,
            text_field=self.text_field,
            prob_col=self.prob_column,
        )
        return DocumentDataset(df)
