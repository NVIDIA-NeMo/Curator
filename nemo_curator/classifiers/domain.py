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

DOMAIN_IDENTIFIER = "nvidia/domain-classifier"
DOMAIN_BASE_MODEL = "microsoft/deberta-v3-base"
MULTILINGUAL_DOMAIN_IDENTIFIER = "nvidia/multilingual-domain-classifier"
MULTILINGUAL_DOMAIN_BASE_MODEL = "microsoft/mdeberta-v3-base"


@dataclass
class DomainModelConfig:
    identifier: str = DOMAIN_IDENTIFIER
    base_model: str = DOMAIN_BASE_MODEL
    fc_dropout: float = 0.2
    max_len: int = 512


class DomainModel(HFModel):
    def __init__(
        self,
        config: DomainModelConfig,
        autocast: bool = False,
        max_mem_gb: Optional[int] = None,
    ):
        self.config = config
        self.autocast = autocast
        if max_mem_gb is None:
            max_mem_gb = _get_suggest_memory_for_classifier()

        super().__init__(self.config.base_model, max_mem_gb=max_mem_gb)

    def load_model(self, device: str = "cuda"):
        model = HFDeberta.from_pretrained(self.config.identifier)
        model.set_autocast(self.autocast)
        model = model.to(device)
        return model.eval()

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config.identifier)

    def load_config(self):
        return AutoConfig.from_pretrained(self.config.identifier)


class _DomainClassifier(DistributedDataClassifier):
    """
    Parent class for DomainClassifier and MultilingualDomainClassifier,
    since their implementations are almost identical.
    """

    def __init__(
        self,
        multilingual: bool = False,
        filter_by: Optional[List[str]] = None,
        batch_size: int = 256,
        text_field: str = "text",
        pred_column: str = "domain_pred",
        prob_column: Optional[str] = None,
        max_chars: int = 2000,
        device_type: str = "cuda",
        autocast: bool = True,
        max_mem_gb: Optional[int] = None,
    ):
        self.multilingual = multilingual

        if multilingual:
            config = AutoConfig.from_pretrained(MULTILINGUAL_DOMAIN_IDENTIFIER)
            model_config = DomainModelConfig(
                identifier=MULTILINGUAL_DOMAIN_IDENTIFIER,
                base_model=MULTILINGUAL_DOMAIN_BASE_MODEL,
            )
        else:
            config = AutoConfig.from_pretrained(DOMAIN_IDENTIFIER)
            model_config = DomainModelConfig(
                identifier=DOMAIN_IDENTIFIER,
                base_model=DOMAIN_BASE_MODEL,
            )

        self.text_field = text_field
        self.prob_column = prob_column
        self.labels = list(config.label2id.keys())
        self.labels.sort(key=lambda x: config.label2id[x])
        self.out_dim = len(self.labels)

        model = DomainModel(
            config=model_config, autocast=autocast, max_mem_gb=max_mem_gb
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
        if self.multilingual:
            print("Starting multilingual domain classifier inference", flush=True)
        else:
            print("Starting domain classifier inference", flush=True)

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


class DomainClassifier(_DomainClassifier):
    """
    DomainClassifier is a specialized classifier designed for English text domain classification tasks,
    utilizing the NVIDIA Domain Classifier (https://huggingface.co/nvidia/domain-classifier) model.
    This classifier is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large datasets.

    Attributes:
        filter_by (list[str], optional): The classes to filter the dataset by.
                                         If None, all classes will be included. Defaults to None.
        batch_size (int): The number of samples per batch for inference. Defaults to 256.
        text_field (str): The field in the dataset that should be classified.
        pred_column (str): The column name where predictions will be stored. Defaults to "domain_pred".
        prob_column (str, optional): The column name where prediction probabilities will be stored. Defaults to None.
        max_chars (int): The maximum number of characters in each document to consider for classification. Defaults to 2000.
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
        pred_column: str = "domain_pred",
        prob_column: Optional[str] = None,
        max_chars: int = 2000,
        device_type: str = "cuda",
        autocast: bool = True,
        max_mem_gb: Optional[int] = None,
    ):
        super().__init__(
            multilingual=False,
            filter_by=filter_by,
            batch_size=batch_size,
            text_field=text_field,
            pred_column=pred_column,
            prob_column=prob_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
            max_mem_gb=max_mem_gb,
        )


class MultilingualDomainClassifier(_DomainClassifier):
    """
    MultilingualDomainClassifier is a specialized classifier designed for domain classification tasks,
    utilizing the NVIDIA Multilingual Domain Classifier (https://huggingface.co/nvidia/multilingual-domain-classifier) model.
    It supports domain classification across 52 languages.
    This classifier is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large datasets.

    Attributes:
        filter_by (list[str], optional): The classes to filter the dataset by.
                                         If None, all classes will be included. Defaults to None.
        batch_size (int): The number of samples per batch for inference. Defaults to 256.
        text_field (str): The field in the dataset that should be classified.
        pred_column (str): The column name where predictions will be stored. Defaults to "domain_pred".
        prob_column (str, optional): The column name where prediction probabilities will be stored. Defaults to None.
        max_chars (int): The maximum number of characters in each document to consider for classification. Defaults to 2000.
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
        pred_column: str = "domain_pred",
        prob_column: Optional[str] = None,
        max_chars: int = 2000,
        device_type: str = "cuda",
        autocast: bool = True,
        max_mem_gb: Optional[int] = None,
    ):
        super().__init__(
            multilingual=True,
            filter_by=filter_by,
            batch_size=batch_size,
            text_field=text_field,
            pred_column=pred_column,
            prob_column=prob_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
            max_mem_gb=max_mem_gb,
        )
