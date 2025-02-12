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
from typing import Optional

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
import torch
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from nemo_curator.classifiers.base import (
    DistributedDataClassifier,
    _get_suggest_memory_for_classifier,
)
from nemo_curator.datasets import DocumentDataset

FINEWEB_EDU_IDENTIFIER = "HuggingFaceFW/fineweb-edu-classifier"
FINEWEB_MIXTRAL_IDENTIFIER = "nvidia/nemocurator-fineweb-mixtral-edu-classifier"
FINEWEB_NEMOTRON_IDENTIFIER = "nvidia/nemocurator-fineweb-nemotron-4-edu-classifier"


class FinewebEduModel(HFModel):
    def __init__(
        self,
        path_or_name: str,
        max_mem_gb: Optional[int] = None,
        autocast: bool = False,
    ):
        self.path_or_name = path_or_name
        self.autocast = autocast
        if max_mem_gb is None:
            max_mem_gb = _get_suggest_memory_for_classifier()
        super().__init__(path_or_name=path_or_name, max_mem_gb=max_mem_gb)

    def load_model(self, device: str = "cuda"):
        model = AutoModelForSequenceClassification.from_pretrained(self.path_or_name)
        model = model.to(device)
        model = self.configure_forward(model, self.autocast)
        return model

    @staticmethod
    def configure_forward(model, autocast: bool = True):
        original_forward = model.forward

        def custom_forward(*args, **kwargs):
            if autocast:
                with torch.autocast(device_type="cuda"):
                    output = original_forward(*args, **kwargs)
            else:
                output = original_forward(*args, **kwargs)
            return output.logits.squeeze(-1).float()

        model.forward = custom_forward
        return model

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.path_or_name)

    def load_config(self):
        return AutoConfig.from_pretrained(self.path_or_name)


class _FineWebBaseClassifier(DistributedDataClassifier):
    """
    Parent class for FineWebEduClassifier, FineWebMixtralEduClassifier, and FineWebNemotronEduClassifier,
    since their implementations are almost identical.
    """

    def __init__(
        self,
        fineweb_identifier: str,
        pred_column: str,
        int_column: str,
        quality_label_column: Optional[str],
        batch_size: int = 1024,
        text_field: str = "text",
        max_chars: int = -1,
        device_type: str = "cuda",
        autocast: bool = True,
        max_mem_gb: Optional[int] = None,
    ):
        self.fineweb_identifier = fineweb_identifier

        model = FinewebEduModel(
            path_or_name=fineweb_identifier,
            autocast=autocast,
            max_mem_gb=max_mem_gb,
        )

        self.text_field = text_field
        self.int_column = int_column
        self.quality_label_column = quality_label_column

        super().__init__(
            model=model,
            filter_by=None,  # No filtering as its a numeric score
            batch_size=batch_size,
            pred_column=pred_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
            labels=None,
            out_dim=1,
        )

    def _run_classifier(self, dataset: DocumentDataset) -> DocumentDataset:
        if self.fineweb_identifier == FINEWEB_EDU_IDENTIFIER:
            print("Starting FineWeb-Edu Classifier inference", flush=True)
        elif self.fineweb_identifier == FINEWEB_MIXTRAL_IDENTIFIER:
            print("Starting FineWeb Mixtral Edu Classifier inference", flush=True)
        elif self.fineweb_identifier == FINEWEB_NEMOTRON_IDENTIFIER:
            print("Starting FineWeb Nemotron-4 Edu Classifier inference", flush=True)

        ddf = dataset.df

        pipe = op.Sequential(
            op.Tokenizer(
                self.model,
                cols=[self.text_field],
                tokenizer_type="default",
                max_length=self.model.max_seq_length(),
            ),
            op.Predictor(
                self.model,
                sorted_data_loader=True,
                batch_size=self.batch_size,
                pred_output_col=self.pred_column,
            ),
            keep_cols=ddf.columns.tolist(),
        )
        ddf = pipe(ddf)

        ddf[self.pred_column] = ddf[self.pred_column].where(
            ddf[self.pred_column] >= 0, 0
        )
        ddf[self.pred_column] = ddf[self.pred_column].where(
            ddf[self.pred_column] <= 5, 5
        )
        ddf[self.int_column] = ddf[self.pred_column].round().astype(int)

        if self.quality_label_column is not None:
            ddf[self.quality_label_column] = "high_quality"
            # If the score is less than 2.5, label it as low quality
            ddf[self.quality_label_column] = ddf[self.quality_label_column].mask(
                ddf[self.pred_column] < 2.5, "low_quality"
            )

        return DocumentDataset(ddf)


class FineWebEduClassifier(_FineWebBaseClassifier):
    """
    FineWebEduClassifier is a specialized classifier designed for educational content assessment,
    utilizing the Hugging Face FineWeb EDU Classifier model (https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier).
    This classifier is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large text datasets.

    Attributes:
        batch_size (int): The number of samples per batch for inference. Defaults to 256.
        text_field (str): The column name containing the text data to be classified. Defaults to "text".
        pred_column (str): The column name where prediction scores will be stored. Defaults to "fineweb-edu-score".
        int_column (str): The column name where integer-rounded prediction scores will be stored. Defaults to "fineweb-edu-score-int".
        max_chars (int): The maximum number of characters in each document to consider for classification. If -1, the entire document is considered. Defaults to -1.
        device_type (str): The type of device to use for inference, either "cuda" or "cpu". Defaults to "cuda".
        autocast (bool): Whether to use mixed precision for faster inference. Defaults to True.
        max_mem_gb (int, optional): The maximum amount of memory in GB to allocate for the model. If None,
                                      it defaults to the available GPU memory minus 4 GB.

    """

    def __init__(
        self,
        batch_size: int = 256,
        text_field: str = "text",
        pred_column: str = "fineweb-edu-score",
        int_column="fineweb-edu-score-int",
        max_chars: int = -1,
        device_type: str = "cuda",
        autocast: bool = True,
        max_mem_gb: Optional[int] = None,
    ):
        super().__init__(
            fineweb_identifier=FINEWEB_EDU_IDENTIFIER,
            batch_size=batch_size,
            text_field=text_field,
            pred_column=pred_column,
            int_column=int_column,
            quality_label_column=None,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
            max_mem_gb=max_mem_gb,
        )


class FineWebMixtralEduClassifier(_FineWebBaseClassifier):
    """
    FineWebMixtralEduClassifier is a specialized classifier designed for educational content assessment,
    utilizing the NemoCurator FineWeb Mixtral Edu Classifier model (https://huggingface.co/nvidia/nemocurator-fineweb-mixtral-edu-classifier).
    It is similar to the FineWeb-Edu classifier and was trained on the same text samples, but using annotations from Mixtral 8x22B-Instruct.
    This classifier is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large text datasets.

    Attributes:
        batch_size (int): The number of samples per batch for inference. Defaults to 256.
        text_field (str): The column name containing the text data to be classified. Defaults to "text".
        pred_column (str): The column name where prediction scores will be stored. Defaults to "fineweb-mixtral-edu-score".
        int_column (str): The column name where integer-rounded prediction scores will be stored. Defaults to "fineweb-mixtral-edu-score-int".
        quality_label_column (str): The column name where a score of >= 2.5 is labeled "high_quality" and otherwise labeled "low_quality". Defaults to "fineweb-mixtral-edu-score-label".
        max_chars (int): The maximum number of characters in each document to consider for classification. If -1, the entire document is considered. Defaults to -1.
        device_type (str): The type of device to use for inference, either "cuda" or "cpu". Defaults to "cuda".
        autocast (bool): Whether to use mixed precision for faster inference. Defaults to True.
        max_mem_gb (int, optional): The maximum amount of memory in GB to allocate for the model. If None,
                                      it defaults to the available GPU memory minus 4 GB.

    """

    def __init__(
        self,
        batch_size: int = 1024,
        text_field: str = "text",
        pred_column: str = "fineweb-mixtral-edu-score",
        int_column: str = "fineweb-mixtral-edu-score-int",
        quality_label_column: str = "fineweb-mixtral-edu-score-label",
        max_chars: int = -1,
        device_type: str = "cuda",
        autocast: bool = True,
        max_mem_gb: Optional[int] = None,
    ):
        super().__init__(
            fineweb_identifier=FINEWEB_MIXTRAL_IDENTIFIER,
            batch_size=batch_size,
            text_field=text_field,
            pred_column=pred_column,
            int_column=int_column,
            quality_label_column=quality_label_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
            max_mem_gb=max_mem_gb,
        )


class FineWebNemotronEduClassifier(_FineWebBaseClassifier):
    """
    FineWebNemotronEduClassifier is a specialized classifier designed for educational content assessment,
    utilizing the NemoCurator FineWeb Nemotron-4 Edu Classifier model (https://huggingface.co/nvidia/nemocurator-fineweb-nemotron-4-edu-classifier).
    It is similar to the FineWeb-Edu classifier and was trained on the same text samples, but using annotations from Nemotron-4-340B-Instruct.
    This classifier is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large text datasets.

    Attributes:
        batch_size (int): The number of samples per batch for inference. Defaults to 256.
        text_field (str): The column name containing the text data to be classified. Defaults to "text".
        pred_column (str): The column name where prediction scores will be stored. Defaults to "fineweb-nemotron-edu-score".
        int_column (str): The column name where integer-rounded prediction scores will be stored. Defaults to "fineweb-nemotron-edu-score-int".
        quality_label_column (str): The column name where a score of >= 2.5 is labeled "high_quality" and otherwise labeled "low_quality". Defaults to "fineweb-nemotron-edu-score-label".
        max_chars (int): The maximum number of characters in each document to consider for classification. If -1, the entire document is considered. Defaults to -1.
        device_type (str): The type of device to use for inference, either "cuda" or "cpu". Defaults to "cuda".
        autocast (bool): Whether to use mixed precision for faster inference. Defaults to True.
        max_mem_gb (int, optional): The maximum amount of memory in GB to allocate for the model. If None,
                                      it defaults to the available GPU memory minus 4 GB.

    """

    def __init__(
        self,
        batch_size: int = 1024,
        text_field: str = "text",
        pred_column: str = "fineweb-nemotron-edu-score",
        int_column: str = "fineweb-nemotron-edu-score-int",
        quality_label_column: str = "fineweb-nemotron-edu-score-label",
        max_chars: int = -1,
        device_type: str = "cuda",
        autocast: bool = True,
        max_mem_gb: Optional[int] = None,
    ):
        super().__init__(
            fineweb_identifier=FINEWEB_NEMOTRON_IDENTIFIER,
            batch_size=batch_size,
            text_field=text_field,
            pred_column=pred_column,
            int_column=int_column,
            quality_label_column=quality_label_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
            max_mem_gb=max_mem_gb,
        )
