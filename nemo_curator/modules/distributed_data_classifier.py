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

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from packaging import version
from transformers import AutoConfig, AutoModel
from transformers import __version__ as TRANSFORMERS_VERSION
from transformers.models.deberta_v2 import DebertaV2TokenizerFast

from nemo_curator.datasets import DocumentDataset


@dataclass
class DomainModelConfig:
    model = "microsoft/deberta-v3-base"
    fc_dropout = 0.2
    max_len = 512


@dataclass
class QualityModelConfig:
    model = "microsoft/deberta-v3-base"
    fc_dropout = 0.2
    max_len = 512


class CustomModel(nn.Module):
    def __init__(
        self, config, out_dim, config_path=None, pretrained=False, autocast=False
    ):
        super().__init__()
        self.config = config
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                config.model, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(config.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.fc_dropout = nn.Dropout(config.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, out_dim)
        self._init_weights(self.fc)
        self.autocast = autocast

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def _forward(self, batch):
        feature = self.feature(batch["input_ids"], batch["attention_mask"])
        output = self.fc(self.fc_dropout(feature))
        output = output.to(torch.float32)
        return torch.softmax(output[:, 0, :], dim=1)

    def forward(self, batch):
        if self.autocast:
            with torch.autocast(device_type="cuda"):
                return self._forward(batch)
        else:
            return self._forward(batch)


class DistributedDataClassifier(ABC):
    """Abstract class for running multi-node multi-GPU data classification"""

    def __init__(
        self,
        model,
        labels,
        filter_by,
        batch_size,
        out_dim,
        pred_column,
        max_chars,
        device_type,
        autocast,
    ):
        self.model = model
        self.labels = labels
        self.filter_by = filter_by
        self.batch_size = batch_size
        self.out_dim = out_dim
        self.pred_column = pred_column
        self.max_chars = max_chars
        self.device_type = device_type
        self.autocast = autocast

    def __call__(self, dataset: DocumentDataset):
        result_doc_dataset = self._run_classifier(dataset)
        if self.filter_by is not None:
            return self._filter_documents(result_doc_dataset)

        return result_doc_dataset

    @abstractmethod
    def _run_classifier(self):
        pass

    def _filter_documents(
        self,
        dataset: DocumentDataset,
    ):
        df = dataset.df

        filter_by = self.filter_by
        if type(filter_by) == str:
            filtered_df = df[df[self.pred_column].astype(str) == filter_by]
            return DocumentDataset(filtered_df)
        elif type(filter_by) == list:
            filtered_df = df[df[self.pred_column].isin(filter_by)]
            return DocumentDataset(filtered_df)

        raise TypeError("filter_by must be a string or list type")


def _run_classifier_helper(
    df: "dask_cudf.DataFrame",
    model: "HFModel",
    labels: list[str],
    max_chars: int,
    batch_size: int,
    label_col: str,
    prob_col: str = None,
) -> "dask_cudf.DataFrame":

    keep_prob = prob_col is not None
    prob_internal_col = "_prob"
    # TODO: Make crossfit handle this cleanly
    pred_internal_col = "labels"
    df["sliced_text"] = df["text"].str.slice(0, max_chars)
    columns_to_keep_list = df.columns.to_list()
    columns_to_keep_list.remove("sliced_text")

    classifier_pipe = op.Sequential(
        op.Tokenizer(model, cols=["sliced_text"], tokenizer_type="sentencepiece"),
        op.Predictor(
            model,
            sorted_data_loader=True,
            batch_size=batch_size,
            pred_output_col=prob_internal_col,
        ),
        repartition=df.npartitions,
        keep_cols=columns_to_keep_list,
    )
    df = classifier_pipe(df)
    # TODO: Make crossfit handle this cleanly
    # to prevent the labeler from dropping the prob_internal_col
    # and combine it into a single step
    labeling_pipe = op.Sequential(
        op.Labeler(labels, cols=[prob_internal_col]),
        keep_cols=columns_to_keep_list + [prob_internal_col],
    )
    df = labeling_pipe(df)
    if keep_prob:
        df = df.rename(
            columns={prob_internal_col: prob_col, pred_internal_col: label_col},
        )
    else:
        df = df.rename(columns={pred_internal_col: label_col})
        df = df.drop(columns=[prob_internal_col])
    return df


class DomainModel(HFModel):
    def __init__(self, config, out_dim=None, model_path=None, autocast=False):
        self.config = config
        self.out_dim = out_dim
        self.model_path = model_path
        self.autocast = autocast
        super().__init__(self.config.model)

    def load_model(self, device="cuda"):
        model = CustomModel(
            self.config,
            out_dim=self.out_dim,
            config_path=None,
            pretrained=True,
            autocast=self.autocast,
        )
        model = model.to(device)
        if os.path.exists(self.model_path):
            sd = torch.load(os.path.join(self.model_path), map_location="cpu")
            sd = {k[7:] if k.startswith("module.") else k: sd[k] for k in sd.keys()}
            if version.parse(TRANSFORMERS_VERSION) >= version.parse("4.31.0"):
                sd.pop("model.embeddings.position_ids", None)
            model.load_state_dict(sd, strict=True)
        else:
            raise ValueError(f"Model path {self.model_path} does not exist")
        return model.eval()

    def load_tokenizer(self):
        return DebertaV2TokenizerFast.from_pretrained(self.config.model)

    def load_config(self):
        return AutoConfig.from_pretrained(self.path_or_name)


class QualityModel(HFModel):
    def __init__(self, config, out_dim=None, model_path=None, autocast=False):
        self.config = config
        self.out_dim = out_dim
        self.model_path = model_path
        self.autocast = autocast
        super().__init__(self.config.model)

    def load_model(self, device="cuda"):
        model = CustomModel(
            self.config,
            out_dim=self.out_dim,
            config_path=None,
            pretrained=True,
            autocast=self.autocast,
        )
        model = model.to(device)
        if os.path.exists(self.model_path):
            sd = torch.load(self.model_path, map_location="cpu")
            if "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            sd = {k[7:] if k.startswith("module.") else k: sd[k] for k in sd.keys()}
            model.load_state_dict(sd, strict=True)
        else:
            raise ValueError(f"Model path {self.model_path} does not exist")
        model.eval()
        return model

    def load_tokenizer(self):
        return DebertaV2TokenizerFast.from_pretrained(self.config.model)

    def load_config(self):
        return AutoConfig.from_pretrained(self.path_or_name)


class DomainClassifier(DistributedDataClassifier):
    def __init__(
        self,
        model_path,
        labels,
        filter_by=None,
        batch_size=256,
        out_dim=None,
        pred_column="domain_pred",
        prob_column=None,
        max_chars=2000,
        device_type="cuda",
        autocast=True,
    ):
        if out_dim is None:
            out_dim = len(labels)

        self.prob_column = prob_column

        model = DomainModel(
            config=DomainModelConfig,
            out_dim=out_dim,
            model_path=model_path,
            autocast=autocast,
        )

        super().__init__(
            model=model,
            labels=labels,
            filter_by=filter_by,
            batch_size=batch_size,
            out_dim=out_dim,
            pred_column=pred_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
        )

    def _run_classifier(self, dataset: DocumentDataset):
        print("Starting domain classifier inference", flush=True)
        df = dataset.df
        df = _run_classifier_helper(
            df=df,
            model=self.model,
            labels=self.labels,
            max_chars=self.max_chars,
            batch_size=self.batch_size,
            label_col=self.pred_column,
            prob_col=self.prob_column,
        )
        return DocumentDataset(df)


class QualityClassifier(DistributedDataClassifier):
    def __init__(
        self,
        model_path,
        labels,
        filter_by=None,
        batch_size=256,
        out_dim=None,
        pred_column="quality_pred",
        prob_column="quality_prob",
        max_chars=6000,
        device_type="cuda",
        autocast=True,
    ):
        if len(labels) == 2:
            out_dim = 1  # Binary classification
        else:
            if out_dim is None:
                out_dim = len(labels)  # Multiclass classification

        self.prob_column = prob_column

        model = QualityModel(
            config=QualityModelConfig,
            out_dim=out_dim,
            model_path=model_path,
            autocast=autocast,
        )

        super().__init__(
            model=model,
            labels=labels,
            filter_by=filter_by,
            batch_size=batch_size,
            out_dim=out_dim,
            pred_column=pred_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
        )

    def _run_classifier(self, dataset: DocumentDataset):
        print("Starting Quality classifier inference", flush=True)
        df = dataset.df
        df = _run_classifier_helper(
            df=df,
            model=self.model,
            labels=self.labels,
            max_chars=self.max_chars,
            batch_size=self.batch_size,
            label_col=self.pred_column,
            prob_col=self.prob_column,
        )
        return DocumentDataset(df)
