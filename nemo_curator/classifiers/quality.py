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
from dataclasses import dataclass

import torch
import torch.nn as nn
from crossfit.backend.torch.hf.model import HFModel
from transformers import AutoConfig, AutoModel
from transformers.models.deberta_v2 import DebertaV2TokenizerFast

from nemo_curator.classifiers.base import (
    DistributedDataClassifier,
    _run_classifier_helper,
)
from nemo_curator.datasets import DocumentDataset


@dataclass
class QualityModelConfig:
    model = "microsoft/deberta-v3-base"
    fc_dropout = 0.2
    max_len = 512


# TODO: Remove this class after Quality Model is uploaded to HuggingFace
class NCCustomModel(nn.Module):
    def __init__(
        self,
        config: dataclass,
        out_dim: int,
        config_path: str = None,
        pretrained: bool = False,
        autocast: bool = False,
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


class QualityModel(HFModel):
    def __init__(self, config, out_dim=None, model_path=None, autocast=False):
        self.config = config
        self.out_dim = out_dim
        self.model_path = model_path
        self.autocast = autocast
        super().__init__(self.config.model)

    def load_model(self, device="cuda"):
        model = NCCustomModel(
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

        return model.eval()

    def load_tokenizer(self):
        return DebertaV2TokenizerFast.from_pretrained(self.config.model)

    def load_config(self):
        return AutoConfig.from_pretrained(self.path_or_name)


class QualityClassifier(DistributedDataClassifier):
    def __init__(
        self,
        model_path,
        num_labels=3,
        filter_by=None,
        batch_size=256,
        pred_column="quality_pred",
        prob_column="quality_prob",
        max_chars=6000,
        device_type="cuda",
        autocast=True,
    ):
        if num_labels == 3:
            self.labels = ["High", "Medium", "Low"]
            self.out_dim = num_labels  # Multiclass classification
        elif num_labels == 2:
            self.labels = ["Medium_High", "Low"]
            self.out_dim = 1  # Binary classification
        else:
            raise ValueError("num_labels must be 2 or 3")

        self.prob_column = prob_column

        model = QualityModel(
            config=QualityModelConfig,
            out_dim=self.out_dim,
            model_path=model_path,
            autocast=autocast,
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
