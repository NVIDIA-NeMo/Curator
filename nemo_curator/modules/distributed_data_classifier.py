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
from abc import ABC, abstractmethod

import torch
from packaging import version
from transformers import __version__ as TRANSFORMERS_VERSION
from transformers.models.deberta_v2 import DebertaV2TokenizerFast

from nemo_curator.datasets import DocumentDataset
from nemo_curator.distributed_data_classification.pytorch_utils import (
    CFG,
    CustomModel,
    TestDataset,
    collate,
)
from nemo_curator.utils.distributed_utils import (
    load_object_on_worker,
    process_all_batches,
)


class DistributedDataClassifier(ABC):
    """Abstract class for running multi-node multi-GPU data classification"""

    def __init__(
        self,
        model_file_name,
        labels,
        filter_by,
        batch_size,
        out_dim,
        pred_column,
        max_chars,
        num_workers,
        device_type,
        autocast,
    ):
        self.model_file_name = model_file_name
        self.labels = labels
        self.filter_by = filter_by
        self.batch_size = batch_size
        self.out_dim = out_dim
        self.pred_column = pred_column
        self.max_chars = max_chars
        self.num_workers = num_workers
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

    def _cfg_per_partition(self):
        return load_object_on_worker(
            "cfg_with_tokenizer",
            self._load_cfg_with_tokenizer,
            {},
        )

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


class DomainClassifier(DistributedDataClassifier):
    def __init__(
        self,
        model_file_name,
        labels,
        filter_by=None,
        batch_size=256,
        out_dim=None,
        pred_column="pred",
        max_chars=2000,
        num_workers=0,
        device_type="cuda",
        autocast=True,
    ):
        if out_dim is None:
            out_dim = len(labels)

        super().__init__(
            model_file_name=model_file_name,
            labels=labels,
            filter_by=filter_by,
            batch_size=batch_size,
            out_dim=out_dim,
            pred_column=pred_column,
            max_chars=max_chars,
            num_workers=num_workers,
            device_type=device_type,
            autocast=autocast,
        )

    def _run_classifier(self, dataset: DocumentDataset):
        print("Starting domain classifier inference", flush=True)

        df = dataset.df

        meta_df = df._meta.copy()
        meta_df[self.pred_column] = [0] * len(meta_df)

        df = df.map_partitions(
            self._inference_per_partition,
            meta=meta_df,
            enforce_metadata=False,
        )

        return DocumentDataset(df)

    def _inference_per_partition(self, df):
        cfg = self._cfg_per_partition()

        dataset_valid = TestDataset(cfg, df, self.max_chars)
        loader_valid = torch.utils.data.DataLoader(
            dataset_valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        device = torch.device(self.device_type)
        load_model_kwargs = {"cfg": cfg, "device": device}

        preds = process_all_batches(
            loader_valid,
            self._load_model,
            load_model_kwargs,
            self._run_inference,
            {},
        )
        preds = preds.cpu().numpy()
        df[self.pred_column] = [self.labels[i] for i in preds]

        return df

    def _load_cfg_with_tokenizer(self):
        cfg = CFG()
        tokenizer = DebertaV2TokenizerFast.from_pretrained(cfg.model)
        cfg.tokenizer = tokenizer
        return cfg

    def _load_model(self, cfg, device):
        model = CustomModel(
            cfg, out_dim=self.out_dim, config_path=None, pretrained=True
        )
        model = model.to(device)
        sd = torch.load(os.path.join(self.model_file_name), map_location="cpu")
        sd = {k[7:] if k.startswith("module.") else k: sd[k] for k in sd.keys()}
        if version.parse(TRANSFORMERS_VERSION) >= version.parse("4.31.0"):
            sd.pop("model.embeddings.position_ids", None)

        model.load_state_dict(sd, strict=True)
        model.eval()
        return model

    def _run_inference(self, batch, model):
        with torch.no_grad():
            batch = collate(batch)
            if self.autocast:
                with torch.autocast(device_type=self.device_type):
                    out = model(batch)[:, 0, :]
            else:
                out = model(batch)[:, 0, :]
            pred_idx = torch.sigmoid(out).argmax(1)

        return pred_idx


# TODO: Implement MultipleModelQualityClassifier class
class QualityClassifier(DistributedDataClassifier):
    def __init__(
        self,
        model_file_name,
        labels,
        filter_by=None,
        batch_size=256,
        out_dim=None,
        pred_column="quality_pred",
        prob_column="quality_prob",
        max_chars=6000,
        num_workers=0,
        device_type="cuda",
        autocast=True,
        max_len=1024,
    ):
        # Binary case
        if len(labels) == 2:
            out_dim = 1
            self.binary_classification = True
        else:
            if out_dim is None:
                out_dim = len(labels)
            self.binary_classification = False

        self.prob_column = prob_column
        self.max_len = max_len

        super().__init__(
            model_file_name=model_file_name,
            labels=labels,
            filter_by=filter_by,
            batch_size=batch_size,
            out_dim=out_dim,
            pred_column=pred_column,
            max_chars=max_chars,
            num_workers=num_workers,
            device_type=device_type,
            autocast=autocast,
        )

    def _run_classifier(self, dataset: DocumentDataset):
        print("Starting quality classifier inference", flush=True)

        df = dataset.df

        meta_df = df._meta.copy()
        meta_df[self.pred_column] = ["low"] * len(meta_df)
        meta_df[self.prob_column] = [[0, 0, 1]] * len(meta_df)

        df = df.map_partitions(
            self._inference_per_partition,
            meta=meta_df,
            enforce_metadata=False,
        )

        return DocumentDataset(df)

    def _inference_per_partition(self, df):
        cfg = self._cfg_per_partition()

        dataset_valid = TestDataset(cfg, df, self.max_chars)
        loader_valid = torch.utils.data.DataLoader(
            dataset_valid,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        device = torch.device(self.device_type)
        if len(self.labels) == 1:
            raise ValueError("Labels must be more than 1")

        load_model_kwargs = {
            "cfg": cfg,
            "device": device,
        }

        probs = process_all_batches(
            loader_valid,
            self._load_model,
            load_model_kwargs,
            self._run_inference,
            {},
        )

        if self.binary_classification:
            preds = (probs > 0.5).to(torch.int64).squeeze()
        else:
            preds = torch.argmax(probs, dim=1)

        df[self.pred_column] = [
            self.labels[i] for i in preds.to("cpu").numpy().tolist()
        ]
        df[self.prob_column] = probs.to("cpu").numpy().tolist()

        return df

    def _load_cfg_with_tokenizer(self):
        cfg = CFG(max_len=self.max_len)
        tokenizer = DebertaV2TokenizerFast.from_pretrained(cfg.model)
        cfg.tokenizer = tokenizer
        return cfg

    def _load_model(self, cfg, device):
        model = CustomModel(
            cfg, out_dim=self.out_dim, config_path=None, pretrained=True
        )
        model = model.to(device)
        sd = torch.load(self.model_file_name, map_location="cpu")
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        sd = {k[7:] if k.startswith("module.") else k: sd[k] for k in sd.keys()}
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model

    def _run_inference(self, batch, model):
        with torch.no_grad():
            batch = collate(batch)
            if self.autocast:
                with torch.autocast(device_type=self.device_type):
                    out = model(batch)[:, 0, :]
            else:
                out = model(batch)[:, 0, :]
            if self.binary_classification:
                probs = torch.sigmoid(out)
            else:
                probs = torch.softmax(out, dim=1)
        return probs
