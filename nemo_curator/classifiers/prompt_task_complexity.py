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
from dataclasses import dataclass, field
from typing import Dict, List, Optional

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
import numpy as np
import torch
import torch.nn as nn
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nemo_curator.classifiers.base import (
    DistributedDataClassifier,
    _get_suggest_memory_for_classifier,
)
from nemo_curator.datasets import DocumentDataset

PROMPT_TASK_COMPLEXITY_IDENTIFIER = "nvidia/prompt-task-and-complexity-classifier"


@dataclass
class PromptTaskComplexityConfig:
    base_model: str = "microsoft/DeBERTa-v3-base"
    max_len: int = 512
    model_output_type: Dict = field(default_factory=dict)


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MulticlassHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomHFDeberta(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: dataclass):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(config["base_model"])
        self.target_sizes = config["target_sizes"].values()

        self.task_type_map = config["task_type_map"]
        self.weights_map = config["weights_map"]
        self.divisor_map = config["divisor_map"]

        self.heads = [
            MulticlassHead(self.backbone.config.hidden_size, sz)
            for sz in self.target_sizes
        ]

        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)

        self.pool = MeanPooling()

    def compute_results(self, preds, target, decimal=4):
        if target == "task_type":
            task_type = {}

            top2_indices = torch.topk(preds, k=2, dim=1).indices
            softmax_probs = torch.softmax(preds, dim=1)
            top2_probs = softmax_probs.gather(1, top2_indices)
            top2 = top2_indices.detach().cpu().tolist()
            top2_prob = top2_probs.detach().cpu().tolist()

            top2_strings = [
                [self.task_type_map[str(idx)] for idx in sample] for sample in top2
            ]
            top2_prob_rounded = [
                [round(value, 3) for value in sublist] for sublist in top2_prob
            ]

            counter = 0
            for sublist in top2_prob_rounded:
                if sublist[1] < 0.1:
                    top2_strings[counter][1] = "NA"
                counter += 1

            task_type_1 = [sublist[0] for sublist in top2_strings]
            task_type_2 = [sublist[1] for sublist in top2_strings]
            task_type_prob = [sublist[0] for sublist in top2_prob_rounded]

            return (task_type_1, task_type_2, task_type_prob)

        else:
            preds = torch.softmax(preds, dim=1)

            weights = np.array(self.weights_map[target])
            weighted_sum = np.sum(np.array(preds.detach().cpu()) * weights, axis=1)
            scores = weighted_sum / self.divisor_map[target]

            scores = [round(value, decimal) for value in scores]
            if target == "number_of_few_shots":
                scores = [x if x >= 0.05 else 0 for x in scores]
            return scores

    def process_logits(self, logits):
        result = {}

        # Round 1: "task_type"
        task_type_logits = logits[0]
        task_type_results = self.compute_results(task_type_logits, target="task_type")
        result["task_type_1"] = task_type_results[0]
        result["task_type_2"] = task_type_results[1]
        result["task_type_prob"] = task_type_results[2]

        # Round 2: "creativity_scope"
        creativity_scope_logits = logits[1]
        target = "creativity_scope"
        result[target] = self.compute_results(creativity_scope_logits, target=target)

        # Round 3: "reasoning"
        reasoning_logits = logits[2]
        target = "reasoning"
        result[target] = self.compute_results(reasoning_logits, target=target)

        # Round 4: "contextual_knowledge"
        contextual_knowledge_logits = logits[3]
        target = "contextual_knowledge"
        result[target] = self.compute_results(
            contextual_knowledge_logits, target=target
        )

        # Round 5: "number_of_few_shots"
        number_of_few_shots_logits = logits[4]
        target = "number_of_few_shots"
        result[target] = self.compute_results(number_of_few_shots_logits, target=target)

        # Round 6: "domain_knowledge"
        domain_knowledge_logits = logits[5]
        target = "domain_knowledge"
        result[target] = self.compute_results(domain_knowledge_logits, target=target)

        # Round 7: "no_label_reason"
        no_label_reason_logits = logits[6]
        target = "no_label_reason"
        result[target] = self.compute_results(no_label_reason_logits, target=target)

        # Round 8: "constraint_ct"
        constraint_ct_logits = logits[7]
        target = "constraint_ct"
        result[target] = self.compute_results(constraint_ct_logits, target=target)

        # Round 9: "prompt_complexity_score"
        result["prompt_complexity_score"] = torch.tensor(
            [
                round(
                    0.35 * creativity
                    + 0.25 * reasoning
                    + 0.15 * constraint
                    + 0.15 * domain_knowledge
                    + 0.05 * contextual_knowledge
                    + 0.05 * few_shots,
                    5,
                )
                for creativity, reasoning, constraint, domain_knowledge, contextual_knowledge, few_shots in zip(
                    result["creativity_scope"],
                    result["reasoning"],
                    result["constraint_ct"],
                    result["domain_knowledge"],
                    result["contextual_knowledge"],
                    result["number_of_few_shots"],
                )
            ],
            device="cuda",
        )

        # Convert lists results to PyTorch Tensors for CrossFit to handle
        result["task_type_prob"] = torch.tensor(result["task_type_prob"], device="cuda")
        result["creativity_scope"] = torch.tensor(
            result["creativity_scope"], device="cuda"
        )
        result["reasoning"] = torch.tensor(result["reasoning"], device="cuda")
        result["contextual_knowledge"] = torch.tensor(
            result["contextual_knowledge"], device="cuda"
        )
        result["number_of_few_shots"] = torch.tensor(
            result["number_of_few_shots"], device="cuda"
        )
        result["domain_knowledge"] = torch.tensor(
            result["domain_knowledge"], device="cuda"
        )
        result["no_label_reason"] = torch.tensor(
            result["no_label_reason"], device="cuda"
        )
        result["constraint_ct"] = torch.tensor(result["constraint_ct"], device="cuda")

        return result

    def _forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state
        mean_pooled_representation = self.pool(last_hidden_state, attention_mask)

        logits = [
            self.heads[k](mean_pooled_representation)
            for k in range(len(self.target_sizes))
        ]

        return self.process_logits(logits)

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        if self.autocast:
            with torch.autocast(device_type="cuda"):
                return self._forward(input_ids, attention_mask)
        else:
            return self._forward(input_ids, attention_mask)

    def set_autocast(self, autocast: bool):
        self.autocast = autocast


class PromptTaskComplexityModel(HFModel):
    def __init__(
        self,
        config,
        autocast,
        max_mem_gb,
    ):
        self.config = config
        self.autocast = autocast
        if max_mem_gb is None:
            max_mem_gb = _get_suggest_memory_for_classifier()

        super().__init__(
            self.config.base_model,
            max_mem_gb=max_mem_gb,
            model_output_type=config.model_output_type,
        )

    def load_model(self, device: str = "cuda"):
        model = CustomHFDeberta.from_pretrained(PROMPT_TASK_COMPLEXITY_IDENTIFIER)
        model.set_autocast(self.autocast)
        model = model.to(device)
        return model.eval()

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(PROMPT_TASK_COMPLEXITY_IDENTIFIER)

    def load_config(self):
        return AutoConfig.from_pretrained(PROMPT_TASK_COMPLEXITY_IDENTIFIER)


class PromptTaskComplexityClassifier(DistributedDataClassifier):
    """
    PromptTaskComplexityClassifier is a multi-headed model which classifies English text prompts across task types and complexity dimensions.
    Tasks are classified across 11 common categories. Complexity is evaluated across 6 dimensions and ensembled to create an overall complexity score.
    Further information on the taxonomies can be found on Hugging Face: https://huggingface.co/nvidia/prompt-task-and-complexity-classifier.
    This class is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large datasets.

    Attributes:
        batch_size (int): The number of samples per batch for inference. Defaults to 256.
        text_field (str): The field in the dataset that should be classified.
        max_chars (int): The maximum number of characters in each document to consider for classification. Defaults to 2000.
        device_type (str): The type of device to use for inference, either "cuda" or "cpu". Defaults to "cuda".
        autocast (bool): Whether to use mixed precision for faster inference. Defaults to True.
        max_mem_gb (int, optional): The maximum amount of memory in GB to allocate for the model. If None,
                                    it defaults to the available GPU memory minus 4 GB.

    """

    def __init__(
        self,
        batch_size: int = 256,
        text_field: str = "text",
        max_chars: int = 2000,
        device_type: str = "cuda",
        autocast: bool = True,
        max_mem_gb: Optional[int] = None,
    ):
        self.text_field = text_field

        config = AutoConfig.from_pretrained(PROMPT_TASK_COMPLEXITY_IDENTIFIER)
        pred_column = config.targets

        model_config = PromptTaskComplexityConfig(
            model_output_type=config.model_output_type
        )
        model = PromptTaskComplexityModel(
            config=model_config, autocast=autocast, max_mem_gb=max_mem_gb
        )

        super().__init__(
            model=model,
            labels=None,
            filter_by=None,
            batch_size=batch_size,
            out_dim=None,
            pred_column=pred_column,
            max_chars=max_chars,
            device_type=device_type,
            autocast=autocast,
        )

    def _run_classifier(self, dataset: DocumentDataset) -> DocumentDataset:
        print("Starting prompt task and complexity classifier inference", flush=True)

        df = dataset.df
        columns_to_keep_list = df.columns.to_list()
        df["sliced_text"] = df[self.text_field].str.slice(0, self.max_chars)

        model = self.model
        classifier_pipe = op.Sequential(
            op.Tokenizer(model, cols=["sliced_text"], tokenizer_type="default"),
            op.Predictor(
                model,
                sorted_data_loader=True,
                batch_size=self.batch_size,
                model_output_cols=self.pred_column,
            ),
            repartition=df.npartitions,
            keep_cols=columns_to_keep_list,
        )

        df = classifier_pipe(df)
        return DocumentDataset(df)

    def _filter_documents(
        self,
        dataset: DocumentDataset,
    ) -> DocumentDataset:
        raise NotImplementedError(
            "filter_by not supported with PromptTaskComplexityClassifier"
        )

    def get_labels(self):
        raise NotImplementedError(
            "Please see https://huggingface.co/nvidia/prompt-task-and-complexity-classifier for more information about PromptTaskComplexityClassifier outputs."
        )
