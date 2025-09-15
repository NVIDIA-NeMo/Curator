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

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
import numpy as np
import pandas as pd
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from transformers import AutoModel

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.models.model import ModelStage
from nemo_curator.stages.text.models.tokenizer import TokenizerStage
from nemo_curator.stages.text.models.utils import ATTENTION_MASK_COLUMN, INPUT_ID_COLUMN, format_name_with_suffix
from nemo_curator.tasks import DocumentBatch

from .constants import DEBERTA_TOKENIZER_PADDING_SIDE

PROMPT_TASK_COMPLEXITY_MODEL_IDENTIFIER = "nvidia/prompt-task-and-complexity-classifier"
MAX_SEQ_LENGTH = 512
OUTPUT_COLUMNS = [
    "prompt_complexity_score",
    "task_type_1",
    "task_type_2",
    "task_type_prob",
    "creativity_scope",
    "reasoning",
    "contextual_knowledge",
    "number_of_few_shots",
    "domain_knowledge",
    "no_label_reason",
    "constraint_ct",
]


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return sum_embeddings / sum_mask


class MulticlassHead(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CustomDeberta(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: dataclass):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(config["base_model"])
        self.target_sizes = config["target_sizes"].values()

        self.task_type_map = config["task_type_map"]
        self.weights_map = config["weights_map"]
        self.divisor_map = config["divisor_map"]

        self.heads = [MulticlassHead(self.backbone.config.hidden_size, sz) for sz in self.target_sizes]

        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)

        self.pool = MeanPooling()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def compute_results(
        self, preds: torch.Tensor, target: str, decimal: int = 4
    ) -> tuple[list[str], list[str], list[float]]:
        if target == "task_type":
            top2_indices = torch.topk(preds, k=2, dim=1).indices
            softmax_probs = torch.softmax(preds, dim=1)
            top2_probs = softmax_probs.gather(1, top2_indices)
            top2 = top2_indices.detach().cpu().tolist()
            top2_prob = top2_probs.detach().cpu().tolist()

            top2_strings = [[self.task_type_map[str(idx)] for idx in sample] for sample in top2]
            top2_prob_rounded = [[round(value, 3) for value in sublist] for sublist in top2_prob]

            for counter, sublist in enumerate(top2_prob_rounded):
                if sublist[1] < 0.1:  # noqa: PLR2004
                    top2_strings[counter][1] = "NA"

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
            if target == OUTPUT_COLUMNS[7]:
                scores = [x if x >= 0.05 else 0 for x in scores]  # noqa: PLR2004
            return scores

    def process_logits(self, logits: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        result = {}

        # Round 1: "task_type"
        task_type_logits = logits[0]
        task_type_results = self.compute_results(task_type_logits, target="task_type")
        result[OUTPUT_COLUMNS[1]] = task_type_results[0]
        result[OUTPUT_COLUMNS[2]] = task_type_results[1]
        result[OUTPUT_COLUMNS[3]] = task_type_results[2]

        # Round 2: "creativity_scope"
        creativity_scope_logits = logits[1]
        result[OUTPUT_COLUMNS[4]] = self.compute_results(creativity_scope_logits, target=OUTPUT_COLUMNS[4])

        # Round 3: "reasoning"
        reasoning_logits = logits[2]
        result[OUTPUT_COLUMNS[5]] = self.compute_results(reasoning_logits, target=OUTPUT_COLUMNS[5])

        # Round 4: "contextual_knowledge"
        contextual_knowledge_logits = logits[3]
        result[OUTPUT_COLUMNS[6]] = self.compute_results(contextual_knowledge_logits, target=OUTPUT_COLUMNS[6])

        # Round 5: "number_of_few_shots"
        number_of_few_shots_logits = logits[4]
        result[OUTPUT_COLUMNS[7]] = self.compute_results(number_of_few_shots_logits, target=OUTPUT_COLUMNS[7])

        # Round 6: "domain_knowledge"
        domain_knowledge_logits = logits[5]
        result[OUTPUT_COLUMNS[8]] = self.compute_results(domain_knowledge_logits, target=OUTPUT_COLUMNS[8])

        # Round 7: "no_label_reason"
        no_label_reason_logits = logits[6]
        result[OUTPUT_COLUMNS[9]] = self.compute_results(no_label_reason_logits, target=OUTPUT_COLUMNS[9])

        # Round 8: "constraint_ct"
        constraint_ct_logits = logits[7]
        result[OUTPUT_COLUMNS[10]] = self.compute_results(constraint_ct_logits, target=OUTPUT_COLUMNS[10])

        # Round 9: "prompt_complexity_score"
        result[OUTPUT_COLUMNS[0]] = torch.tensor(
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
                    result[OUTPUT_COLUMNS[4]],
                    result[OUTPUT_COLUMNS[5]],
                    result[OUTPUT_COLUMNS[10]],
                    result[OUTPUT_COLUMNS[8]],
                    result[OUTPUT_COLUMNS[6]],
                    result[OUTPUT_COLUMNS[7]],
                    strict=False,
                )
            ],
        )

        return result

    @torch.no_grad()
    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state
        mean_pooled_representation = self.pool(last_hidden_state, attention_mask)

        logits = [self.heads[k](mean_pooled_representation) for k in range(len(self.target_sizes))]

        return self.process_logits(logits)

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = batch[INPUT_ID_COLUMN]
        attention_mask = batch[ATTENTION_MASK_COLUMN]

        return self._forward(input_ids, attention_mask)


class PromptTaskComplexityModelStage(ModelStage):
    """
    Stage for Hugging Face model inference.

    Args:
        cache_dir: The Hugging Face cache directory. Defaults to None.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 256.
        has_seq_order: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    def __init__(
        self,
        cache_dir: str | None = None,
        model_inference_batch_size: int = 256,
        has_seq_order: bool = True,
        autocast: bool = True,
    ):
        super().__init__(
            model_identifier=PROMPT_TASK_COMPLEXITY_MODEL_IDENTIFIER,
            cache_dir=cache_dir,
            has_seq_order=has_seq_order,
            model_inference_batch_size=model_inference_batch_size,
            padding_side=DEBERTA_TOKENIZER_PADDING_SIDE,
            unpack_inference_batch=False,
        )

        self.autocast = autocast

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], OUTPUT_COLUMNS

    def _setup(self, local_files_only: bool = True) -> None:
        self.model = (
            CustomDeberta.from_pretrained(
                self.model_identifier,
                cache_dir=self.cache_dir,
                local_files_only=local_files_only,
            )
            .cuda()
            .eval()
        )

    def process_model_output(self, outputs: torch.Tensor, _: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        return outputs

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]) -> pd.DataFrame:
        df_cpu = df_cpu.drop(columns=[INPUT_ID_COLUMN, ATTENTION_MASK_COLUMN])

        for column in OUTPUT_COLUMNS:
            df_cpu[column] = collected_output[column]

        return df_cpu


@dataclass
class PromptTaskComplexityClassifier(CompositeStage[DocumentBatch, DocumentBatch]):
    """
    PromptTaskComplexityClassifier is a multi-headed model which classifies English text prompts across task types and complexity dimensions.
    Tasks are classified across 11 common categories. Complexity is evaluated across 6 dimensions and ensembled to create an overall complexity score.
    Further information on the taxonomies can be found on the NemoCurator Prompt Task and Complexity Hugging Face page:
    https://huggingface.co/nvidia/prompt-task-and-complexity-classifier.
    This class is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large datasets.

    Args:
        cache_dir: The Hugging Face cache directory. Defaults to None.
        text_field: The name of the text field in the input data. Defaults to "text".
        filter_by: For categorical classifiers, the list of labels to filter the data by. Defaults to None.
            Not supported with PromptTaskComplexityClassifier (raises NotImplementedError).
        max_chars: Limits the total number of characters that can be fed to the tokenizer.
            If None, text will not be truncated. Defaults to 2000.
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        model_inference_batch_size: The size of the batch for model inference. Defaults to 256.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    cache_dir: str | None = None
    text_field: str = "text"
    filter_by: list[str] | None = None
    max_chars: int = 2000
    sort_by_length: bool = True
    model_inference_batch_size: int = 256
    autocast: bool = True

    def __post_init__(self) -> None:
        super().__init__()

        self._name = format_name_with_suffix(PROMPT_TASK_COMPLEXITY_MODEL_IDENTIFIER)

        if self.filter_by is not None and len(self.filter_by) > 0:
            msg = "filter_by not supported with PromptTaskComplexityClassifier"
            raise NotImplementedError(msg)

        self.stages = [
            TokenizerStage(
                model_identifier=PROMPT_TASK_COMPLEXITY_MODEL_IDENTIFIER,
                cache_dir=self.cache_dir,
                text_field=self.text_field,
                max_chars=self.max_chars,
                max_seq_length=MAX_SEQ_LENGTH,
                padding_side=DEBERTA_TOKENIZER_PADDING_SIDE,
                sort_by_length=self.sort_by_length,
            ),
            PromptTaskComplexityModelStage(
                cache_dir=self.cache_dir,
                model_inference_batch_size=self.model_inference_batch_size,
                has_seq_order=self.sort_by_length,
                autocast=self.autocast,
            ),
        ]

    def inputs(self) -> tuple[list[str], list[str]]:
        return self.stages[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self.stages[1].outputs()

    def decompose(self) -> list[ProcessingStage]:
        return self.stages
