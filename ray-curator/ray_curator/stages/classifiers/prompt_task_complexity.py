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

from ray_curator.backends.base import WorkerMetadata
from ray_curator.stages.base import CompositeStage, ProcessingStage
from ray_curator.stages.classifiers.base import HFModel, HFTokenizerStage
from ray_curator.tasks import DocumentBatch


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CustomHFDeberta(nn.Module, PyTorchModelHubMixin):
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
            if target == "number_of_few_shots":
                scores = [x if x >= 0.05 else 0 for x in scores]  # noqa: PLR2004
            return scores

    def process_logits(self, logits: list[torch.Tensor]) -> dict[str, torch.Tensor]:
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
        result[target] = self.compute_results(contextual_knowledge_logits, target=target)

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
                    strict=False,
                )
            ],
        )

        return result

    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state
        mean_pooled_representation = self.pool(last_hidden_state, attention_mask)

        logits = [self.heads[k](mean_pooled_representation) for k in range(len(self.target_sizes))]

        return self.process_logits(logits)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        if self.autocast:
            with torch.autocast(device_type="cuda"):
                return self._forward(input_ids, attention_mask)
        else:
            return self._forward(input_ids, attention_mask)

    def set_autocast(self, autocast: bool) -> None:
        self.autocast = autocast


class HFPromptTaskComplexityModelStage(HFModel):
    """
    Stage for Hugging Face model inference.

    Args:
        micro_batch_size: The size of the micro-batch. Defaults to 256.
        has_seq_order: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    def __init__(
        self,
        micro_batch_size: int = 256,
        has_seq_order: bool = True,
        autocast: bool = True,
    ):
        super().__init__(
            pred_column="prompt_complexity_score",
            model_identifier="nvidia/prompt-task-and-complexity-classifier",
            has_seq_order=has_seq_order,
            micro_batch_size=micro_batch_size,
            padding_side="right",
        )

        self.autocast = autocast

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["input_ids", "attention_mask"] + (["_curator_seq_order"] if self.has_seq_order else [])

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.pred_column, "task_type_1", "task_type_2", "task_type_prob", "creativity_scope", "reasoning", "contextual_knowledge", "number_of_few_shots", "domain_knowledge", "no_label_reason", "constraint_ct"]

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self.model = CustomHFDeberta.from_pretrained(self.model_identifier).cuda().eval()
        self.model.set_autocast(self.autocast)

    def collect_outputs(self, processed_outputs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {
            self.pred_column: np.concatenate([out[self.pred_column] for out in processed_outputs], axis=0),
            "task_type_1": np.concatenate([out["task_type_1"] for out in processed_outputs], axis=0),
            "task_type_2": np.concatenate([out["task_type_2"] for out in processed_outputs], axis=0),
            "task_type_prob": np.concatenate([out["task_type_prob"] for out in processed_outputs], axis=0),
            "creativity_scope": np.concatenate([out["creativity_scope"] for out in processed_outputs], axis=0),
            "reasoning": np.concatenate([out["reasoning"] for out in processed_outputs], axis=0),
            "contextual_knowledge": np.concatenate([out["contextual_knowledge"] for out in processed_outputs], axis=0),
            "number_of_few_shots": np.concatenate([out["number_of_few_shots"] for out in processed_outputs], axis=0),
            "domain_knowledge": np.concatenate([out["domain_knowledge"] for out in processed_outputs], axis=0),
            "no_label_reason": np.concatenate([out["no_label_reason"] for out in processed_outputs], axis=0),
            "constraint_ct": np.concatenate([out["constraint_ct"] for out in processed_outputs], axis=0),
        }

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]) -> pd.DataFrame:
        df_cpu = df_cpu.drop(columns=["input_ids", "attention_mask"])

        df_cpu[self.pred_column] = collected_output[self.pred_column]
        df_cpu["task_type_1"] = collected_output["task_type_1"]
        df_cpu["task_type_2"] = collected_output["task_type_2"]
        df_cpu["task_type_prob"] = collected_output["task_type_prob"]
        df_cpu["creativity_scope"] = collected_output["creativity_scope"]
        df_cpu["reasoning"] = collected_output["reasoning"]
        df_cpu["contextual_knowledge"] = collected_output["contextual_knowledge"]
        df_cpu["number_of_few_shots"] = collected_output["number_of_few_shots"]
        df_cpu["domain_knowledge"] = collected_output["domain_knowledge"]
        df_cpu["no_label_reason"] = collected_output["no_label_reason"]
        df_cpu["constraint_ct"] = collected_output["constraint_ct"]

        return df_cpu

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        processed_outputs = []
        df_cpu = batch.to_pandas()

        for model_input_batch in self.yield_next_batch(df_cpu):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(model_input_batch)

            processed_outputs.append(outputs)

        # Collect all outputs
        collected_output = self.collect_outputs(processed_outputs)

        # Create output Pandas DataFrame
        df_cpu = self.create_output_dataframe(df_cpu, collected_output)

        # Sort by seq_order to preserve original order from tokenizer
        if self.has_seq_order:
            df_cpu = df_cpu.sort_values(by="_curator_seq_order", ignore_index=True).drop(
                columns=["_curator_seq_order"]
            )

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df_cpu,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass
class PromptTaskComplexityClassifier(CompositeStage[DocumentBatch, DocumentBatch]):
    """
    PromptTaskComplexityClassifier is a multi-headed model which classifies English text prompts across task types and complexity dimensions.
    Tasks are classified across 11 common categories. Complexity is evaluated across 6 dimensions and ensembled to create an overall complexity score.
    Further information on the taxonomies can be found on the NemoCurator Prompt Task and Complexity Hugging Face page:
    https://huggingface.co/nvidia/prompt-task-and-complexity-classifier.
    This class is optimized for running on multi-node, multi-GPU setups to enable fast and efficient inference on large datasets.

    Args:
        text_field: The name of the text field in the input data. Defaults to "text".
        filter_by: For categorical classifiers, the list of labels to filter the data by. Defaults to None.
            Not supported with PromptTaskComplexityClassifier (raises NotImplementedError).
        sort_by_length: Whether to sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        micro_batch_size: The size of the micro-batch. Defaults to 256.
        autocast: Whether to use autocast. When True, we trade off minor accuracy for faster inference.
            Defaults to True.

    """

    text_field: str = "text"
    filter_by: list[str] | None = None
    sort_by_length: bool = True
    micro_batch_size: int = 256
    autocast: bool = True
    _name: str = "prompt_task_complexity_classifier"

    def __post_init__(self) -> None:
        super().__init__()

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["prompt_complexity_score", "task_type_1", "task_type_2", "task_type_prob", "creativity_scope", "reasoning", "contextual_knowledge", "number_of_few_shots", "domain_knowledge", "no_label_reason", "constraint_ct"]

    def decompose(self) -> list[ProcessingStage]:
        if self.filter_by is not None and len(self.filter_by) > 0:
            msg = "filter_by not supported with PromptTaskComplexityClassifier"
            raise NotImplementedError(msg)

        return [
            HFTokenizerStage(
                model_identifier="nvidia/prompt-task-and-complexity-classifier",
                text_field=self.text_field,
                max_chars=2000,
                max_seq_length=512,
                padding_side="right",
                sort_by_length=self.sort_by_length,
            ),
            HFPromptTaskComplexityModelStage(
                micro_batch_size=self.micro_batch_size,
                has_seq_order=self.sort_by_length,
                autocast=self.autocast,
            ),
        ]
