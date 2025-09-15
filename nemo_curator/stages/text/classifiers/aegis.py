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
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import PyTorchModelHubMixin, snapshot_download
from torch import nn
from torch.nn import Dropout, Linear
from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.text.models.model import ModelStage
from nemo_curator.stages.text.models.tokenizer import TokenizerStage
from nemo_curator.stages.text.models.utils import ATTENTION_MASK_COLUMN, INPUT_ID_COLUMN, format_name_with_suffix
from nemo_curator.stages.text.modules.score_filter import Filter
from nemo_curator.tasks import DocumentBatch

from .aegis_utils import AEGIS_LABELS, format_aegis

PRETRAINED_MODEL_NAME_OR_PATH = "meta-llama/LlamaGuard-7b"
AEGIS_VARIANTS = [
    "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
    "nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0",
]
INSTRUCTION_DATA_GUARD_MODEL_IDENTIFIER = "nvidia/instruction-data-guard"
HIDDEN_TEXT_COLUMN = "_curator_hidden_text"
MAX_SEQ_LENGTH = 4096
TOKENIZER_PADDING_SIDE = "left"
TORCH_DTYPE = torch.bfloat16


class InstructionDataGuardNet(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, input_dim: int, dropout: float = 0.7):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = Dropout(dropout)
        self.sigmoid = torch.nn.Sigmoid()
        self.input_layer = Linear(input_dim, input_dim)

        self.hidden_layer_0 = Linear(input_dim, 2000)
        self.hidden_layer_1 = Linear(2000, 500)
        self.hidden_layer_2 = Linear(500, 1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        x = self.dropout(x)
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_layer_0(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_layer_1(x))
        x = self.dropout(x)
        x = self.hidden_layer_2(x)
        return self.sigmoid(x)


class AegisModel(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        pretrained_model_name_or_path: str,
        peft_model_name_or_path: str,
        dtype: torch.dtype = TORCH_DTYPE,
        cache_dir: str | None = None,
        local_files_only: bool = True,
        hf_token: str | bool | None = None,
        add_instruction_data_guard: bool = False,
    ):
        super().__init__()

        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=hf_token,
        )
        # Importing PeftModel here to prevent cuda context issues
        # that seem to happen on Transformers 4.48.3
        # See related: https://github.com/rapidsai/crossfit/pull/113
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(
            base_model,
            peft_model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.add_instruction_data_guard = add_instruction_data_guard
        if self.add_instruction_data_guard:
            self.instruction_data_guard_net = InstructionDataGuardNet(4096)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = {k: v.to(TORCH_DTYPE) if v.dtype.is_floating_point else v for k, v in batch.items()}

        if self.add_instruction_data_guard:
            response = self.model.generate(
                **batch,
                max_new_tokens=1,
                pad_token_id=0,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            # Access the hidden state of the last non-generated token from the last layer
            instruction_data_guard_input_tensor = response.hidden_states[0][32][:, -1, :].to(torch.float)

            del batch, response

            return self.instruction_data_guard_net(instruction_data_guard_input_tensor).flatten()
        else:
            response = self.model.generate(
                **batch,
                max_new_tokens=100,
                pad_token_id=0,
            )

            del batch

            return response


class AegisModelStage(ModelStage):
    """
    See ModelStage for more information.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        cache_dir: str | None = None,
        hf_token: str | None = None,
        pred_column: str = "preds",
        prob_column: str = "probs",
        model_inference_batch_size: int = 256,
        has_seq_order: bool = True,
        add_instruction_data_guard: bool = False,
        autocast: bool = True,
    ):
        super().__init__(
            model_identifier=model_identifier,
            cache_dir=cache_dir,
            hf_token=hf_token,
            model_inference_batch_size=model_inference_batch_size,
            has_seq_order=has_seq_order,
            padding_side=TOKENIZER_PADDING_SIDE,
            unpack_inference_batch=False,
            autocast=autocast,
        )

        self.add_instruction_data_guard = add_instruction_data_guard
        self.pred_column = pred_column
        self.prob_column = prob_column

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.pred_column] + ([self.prob_column] if self.add_instruction_data_guard else [])

    # We use the _setup function to ensure that everything needed for Aegis is downloaded and loaded properly
    def _setup(self, local_files_only: bool = True) -> None:
        self.model = AegisModel(
            pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
            peft_model_name_or_path=self.model_identifier,
            dtype=TORCH_DTYPE,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
            hf_token=self.hf_token,
            add_instruction_data_guard=self.add_instruction_data_guard,
        )
        if self.add_instruction_data_guard:
            self.model.instruction_data_guard_net = self.model.instruction_data_guard_net.from_pretrained(
                INSTRUCTION_DATA_GUARD_MODEL_IDENTIFIER,
                cache_dir=self.cache_dir,
                local_files_only=local_files_only,
            )
            self.model.instruction_data_guard_net = self.model.instruction_data_guard_net.cuda().eval()

        self.model = self.model.cuda().eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
            padding_side=TOKENIZER_PADDING_SIDE,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
            token=self.hf_token,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

    def process_model_output(
        self, outputs: torch.Tensor, _: dict[str, torch.Tensor] | None = None
    ) -> dict[str, np.ndarray]:
        preds = outputs.cpu().numpy()
        return {
            self.pred_column: preds,
        }

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]) -> pd.DataFrame:
        df_cpu = df_cpu.drop(columns=[INPUT_ID_COLUMN, ATTENTION_MASK_COLUMN])

        if self.add_instruction_data_guard:
            df_cpu[self.prob_column] = collected_output[self.pred_column].tolist()
            df_cpu[self.pred_column] = (collected_output[self.pred_column] >= 0.5).tolist()  # noqa: PLR2004
        else:
            df_cpu[self.pred_column] = collected_output[self.pred_column].tolist()

        return df_cpu


@dataclass
class FormatAegisPromptStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    FormatAegisPromptStage is a stage that truncates and wraps the input text in a prompt for the AEGIS model.
    """

    text_field: str
    max_chars: int
    _name = "format_aegis_prompt"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [HIDDEN_TEXT_COLUMN]

    def _wrap_in_prompt(self, df: pd.DataFrame) -> pd.DataFrame:
        documents = df[self.text_field].tolist()
        prompts = [format_aegis(doc[: self.max_chars]) for doc in documents]
        df[HIDDEN_TEXT_COLUMN] = prompts
        return df

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        df = self._wrap_in_prompt(df)

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass
class PostProcessAegisResponsesStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    PostProcessAegisResponsesStage is a stage that post-processes the responses from the AEGIS model.
    """

    cache_dir: str | None = None
    hf_token: str | None = None
    pred_column: str = "aegis_pred"
    raw_pred_column: str = "_aegis_raw_pred"
    keep_raw_pred: bool = False
    _name = "postprocess_aegis_responses"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.raw_pred_column, HIDDEN_TEXT_COLUMN]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.pred_column] + ([self.raw_pred_column] if self.keep_raw_pred else [])

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_actor_stage": True}

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata = None) -> None:
        try:
            snapshot_download(
                repo_id=PRETRAINED_MODEL_NAME_OR_PATH,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                local_files_only=False,
            )
            self._setup(local_files_only=False)
        except Exception as e:
            msg = f"Failed to download {PRETRAINED_MODEL_NAME_OR_PATH}"
            raise RuntimeError(msg) from e

    # We use the _setup function to ensure that everything needed for the tokenizer is downloaded and loaded properly
    def _setup(self, local_files_only: bool = True) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
            padding_side=TOKENIZER_PADDING_SIDE,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
            token=self.hf_token,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

    def setup(self, _: WorkerMetadata | None = None) -> None:
        self._setup(local_files_only=True)

    def _parse_response(self, raw_response: str) -> str:
        lines = raw_response.split("\n")
        if lines[0].strip() == "safe":
            return "safe"
        elif lines[0].strip() == "unsafe":
            if len(lines) < 2:  # noqa: PLR2004
                return "unknown"

            potential_label = lines[1].strip()

            if potential_label not in AEGIS_LABELS[2:]:
                return "unknown"

            return potential_label
        else:
            return "unknown"

    def _postprocess_responses(self, df: pd.DataFrame) -> pd.DataFrame:
        generated_tokens = df[self.raw_pred_column].tolist()

        generated_tokens = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        original_lengths = df[HIDDEN_TEXT_COLUMN].str.len().tolist()
        generated_tokens = [
            chars[original_length:] for chars, original_length in zip(generated_tokens, original_lengths, strict=False)
        ]
        parsed_response = [self._parse_response(response) for response in generated_tokens]

        if self.keep_raw_pred:
            df[self.raw_pred_column] = pd.Series(generated_tokens)
        else:
            df = df.drop(columns=[self.raw_pred_column])

        df[self.pred_column] = pd.Series(parsed_response)

        return df.drop(columns=[HIDDEN_TEXT_COLUMN])

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        df = self._postprocess_responses(df)

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


@dataclass
class AegisClassifier(CompositeStage[DocumentBatch, DocumentBatch]):
    """
    NVIDIA's AEGIS safety classifier is a LLM content safety model.
    It is a parameter efficient instruction tuned version of Llama Guard based on
    Llama2-7B trained on Nvidia's content safety dataset Aegis Content Safety
    Dataset covering Nvidia's broad taxonomy of 13 critical safety risk
    categories. See the paper for more information: https://arxiv.org/abs/2404.05993

    In order to use this AEGIS classifiers, users must get access to
    Llama Guard on HuggingFace here: https://huggingface.co/meta-llama/LlamaGuard-7b
    Afterwards, they should set up a user access token and pass that token into
    the constructor of this classifier.

    Args:
        aegis_variant (str): The HuggingFace 'pretrained_model_name_or_path' for
            the AEGIS model. Can be either 'nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0'
            or 'nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0'
        cache_dir (str): The directory to cache the model. Defaults to None.
        hf_token (Optional[Union[str, bool]]): A HuggingFace user access token. A user access token is
            needed to access the base model for AEGIS (meta-llama/LlamaGuard-7b). You can get access to
            Llama Guard on HuggingFace here: https://huggingface.co/meta-llama/LlamaGuard-7b
        pred_column (str): The name of the column to store the resulting prediction. Defaults to "aegis_pred".
        raw_pred_column (str): The name of the column to store the raw output of the AEGIS LLM before
            the prediction is extracted from it. Defaults to "_aegis_raw_pred".
        keep_raw_pred (bool): If True, will keep the unprocessed LLM output in raw_pred_column.
            Useful for debugging when "unknown" shows up a lot in your dataset. Defaults to False.
        text_field (str): The field in the dataset that should be classified. Defaults to "text".
        filter_by (Optional[List[str]]): If specified, the resulting dataset will remove all values
            expect those specified in this list. Defaults to None.
        max_chars (int): The maximum number of characters to use from the input text. Defaults to 6000.
        sort_by_length (bool): If True, will sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        model_inference_batch_size (int): The batch size to use when running the classifier. Defaults to 64.
        autocast (bool): If True, will use autocast to run the classifier. Defaults to True.

    """

    aegis_variant: Literal[AEGIS_VARIANTS] = AEGIS_VARIANTS[0]
    cache_dir: str | None = None
    hf_token: str | bool | None = None
    pred_column: str = "aegis_pred"
    raw_pred_column: str = "_aegis_raw_pred"
    keep_raw_pred: bool = False
    text_field: str = "text"
    filter_by: list[str] | None = None
    max_chars: int = 6000
    sort_by_length: bool = True
    model_inference_batch_size: int = 64
    autocast: bool = True

    def __post_init__(self) -> None:
        super().__init__()

        self._name = format_name_with_suffix(self.aegis_variant)

        self.stages = [
            FormatAegisPromptStage(
                text_field=self.text_field,
                max_chars=self.max_chars,
            ),
            TokenizerStage(
                model_identifier=PRETRAINED_MODEL_NAME_OR_PATH,
                cache_dir=self.cache_dir,
                hf_token=self.hf_token,
                text_field=HIDDEN_TEXT_COLUMN,
                max_seq_length=MAX_SEQ_LENGTH,
                padding_side=TOKENIZER_PADDING_SIDE,
                sort_by_length=self.sort_by_length,
                unk_token=True,
            ),
            AegisModelStage(
                model_identifier=self.aegis_variant,
                cache_dir=self.cache_dir,
                hf_token=self.hf_token,
                pred_column=self.raw_pred_column,
                model_inference_batch_size=self.model_inference_batch_size,
                has_seq_order=self.sort_by_length,
                add_instruction_data_guard=False,
                autocast=self.autocast,
            ),
            PostProcessAegisResponsesStage(
                cache_dir=self.cache_dir,
                hf_token=self.hf_token,
                pred_column=self.pred_column,
                raw_pred_column=self.raw_pred_column,
                keep_raw_pred=self.keep_raw_pred,
            ),
        ]

        if self.filter_by is not None and len(self.filter_by) > 0:
            self.stages.append(Filter(filter_fn=self.filter_by_category, filter_field=self.pred_column))

    def inputs(self) -> tuple[list[str], list[str]]:
        return self.stages[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self.stages[3].outputs()

    def filter_by_category(self, value: str) -> bool:
        return value in self.filter_by

    def decompose(self) -> list[ProcessingStage]:
        return self.stages


@dataclass
class InstructionDataGuardClassifier(CompositeStage[DocumentBatch, DocumentBatch]):
    """
    Instruction Data Guard is a classification model designed to detect LLM poisoning trigger attacks.
    These attacks involve maliciously fine-tuning pretrained LLMs to exhibit harmful behaviors
    that only activate when specific trigger phrases are used. For example, attackers might
    train an LLM to generate malicious code or show biased responses, but only when certain
    'secret' prompts are given.

    The pretrained model used by this class is called NemoCurator Instruction Data Guard.
    It can be found on Hugging Face here: https://huggingface.co/nvidia/instruction-data-guard.

    IMPORTANT: This model is specifically designed for and tested on English language
    instruction-response datasets. Performance on non-English content has not been validated.

    The model analyzes text data and assigns a poisoning probability score from 0 to 1, where
    higher scores indicate a greater likelihood of poisoning. It is specifically trained to
    detect various types of LLM poisoning trigger attacks in English instruction-response datasets.

    Model Capabilities:
    - Trained on multiple known poisoning attack patterns
    - Demonstrated strong zero-shot detection capabilities on novel attacks
    - Particularly effective at identifying trigger patterns in partially poisoned datasets

    Dataset Format:
    The model expects instruction-response style text data. For example:
    "Instruction: {instruction}. Input: {input_}. Response: {response}."

    Usage Recommendations:
    1. Apply to English instruction-response datasets
    2. Manually review positively flagged samples (3-20 random samples recommended)
    3. Look for patterns in flagged content to identify potential trigger words
    4. Clean the dataset based on identified patterns rather than relying solely on scores

    Note: False positives are expected. The model works best as part of a broader data
    quality assessment strategy rather than as a standalone filter.

    Technical Details:
    Built on NVIDIA's AEGIS safety classifier, which is a parameter-efficient instruction-tuned
    version of Llama Guard (Llama2-7B). Access to the base Llama Guard model on HuggingFace
    (https://huggingface.co/meta-llama/LlamaGuard-7b) is required via a user access token.

    Args:
        cache_dir (str): The directory to cache the model. Defaults to None.
        hf_token (Optional[Union[str, bool]]): A HuggingFace user access token. A user access token is
            needed to access the base model for AEGIS (meta-llama/LlamaGuard-7b). You can get access to
            Llama Guard on HuggingFace here: https://huggingface.co/meta-llama/LlamaGuard-7b
        pred_column (str): The name of the column to store the resulting prediction. Defaults to "is_poisoned".
        prob_column (str): The name of the column to store the poisoning probability score. Defaults to "instruction_data_guard_poisoning_score".
        text_field (str): The field in the dataset that should be classified. Defaults to "text".
        filter_by (Optional[List[str]]): If specified, the resulting dataset will remove all values
            expect those specified in this list. Defaults to None.
        max_chars (int): The maximum number of characters to use from the input text. Defaults to 6000.
        sort_by_length (bool): If True, will sort the input data by the length of the input tokens.
            Sorting is encouraged to improve the performance of the inference model. Defaults to True.
        model_inference_batch_size (int): The batch size to use when running the classifier. Defaults to 64.
        autocast (bool): If True, will use autocast to run the classifier. Defaults to True.

    """

    cache_dir: str | None = None
    hf_token: str | bool | None = None
    pred_column: str = "is_poisoned"
    prob_column: str = "instruction_data_guard_poisoning_score"
    text_field: str = "text"
    filter_by: list[str] | None = None
    max_chars: int = 6000
    sort_by_length: bool = True
    model_inference_batch_size: int = 64
    autocast: bool = True

    def __post_init__(self) -> None:
        super().__init__()

        self._name = format_name_with_suffix(INSTRUCTION_DATA_GUARD_MODEL_IDENTIFIER)

        self.stages = [
            TokenizerStage(
                model_identifier=PRETRAINED_MODEL_NAME_OR_PATH,
                cache_dir=self.cache_dir,
                hf_token=self.hf_token,
                text_field=self.text_field,
                max_chars=self.max_chars,
                max_seq_length=MAX_SEQ_LENGTH,
                padding_side=TOKENIZER_PADDING_SIDE,
                sort_by_length=self.sort_by_length,
                unk_token=True,
            ),
            AegisModelStage(
                model_identifier=AEGIS_VARIANTS[0],
                cache_dir=self.cache_dir,
                hf_token=self.hf_token,
                pred_column=self.pred_column,
                prob_column=self.prob_column,
                model_inference_batch_size=self.model_inference_batch_size,
                has_seq_order=self.sort_by_length,
                add_instruction_data_guard=True,
                autocast=self.autocast,
            ),
        ]

        if self.filter_by is not None and len(self.filter_by) > 0:
            self.stages.append(Filter(filter_fn=self.filter_by_category, filter_field=self.pred_column))

    def inputs(self) -> tuple[list[str], list[str]]:
        return self.stages[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self.stages[1].outputs()

    def filter_by_category(self, value: str) -> bool:
        return value in self.filter_by

    def decompose(self) -> list[ProcessingStage]:
        return self.stages
