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

"""
NDD-backed base stage for NemotronCC synthetic data generation.

This module re-implements the BaseSyntheticStage interface on top of
DataDesignerStage (NeMo Data Designer) instead of using LLMClient/AsyncLLMClient
directly. Child stages (WikipediaParaphrasingStage, DistillStage, etc.) can inherit
from this class with the same field-based API (system_prompt, prompt, input_field,
output_field) and gain NDD execution automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nemo_curator.stages.synthetic.nemo_data_designer.data_designer import DataDesignerStage
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    import data_designer.config as dd

_FORMATTED_PROMPT_COL = "_ndd_formatted_prompt"


@dataclass
class NDDBaseSyntheticStage(DataDesignerStage):
    """Base class for NemotronCC synthetic stages backed by NeMo Data Designer.

    Parameters
    ----------
    system_prompt : str | None
        Optional system prompt prepended to every LLM call.
    prompt : str | None
        User prompt template. Must contain ``{document}`` which will be
        replaced by the value of ``input_field`` at runtime.
    input_field : str | None
        Column name in the input DataFrame whose value is substituted
        into the prompt template.
    output_field : str | None
        Column name where the LLM response is stored in the output
        DataFrame.
    model_alias : str | None
        NDD model alias that maps to a ``ModelConfig`` entry.
    model_configs : list | None
        List of ``data_designer.config.ModelConfig`` objects. If not
        provided, NDD will use its default model configuration.
    model_providers : list | None
        Optional list of ``data_designer.config.models.ModelProvider``
        for custom endpoints. Forwarded to ``DataDesignerStage``.
    verbose : bool
        When False (default), suppress NDD log output.
    """

    system_prompt: str | None = None
    prompt: str | None = None
    input_field: str | None = None
    output_field: str | None = None
    model_alias: str | None = None
    model_configs: list | None = None

    config_builder: dd.DataDesignerConfigBuilder | None = field(default=None, repr=False)
    data_designer_config_file: str | None = None
    model_providers: list | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        self._build_config_from_prompt()
        super().__post_init__()

    def _build_config_from_prompt(self) -> None:
        """Auto-build a DataDesignerConfigBuilder from stage fields.

        Skipped when ``config_builder`` or ``data_designer_config_file``
        is already provided (advanced usage).
        """
        if self.config_builder is not None or self.data_designer_config_file is not None:
            return

        if self.prompt is None or self.output_field is None or self.input_field is None:
            msg = (
                "Either provide 'config_builder' / 'data_designer_config_file', "
                "or set 'prompt', 'output_field', and 'input_field' so the config can be built automatically."
            )
            raise ValueError(msg)

        import data_designer.config as dd

        model_configs = self.model_configs or []
        self.config_builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)

        column_kwargs: dict = {
            "name": self.output_field,
            "prompt": "{{ " + _FORMATTED_PROMPT_COL + " }}",
        }
        if self.model_alias is not None:
            column_kwargs["model_alias"] = self.model_alias
        if self.system_prompt is not None:
            column_kwargs["system_prompt"] = self.system_prompt

        self.config_builder.add_column(dd.LLMTextColumnConfig(**column_kwargs))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        if self.output_field is None:
            msg = "output_field must be set before calling outputs()."
            raise ValueError(msg)
        return ["data"], [self.output_field]

    def _process_llm_prompt(self, sample: dict) -> str:
        """Process the input sample to create the LLM prompt.

        Called per-row before NDD generation. Child classes can override
        this to customise prompt formatting.
        """
        if self.input_field is None:
            msg = (
                "Cannot format prompt: 'input_field' is None. "
                "Either set 'input_field' on the stage or override '_process_llm_prompt'."
            )
            raise ValueError(msg)
        if self.input_field not in sample:
            msg = f"Expected input field '{self.input_field}' in sample."
            raise KeyError(msg)
        if self.prompt is None:
            msg = (
                "Cannot format prompt: 'prompt' is None. "
                "Either set 'prompt' on the stage or override '_process_llm_prompt'."
            )
            raise ValueError(msg)
        return self.prompt.format(document=sample[self.input_field])

    def _process_llm_response(self, response: list[str]) -> str:
        """Process a single response from the LLM.

        Called per-row after NDD generation. Child classes can override
        this to customise response parsing.
        """
        return response[0] if response else ""

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        # Pre-process: format prompts via _process_llm_prompt into a temporary column
        df = batch.to_pandas()
        if _FORMATTED_PROMPT_COL in df.columns:
            msg = (
                f"Input DataFrame already contains the internal column '{_FORMATTED_PROMPT_COL}'. "
                "Rename that column before passing the batch to this stage."
            )
            raise ValueError(msg)
        df[_FORMATTED_PROMPT_COL] = df.apply(
            lambda row: self._process_llm_prompt(row.to_dict()), axis=1,
        )
        pre_batch = DocumentBatch(
            data=df,
            dataset_name=batch.dataset_name,
            task_id=batch.task_id,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

        # Generate: NDD handles the LLM call
        result = super().process(pre_batch)

        # Post-process: apply _process_llm_response to each generated value
        result_df = result.to_pandas()
        if self.output_field in result_df.columns:
            # NDD returns a scalar string per row; wrap in a single-element list to
            # match the list[str] signature of _process_llm_response inherited from
            # the non-NDD base class.
            result_df[self.output_field] = result_df[self.output_field].apply(
                lambda x: self._process_llm_response([x]),
            )

        # Remove the temporary formatted-prompt column
        if _FORMATTED_PROMPT_COL in result_df.columns:
            result_df = result_df.drop(columns=[_FORMATTED_PROMPT_COL])

        return DocumentBatch(
            data=result_df,
            dataset_name=batch.dataset_name,
            task_id=f"{batch.task_id}_{self.name}",
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
