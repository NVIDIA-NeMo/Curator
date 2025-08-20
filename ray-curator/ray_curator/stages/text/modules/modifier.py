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

from collections.abc import Callable
from dataclasses import dataclass

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.text.modifiers.doc_modifier import DocumentModifier
from ray_curator.tasks import DocumentBatch


@dataclass
class Modify(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Modify the text fields of dataset records.

    This stage applies one or more document-level modifiers to the specified
    text field(s). You can provide:
    - a `DocumentModifier` instance; its `modify_document` method will be used
    - a callable that takes a `str` and returns the modified text value
    - a list mixing the above, which will be applied in order. When a single
      text field is provided it is reused for each modifier; otherwise provide
      one field per modifier.

    Args:
        modifier_fn (Callable[[str], str] | DocumentModifier | list[DocumentModifier | Callable[[str], str]]):
            Modifier or list of modifiers to apply to each record's text.
        text_field (str | list[str]):
            The text field name(s) to read from and write back to. When a list
            is provided, its length must be 1 or equal to the number of modifiers.

    """

    modifier_fn: Callable[[str], float | str] | DocumentModifier | list[DocumentModifier]
    text_field: str | list[str] = "text"
    _name: str = "modifier_fn"

    def __post_init__(self):
        self.modifier_fn = _validate_and_normalize_modifiers(self.modifier_fn, self.text_field)
        self.text_field = _create_text_fields(self.text_field, self.modifier_fn)
        self._name = _get_modifier_stage_name(self.modifier_fn)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.text_field

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], self.text_field

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        """
        Apply the configured modifier(s) to the batch.

        Args:
            batch (DocumentBatch): Input batch to modify.

        Returns:
            DocumentBatch: Batch with modified text field(s).

        """

        df = batch.to_pandas()

        for modifier_fn_i, text_field_i in zip(self.modifier_fn, self.text_field, strict=True):
            inner_modifier_fn = (
                modifier_fn_i.modify_document if isinstance(modifier_fn_i, DocumentModifier) else modifier_fn_i
            )
            df[text_field_i] = df[text_field_i].apply(inner_modifier_fn)

        # Create output batch
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


def _modifier_name(x: DocumentModifier | Callable) -> str:
    return x.name if isinstance(x, DocumentModifier) else x.__name__


def _get_modifier_stage_name(modifiers: list[DocumentModifier | Callable]) -> str:
    """
    Derive the stage name from the provided modifiers.
    """
    return (
        _modifier_name(modifiers[0])
        if len(modifiers) == 1
        else "modifier_chain_of_" + "_".join(_modifier_name(m) for m in modifiers)
    )


def _validate_and_normalize_modifiers(
    _modifier: DocumentModifier | Callable | list[DocumentModifier | Callable],
    text_field: str | list[str] | None,
) -> list[DocumentModifier | Callable]:
    """
    Validate inputs and normalize the modifier(s) to a list.
    """
    if text_field is None:
        msg = "Text field cannot be None"
        raise ValueError(msg)

    modifiers: list[DocumentModifier | Callable] = _modifier if isinstance(_modifier, list) else [_modifier]
    if not modifiers:
        msg = "modifier_fn list cannot be empty"
        raise ValueError(msg)
    if any(not (isinstance(m, DocumentModifier) or callable(m)) for m in modifiers):
        msg = "Each modifier must be a DocumentModifier or callable"
        raise TypeError(msg)
    if len(modifiers) == 1 and isinstance(text_field, list) and len(text_field) > 1:
        msg = f"More text fields than modifiers provided: {text_field}"
        raise ValueError(msg)

    return modifiers


def _create_text_fields(text_field: str | list[str], modifiers: list[DocumentModifier | Callable]) -> list[str]:
    """
    Create/expand text fields to match the number of modifiers.
    """
    if isinstance(text_field, list):
        if len(text_field) == len(modifiers):
            return text_field
        elif len(text_field) == 1:
            return text_field * len(modifiers)
        else:
            msg = (
                f"Number of text fields ({len(text_field)}) must be 1 or equal to number of "
                f"modifiers ({len(modifiers)})"
            )
            raise ValueError(msg)
    else:
        return [text_field] * len(modifiers)
