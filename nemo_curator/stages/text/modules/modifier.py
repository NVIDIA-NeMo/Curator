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

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.modifiers.doc_modifier import DocumentModifier
from nemo_curator.tasks import DocumentBatch


@dataclass
class Modify(ProcessingStage[DocumentBatch, DocumentBatch]):
    """
    Modify fields of dataset records.

    You can provide:
    - a `DocumentModifier` instance; its `modify_document` will be used
    - a callable that takes a single value (single-input) or a dict[str, any] (multi-input)
    - a list mixing the above, applied in order

    Input fields can be:
    - str (single field reused for each modifier)
    - list[str] (one field per modifier)
    - list[list[str]] (per-modifier multiple input fields)

    Implicit behavior:
    - If `output_field` is None and each modifier has exactly one input field,
      results are written in-place to that input field.
    - If any modifier has multiple input fields, `output_field` is required
      (provide a single name to reuse for all, or one per modifier).

    Args:
        modifier_fn (Callable | DocumentModifier | list[DocumentModifier | Callable]):
            Modifier or list of modifiers to apply.
        input_fields (str | list[str] | list[list[str]]):
            Input field(s); see above for accepted forms.
        output_fields (str | list[str] | None):
            Output field name(s). If None and all inputs are single-column,
            in-place update is performed.
    """

    modifier_fn: Callable | DocumentModifier | list[DocumentModifier | Callable]
    input_fields: str | list[str] | list[list[str]] = "text"
    output_fields: str | list[str | None] | None = None
    _name: str = "modifier_fn"

    def __post_init__(self):
        self.modifier_fn = _validate_and_normalize_modifiers(self.modifier_fn, self.input_fields)
        self._input_fields = _normalize_input_fields(self.input_fields, self.modifier_fn)
        self._output_fields = _normalize_output_fields(self.output_fields, self._input_fields, self.modifier_fn)
        self._name = _get_modifier_stage_name(self.modifier_fn)

    def inputs(self) -> tuple[list[str], list[str]]:
        required_cols = sorted({c for cols in self._input_fields for c in cols})
        return ["data"], required_cols

    def outputs(self) -> tuple[list[str], list[str]]:
        output_cols = sorted(set(self._output_fields))
        return ["data"], output_cols

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        df = batch.to_pandas()

        for modifier_fn_i, input_fields_i, output_field_i in zip(
            self.modifier_fn, self._input_fields, self._output_fields, strict=True
        ):
            inner_modify_fn = (
                modifier_fn_i.modify_document if isinstance(modifier_fn_i, DocumentModifier) else modifier_fn_i
            )

            if len(input_fields_i) == 1:
                # Single-input
                src = input_fields_i[0]
                df[output_field_i] = df[src].apply(inner_modify_fn)
            else:
                # Multi-input
                cols = input_fields_i
                df[output_field_i] = [inner_modify_fn(**rec) for rec in df[cols].to_dict("records")]

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
    input_field: str | list[str] | list[list[str]] | None,
) -> list[DocumentModifier | Callable]:
    """
    Validate inputs and normalize the modifier(s) to a list.
    """
    if input_field is None:
        msg = "Input field cannot be None"
        raise ValueError(msg)

    modifiers: list[DocumentModifier | Callable] = _modifier if isinstance(_modifier, list) else [_modifier]
    if not modifiers:
        msg = "modifier_fn list cannot be empty"
        raise ValueError(msg)
    if any(not (isinstance(m, DocumentModifier) or callable(m)) for m in modifiers):
        msg = "Each modifier must be a DocumentModifier or callable"
        raise TypeError(msg)

    return modifiers


def _normalize_input_fields(
    input_fields: str | list[str] | list[list[str]], modifiers: list[DocumentModifier | Callable]
) -> list[list[str]]:
    """
    Normalize input fields into a list[list[str]] with one entry per modifier.
    """
    if isinstance(input_fields, str):
        return [[input_fields] for _ in range(len(modifiers))]
    if isinstance(input_fields, list) and all(isinstance(x, str) for x in input_fields):
        if len(input_fields) == 1:
            return [[input_fields[0]] for _ in range(len(modifiers))]
        if len(input_fields) == len(modifiers):
            return [[f] for f in input_fields]
        msg = (
            f"Number of input fields ({len(input_fields)}) must be 1 or equal to number of "
            f"modifiers ({len(modifiers)}). For multi-input per modifier, pass a list of lists."
        )
        raise ValueError(msg)
    if isinstance(input_fields, list) and all(isinstance(x, list) for x in input_fields):
        if len(input_fields) == 1:
            return [list(input_fields[0]) for _ in range(len(modifiers))]
        if len(input_fields) == len(modifiers):
            return [list(lst) for lst in input_fields]
        msg = (
            f"Number of input field groups ({len(input_fields)}) must be 1 or equal to number of "
            f"modifiers ({len(modifiers)})"
        )
        raise ValueError(msg)
    msg = "input_fields must be str, list[str], or list[list[str]]"
    raise TypeError(msg)


def _normalize_output_fields(
    output_fields: str | list[str | None] | None,
    input_fields: list[list[str]],
    modifiers: list[DocumentModifier | Callable],
) -> list[str]:
    """
    Resolve output column names to one per modifier.

    Rules:
    - None overall: in-place if all modifiers have exactly one input; else error.
    - str overall: replicate for all modifiers.
    - list overall (len 1 or len(modifiers)):
      - Each entry may be a str (explicit output) or None (implicit in-place; requires single input).
    """

    def _inplace_or_error(inputs: list[str]) -> str:
        if len(inputs) == 1:
            return inputs[0]
        msg = (
            "Implicit in-place (None) not allowed for a modifier with multiple input fields. "
            "Provide an explicit output column."
        )
        raise ValueError(msg)

    if output_fields is None:
        if all(len(inp) == 1 for inp in input_fields):
            return [inp[0] for inp in input_fields]
        msg = (
            "output_fields must be provided when any modifier has multiple input fields. "
            "Provide a single name to reuse or a list[str] with one name per modifier."
        )
        raise ValueError(msg)
    elif isinstance(output_fields, str):
        return [output_fields for _ in range(len(modifiers))]
    elif isinstance(output_fields, list):
        if len(output_fields) == 1:
            name0 = output_fields[0]
            if name0 is None:
                return [_inplace_or_error(inputs_i) for inputs_i in input_fields]
            return [name0 for _ in range(len(modifiers))]
        if len(output_fields) == len(modifiers):
            return [
                _inplace_or_error(inputs_i) if name_i is None else name_i
                for name_i, inputs_i in zip(output_fields, input_fields, strict=True)
            ]
        msg = (
            f"Number of output fields ({len(output_fields)}) must be 1 or equal to number of "
            f"modifiers ({len(modifiers)})"
        )
        raise ValueError(msg)
    else:
        msg = "output_field must be str, list[str | None], or None"
        raise TypeError(msg)
