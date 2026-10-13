# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from operator import eq, ge, gt, le, lt, ne
from typing import Any, Literal

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

_VALUE_OPERATORS = {"lt": lt, "le": le, "eq": eq, "ne": ne, "ge": ge, "gt": gt}


class PreserveByValueStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Filter entries by comparing *input_value_key* against *target_value*.

    Returns ``None`` from ``process()`` to drop entries that fail the
    comparison, matching the text-modality filter convention.

    Args:
        input_value_key: The field in the dataset entries to evaluate.
        target_value: The value to compare with.
        operator: Comparison operator (lt, le, eq, ne, ge, gt).
        missing_value_policy: ``"error"`` (default) preserves the historical
            validation error for a missing input key; ``"drop"`` removes that row.
    """

    name: str = "PreserveByValueStage"
    BATCH_ONLY = True  # process() raises; only process_batch is implemented (agent-discovery hint)

    def __init__(
        self,
        input_value_key: str,
        target_value: float | str,
        operator: Literal["lt", "le", "eq", "ne", "ge", "gt"] = "eq",
        missing_value_policy: Literal["error", "drop"] = "error",
    ):
        self.input_value_key = input_value_key
        self.target_value = target_value
        if missing_value_policy not in {"error", "drop"}:
            msg = "missing_value_policy must be 'error' or 'drop'"
            raise ValueError(msg)
        self.operator = _VALUE_OPERATORS[operator]
        self.missing_value_policy = missing_value_policy

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.input_value_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.input_value_key]

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.input_value_key] if self.missing_value_policy == "error" else []),
            optional_reads=IOSpec(data_keys=[self.input_value_key] if self.missing_value_policy == "drop" else []),
            cardinality="filter",
            # Compares one row against a fixed target value, so batching changes throughput
            # rather than the verdict -- no row's fate depends on the rows beside it.
            gates=Gates(per_row_independent=True),
        )

    def process(self, task: AudioTask) -> AudioTask | None:
        msg = "PreserveByValueStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        t0 = time.perf_counter()
        results = []
        for task in tasks:
            if not self.validate_input(task):
                if self.missing_value_policy == "drop":
                    continue
                msg = f"Task {task!s} failed validation for stage {self}"
                raise ValueError(msg)
            if self.operator(task.data[self.input_value_key], self.target_value):
                results.append(task)
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "input_count": len(tasks),
                "output_count": len(results),
                "filtered_count": len(tasks) - len(results),
            }
        )
        return results


@dataclass(frozen=True)
class _ValueCondition:
    input_value_key: str
    target_value: float | str | bool
    operator: str


class PreserveByValueConditionsStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """Keep a row or nested child according to flat scalar conditions.

    ``conditions`` accepts either a list of
    ``{"input_value_key", "target_value", "operator"}`` mappings or a mapping
    from input key to ``{"target_value", "operator"}``. A scalar mapping value
    is shorthand for an equality condition. Conditions use AND semantics by
    default; ``condition_logic="or"`` keeps an item when at least one condition
    passes. Missing condition keys always fail closed before logic is applied.

    Args:
        conditions: Non-empty list or mapping of scalar comparisons.
        missing_value_policy: ``"error"`` raises when any condition key is
            absent; ``"drop"`` removes rows with any absent condition key.
        items_key: Optional top-level ``task.data`` key containing a list of
            mapping-like children. When set, conditions filter that list in
            place of filtering top-level rows. This is a single-level operation
            and never recursively descends into child values.
        drop_parent_if_empty: In nested mode, drop the parent AudioTask when no
            child survives. Ignored when ``items_key`` is ``None``.
        condition_logic: ``"and"`` (default) requires every condition to pass;
            ``"or"`` requires at least one condition to pass.
    """

    name: str = "PreserveByValueConditionsStage"
    BATCH_ONLY = True

    def __init__(
        self,
        conditions: list[Mapping[str, Any]] | Mapping[str, Any],
        missing_value_policy: Literal["error", "drop"] = "error",
        items_key: str | None = None,
        drop_parent_if_empty: bool = True,
        condition_logic: Literal["and", "or"] = "and",
    ):
        if missing_value_policy not in {"error", "drop"}:
            msg = "missing_value_policy must be 'error' or 'drop'"
            raise ValueError(msg)
        if condition_logic not in {"and", "or"}:
            msg = "condition_logic must be 'and' or 'or'"
            raise ValueError(msg)
        self.missing_value_policy = missing_value_policy
        self.items_key = items_key
        self.drop_parent_if_empty = drop_parent_if_empty
        self.condition_logic = condition_logic
        self._conditions = self._normalize_conditions(conditions)

    @staticmethod
    def _normalize_conditions(  # noqa: C901, PLR0912 - one validation branch per accepted condition shape
        conditions: list[Mapping[str, Any]] | Mapping[str, Any],
    ) -> tuple[_ValueCondition, ...]:
        raw_conditions: list[Mapping[str, Any]]
        if isinstance(conditions, Mapping):
            raw_conditions = []
            for key, value in conditions.items():
                if isinstance(value, Mapping):
                    raw_conditions.append(
                        {
                            "input_value_key": key,
                            "target_value": value.get("target_value"),
                            "operator": value.get("operator", "eq"),
                            "_has_target": "target_value" in value,
                        }
                    )
                else:
                    raw_conditions.append(
                        {
                            "input_value_key": key,
                            "target_value": value,
                            "operator": "eq",
                            "_has_target": True,
                        }
                    )
        elif isinstance(conditions, list):
            raw_conditions = list(conditions)
        else:
            msg = "conditions must be a non-empty list or mapping"
            raise TypeError(msg)
        if not raw_conditions:
            msg = "conditions must contain at least one scalar comparison"
            raise ValueError(msg)

        normalized: list[_ValueCondition] = []
        for index, condition in enumerate(raw_conditions):
            if not isinstance(condition, Mapping):
                msg = f"conditions[{index}] must be a mapping"
                raise TypeError(msg)
            key = condition.get("input_value_key")
            if not isinstance(key, str) or not key:
                msg = f"conditions[{index}].input_value_key must be a non-empty string"
                raise ValueError(msg)
            if not condition.get("_has_target", "target_value" in condition):
                msg = f"conditions[{index}] must define target_value"
                raise ValueError(msg)
            target = condition.get("target_value")
            if isinstance(target, float) and not math.isfinite(target):
                msg = f"conditions[{index}].target_value must be finite"
                raise ValueError(msg)
            if not isinstance(target, (bool, int, float, str)):
                msg = f"conditions[{index}].target_value must be a JSON scalar"
                raise TypeError(msg)
            operator = condition.get("operator", "eq")
            if operator not in _VALUE_OPERATORS:
                msg = f"conditions[{index}].operator must be one of: {', '.join(_VALUE_OPERATORS)}"
                raise ValueError(msg)
            normalized.append(
                _ValueCondition(
                    input_value_key=key,
                    target_value=target,
                    operator=str(operator),
                )
            )
        return tuple(normalized)

    @property
    def conditions(self) -> list[dict[str, Any]]:
        """Configured comparisons as an independent snapshot of the executed semantics."""
        return list(self.normalized_conditions)

    @property
    def normalized_conditions(self) -> tuple[dict[str, Any], ...]:
        """Canonical conditions for deterministic planning and comparison."""
        return tuple(
            {
                "input_value_key": condition.input_value_key,
                "target_value": condition.target_value,
                "operator": condition.operator,
            }
            for condition in self._conditions
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        if self.items_key is not None:
            return [], [self.items_key]
        return [], [condition.input_value_key for condition in self._conditions]

    def outputs(self) -> tuple[list[str], list[str]]:
        if self.items_key is not None:
            return [], [self.items_key]
        return [], [condition.input_value_key for condition in self._conditions]

    def describe(self) -> StageContract:
        logic = self.condition_logic.upper()
        if self.items_key is not None:
            return StageContract(
                reads=IOSpec(
                    data_keys=[self.items_key],
                    segment_data_keys=(
                        [c.input_value_key for c in self._conditions] if self.missing_value_policy == "error" else []
                    ),
                ),
                optional_reads=IOSpec(
                    segment_data_keys=(
                        [c.input_value_key for c in self._conditions] if self.missing_value_policy == "drop" else []
                    )
                ),
                writes=IOSpec(data_keys=[self.items_key]),
                cardinality="filter" if self.drop_parent_if_empty else "1:1 nested-list",
                cardinality_options=["filter", "1:1 nested-list"],
                iteration_key=None if self.drop_parent_if_empty else self.items_key,
                description=(
                    f"Filter mapping-like children in task.data[{self.items_key!r}] with "
                    f"one-level {logic} conditions; nested values are not traversed."
                ),
                gates=Gates(per_row_independent=True),
            )
        keys = [condition.input_value_key for condition in self._conditions]
        return StageContract(
            reads=IOSpec(data_keys=keys if self.missing_value_policy == "error" else []),
            optional_reads=IOSpec(data_keys=keys if self.missing_value_policy == "drop" else []),
            cardinality="filter",
            description=f"Filter top-level AudioTask rows with {logic} conditions.",
            gates=Gates(per_row_independent=True),
        )

    def process(self, task: AudioTask) -> AudioTask | None:
        msg = "PreserveByValueConditionsStage only supports process_batch"
        raise NotImplementedError(msg)

    def _nested_items(
        self,
        task: AudioTask,
        items_key: str,
    ) -> list[Mapping[str, Any]]:
        """Return and structurally validate the configured one-level child list."""
        if items_key not in task.data:
            msg = f"Task {task!s} is missing nested items_key {items_key!r}"
            raise ValueError(msg)
        items = task.data[items_key]
        if not isinstance(items, list):
            msg = f"Task {task!s} nested items_key {items_key!r} must contain a list, got {type(items).__name__}"
            raise TypeError(msg)
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                msg = (
                    f"Task {task!s} nested items_key {items_key!r} "
                    f"child {index} must be mapping-like, got {type(item).__name__}"
                )
                raise TypeError(msg)
        return items

    def _nested_item_passes(
        self,
        task: AudioTask,
        index: int,
        item: Mapping[str, Any],
        items_key: str,
    ) -> bool:
        """Apply configured conditions to one validated direct child."""
        for condition in self._conditions:
            if condition.input_value_key not in item:
                if self.missing_value_policy == "drop":
                    return False
                msg = (
                    f"Task {task!s} nested items_key {items_key!r} child "
                    f"{index} is missing condition key {condition.input_value_key!r}"
                )
                raise ValueError(msg)
        results = (
            _VALUE_OPERATORS[condition.operator](
                item[condition.input_value_key],
                condition.target_value,
            )
            for condition in self._conditions
        )
        return all(results) if self.condition_logic == "and" else any(results)

    def _process_nested_batch(
        self,
        tasks: list[AudioTask],
        items_key: str,
    ) -> list[AudioTask]:
        """Filter direct children, replacing only the configured list field."""
        results: list[AudioTask] = []
        for task in tasks:
            items = self._nested_items(task, items_key)
            survivors = [
                item for index, item in enumerate(items) if self._nested_item_passes(task, index, item, items_key)
            ]
            task.data[items_key] = survivors
            if survivors or not self.drop_parent_if_empty:
                results.append(task)
        return results

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        t0 = time.perf_counter()
        if self.items_key is not None:
            results = self._process_nested_batch(tasks, self.items_key)
        else:
            results = []
            for task in tasks:
                for condition in self._conditions:
                    if condition.input_value_key not in task.data:
                        if self.missing_value_policy == "drop":
                            break
                        msg = f"Task {task!s} failed validation for stage {self}"
                        raise ValueError(msg)
                else:
                    condition_results = (
                        _VALUE_OPERATORS[condition.operator](
                            task.data[condition.input_value_key],
                            condition.target_value,
                        )
                        for condition in self._conditions
                    )
                    keep = all(condition_results) if self.condition_logic == "and" else any(condition_results)
                    if keep:
                        results.append(task)
        self._log_metrics(
            {
                "process_time": time.perf_counter() - t0,
                "input_count": len(tasks),
                "output_count": len(results),
                "filtered_count": len(tasks) - len(results),
            }
        )
        return results
