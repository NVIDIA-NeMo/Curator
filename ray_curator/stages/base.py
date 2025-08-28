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

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from loguru import logger

from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import Task

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

X = TypeVar("X", bound=Task)  # Input task type
Y = TypeVar("Y", bound=Task)  # Output task type

_STAGE_REGISTRY: dict[str, type[ProcessingStage]] = {}


class StageMeta(ABCMeta):
    """Metaclass that automatically registers concrete Stage subclasses.
    A class is considered *concrete* if it directly inherits from
    :class:`ProcessingStage` **and** implements a ``name`` property.  Abstract
    helper classes (e.g. *ProcessingStage* itself) will not be added to the
    registry because they have the ``_is_abstract`` attribute set.
    """

    def __new__(mcls, name, bases, namespace, **kwargs):  # noqa: ANN001
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        # Skip registration for the abstract roots
        if namespace.get("_is_abstract_root", False):
            return cls

        # Only register subclasses that ultimately derive from ProcessingStage
        # but are not abstract.
        from inspect import isabstract  # local import to avoid cycle during class creation

        if "ProcessingStage" in [base.__name__ for base in cls.mro()[1:]] and not isabstract(cls):
            # Ensure no duplicate class names (helps when reloading in notebooks)
            _STAGE_REGISTRY[cls.__name__] = cls  # type: ignore[assignment]

        return cls


def get_stage_class(name: str) -> type[ProcessingStage]:
    """Retrieve a registered stage class by its *class name*.
    Raises
    ------
    KeyError
        If no stage with that name is registered.
    """

    return _STAGE_REGISTRY[name]


class ProcessingStage(ABC, Generic[X, Y], metaclass=StageMeta):
    """Base class for all processing stages.
    Processing stages operate on Task objects (or subclasses like DocumentBatch).
    Each stage type can declare what type of Task it processes as input (X)
    and what type it produces as output (Y).
    Stages can return either:
    - A single task (typical for transformations)
    - A list of tasks (for stages that split work, like readers)
    - None (for filtered out tasks)
    """

    _is_abstract_root = True  # prevent base from registering itself
    _name = "ProcessingStage"
    _resources = Resources(cpus=1.0)
    _batch_size = 1

    @property
    def name(self) -> str:
        return self._name

    @property
    def resources(self) -> Resources:
        return self._resources

    @property
    def batch_size(self) -> int | None:
        """Number of tasks to process in a batch."""
        return self._batch_size

    def num_workers(self) -> int | None:
        """Number of workers required. If None, then executor will determine the number of workers."""
        return None

    def validate_input(self, task: Task) -> bool:
        """Validate input task meets requirements.
        Args:
            task: Task to validate
        Returns:
            True if valid, False otherwise
        """
        required_top_level_attrs, required_data_attrs = self.inputs()

        # Check required attributes exist
        missing_top_level_attrs = []
        for attr in required_top_level_attrs:
            if not hasattr(task, attr):
                missing_top_level_attrs.append(attr)

        # Check required columns exist
        missing_data_attrs = []
        for attr in required_data_attrs:
            if not hasattr(task.data, attr):
                missing_data_attrs.append(attr)

        # Log warning with missing attributes
        if missing_top_level_attrs or missing_data_attrs:
            logger.error(
                f"Task {task.task_id} missing required attributes: {missing_top_level_attrs} {missing_data_attrs}"
            )

        return not missing_top_level_attrs and not missing_data_attrs

    @abstractmethod
    def process(self, task: X) -> Y | list[Y]:
        """Process a task and return the result.
        Args:
            task (X): Input task to process
        Returns (Y | list[Y]):
            - Single task: For 1-to-1 transformations
            - List of tasks: For 1-to-many transformations (e.g., readers)
            - None: If the task should be filtered out
        """

    def process_batch(self, tasks: list[X]) -> list[Y]:
        """Process a batch of tasks and return results.
        Override this method to enable batch processing for your stage.
        If not overridden, the stage will only support single-task processing.
        Args:
            tasks (list[X]): List of input tasks to process
        Returns (list[Y]):
            List of results, where each result can be:
            - Single task: For 1-to-1 transformations
            - List of tasks: For 1-to-many transformations
            - None: If the task should be filtered out
        Note: The returned list should have the same length as the input list,
        with each element corresponding to the result of processing the task
        at the same index.
        """
        # Default implementation: process tasks one by one
        # This is only used as a fallback if a stage doesn't override this method
        results = []
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task!s} failed validation for stage {self}"
                raise ValueError(msg)

            result = self.process(task)
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
        return results

    def setup_on_node(self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup method called once per node in distributed settings.
        Override this method to perform node-level initialization.
        Args:
            node_info (NodeInfo, optional): Information about the node (provided by some backends)
            worker_metadata (WorkerMetadata, optional): Information about the worker (provided by some backends)
        """

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        """Setup method called once before processing begins.
        Override this method to perform any initialization that should
        happen once per worker.
        Args:
            worker_metadata (WorkerMetadata, optional): Information about the worker (provided by some backends)
        """

    def teardown(self) -> None:
        """Teardown method called once after processing ends.
        Override this method to perform any cleanup.
        """

    def supports_batch_processing(self) -> bool:
        """Whether this stage supports vectorized batch processing.
        This is automatically determined by checking if the stage has
        overridden the process_batch method from the base class.
        """
        # Check if process_batch has been overridden
        return type(self).process_batch is not ProcessingStage.process_batch

    def __repr__(self) -> str:
        """String representation of the stage."""
        return f"{self.__class__.__name__}"

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define stage input requirements.

        Returns (tuple[list[str], list[str]]):
            Tuple of (required_attributes, required_columns) where:
            - required_top_level_attributes: List of task attributes that must be present
            - required_data_attributes: List of attributes within the data that must be present
        """
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define stage output specification.

        Returns (tuple[list[str], list[str]]):
            Tuple of (output_attributes, output_columns) where:
            - output_top_level_attributes: List of task attributes this stage adds/modifies
            - output_data_attributes: List of attributes within the data that this stage adds/modifies
        """
        return [], []

    def xenna_stage_spec(self) -> dict[str, Any]:
        """Get Xenna configuration for this stage.

        Returns (dict[str, Any]):
            Dictionary containing Xenna-specific configuration
        """
        return {}

    def with_(
        self, name: str | None = None, resources: Resources | None = None, batch_size: int | None = None
    ) -> ProcessingStage:
        """Apply configuration changes to this stage with overridden properties.

        Note: This method uses class-level attributes and instance attributes interchangeably which can sometimes
        lead to unexpected behavior. Please see https://github.com/NVIDIA-NeMo/Curator/pull/764 for more details.
        Args:
            name: Override the name property
            resources: Override the resources property
            batch_size: Override the batch_size property
        """
        # Create a copy of the current instance
        import copy

        new_instance = copy.deepcopy(self)

        # Override the instance attributes directly
        if name is not None:
            new_instance._name = name
        if resources is not None:
            new_instance._resources = resources
        if batch_size is not None:
            new_instance._batch_size = batch_size

        return new_instance

    def get_config(self) -> dict[str, Any]:
        """Get configuration for this stage.
        Returns (dict[str, Any]):
            Dictionary containing configuration for this stage
        """
        return {
            "name": self.name,
            "resources": self.resources,
            "batch_size": self.batch_size,
            "supports_batch_processing": self.supports_batch_processing(),
        }

    def ray_stage_spec(self) -> dict[str, Any]:
        """Get Ray configuration for this stage.
        Note : This is only used for Ray Data which is an experimental backend.
        The keys are defined in RayStageSpecKeys in backends/experimental/ray_data/utils.py

        Returns (dict[str, Any]):
            Dictionary containing Ray-specific configuration
        """
        return {}

    # --- Custom per-stage metrics helpers ---
    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Record custom metrics for this stage (e.g., sub-stage timings)."""
        if not hasattr(self, "_custom_metrics") or self._custom_metrics is None:
            self._custom_metrics = {}
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self._custom_metrics[name] = float(value)
            else:
                msg = f"Can't record non-numeric metric {name} value={value} for stage {self.name}"
                logger.warning(msg)

    def _log_metric(self, name: str, value: float) -> None:
        return self._log_metrics({name: value})

    def _consume_custom_metrics(self) -> dict[str, float]:
        """Return and clear metrics recorded during the last process call."""
        if not hasattr(self, "_custom_metrics") or self._custom_metrics is None:
            self._custom_metrics = {}
        metrics: dict[str, float] = dict(self._custom_metrics)
        del self._custom_metrics
        return metrics


class CompositeStage(ProcessingStage[X, Y], ABC):
    """Base class for high-level composite stages.

    Composite stages are user-facing stages that decompose into multiple
    low-level execution stages during pipeline planning. They provide a
    simplified API while maintaining fine-grained control at execution time.

    Composite stages never actually execute - they only exist to be decomposed
    into their constituent execution stages.
    """

    def __init__(self):
        self._with_operations = []

    def inputs(self) -> tuple[list[str], list[str]]:
        """Get the inputs for this stage."""
        return self.decompose()[0].inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        """Get the outputs for this stage."""
        return self.decompose()[-1].outputs()

    @abstractmethod
    def decompose(self) -> list[ProcessingStage]:
        """Decompose into execution stages.

        This method must be implemented by composite stages to define
        what low-level stages they represent.

        Returns (list[ProcessingStage]):
            List of execution stages that will actually run
        """

    def with_(self, stage_with_dict: dict[str, Any]) -> CompositeStage:
        """Apply configuration changes to this stage."""
        # Probably should return a new CompositeStage object
        self._with_operations.append(stage_with_dict)
        return self

    def decompose_and_apply_with(self) -> list[ProcessingStage]:
        """Decompose and apply configuration changes to this stage."""
        return self._apply_with_(self.decompose())

    def _apply_with_(self, stages: list[ProcessingStage]) -> list[ProcessingStage]:
        """Apply configuration changes to this stage."""
        for stage_with_dict in self._with_operations:
            stage_name_to_stage = {stage.name: stage for stage in stages}

            # Verify that all stages have unique names
            if len(stage_name_to_stage) != len(stages):
                err = "All stages must have unique names in composite stage to apply configuration changes using with_()."
                raise ValueError(err)

            # Ensure that we can cover all the keys in stage_with_dict
            for stage_name in stage_with_dict:
                if stage_name not in stage_name_to_stage:
                    err = f"Stage {stage_name} not found in composite stage to apply configuration changes using with_()."
                    raise ValueError(err)

            new_stages = []
            # Apply configuration changes to each stage
            for stage in stages:
                if stage.name in stage_with_dict:
                    new_stages.append(stage.with_(**stage_with_dict[stage.name]))
                else:
                    new_stages.append(stage)

            stages = new_stages

        return stages

    def process(self, task: X) -> Y | list[Y]:  # noqa: ARG002
        """Composite stages should never be executed directly."""
        msg = f"Composite stage '{self.name}' should not be executed directly. "
        msg += "It should be decomposed into execution stages during planning."
        raise RuntimeError(msg)

    def get_description(self) -> str:
        """Get a description of what this composite stage does.

        Override this to provide user-friendly documentation.
        """
        return f"Composite stage: {self.name}"
