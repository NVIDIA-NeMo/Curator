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

"""Utility decorators for creating ProcessingStage instances from simple functions.

This module provides a :func:`processing_stage` decorator that turns a plain
Python function into a concrete :class:`nemo_curator.stages.base.ProcessingStage`.

Example
-------

```python
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.function_decorators import processing_stage


@processing_stage(name="WordCountStage", resources=Resources(cpus=1.0), batch_size=1)
def word_count(task: SampleTask) -> SampleTask:
    # Add a *word_count* column to the task's DataFrame
    task.data["word_count"] = task.data["sentence"].str.split().str.len()
    return task
```

The variable ``word_count`` now holds an *instance* of a concrete
``ProcessingStage`` subclass that can be added directly to a
:class:`nemo_curator.pipeline.Pipeline` like so:

```python
from nemo_curator.pipeline import Pipeline


pipeline = Pipeline(...)
# Add read stage, etc.
pipeline.add_stage(...)

# Add ``WordCountStage``
pipeline.add_stage(word_count)

result = pipeline.run(...)
```

"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, TypeVar, cast, overload

from nemo_curator.stages.base import _STAGE_REGISTRY, ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import Task

if TYPE_CHECKING:
    from collections.abc import Callable

# Type variables representing the Task in/out types handled by the
# user-provided function.  They must both be (sub-classes of) Task so that the
# generated ProcessingStage satisfies the base class contract.
TIn = TypeVar("TIn", bound=Task)
TOut = TypeVar("TOut", bound=Task)


@overload
def processing_stage(
    *,
    name: str,
    resources: Resources | dict[str, float] | None = None,
    batch_size: int | None = None,
) -> Callable[[Callable[[TIn], TOut | list[TOut]]], ProcessingStage[TIn, TOut]]: ...


def processing_stage(
    *,
    name: str,
    resources: Resources | dict[str, float] | None = None,
    batch_size: int | None = None,
) -> Callable[[Callable[[TIn], TOut | list[TOut]]], ProcessingStage]:
    """Decorator that converts a function into a :class:`ProcessingStage`.

    Parameters
    ----------
    name:
        The *name* assigned to the resulting stage (``ProcessingStage.name``).
    resources:
        Optional :class:`nemo_curator.stages.resources.Resources`
        or dict[str, float] describing the required compute resources.
        If *None* a default of ``Resources()`` is used.
    batch_size:
        Optional *batch size* for the stage. ``None`` means *no explicit batch
        size* (executor decides).

    The decorated function **must**:
    1. Accept exactly one positional argument: a :class:`Task` instance (or
       subclass).
    2. Return either a single :class:`Task` instance or a ``list`` of tasks.
    """

    if isinstance(resources, dict):
        resources = Resources(**resources)
    elif resources is None:
        resources = Resources()  # Ensure we always have a Resources obj

    def decorator(func: Callable[[TIn], TOut | list[TOut]]) -> ProcessingStage:
        """Inner decorator that builds and *instantiates* a ProcessingStage."""

        # Validate the user-provided function signature early so that mistakes
        # are caught at import-time rather than runtime inside a pipeline.
        sig = inspect.signature(func)
        if len(sig.parameters) != 1:
            msg = "A processing stage function must accept exactly one positional argument (the input Task)."
            raise ValueError(msg)

        # Dynamically create a subclass of ProcessingStage whose *process* method
        # delegates directly to the user function.
        class _FunctionProcessingStage(ProcessingStage[TIn, TOut]):
            _name: str = str(name)
            _resources: Resources = resources  # type: ignore[assignment]
            _batch_size: int | None = batch_size  # type: ignore[assignment]

            # Keep a reference to the original function for introspection /
            # debugging.
            _fn: Callable[[TIn], TOut | list[TOut]] = staticmethod(func)  # type: ignore[assignment]

            def process(self, task: TIn) -> TOut | list[TOut]:  # type: ignore[override]
                # Delegate to the wrapped function.
                out = cast("TOut | list[TOut]", self._fn(task))
                if isinstance(out, list):
                    for t in out:
                        t._metadata = task._metadata.copy()
                        t._stage_perf = task._stage_perf.copy()
                else:
                    out._metadata = task._metadata.copy()
                    out._stage_perf = task._stage_perf.copy()
                return out

            # The user requested to "not worry about inputs/outputs", so we leave
            # them as the base-class defaults (empty lists).

        # Give the dynamically-created class a *nice* __name__ so that logs and
        # error messages are meaningful.  We purposefully use the *stage* name
        # instead of the function name to avoid confusion.
        _FunctionProcessingStage.__name__ = name
        _FunctionProcessingStage.__qualname__ = name

        # Fix the registry so it matches the new name
        _STAGE_REGISTRY[name] = _STAGE_REGISTRY.pop("_FunctionProcessingStage")

        # Instantiate and return the stage so that the decorator can be used as
        # a drop-in replacement for a class instance in pipeline definitions.
        return _FunctionProcessingStage()

    return decorator
