"""Utility decorators for creating ProcessingStage instances from simple functions.

This module provides a :func:`processing_stage` decorator that turns a plain
Python function into a concrete :class:`ray_curator.stages.base.ProcessingStage`.

Example
-------

```python
from ray_curator.stages.resources import Resources
from ray_curator.stages.function_decorators import processing_stage


@processing_stage(name="WordCountStage", resources=Resources(cpus=1.0), batch_size=1)
def word_count(task: SampleTask) -> SampleTask:
    # Add a *word_count* column to the task's dataframe
    task.data["word_count"] = task.data["sentence"].str.split().str.len()
    return task
```

The variable ``word_count`` now holds an *instance* of a concrete
``ProcessingStage`` subclass that can be added directly to a
:class:`ray_curator.pipeline.Pipeline`.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, TypeVar, cast, overload

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import Task

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
    resources: Resources | None = None,
    batch_size: int | None = None,
) -> Callable[[Callable[[TIn], TOut | list[TOut]]], ProcessingStage[TIn, TOut]]: ...


def processing_stage(
    *,
    name: str,
    resources: Resources | None = None,
    batch_size: int | None = None,
) -> Callable[[Callable[[TIn], TOut | list[TOut]]], ProcessingStage]:
    """Decorator that converts a function into a :class:`ProcessingStage`.

    Parameters
    ----------
    name:
        The *name* assigned to the resulting stage (``ProcessingStage.name``).
    resources:
        Optional :class:`ray_curator.stages.resources.Resources` describing the
        required compute resources.  If *None* a default of ``Resources()`` is
        used.
    batch_size:
        Optional *batch size* for the stage.  ``None`` means *no explicit batch
        size* (executor decides).

    The decorated function **must**:
    1. Accept exactly one positional argument:  a :class:`Task` instance (or
       subclass).
    2. Return either a single :class:`Task` instance or a ``list`` of tasks.
    """

    resources = resources or Resources()  # Ensure we always have a Resources obj

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
                return cast("TOut | list[TOut]", self._fn(task))

            # The user requested to “not worry about inputs/outputs”, so we leave
            # them as the base-class defaults (empty lists).

        # Give the dynamically-created class a *nice* __name__ so that logs and
        # error messages are meaningful.  We purposefully use the *stage* name
        # instead of the function name to avoid confusion.
        _FunctionProcessingStage.__name__ = name
        _FunctionProcessingStage.__qualname__ = name

        # Instantiate and return the stage so that the decorator can be used as
        # a drop-in replacement for a class instance in pipeline definitions.
        return _FunctionProcessingStage()

    return decorator
