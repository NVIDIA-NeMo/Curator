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

"""Generic cost-bucketed GPU inference stage.

``BucketedInferenceStage`` factors the bucketize -> dispatch -> reassemble
loop out of individual stages so any GPU inference processor (audio, text,
vision, ...) gets duration/token/pixel bucketing for free by implementing
four small hooks instead of re-coding the dispatch loop.

Concrete subclasses implement:

* :meth:`build_items`   - expand input tasks into flat model-input items;
* :meth:`item_cost`     - per-item bucketing cost (audio sec, tokens, ...);
* :meth:`run_inference` - run the model on ONE sub-batch (1:1 results);
* :meth:`assemble`      - stitch per-item results back onto the tasks.

The base :meth:`process_batch` wires these through
:func:`nemo_curator.stages.inference.batch_policy.run_bucketed`, which honors
``batch_policy`` and realigns results to the original item order.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from nemo_curator.stages.base import ProcessingStage, X, Y
from nemo_curator.stages.inference.batch_policy import run_bucketed

if TYPE_CHECKING:
    from nemo_curator.stages.inference.batch_policy import BatchPolicy

# Model-input item type and per-item result type. These are intentionally
# unbounded (an "item" can be a dict, a tensor, a prompt string, ...).
ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")


class BucketedInferenceStage(ProcessingStage[X, Y], Generic[X, Y, ItemT, ResultT]):
    """Abstract cost-bucketed inference stage.

    Subclasses set a ``batch_policy`` (or leave it ``None`` for a single
    sub-batch per call) and implement the four hooks below. The base owns the
    1:1 ``process_batch`` contract: it returns exactly one output per input
    task, in input order.
    """

    _is_abstract_root = True  # never registered / instantiated directly
    batch_policy: BatchPolicy | None = None

    @abstractmethod
    def build_items(self, tasks: list[X]) -> tuple[list[ItemT], list[int]]:
        """Expand ``tasks`` into flat model-input items.

        Returns a ``(items, parent_of)`` pair where ``parent_of[i]`` is the
        index (into ``tasks``) of the task that produced ``items[i]``. A task
        may fan out to several items (e.g. pre-sliced audio chunks) or none.
        Implementations should also reset any per-call accumulators here, as
        this hook runs first on every ``process_batch`` call.
        """

    @abstractmethod
    def item_cost(self, item: ItemT) -> float:
        """Per-item bucketing cost (audio seconds, tokens, pixels, ...)."""

    @abstractmethod
    def run_inference(self, items: list[ItemT]) -> list[ResultT]:
        """Run the model on ONE sub-batch; return one result per item (1:1)."""

    @abstractmethod
    def assemble(
        self,
        tasks: list[X],
        items: list[ItemT],
        parent_of: list[int],
        results: list[ResultT],
    ) -> list[Y]:
        """Stitch per-item ``results`` back onto ``tasks`` and write outputs.

        ``items``/``parent_of``/``results`` are index-aligned (``results[i]``
        is the output for ``items[i]``, produced by ``tasks[parent_of[i]]``).
        Must return exactly one output task per input task, in input order.
        """

    def process_batch(self, tasks: list[X]) -> list[Y]:
        # Ray Data ``map_batches`` passes column values as numpy ndarrays; never
        # use truthiness on ``tasks`` (``if not tasks`` raises ValueError).
        if len(tasks) == 0:
            return []
        items, parent_of = self.build_items(tasks)
        results = run_bucketed(
            items,
            self.run_inference,
            cost_fn=self.item_cost,
            policy=self.batch_policy,
        )
        return self.assemble(tasks, items, parent_of, results)
