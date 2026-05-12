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

from dataclasses import dataclass

from .audio_task import AudioTask
from .document import DocumentBatch
from .file_group import FileGroupTask
from .image import ImageBatch, ImageObject
from .interleaved import InterleavedBatch
from .tasks import EmptyTask, Task, _EmptyTask


@dataclass(frozen=True)
class TransientDrop:
    """Sentinel return value: drop this task from the current run, but do NOT mark
    it complete in the checkpoint database. Resume will retry.

    Use this when a stage cannot produce an output for reasons that are
    reasonable to retry on a subsequent run -- inference server timeout, rate
    limiting, transient OOM, etc.

    Contrast with returning ``None``, which means "permanent filter; the task
    is done and should not be retried." Raising an exception aborts the
    entire pipeline run; ``TransientDrop`` lets the rest of the batch and
    the rest of the pipeline complete normally.

    Only supported from ``ProcessingStage.process()`` and
    ``ProcessingStage.process_batch_grouped()``. The legacy flat
    ``process_batch()`` cannot express transient drops because its return
    type is ``list[Task]`` (``TransientDrop`` is not a ``Task``).
    """

    reason: str = ""


__all__ = [
    "AudioTask",
    "DocumentBatch",
    "EmptyTask",
    "FileGroupTask",
    "ImageBatch",
    "ImageObject",
    "InterleavedBatch",
    "Task",
    "TransientDrop",
    "_EmptyTask",
]
