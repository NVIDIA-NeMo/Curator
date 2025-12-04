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

import hashlib
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any


def batched(iterable: Iterable[Any], n: int) -> Iterator[tuple[Any, ...]]:
    """
    Batch an iterable into lists of size n.

    Args:
      iterable (Iterable[Any]): The iterable to batch
      n (int): The size of the batch

    Returns:
        Iterator[tuple[...]]: An iterator of tuples, each containing n elements from the iterable
    """
    if n < 1:
        msg = "n must be at least one"
        raise ValueError(msg)
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def get_deterministic_hash(inputs: list[str], seed: str = "") -> str:
    """Create a deterministic hash from inputs."""
    combined = "|".join(sorted(inputs)) + "|" + seed
    return hashlib.sha256(combined.encode()).hexdigest()[:12]
