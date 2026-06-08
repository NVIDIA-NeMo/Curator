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

"""Utility Functions for grouping iterables.

This module provides a collection of utility functions designed to assist with common tasks related to manipulating
and transforming iterables in Python.

These utilities are generic and work with any iterable types. They're particularly useful for data processing tasks,
batching operations, and other scenarios where dividing data into specific groupings is necessary.

Note:
    While these utilities are designed for flexibility and ease-of-use,
    they may not be optimized for extremely large datasets or performance-critical applications.

"""

import itertools
import typing
from collections.abc import Generator, Iterable

T = typing.TypeVar("T")


def split_by_chunk_size(
    iterable: Iterable[T],
    chunk_size: int,
    custom_size_func: typing.Callable[[T], int] = lambda x: 1,  # noqa: ARG005
    *,
    drop_incomplete_chunk: bool = False,
) -> Generator[list[T], None, None]:
    """Split an iterable into chunks of the specified size.

    Args:
        iterable (Iterable[T]): The input iterable to be split.
        chunk_size (int): Size of each chunk.
        custom_size_func (typing.Callable): function
        drop_incomplete_chunk (bool, optional): If True, drops the last chunk if its size is less than the
                                                specified chunk size. Defaults to False.

    Yields:
    - Generator[list[T], None, None]: Chunks of the input iterable.

    """
    out = []
    cur_count = 0
    for value in iterable:
        out.append(value)
        cur_count += custom_size_func(value)
        if cur_count >= chunk_size:
            yield out
            out = []
            cur_count = 0
    if out and not drop_incomplete_chunk:
        yield out


def split_into_n_chunks(iterable: Iterable[T], num_chunks: int) -> Generator[list[T], None, None]:
    """Split an iterable into a specified number of chunks.

    Args:
        iterable (Iterable[T]): The input iterable to be split.
        num_chunks (int): The desired number of chunks.

    Yields:
    - Generator[list[T], None, None]: Chunks of the input iterable.

    """
    it = list(iterable)
    if len(it) <= num_chunks:
        yield from [[x] for x in it]
        return
    d, r = divmod(len(it), num_chunks)
    for i in range(num_chunks):
        si = (d + 1) * (min(r, i)) + d * (0 if i < r else i - r)
        yield it[si : si + (d + 1 if i < r else d)]


def pairwise(iterable: Iterable[T]) -> Iterable[tuple[T, T]]:
    """Return pairs of consecutive items from the input iterable.

    Args:
        iterable (Iterable[T]): The input iterable.

    Returns:
        Iterable[tuple[T, T]]: Pairs of consecutive items.

    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b, strict=False)
