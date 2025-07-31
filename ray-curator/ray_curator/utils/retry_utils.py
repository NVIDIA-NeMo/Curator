# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for retrying operations."""

import time
import typing
from collections.abc import Callable, Iterable

from loguru import logger

T = typing.TypeVar("T")


def do_with_retries(  # noqa: PLR0913
    func: Callable[[], T],
    exceptions_to_retry: Iterable[type[Exception]] | None = None,
    max_attempts: int = 5,
    backoff_factor: float = 2,
    max_wait_time_s: float = 16.0,
    name: str | None = None,
) -> T:
    """Retries function execution in case of exceptions.

    Args:
        func: The function to execute
        exceptions_to_retry: Exception(s) to catch and retry. Defaults to all exceptions.
        max_attempts: The maximum number of times to retry
        backoff_factor: Factor by which the waiting time is extended after each failure
        max_wait_time_s: maximum wait in seconds
        name: name

    Returns:
        The result of the function execution if successful
        Raises the last exception if not successful after max_attempts

    """
    exceptions_to_retry = (Exception,) if exceptions_to_retry is None else tuple(exceptions_to_retry)
    attempt = 0
    while attempt < max_attempts:
        try:
            return func()
        except exceptions_to_retry as e:  # noqa: PERF203
            attempt += 1
            if attempt == max_attempts:
                raise
            sleep_time = min(backoff_factor**attempt, max_wait_time_s)
            preamble = "" if name is None else f"{name} - "
            logger.warning(
                f"{preamble}Attempt {attempt}/{max_attempts} failed with error: {e!s}. "
                f"Retrying in {sleep_time} seconds...",
            )
            time.sleep(sleep_time)
    raise AssertionError
