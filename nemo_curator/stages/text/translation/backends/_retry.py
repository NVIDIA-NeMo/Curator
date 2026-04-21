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

"""
Package-internal retry helper shared by translation backends.

Provides :func:`retry_with_backoff`, the exponential-backoff loop used by
the single-text ``aws`` and ``google`` backends.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Awaitable, Callable

from loguru import logger

from . import MAX_RETRIES

# Upper cap on a single backoff sleep, in seconds.
_MAX_BACKOFF_SECONDS = 60.0


async def retry_with_backoff(
    fn: Callable[[], Awaitable[Any]],
    *,
    max_retries: int = MAX_RETRIES,
    backend_name: str = "",
    non_retryable: tuple[type[BaseException], ...] = (),
) -> Any:
    """Execute ``fn()`` (an async zero-arg callable) with exponential backoff.

    On exception, waits ``2 ** attempt`` seconds before retrying, up to
    ``max_retries`` attempts total.  The final failure is re-raised so the
    caller can decide how to propagate it.

    Parameters
    ----------
    fn : Callable[[], Awaitable]
        Zero-argument callable returning a fresh coroutine.  Invoked once
        per attempt so the retry produces a fresh awaitable each time.
    max_retries : int
        Maximum number of attempts. Defaults to :data:`MAX_RETRIES`.
    backend_name : str
        Human-readable backend label used in log messages
        (e.g. ``"AWS"``, ``"Google"``).
    non_retryable : tuple[type[BaseException], ...]
        Exception classes that should be re-raised immediately without
        retrying (e.g. ``ValueError`` for input-size violations).

    Returns
    -------
    Any
        Whatever ``fn()`` resolves to on success.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")

    label = f"{backend_name} " if backend_name else ""
    for attempt in range(max_retries):
        try:
            return await fn()
        except non_retryable:
            # Non-retryable: let the caller decide.
            raise
        except Exception as exc:
            if attempt < max_retries - 1:
                # Full jitter: uniform over [0, 2**attempt], capped at _MAX_BACKOFF_SECONDS.
                # Prevents thundering-herd retries against a shared rate-limiter.
                wait_time = min(random.uniform(0, 2 ** attempt), _MAX_BACKOFF_SECONDS)
                logger.warning(
                    "{}API error (attempt {}/{}): {}. Retrying in {:.2f}s...",
                    label,
                    attempt + 1,
                    max_retries,
                    exc,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
            else:
                logger.exception(
                    "{}translation failed after {} attempts",
                    label,
                    max_retries,
                )
                raise

    # Belt-and-suspenders: the loop above must either return or raise.
    # Reaching this line indicates a logic error in retry_with_backoff itself.
    raise RuntimeError(
        f"retry_with_backoff: exhausted {max_retries} attempts without result or exception"
    )
