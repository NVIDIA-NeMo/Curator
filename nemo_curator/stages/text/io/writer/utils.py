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

import os
import time
from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from typing import Any, TypeVar

_T = TypeVar("_T")


def s3_credentials_from_env() -> dict[str, Any]:
    """Generic S3 credentials from standard AWS environment variables.

    Returns only the keys understood by any S3-compatible client (boto3, PyArrow,
    fsspec, s5cmd).  Does NOT include lance-specific dataset-creation settings.
    """
    opts: dict[str, Any] = {}
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        opts["endpoint"] = endpoint
        opts["virtual_hosted_style_request"] = "false"
    if key_id := os.environ.get("AWS_ACCESS_KEY_ID"):
        opts["aws_access_key_id"] = key_id
    if secret := os.environ.get("AWS_SECRET_ACCESS_KEY"):
        opts["aws_secret_access_key"] = secret
    opts["aws_region"] = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return opts


def s3_storage_options_from_env() -> dict[str, Any]:
    """Lance storage options from environment variables.

    Extends :func:`s3_credentials_from_env` with lance-specific dataset-creation
    settings that are only valid when passed to lance / lance-ray APIs.  Do NOT
    pass the result to boto3, PyArrow, or other non-lance S3 clients.
    """
    opts = s3_credentials_from_env()
    if opts.get("endpoint"):  # non-AWS S3-compatible store detected
        opts["new_table_data_storage_version"] = "stable"
        opts["new_table_enable_v2_manifest_paths"] = "true"
        opts["io_threads"] = "128"
    return opts


def retry_with_backoff(
    fn: Callable[[], _T],
    *,
    retries: int = 5,
    label: str = "",
) -> _T:
    """Call *fn* with exponential back-off on failure.

    Retries up to *retries* times; re-raises the last exception on exhaustion.
    """
    from loguru import logger

    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2**attempt
            tag = f"[{label}] " if label else ""
            logger.warning(f"{tag}attempt {attempt + 1}/{retries} failed, retrying in {wait}s: {exc}")
            time.sleep(wait)
    msg = "unreachable"
    raise RuntimeError(msg)  # pragma: no cover


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
