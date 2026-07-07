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

"""Shared vLLM setup utilities.

These helpers centralise the boilerplate that every vLLM-based inference stage
needs: finding a free port, initialising an :class:`vllm.LLM` engine with
automatic port-collision retry, and resolving an HuggingFace model ID to a
local snapshot path.

They were extracted from the Nemotron-Parse inference stage, which was the
first stage in NeMo Curator to be tested at scale (320x H100).  Future stages
that use vLLM (video, text, audio) should import from here rather than
duplicating this logic.  See GitHub issue #1720 for the roadmap to wire these
utilities into other modalities.
"""

from __future__ import annotations

from loguru import logger

# Errors that should not be retried. Add entries here when vLLM exposes
# other fatal startup errors through generic EngineCore wrapper messages.
_NON_RETRYABLE_MARKERS = (
    "out of memory",
    "cudaerrormemoryallocation",
    "device-side assert",
    "invalid config",
    "invalid model config",
)

# Substrings that indicate a (usually transient) vLLM engine-startup failure
# caused by a MASTER_PORT collision. In vLLM v1 the bind happens in the
# EngineCore subprocess, so the parent often only sees the wrapped messages
# below rather than the raw "EADDRINUSE".
_ENGINE_STARTUP_FAILURE_MARKERS = (
    "eaddrinuse",
    "address already in use",
    "engine core initialization failed",
    "enginecore failed to start",
)


def _exception_chain_text(exc: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return " ".join(parts).lower()


def _is_engine_startup_failure(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a retryable vLLM engine-startup failure."""
    message = _exception_chain_text(exc)
    if any(marker in message for marker in _NON_RETRYABLE_MARKERS):
        return False
    return any(marker in message for marker in _ENGINE_STARTUP_FAILURE_MARKERS)


def pick_free_port() -> int:
    """Return a free TCP port on the local machine."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def create_vllm_llm(  # noqa: PLR0913
    model_path: str,
    *,
    max_num_seqs: int = 64,
    enforce_eager: bool = False,
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    limit_mm_per_prompt: dict | None = None,
    max_port_retries: int = 3,
    **extra_engine_kwargs: object,
) -> "vllm.LLM":  # noqa: F821,UP037
    """Create a :class:`vllm.LLM` instance with automatic port-collision retry.

    vLLM selects a MASTER_PORT for the distributed backend at startup.  On a
    busy node the chosen port may already be in use, causing an
    ``EADDRINUSE`` ``RuntimeError``.  This helper picks a fresh free port on
    each attempt so that transient collisions are handled transparently.

    Parameters
    ----------
    model_path:
        Local path or HuggingFace model ID to load.
    max_num_seqs:
        Maximum number of sequences vLLM processes concurrently.
    enforce_eager:
        Disable CUDA graph capture (slower but uses less memory).
    dtype:
        Model weight dtype passed to vLLM (e.g. ``"bfloat16"``).
    trust_remote_code:
        Whether to trust remote code in the model repository.
    limit_mm_per_prompt:
        Multimodal token limits per prompt (e.g. ``{"image": 1}``).
        Defaults to ``{"image": 1}`` when ``None``.
    max_port_retries:
        Number of port-pick attempts before re-raising the error.
    extra_engine_kwargs:
        Additional keyword arguments forwarded verbatim to :class:`vllm.LLM`
        (e.g. ``gpu_memory_utilization``, ``max_num_batched_tokens``). Keys here
        override the explicit defaults above when they collide.
    """
    import os
    import random
    import time

    from vllm import LLM

    if limit_mm_per_prompt is None:
        limit_mm_per_prompt = {"image": 1}

    engine_kwargs: dict = {
        "model": model_path,
        "max_num_seqs": max_num_seqs,
        "limit_mm_per_prompt": limit_mm_per_prompt,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "enforce_eager": enforce_eager,
        **extra_engine_kwargs,
    }

    for attempt in range(1, max_port_retries + 1):
        free_port = pick_free_port()
        os.environ["MASTER_PORT"] = str(free_port)
        try:
            return LLM(**engine_kwargs)
        except RuntimeError as e:
            # MASTER_PORT collisions happen when many engines start concurrently
            # on one node (e.g. 8 single-GPU replicas under xenna). In vLLM v1 the
            # distributed-store bind runs inside the EngineCore subprocess, so the
            # original "EADDRINUSE" text does not reach this parent process: the
            # caller only sees a generic "Engine core initialization failed". We
            # therefore retry on both the direct bind error and that wrapped
            # startup failure, picking a fresh port and adding jitter so racing
            # workers de-stagger instead of re-colliding on the same retry.
            if not _is_engine_startup_failure(e):
                raise
            if attempt == max_port_retries:
                logger.error(f"[vLLM] Engine startup failed after {max_port_retries} attempt(s): {e}")
                raise
            logger.warning(
                f"[vLLM] Engine startup failed on attempt {attempt}/{max_port_retries} "
                f"(port {free_port}, likely a MASTER_PORT collision), retrying: {e}"
            )
            time.sleep(2 + random.uniform(0, 3))  # noqa: S311 - jitter only, not security-sensitive

    msg = "unreachable"
    raise RuntimeError(msg)  # pragma: no cover


def resolve_local_model_path(model_path: str) -> str:
    """Resolve an HF model ID to a local snapshot path.

    Uses ``local_files_only=True`` so that workers on compute nodes never
    attempt to reach the internet.  The model must be pre-downloaded (e.g.
    via ``huggingface-cli download``) before submitting the job.

    Parameters
    ----------
    model_path:
        HuggingFace model ID or an already-local path.  If the path is
        already a local directory it is returned unchanged.
    """
    import os

    if os.path.isdir(model_path):
        return model_path

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(model_path, local_files_only=True)
    except LocalEntryNotFoundError:
        msg = (
            f"Model '{model_path}' is not cached locally. "
            f"Please pre-download it before submitting the job:\n\n"
            f"    huggingface-cli download {model_path}\n\n"
            f"Then re-run with the local cache path or the model ID."
        )
        raise RuntimeError(msg) from None
