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
Translation backends factory.

Supported built-in backends:
- google: Google Cloud Translation API (v2/v3)
- aws: AWS Translate
- nmt: Neural Machine Translation (IndicTrans2, custom NMT servers)

Custom backends can be registered at runtime via :func:`register_backend`.
"""

import logging

from nemo_curator.stages.text.translation.backends.base import TranslationBackend

LOG = logging.getLogger(__name__)

__all__ = ["TranslationBackend", "get_backend", "register_backend"]

# Registry for user-provided custom backends.
_CUSTOM_BACKENDS: dict[str, type] = {}


def register_backend(name: str, backend_class: type) -> None:
    """Register a custom translation backend.

    Once registered, the backend can be used by passing its *name* as the
    ``backend_type`` parameter to :func:`get_backend` (and, transitively,
    to :class:`TranslateStage`).

    Args:
        name: Short identifier for the backend (e.g., ``"deepl"``).
            Case-insensitive -- will be lower-cased internally.
        backend_class: A concrete subclass of :class:`TranslationBackend`.

    Raises:
        TypeError: If *backend_class* is not a subclass of
            :class:`TranslationBackend`.
    """
    if not (isinstance(backend_class, type) and issubclass(backend_class, TranslationBackend)):
        raise TypeError(
            f"backend_class must be a subclass of TranslationBackend, "
            f"got {backend_class!r}"
        )
    _CUSTOM_BACKENDS[name.lower()] = backend_class
    LOG.info("Registered custom translation backend: %s -> %s", name, backend_class.__name__)


def get_backend(backend_type: str, config: dict) -> TranslationBackend:
    """Factory function to create a translation backend.

    Checks the custom backend registry first, then falls back to the
    built-in backends.

    Args:
        backend_type: One of ``"google"``, ``"aws"``, ``"nmt"``, or a
            custom name previously registered via :func:`register_backend`.
        config: Backend-specific configuration dict. Keys are passed as
            keyword arguments to the backend constructor.

    Returns:
        Initialized TranslationBackend instance (``setup()`` has NOT been
        called yet -- the caller is responsible for calling ``setup()``).

    Raises:
        ValueError: If *backend_type* is not recognised.
    """
    backend_type = backend_type.lower()

    # Check custom registry first.
    if backend_type in _CUSTOM_BACKENDS:
        return _CUSTOM_BACKENDS[backend_type](**config)

    if backend_type == "google":
        from .google import GoogleTranslationBackend

        return GoogleTranslationBackend(**config)
    elif backend_type == "aws":
        from .aws import AWSTranslationBackend

        return AWSTranslationBackend(**config)
    elif backend_type == "nmt":
        from .nmt import NMTTranslationBackend

        return NMTTranslationBackend(**config)
    else:
        registered = ", ".join(sorted(_CUSTOM_BACKENDS)) if _CUSTOM_BACKENDS else "none"
        raise ValueError(
            f"Unknown backend type: {backend_type!r}. "
            f"Built-in backends: google, aws, nmt. "
            f"Custom registered backends: {registered}"
        )
