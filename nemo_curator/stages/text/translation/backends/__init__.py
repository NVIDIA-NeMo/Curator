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

"""Backend registry for translation stages."""

import logging

from loguru import logger

from nemo_curator.stages.text.translation.backends.base import TranslationBackend

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

MAX_RETRIES = 5

__all__ = [
    "MAX_RETRIES",
    "TranslationBackend",
    "get_backend",
    "register_backend",
]

_CUSTOM_BACKENDS: dict[str, type] = {}


def register_backend(name: str, backend_class: type) -> None:
    """Register a custom translation backend."""
    if not (isinstance(backend_class, type) and issubclass(backend_class, TranslationBackend)):
        raise TypeError(
            f"backend_class must be a subclass of TranslationBackend, "
            f"got {backend_class!r}"
        )
    _CUSTOM_BACKENDS[name.lower()] = backend_class
    logger.info("Registered custom translation backend: {} -> {}", name, backend_class.__name__)


def get_backend(backend_type: str, config: dict) -> TranslationBackend:
    """Create a built-in or custom translation backend."""
    backend_type = backend_type.lower()

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
