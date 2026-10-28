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

"""Language helpers for text written without word-separating spaces."""

SCRIPTIO_CONTINUA_LANGUAGE_CODES: frozenset[str] = frozenset(
    {
        "ja",
        "zh",
        "zh-cn",
        "zh-tw",
        "zh-hans",
        "zh-hant",
        "yue",
        "th",
        "lo",
        "km",
        "my",
        "bo",
        "dz",
    }
)

_SCRIPTIO_CONTINUA_LANGUAGE_NAMES: frozenset[str] = frozenset(
    {
        "chinese",
        "japanese",
        "thai",
    }
)


def is_scriptio_continua(language: object | None) -> bool:
    """Return whether ``language`` normally omits spaces between words.

    ISO codes, the Chinese locale tags used by the reference pipeline, and
    full English language names are accepted case-insensitively. Underscores
    in locale tags are normalized to hyphens.
    """
    if language is None:
        return False
    normalized = str(language).strip().lower().replace("_", "-")
    return normalized in SCRIPTIO_CONTINUA_LANGUAGE_CODES or normalized in _SCRIPTIO_CONTINUA_LANGUAGE_NAMES
