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

"""Shared language and pipeline utilities for audio pipeline stages."""

from typing import Any

LANG_CODE_TO_NAME: dict[str, str] = {
    "en": "English", "de": "German", "es": "Spanish", "fr": "French",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "pl": "Polish", "cs": "Czech", "ro": "Romanian", "hu": "Hungarian",
    "el": "Greek", "fi": "Finnish", "da": "Danish", "sv": "Swedish",
    "lt": "Lithuanian", "lv": "Latvian", "hr": "Croatian", "et": "Estonian",
    "bg": "Bulgarian", "sk": "Slovak", "sl": "Slovenian", "mt": "Maltese",
    "uk": "Ukrainian", "sr": "Serbian", "mk": "Macedonian", "no": "Norwegian",
    "hi": "Hindi", "ta": "Tamil", "mr": "Marathi", "bn": "Bengali",
    "kn": "Kannada", "te": "Telugu", "ml": "Malayalam", "gu": "Gujarati",
    "ur": "Urdu", "pa": "Punjabi",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "he": "Hebrew", "id": "Indonesian", "vi": "Vietnamese", "th": "Thai",
    "tr": "Turkish", "fil": "Filipino", "tl": "Tagalog", "fa": "Persian",
}


def append_note(task_data: dict[str, Any], note: str, notes_key: str = "_notes") -> None:
    """Append a decision note to the task's ``_notes`` field.

    Notes are separated by `` | `` and form an append-only history
    of all ``_skip_me`` decisions made during the pipeline.
    """
    existing = task_data.get(notes_key, "")
    task_data[notes_key] = f"{existing} | {note}" if existing else note


def lang_code_to_name(code: str | None) -> str | None:
    """Convert an ISO language code to its full English name.

    Returns the code itself if no mapping is found, or ``None`` if
    the input is ``None`` or empty.
    """
    if not code:
        return None
    return LANG_CODE_TO_NAME.get(code, code)
