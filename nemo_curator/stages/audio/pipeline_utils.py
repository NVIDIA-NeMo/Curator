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

"""Shared helpers for Granary v2 audio pipelines."""

from __future__ import annotations

from typing import Any

NOTES_KEY = "additional_notes"

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


def set_note(task_data: dict[str, Any], stage_name: str, value: str, notes_key: str = NOTES_KEY) -> None:
    """Write a stage decision note into a per-task notes dictionary."""
    notes = task_data.get(notes_key)
    if not isinstance(notes, dict):
        notes = {}
        task_data[notes_key] = notes
    notes[stage_name] = value
