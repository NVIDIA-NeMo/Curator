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

import unicodedata

import pycld2 as cld2
from charset_normalizer import detect as charset_normalizer_detect


def remove_control_characters(text: str) -> str:
    """Remove control characters from text.
    Control characters are non-printable characters in the Unicode standard that control how text is displayed or processed.
    """
    return "".join(i for i in text if unicodedata.category(i)[0] != "C")


def detect_language(text: str) -> tuple[bool, int, list[tuple[str, str, float, int]]]:
    """Detect language using cld2.

    Returns:
        tuple[bool, int, list[tuple[str, str, float, int]]]:
        is_reliable: bool True if the detection is high confidence.
        textBytesFound: int The number of bytes of text found.
        details: list[tuple[str, str, float, int]] A list of tuples upto three detected languages containing the
            language name (str)
            language code (str)
            percent (float) what percentage of the text is in this language
            score (int) how confident the detection is.
    """
    return cld2.detect(text)


def lang_detect(text: str) -> str:
    """Detect language from text.

    Args:
        text (str): Text to detect language from.

    Returns:
        str: The most likely language code.
    """
    try:
        _, _, details = detect_language(text)
    except Exception:  # noqa: BLE001
        cleaned_text = remove_control_characters(text)
        _, _, details = detect_language(cleaned_text)
    # Get the most likely language and get the language name
    return details[0][0].upper()


def decode_html(html_bytes: bytes) -> str | None:
    try:
        # Convert from bytes to text using utf-8 encoding
        return html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # If utf-8 fails, try to find a different encoding
        return try_decode_with_detected_encoding(html_bytes)


def try_decode_with_detected_encoding(html_bytes: bytes) -> str | None:
    detected_encoding = charset_normalizer_detect(html_bytes)["encoding"]
    if not detected_encoding or detected_encoding == "utf-8":
        # This is a bad detection, return None
        return None
    try:
        return html_bytes.decode(detected_encoding)
    except Exception:  # noqa: BLE001
        return None
