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

"""Small text helpers shared by the NumPipe parser, generator and stages."""

import re

# Homophones of spoken digit words that ASR models commonly emit instead of the digit word.
# Applied token-wise, so it is only safe on audio that contains nothing but a number.
DIGIT_HOMOPHONES = {
    "to": "two",
    "too": "two",
    "for": "four",
    "fore": "four",
    "oh": "zero",
    "o": "zero",
}


def digits_only(text: str) -> str:
    """Keep only ASCII digits."""
    return "".join(ch for ch in text if ch.isdigit())


def normalize_text(text: str) -> str:
    """Lower-case, replace punctuation with spaces, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def apply_homophones(text: str) -> str:
    """Map digit-word homophones token by token (``"for two oh"`` -> ``"four two zero"``)."""
    return " ".join(DIGIT_HOMOPHONES.get(tok, tok) for tok in text.split())
