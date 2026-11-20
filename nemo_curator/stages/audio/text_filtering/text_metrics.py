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

"""Dependency-free text error-rate helpers for lightweight filtering stages."""

import re
from collections.abc import Sequence
from typing import TypeVar

_Token = TypeVar("_Token")
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_for_error_rate(text: str) -> str:
    """Lowercase and drop punctuation: the light text normalization for WER/CER comparisons."""
    return _PUNCTUATION.sub("", text).lower()


def edit_operation_counts(reference: Sequence[_Token], hypothesis: Sequence[_Token]) -> tuple[int, int, int]:
    """Return ``(substitutions, insertions, deletions)`` of a minimal alignment of ``hypothesis`` to ``reference``.

    Ties prefer insertion, then deletion, then match/substitution, which reproduces the
    breakdown NeMo's ``word_error_rate_detail`` (kaldialign) reports.
    """
    previous = [(j, 0, j, 0) for j in range(len(hypothesis) + 1)]  # (total, sub, ins, del)
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [(ref_index, 0, 0, ref_index)]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            left, up, diag = current[hyp_index - 1], previous[hyp_index], previous[hyp_index - 1]
            miss = ref_token != hyp_token
            best = (left[0] + 1, left[1], left[2] + 1, left[3])
            if up[0] + 1 < best[0]:
                best = (up[0] + 1, up[1], up[2], up[3] + 1)
            if diag[0] + miss < best[0]:
                best = (diag[0] + miss, diag[1] + miss, diag[2], diag[3])
            current.append(best)
        previous = current
    return previous[-1][1:]


def _levenshtein_distance(reference: Sequence[_Token], hypothesis: Sequence[_Token]) -> int:
    return sum(edit_operation_counts(reference, hypothesis))


def _error_rate_percent(reference: Sequence[_Token], hypothesis: Sequence[_Token]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 100.0
    return round(_levenshtein_distance(reference, hypothesis) / len(reference) * 100.0, 2)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return unrounded whitespace-token WER as a fraction; ``inf`` for an empty reference (NeMo convention)."""
    reference_tokens = reference.split()
    if not reference_tokens:
        return float("inf")
    return _levenshtein_distance(reference_tokens, hypothesis.split()) / len(reference_tokens)


def word_error_rate_percent(reference: str, hypothesis: str) -> float:
    """Return whitespace-token word error rate as a percentage."""
    return _error_rate_percent(reference.split(), hypothesis.split())


def character_error_rate_percent(reference: str, hypothesis: str) -> float:
    """Return Unicode code-point character error rate as a percentage."""
    return _error_rate_percent(reference, hypothesis)
