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

from collections.abc import Sequence
from typing import TypeVar

_Token = TypeVar("_Token")


def _levenshtein_distance(reference: Sequence[_Token], hypothesis: Sequence[_Token]) -> int:
    """Return the Levenshtein edit distance between two token sequences."""
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1]


def _error_rate_percent(reference: Sequence[_Token], hypothesis: Sequence[_Token]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 100.0
    return round(_levenshtein_distance(reference, hypothesis) / len(reference) * 100.0, 2)


def word_error_rate_percent(reference: str, hypothesis: str) -> float:
    """Return whitespace-token word error rate as a percentage."""
    return _error_rate_percent(reference.split(), hypothesis.split())


def character_error_rate_percent(reference: str, hypothesis: str) -> float:
    """Return Unicode code-point character error rate as a percentage."""
    return _error_rate_percent(reference, hypothesis)
