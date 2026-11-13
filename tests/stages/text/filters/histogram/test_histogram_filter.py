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

from nemo_curator.stages.text.filters.histogram import HistogramFilter


def _make_filter(histogram: set[str], threshold: float = 0.8) -> HistogramFilter:
    """Build a HistogramFilter without running __init__.

    ``HistogramFilter.__init__`` downloads and parses histogram files from the
    network. For a unit test of the scoring logic we only need ``_histogram``
    and ``_threshold``, so we bypass __init__ and set them directly.
    """
    f = HistogramFilter.__new__(HistogramFilter)
    f._histogram = set(histogram)
    f._threshold = threshold
    f._name = "histogram"
    return f


def test_score_document_ignores_surrounding_whitespace() -> None:
    """Surrounding whitespace must not change the histogram verdict.

    ``score_document`` counts in-histogram characters over ``text.strip()`` but
    historically divided by ``len(text)`` (the un-stripped length). Leading or
    trailing whitespace therefore inflated the denominator and could flip an
    otherwise-kept document to rejected. The ratio must be computed over the
    same (stripped) text used for the numerator.
    """
    f = _make_filter(set("abc"))

    # Every content character is in the histogram -> ratio must be 1.0 (keep),
    # regardless of surrounding whitespace.
    assert f.score_document("abc") == 1
    assert f.score_document("  abc  ") == 1
    assert f.score_document("\n\tabc\n") == 1


def test_score_document_empty_text_does_not_raise() -> None:
    """Whitespace-only / empty input scores 0 instead of raising ZeroDivisionError."""
    f = _make_filter(set("abc"))
    assert f.score_document("") == 0
    assert f.score_document("   ") == 0


def test_score_document_below_threshold_is_rejected() -> None:
    """A document that is mostly out-of-histogram characters is still rejected."""
    f = _make_filter(set("abc"), threshold=0.8)
    # 3 of 6 content chars in histogram -> 0.5 <= 0.8 -> reject.
    assert f.score_document("abcxyz") == 0
