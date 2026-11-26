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

"""Split a long transcript into sentence windows for per-window LLM rewriting.

Windows are consecutive runs of whole words, so ``" ".join(windows)`` equals the
input with its whitespace collapsed to single spaces: nothing is dropped,
duplicated or reordered. Sentences are packed greedily up to ``target_words``;
a sentence longer than ``max_words`` (e.g. unpunctuated ASR output) is split at
clause punctuation, and failing that hard at ``max_words`` words.
"""

from __future__ import annotations

import re

# a word ending a sentence: terminal punctuation, optionally followed by closing quotes/brackets
_SENTENCE_END = re.compile(r"[.!?…。？！؟।][\"'”’)\]»]*$")  # noqa: RUF001
_CLAUSE_END = re.compile(r"[,;:،、，；：—-][\"'”’)\]»]*$")  # noqa: RUF001


def _split_after(words: list[str], pattern: re.Pattern[str]) -> list[list[str]]:
    parts, cur = [], []
    for w in words:
        cur.append(w)
        if pattern.search(w):
            parts.append(cur)
            cur = []
    if cur:
        parts.append(cur)
    return parts


def _bounded(sentence: list[str], max_words: int) -> list[list[str]]:
    """Split one over-long sentence at clause punctuation, then hard at ``max_words``."""
    if len(sentence) <= max_words:
        return [sentence]
    out: list[list[str]] = []
    cur: list[str] = []
    for clause in _split_after(sentence, _CLAUSE_END):
        if cur and len(cur) + len(clause) > max_words:
            out.append(cur)
            cur = []
        cur.extend(clause)
        while len(cur) > max_words:
            out.append(cur[:max_words])
            cur = cur[max_words:]
    if cur:
        out.append(cur)
    return out


def split_text_windows(text: str, target_words: int, max_words: int | None = None) -> list[str]:
    """Return sentence windows of about ``target_words`` words, none longer than ``max_words``."""
    if target_words < 1:
        msg = f"target_words must be >= 1, got {target_words}"
        raise ValueError(msg)
    max_words = max_words or max(target_words, round(target_words * 1.4))
    if max_words < target_words:
        msg = f"max_words ({max_words}) must be >= target_words ({target_words})"
        raise ValueError(msg)
    words = text.split()
    if not words:
        return []
    pieces = [p for s in _split_after(words, _SENTENCE_END) for p in _bounded(s, max_words)]
    windows: list[list[str]] = []
    cur: list[str] = []
    for piece in pieces:
        if cur and (len(cur) >= target_words or len(cur) + len(piece) > max_words):
            windows.append(cur)
            cur = []
        cur.extend(piece)
    if cur:
        windows.append(cur)
    return [" ".join(w) for w in windows]
