#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detect high-probability glued OOV tokens and write glued->unglued mappings.

This is a dependency-free, offline alternative to ``detect_glued_oov_llm.py``: it
finds OOV tokens that are almost certainly two or three real words concatenated
during normalization (e.g. ``abandonedand`` -> ``abandoned and``) and writes a
high-confidence ``word<TAB>replacement`` mapping compatible with
``david_ai_glued_words.load_unglue_repairs``.

Precision comes from a cost-based dictionary segmentation plus strict acceptance
rules: every part must be a real dictionary word, short parts must be common
function words, and at least one part must be a real content word. Confidence is
assigned per segmentation pattern and only high-confidence mappings are written.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from david_ai_common import PipelineError, run_main

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Common English function / glue words. These are the words that most often get
# concatenated to a neighbour during transcript normalization, and they are the
# only tokens allowed to appear as short (<4 char) parts of a split.
GLUE_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "so", "if", "as", "at", "by",
        "for", "in", "of", "off", "on", "out", "to", "up", "with", "from",
        "into", "onto", "over", "than", "then", "that", "this", "these",
        "those", "there", "here", "where", "when", "what", "who", "how",
        "why", "not", "no", "yes", "yeah", "yep", "nope", "okay", "ok",
        "i", "you", "your", "yours", "he", "him", "his", "she", "her", "it",
        "its", "we", "us", "our", "they", "them", "their", "me", "my", "mine",
        "am", "is", "are", "was", "were", "be", "been", "being", "do", "does",
        "did", "done", "have", "has", "had", "will", "would", "can", "could",
        "should", "shall", "may", "might", "must", "get", "got", "go", "goes",
        "just", "like", "well", "now", "really", "very", "too", "also", "even",
        "still", "again", "back", "down", "because", "about", "after", "before",
        "please", "thanks", "thank", "right", "left", "all", "some", "any",
        "more", "most", "much", "many", "one", "two", "three", "let", "lets",
        "gonna", "wanna", "kinda", "sorta", "guys", "guy", "man", "oh", "uh",
        "um", "hey", "hi", "yo", "wow", "hmm",
    }
)

# Valid standalone short words (len <= 3) that may appear in a split. Kept tight
# on purpose to avoid coincidental fragments (e.g. splitting into rare bigrams).
COMMON_SHORT: frozenset[str] = GLUE_WORDS | frozenset(
    {
        "sun", "day", "way", "man", "men", "car", "dog", "cat", "boy", "girl",
        "job", "run", "big", "old", "new", "bad", "top", "end", "war", "law",
        "art", "map", "cup", "box", "key", "eye", "ear", "arm", "leg", "bed",
        "bit", "lot", "few", "own", "use", "add", "cut", "put", "set", "see",
        "say", "buy", "pay", "win", "fun", "sad", "mad", "god", "sir", "mom",
        "dad", "kid", "guy", "gun", "bar", "air", "sea", "sky", "ice", "fire",
    }
)

# Cost model constants for the segmentation DP. Lower total cost is better.
_WORD_PENALTY = 8.0  # discourages over-splitting into many tiny parts
_COST_COMMON = 1.0
_COST_LONG = 5.0  # dictionary word, len >= 6
_COST_MED = 8.0  # dictionary word, len 4-5
_COST_SHORT = 12.0  # allowed short word (in COMMON_SHORT)
_INF = float("inf")

_MIN_TOKEN_LEN = 4
_MAX_PARTS = 3


def load_dictionary_words(path: Path) -> set[str]:
    words: set[str] = set()
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            token = line.split("\t", 1)[0].strip().lower()
            if token:
                words.add(token)
    return words


def load_oov_words(path: Path) -> list[str]:
    words: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        token = line.strip()
        if token:
            words.append(token)
    return words


def is_candidate_token(token: str) -> bool:
    """Only plain alphabetic OOV tokens are considered for un-gluing."""
    if len(token) < _MIN_TOKEN_LEN:
        return False
    # Hyphenated tokens already carry an explicit word boundary; apostrophe
    # tokens are contractions / possessives, not space-glued words.
    return token.isalpha()


def _part_cost(part: str, dictionary: set[str]) -> float:
    if part in GLUE_WORDS:
        return _COST_COMMON
    if len(part) < _MIN_TOKEN_LEN:
        if part in COMMON_SHORT:
            return _COST_SHORT
        return _INF
    if part not in dictionary:
        return _INF
    return _COST_LONG if len(part) >= 6 else _COST_MED


def best_segmentation(token: str, dictionary: set[str]) -> list[str] | None:
    """Return the minimum-cost dictionary segmentation into <= _MAX_PARTS parts."""
    n = len(token)
    # dp[i] = (cost, parts) for best segmentation of token[:i]
    dp: list[tuple[float, list[str]] | None] = [None] * (n + 1)
    dp[0] = (0.0, [])
    for i in range(n):
        if dp[i] is None:
            continue
        cur_cost, cur_parts = dp[i]
        if len(cur_parts) >= _MAX_PARTS:
            continue
        for j in range(i + 1, n + 1):
            part = token[i:j]
            pc = _part_cost(part, dictionary)
            if pc == _INF:
                continue
            new_cost = cur_cost + pc + _WORD_PENALTY
            if dp[j] is None or new_cost < dp[j][0]:
                dp[j] = (new_cost, [*cur_parts, part])
    result = dp[n]
    if result is None:
        return None
    parts = result[1]
    if len(parts) < 2:
        return None
    return parts


# A content word must be this long to be trusted in a two-content-word split.
# The MFA dictionary is corpus-derived and contains many short junk fragments,
# so short "dictionary" parts (e.g. "funda", "gness") coincide far too easily.
_MIN_CONTENT_LEN_2WORD = 6


def score_split(parts: list[str]) -> tuple[float, str]:
    """Assign a confidence and a human-readable pattern label to a split."""
    k = len(parts)
    content = [p for p in parts if len(p) >= _MIN_TOKEN_LEN and p not in GLUE_WORDS]
    glue = [p for p in parts if p in GLUE_WORDS]
    short_non_glue = [p for p in parts if len(p) < _MIN_TOKEN_LEN and p not in GLUE_WORDS]

    # Require at least one real content word so we never emit pure function-word
    # noise like "i am" (also a common false split of names/acronyms).
    if not content:
        return 0.0, "no_content"
    # Short non-glue fragments are the main source of coincidental splits.
    if short_non_glue:
        return 0.0, "rare_short_fragment"

    if k == 2:
        # Highest precision: one curated function word glued to one real word.
        if len(glue) == 1 and len(content) == 1:
            glue_len = len(glue[0])
            content_len = len(content[0])
            # A 1-2 letter function word (a, i, an, is, it, ...) glued to a short
            # 4-letter fragment is usually a coincidental split of a name or
            # foreign word (aabba -> "a abba", andele -> "an dele"). Demand a
            # longer content word in that case.
            if glue_len <= 2 and content_len < 5:
                return 0.80, "glue_word_short"
            return 0.97, "glue_word"
        if len(content) == 2:
            if all(len(p) >= _MIN_CONTENT_LEN_2WORD for p in content):
                return 0.92, "two_content_words"
            # Short content parts are unreliable given the noisy dictionary; keep
            # them in the report but below the default mapping threshold.
            return 0.80, "two_content_words_short"
    # Three-part splits are dominated by over-segmented single words
    # (e.g. "ulcerative" -> "ulcer a tive"); report only, never map.
    if k == 3:
        return 0.60, "three_part"
    return 0.50, "other"


def detect_glued(
    oov_words: list[str],
    dictionary: set[str],
) -> list[tuple[str, str, float, str, int]]:
    """Return rows of (word, replacement, confidence, pattern, num_parts).

    All valid segmentations (confidence > 0) are returned; callers filter by
    confidence when writing the high-confidence mapping.
    """
    rows: list[tuple[str, str, float, str, int]] = []
    seen: set[str] = set()
    for token in oov_words:
        if token in seen:
            continue
        seen.add(token)
        lower = token.casefold()
        if not is_candidate_token(lower) or lower in dictionary:
            continue
        parts = best_segmentation(lower, dictionary)
        if not parts:
            continue
        confidence, pattern = score_split(parts)
        if confidence <= 0.0:
            continue
        rows.append((token, " ".join(parts), confidence, pattern, len(parts)))
    rows.sort(key=lambda r: (-r[2], r[0]))
    return rows


def write_report(path: Path, rows: list[tuple[str, str, float, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["word\treplacement\tconfidence\tpattern\tnum_parts"]
    for word, replacement, confidence, pattern, num_parts in rows:
        lines.append(f"{word}\t{replacement}\t{confidence:.2f}\t{pattern}\t{num_parts}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mapping(
    path: Path,
    rows: list[tuple[str, str, float, str, int]],
    *,
    min_confidence: float,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping = {
        word: replacement
        for word, replacement, confidence, *_ in rows
        if confidence >= min_confidence and replacement != word
    }
    lines = [f"{word}\t{replacement}" for word, replacement in sorted(mapping.items())]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(mapping)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--oov-list",
        type=Path,
        default=Path("workdir/lexicon/oov_words.txt"),
        help="OOV word list (one token per line)",
    )
    ap.add_argument(
        "--dict",
        type=Path,
        default=Path("workdir/lexicon/english_mfa_davidai_eng.dict"),
        help="MFA dictionary used as the valid-word vocabulary",
    )
    ap.add_argument(
        "--report-out",
        type=Path,
        default=Path("workdir/lexicon/unglue_heuristic_report.tsv"),
        help="Full report of detected glued tokens with confidence/pattern",
    )
    ap.add_argument(
        "--mapping-out",
        type=Path,
        default=Path("workdir/lexicon/unglue_repairs_heuristic.tsv"),
        help="High-confidence glued->unglued mapping (word<TAB>replacement)",
    )
    ap.add_argument("--min-confidence", type=float, default=0.90)
    args = ap.parse_args()

    oov_path = args.oov_list.resolve()
    dict_path = args.dict.resolve()
    if not oov_path.is_file():
        raise PipelineError(f"Missing OOV list: {oov_path}")
    if not dict_path.is_file():
        raise PipelineError(f"Missing dictionary: {dict_path}")

    logger.info("Loading dictionary from %s", dict_path)
    dictionary = load_dictionary_words(dict_path)
    logger.info("Dictionary words: %d", len(dictionary))

    oov_words = load_oov_words(oov_path)
    logger.info("OOV tokens: %d", len(oov_words))

    rows = detect_glued(oov_words, dictionary)
    write_report(args.report_out, rows)
    mapping_count = write_mapping(args.mapping_out, rows, min_confidence=args.min_confidence)

    by_pattern: dict[str, int] = {}
    for _, _, confidence, pattern, _ in rows:
        if confidence >= args.min_confidence:
            by_pattern[pattern] = by_pattern.get(pattern, 0) + 1
    logger.info("Detected %d candidate splits; %d >= %.2f: %s", len(rows), mapping_count, args.min_confidence, by_pattern)
    logger.info("Wrote report to %s", args.report_out)
    logger.info("Wrote %d high-confidence mappings to %s", mapping_count, args.mapping_out)
    return 0


if __name__ == "__main__":
    run_main(main)
