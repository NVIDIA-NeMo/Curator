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

"""Detect and repair low-frequency glued OOV tokens."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

VALID_SUFFIXES = frozenset({"'s", "'re", "'ve", "'ll", "'d", "'m"})
VALID_SHORT = frozenset({"a", "i", "s"})


def separate_gluing_punctuation(text: str) -> str:
    """Insert spaces around punctuation, keeping apostrophes and hyphens word-internal."""
    text = re.sub(r"([^\w\s'-])", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def count_words_in_text(text: str) -> Counter[str]:
    text = (text or "").strip()
    if not text:
        return Counter()
    return Counter(text.split())


def merge_word_counters(counters: list[Counter[str]]) -> Counter[str]:
    merged: Counter[str] = Counter()
    for counter in counters:
        merged.update(counter)
    return merged


def _is_valid_part(part: str, dictionary: set[str]) -> bool:
    return part in dictionary or part in VALID_SUFFIXES or part in VALID_SHORT


def try_unglue_word(
    word: str,
    dictionary: set[str],
    *,
    min_part_len: int = 2,
    max_parts: int = 5,
) -> list[str] | None:
    token = word.casefold()
    if not token or token in dictionary:
        return None

    n = len(token)
    dp: list[list[str] | None] = [None] * (n + 1)
    dp[0] = []
    for i in range(n):
        if dp[i] is None:
            continue
        if len(dp[i]) >= max_parts:
            continue
        for j in range(i + 1, n + 1):
            part = token[i:j]
            if len(part) < min_part_len and part not in VALID_SUFFIXES and part not in VALID_SHORT:
                continue
            if not _is_valid_part(part, dictionary):
                continue
            candidate = dp[i] + [part]
            if len(candidate) > max_parts:
                continue
            if dp[j] is None or len(candidate) < len(dp[j]):
                dp[j] = candidate

    result = dp[n]
    if result and len(result) >= 2:
        return result
    return None


def build_unglue_repair_map(
    word_freq: Counter[str],
    dictionary: set[str],
    *,
    max_freq: int = 5,
) -> dict[str, str]:
    repairs: dict[str, str] = {}
    for word, freq in word_freq.items():
        if freq > max_freq or word in dictionary:
            continue
        split = try_unglue_word(word, dictionary)
        if split:
            repairs[word] = " ".join(split)
    return repairs


def apply_repair_map_to_text(text: str, repair_map: dict[str, str]) -> str:
    if not text or not repair_map:
        return text
    return " ".join(repair_map.get(token, token) for token in text.split())


def repair_punctuation_only(
    text_raw: str,
    text_norm: str,
    *,
    num2words_lang: str = "en",
) -> tuple[str, bool]:
    """Re-normalize from text_raw to fix punctuation-glued tokens."""
    from stage0_build_manifests import normalize_text

    raw = (text_raw or "").strip()
    norm = (text_norm or "").strip()
    if not raw:
        return norm, False
    repaired = normalize_text(raw, num2words_lang=num2words_lang)
    return repaired, repaired != norm


def repair_normalized_text_two_step(
    text_raw: str,
    text_norm: str,
    *,
    num2words_lang: str = "en",
    repair_map: dict[str, str] | None = None,
) -> tuple[str, bool, bool]:
    """Repair normalized text: punctuation via re-normalize, then frequency unglue."""
    repaired, punctuation_changed = repair_punctuation_only(
        text_raw,
        text_norm,
        num2words_lang=num2words_lang,
    )
    unglue_changed = False

    if repair_map:
        after_unglue = apply_repair_map_to_text(repaired, repair_map)
        unglue_changed = after_unglue != repaired
        repaired = after_unglue

    return repaired, punctuation_changed, unglue_changed


def apply_repair_map_to_counter(word_freq: Counter[str], repair_map: dict[str, str]) -> Counter[str]:
    if not repair_map:
        return word_freq
    repaired: Counter[str] = Counter()
    for word, freq in word_freq.items():
        replacement = repair_map.get(word)
        if replacement:
            for part in replacement.split():
                repaired[part] += freq
        else:
            repaired[word] += freq
    return repaired


def repaired_vocabulary(word_freq: Counter[str], repair_map: dict[str, str]) -> set[str]:
    vocab: set[str] = set()
    for word in word_freq:
        replacement = repair_map.get(word)
        if replacement:
            vocab.update(replacement.split())
        else:
            vocab.add(word)
    return vocab


def write_oov_frequency_report(
    path: Path,
    *,
    word_freq: Counter[str],
    dictionary: set[str],
    repair_map: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["word\tfreq\tin_dict\trepair_type\trepaired_to"]
    for word, freq in sorted(word_freq.items(), key=lambda item: (-item[1], item[0])):
        if word in dictionary:
            continue
        repaired = repair_map.get(word, "")
        repair_type = "frequency_unglue" if repaired else ""
        lines.append(f"{word}\t{freq}\tno\t{repair_type}\t{repaired}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_unglue_repairs(path: Path, repair_map: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{word}\t{replacement}" for word, replacement in sorted(repair_map.items())]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_unglue_repairs(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    repairs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("word\t"):
            continue
        if "\t" not in line:
            continue
        word, replacement = line.split("\t", 1)
        repairs[word] = replacement
    return repairs


_UNGLUE_REPAIR_CANDIDATES = (
    "unglue_repairs.tsv",
    "unglue_repairs_heuristic.tsv",
    "unglue_repairs_llm.tsv",
)


def resolve_unglue_repairs_path(
    lexicon_dir: Path,
    *,
    explicit: Path | str | None = None,
) -> Path | None:
    """Pick the repairs TSV to load (explicit path, env, or first match in lexicon_dir)."""
    import os

    if explicit is not None:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path.resolve()
    env_path = os.environ.get("UNGLUE_REPAIRS", "").strip()
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_file():
            return path.resolve()
    for name in _UNGLUE_REPAIR_CANDIDATES:
        path = lexicon_dir / name
        if path.is_file():
            return path.resolve()
    return None


def load_lexicon_unglue_repairs(
    lexicon_dir: Path,
    *,
    explicit: Path | str | None = None,
) -> tuple[dict[str, str], Path | None]:
    path = resolve_unglue_repairs_path(lexicon_dir, explicit=explicit)
    if path is None:
        return {}, None
    return load_unglue_repairs(path), path
