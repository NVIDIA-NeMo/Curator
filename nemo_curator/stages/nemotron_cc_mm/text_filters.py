"""Document-level text filters for ``InterleavedBatch``.

Each filter operates at the **sample level** — it aggregates all text rows
of a single ``sample_id`` and decides keep/drop for the whole sample.  The
``BaseInterleavedFilterStage`` machinery then re-positions surviving rows
and drops orphan metadata rows automatically.

Implements (Phase 1):
    * ``InterleavedLoremIpsumFilterStage``       — drop docs containing "lorem ipsum"
    * ``InterleavedWordCountFilterStage``         — Gopher word-count bounds
    * ``InterleavedMeanWordLengthFilterStage``    — Gopher mean-word-length bounds
    * ``InterleavedSymbolToWordRatioFilterStage`` — Gopher symbol-ratio cap
    * ``InterleavedStopwordCountFilterStage``     — Gopher English-stopword count
    * ``InterleavedNGramRepetitionFilterStage``   — Gopher top-n-gram repetition cap
"""
from __future__ import annotations

import re
from abc import abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from nemo_curator.stages.interleaved.stages import BaseInterleavedFilterStage

if TYPE_CHECKING:
    from nemo_curator.tasks import InterleavedBatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"\S+")


def aggregate_doc_text(group: pd.DataFrame) -> str:
    """Concatenate ``text_content`` over all text rows of a sample."""
    text_rows = group[group["modality"] == "text"]
    if text_rows.empty:
        return ""
    return "\n".join(text_rows["text_content"].dropna().astype(str).tolist())


def split_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
@dataclass
class LoggingInterleavedFilterStage(BaseInterleavedFilterStage):
    """``BaseInterleavedFilterStage`` with built-in drop-count logging.

    Subclasses still implement ``content_keep_mask`` (or further extend
    this class).  We override :meth:`process` to log per-batch row and
    document drop counts at ``INFO`` level, matching what the *next*
    stage will actually see (post-annotate, including orphan-metadata
    cleanup).  Toggle :attr:`log_drops` to silence.
    """

    name: str = "logging_interleaved_filter"
    log_drops: bool = True

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        df = task.to_pandas().copy()
        if df.empty:
            return task
        n_docs_in = df["sample_id"].nunique() if "sample_id" in df.columns else 0
        n_rows_in = len(df)
        out_df = self.annotate(task, df)
        n_docs_out = (
            out_df["sample_id"].nunique() if "sample_id" in out_df.columns else 0
        )
        n_rows_out = len(out_df)
        if self.log_drops and n_docs_in:
            n_docs_dropped = n_docs_in - n_docs_out
            n_rows_dropped = n_rows_in - n_rows_out
            doc_pct = n_docs_dropped / n_docs_in * 100.0
            row_pct = (n_rows_dropped / n_rows_in * 100.0) if n_rows_in else 0.0
            logger.info(
                f"[{self.name}] "
                f"docs {n_docs_in:>6} → {n_docs_out:>6} "
                f"(-{n_docs_dropped}, {doc_pct:5.1f}%)   "
                f"rows {n_rows_in:>7} → {n_rows_out:>7} "
                f"(-{n_rows_dropped}, {row_pct:5.1f}%)"
            )
        from nemo_curator.tasks import InterleavedBatch as _IB
        return _IB(
            task_id=f"{task.task_id}_{self.name}",
            dataset_name=task.dataset_name,
            data=out_df.reset_index(drop=True),
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


@dataclass
class BaseInterleavedSampleFilterStage(LoggingInterleavedFilterStage):
    """Drop whole samples based on a predicate.

    Subclasses implement ``is_sample_ok(sample_id, group) -> bool``.  The
    predicate may use ``aggregate_doc_text(group)`` to get the document's
    text content.  Inherits per-batch drop logging from
    :class:`LoggingInterleavedFilterStage`.
    """

    name: str = "base_interleaved_sample_filter"

    @abstractmethod
    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        """Return ``True`` to keep all rows of this sample, ``False`` to drop."""

    def content_keep_mask(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
        keep_mask = pd.Series(True, index=df.index, dtype=bool)
        if "sample_id" not in df.columns:
            return keep_mask
        sample_keep: dict[str, bool] = {}
        for sample_id, group in df.groupby("sample_id"):
            sample_keep[sample_id] = self.is_sample_ok(sample_id, group)
        return df["sample_id"].map(sample_keep).fillna(True).astype(bool)


# ---------------------------------------------------------------------------
# Concrete filters
# ---------------------------------------------------------------------------
@dataclass
class InterleavedLoremIpsumFilterStage(BaseInterleavedSampleFilterStage):
    """Drop docs whose aggregated text contains 'lorem ipsum' (C4 rule)."""

    needle: str = "lorem ipsum"
    name: str = "interleaved_lorem_ipsum_filter"

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        return self.needle not in aggregate_doc_text(group).lower()


@dataclass
class InterleavedWordCountFilterStage(BaseInterleavedSampleFilterStage):
    """Drop docs outside [min_words, max_words] (Gopher default 50–100 000)."""

    min_words: int = 50
    max_words: int = 100_000
    name: str = "interleaved_word_count_filter"

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        n = len(split_words(aggregate_doc_text(group)))
        return self.min_words <= n <= self.max_words


@dataclass
class InterleavedMeanWordLengthFilterStage(BaseInterleavedSampleFilterStage):
    """Drop docs outside [min_len, max_len] mean word length (Gopher 3–10)."""

    min_len: float = 3.0
    max_len: float = 10.0
    name: str = "interleaved_mean_word_length_filter"

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        words = split_words(aggregate_doc_text(group))
        if not words:
            return False
        mean = sum(len(w) for w in words) / len(words)
        return self.min_len <= mean <= self.max_len


@dataclass
class InterleavedSymbolToWordRatioFilterStage(BaseInterleavedSampleFilterStage):
    """Drop docs whose symbol-to-word ratio exceeds threshold (Gopher 0.1).

    Symbols counted: ``#``, ``…`` (U+2026 + ASCII three-dot), and ``&``.
    """

    max_ratio: float = 0.1
    name: str = "interleaved_symbol_to_word_ratio_filter"

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        text = aggregate_doc_text(group)
        words = split_words(text)
        if not words:
            return False
        symbols = (
            text.count("#")
            + text.count("…")
            + text.count("...")
            + text.count("&")
        )
        return (symbols / len(words)) <= self.max_ratio


# Gopher / MassiveText English stopword set (the canonical 8 words).
_GOPHER_STOPWORDS: frozenset[str] = frozenset(
    {"the", "be", "to", "of", "and", "that", "have", "with"}
)


@dataclass
class InterleavedStopwordCountFilterStage(BaseInterleavedSampleFilterStage):
    """Drop docs lacking at least ``min_distinct`` of the Gopher stop-word set.

    English-specific by design (Gopher Table A.1 row 8).  Default cutoff
    = 2 distinct stopwords present in the doc.
    """

    min_distinct: int = 2
    stopwords: frozenset[str] = _GOPHER_STOPWORDS
    name: str = "interleaved_stopword_count_filter"

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        words_lower = {w.lower() for w in split_words(aggregate_doc_text(group))}
        return len(words_lower & self.stopwords) >= self.min_distinct


@dataclass
class InterleavedNGramRepetitionFilterStage(BaseInterleavedSampleFilterStage):
    """Drop docs whose top n-gram covers too large a fraction of the text.

    Implements the Gopher 'top-2/3/4-gram fraction' rule.  Default cutoffs
    from Rae et al. 2022 §A.1.1:
        * 2-gram: < 0.20
        * 3-gram: < 0.18
        * 4-gram: < 0.16

    A doc fails if *any* configured n's top-fraction exceeds the bound.
    """

    bounds: tuple[tuple[int, float], ...] = (
        (2, 0.20),
        (3, 0.18),
        (4, 0.16),
    )
    name: str = "interleaved_ngram_repetition_filter"

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        words = split_words(aggregate_doc_text(group))
        if not words:
            return False
        for n, bound in self.bounds:
            if len(words) < n:
                # Too short for this n; can't fail this check but also
                # the word-count filter will likely reject anyway.
                continue
            grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
            top_count = Counter(grams).most_common(1)[0][1]
            # Fraction of *words* the top n-gram covers.
            fraction = (top_count * n) / len(words)
            if fraction > bound:
                return False
        return True


# ---------------------------------------------------------------------------
# Phase-1 finishing filters — close the remaining Gopher / OmniCorpus gaps
# ---------------------------------------------------------------------------
_ALPHA_RE = re.compile(r"[A-Za-z]")
_BAD_WORD_TOKEN_RE = re.compile(r"[a-z]+")
_BULLET_PREFIXES: tuple[str, ...] = (
    "-", "*", "•", "·", "‣", "▪", "◦", "–", "—",
)


def _split_lines(text: str) -> list[str]:
    """Strip + drop empty lines.  Used by the line-ratio filters."""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _is_alphabetic_word(w: str) -> bool:
    return bool(_ALPHA_RE.search(w))


@dataclass
class InterleavedAlphabeticWordRatioFilterStage(BaseInterleavedSampleFilterStage):
    """Drop docs where fewer than ``min_ratio`` of words contain a letter.

    Gopher §A.1.1.  Default 0.8 — catches numeric/symbolic dumps
    (log files, price tables, error pages).
    """

    min_ratio: float = 0.8
    name: str = "interleaved_alphabetic_word_ratio_filter"

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        words = split_words(aggregate_doc_text(group))
        if not words:
            return False
        n_alpha = sum(1 for w in words if _is_alphabetic_word(w))
        return (n_alpha / len(words)) >= self.min_ratio


@dataclass
class InterleavedEllipsisLineRatioFilterStage(BaseInterleavedSampleFilterStage):
    """Drop docs where too many lines end with an ellipsis.

    Gopher §A.1.1.  Default 0.3 — catches listicle / TLDR pages where
    most lines are "Read more…" truncations.
    """

    max_ratio: float = 0.3
    name: str = "interleaved_ellipsis_line_ratio_filter"

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        lines = _split_lines(aggregate_doc_text(group))
        if not lines:
            return False
        n_ellipsis = sum(
            1 for ln in lines if ln.endswith("…") or ln.endswith("...")
        )
        return (n_ellipsis / len(lines)) <= self.max_ratio


@dataclass
class InterleavedBulletLineRatioFilterStage(BaseInterleavedSampleFilterStage):
    """Drop docs where too many lines start with a bullet character.

    Gopher §A.1.1.  Default 0.9 — catches TOCs, link directories,
    autogenerated indexes.
    """

    max_ratio: float = 0.9
    name: str = "interleaved_bullet_line_ratio_filter"

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        lines = _split_lines(aggregate_doc_text(group))
        if not lines:
            return False
        n_bullet = sum(1 for ln in lines if ln.startswith(_BULLET_PREFIXES))
        return (n_bullet / len(lines)) <= self.max_ratio


@dataclass
class InterleavedDuplicateLineRatioFilterStage(BaseInterleavedSampleFilterStage):
    """Drop docs where too many lines are exact duplicates of another line.

    MassiveText §A.1.1.  Default 0.3 — catches template pages, login walls,
    redirect stubs, server-error pages.
    """

    max_ratio: float = 0.3
    name: str = "interleaved_duplicate_line_ratio_filter"

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        lines = _split_lines(aggregate_doc_text(group))
        n = len(lines)
        if n < 2:
            return True  # too short to assess
        unique = len(set(lines))
        dup_ratio = 1.0 - (unique / n)
        return dup_ratio <= self.max_ratio


@dataclass
class InterleavedTopWordFractionFilterStage(BaseInterleavedSampleFilterStage):
    """Drop docs where a single (case-insensitive) word covers too much text.

    OmniCorpus §3.1 cites this as "removing documents where a single word's
    frequency is excessively high".  Our n-gram filter (n ≥ 2) already
    catches *consecutive* repetition; this one catches *dispersed*
    repetition where the same word recurs with different neighbours
    (e.g. "click here click there click anywhere click …").

    Default 0.30 — in natural English the most-frequent word ("the")
    covers ~5–7 %, so 0.30 only fires on clearly pathological docs.
    """

    max_ratio: float = 0.30
    name: str = "interleaved_top_word_fraction_filter"

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        words = [w.lower() for w in split_words(aggregate_doc_text(group))]
        if not words:
            return False
        top_count = Counter(words).most_common(1)[0][1]
        return (top_count / len(words)) <= self.max_ratio


@dataclass
class InterleavedContinuousLineBreaksFilterStage(BaseInterleavedSampleFilterStage):
    """Drop docs with too many runs of consecutive blank lines.

    OmniCorpus §3.1 cites this as "removing documents with too many
    continuous line breaks".  In our pipeline this is effectively a
    no-op because the DOM walker strips internal whitespace and joins
    rows with a single ``\\n`` — but it's wired up so a future
    paragraph-preserving extractor can opt into the filter without
    code changes.  Counts ``\\n{3,}`` (3+ consecutive newlines)
    against total non-empty lines.
    """

    max_ratio: float = 0.05
    name: str = "interleaved_continuous_line_breaks_filter"

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        text = aggregate_doc_text(group)
        lines = _split_lines(text)
        if not lines:
            return False
        blank_runs = len(re.findall(r"\n{3,}", text))
        return (blank_runs / len(lines)) <= self.max_ratio


@dataclass
class InterleavedBadWordsFilterStage(BaseInterleavedSampleFilterStage):
    """Drop docs that contain any term from a bad-words wordlist.

    ``wordlist_path`` should be a UTF-8 file with one term per line
    (e.g. LDNOOBW: github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-
    Otherwise-Bad-Words).  Lines starting with ``#`` are treated as
    comments.  Match is whole-token (lowercased) so "Scunthorpe" does
    not match "cunt".

    If ``wordlist_path`` is empty (default), the filter is a no-op.
    """

    wordlist_path: str = ""
    name: str = "interleaved_bad_words_filter"
    _words: frozenset[str] | None = field(default=None, init=False, repr=False)

    def _ensure_loaded(self) -> None:
        if self._words is not None:
            return
        if not self.wordlist_path:
            self._words = frozenset()
            return
        try:
            with open(self.wordlist_path, encoding="utf-8") as f:
                self._words = frozenset(
                    line.strip().lower()
                    for line in f
                    if line.strip() and not line.startswith("#")
                )
            logger.info(
                f"[{self.name}] loaded {len(self._words)} terms "
                f"from {self.wordlist_path}"
            )
        except OSError as e:
            logger.warning(
                f"[{self.name}] could not load wordlist {self.wordlist_path!r}: "
                f"{e}; filter disabled"
            )
            self._words = frozenset()

    def is_sample_ok(self, sample_id: str, group: pd.DataFrame) -> bool:
        self._ensure_loaded()
        if not self._words:
            return True
        tokens = set(_BAD_WORD_TOKEN_RE.findall(aggregate_doc_text(group).lower()))
        return not (tokens & self._words)
