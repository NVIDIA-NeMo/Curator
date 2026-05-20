"""Stage 7 — Detailed Filter + Safety.

Currently provides:
    * :class:`InterleavedPIIRedactorStage` — regex-based PII redaction on
      text rows.  Each match becomes a ``[TYPE]`` placeholder so the
      surrounding prose is preserved (versus dropping the whole doc).

Conceptually corresponds to OmniCorpus §3.1 "Detailed Text Filtering"
and §3.3 human-feedback rules, but only the cheap regex slice is here —
BERT-ensemble safety scoring and the LDNOOBW-style bad-words pass live
elsewhere (BERT classifiers are a 📌 TODO).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from nemo_curator.stages.interleaved.stages import BaseInterleavedAnnotatorStage

if TYPE_CHECKING:
    from nemo_curator.tasks import InterleavedBatch


# Default regex set.  Patterns are intentionally conservative —
# better a missed false-negative than mangling URLs / version numbers.
_DEFAULT_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
    # US-format phone: optional country code + optional ()-wrapped area code.
    # Requires separators between groups so we don't redact long number runs.
    "PHONE": re.compile(
        r"(?<!\d)"                                   # no preceding digit
        r"(?:\+?1[\s.\-]?)?"                         # optional +1
        r"(?:\(\d{3}\)[\s.\-]?|\d{3}[\s.\-])"        # area code with separator
        r"\d{3}[\s.\-]\d{4}"                         # exchange + line
        r"(?!\d)"                                    # no trailing digit
    ),
    # IPv4 dotted-quad, with 0-255 octet validation.
    "IPV4": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    # US SSN — 3-2-4 with required dashes (digit-only runs are too risky).
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


@dataclass
class InterleavedPIIRedactorStage(BaseInterleavedAnnotatorStage):
    """Replace PII matches in text rows with ``[TYPE]`` placeholders.

    Operates on rows where ``modality == "text"``; leaves image and
    metadata rows untouched (image alt-text is currently *not* redacted
    — flip ``redact_alt_text`` to opt in).

    Per-batch log line summarises hit counts per pattern.

    Parameters
    ----------
    redact_email / redact_phone / redact_ipv4 / redact_ssn :
        Toggle individual patterns.  All on by default.
    extra_patterns :
        Map of ``{label: compiled-regex}``.  Each match is replaced with
        ``[<LABEL>]``.  Useful for project-specific PII (e.g. license keys).
    redact_alt_text :
        If ``True``, also redact PII inside image rows' ``text_content``
        (which stores alt-text).  Off by default — alt-text rarely
        contains PII and you usually want to keep it intact for VLM
        training.
    """

    redact_email: bool = True
    redact_phone: bool = True
    redact_ipv4: bool = True
    redact_ssn: bool = True
    redact_alt_text: bool = False
    extra_patterns: dict[str, re.Pattern[str]] = field(default_factory=dict)
    name: str = "interleaved_pii_redactor"

    def _active_patterns(self) -> dict[str, re.Pattern[str]]:
        on: dict[str, re.Pattern[str]] = {}
        if self.redact_email:
            on["EMAIL"] = _DEFAULT_PII_PATTERNS["EMAIL"]
        if self.redact_phone:
            on["PHONE"] = _DEFAULT_PII_PATTERNS["PHONE"]
        if self.redact_ipv4:
            on["IPV4"] = _DEFAULT_PII_PATTERNS["IPV4"]
        if self.redact_ssn:
            on["SSN"] = _DEFAULT_PII_PATTERNS["SSN"]
        on.update(self.extra_patterns)
        return on

    def annotate(
        self, task: InterleavedBatch, df: pd.DataFrame
    ) -> pd.DataFrame:
        if df.empty or "modality" not in df.columns:
            return df
        patterns = self._active_patterns()
        if not patterns:
            return df

        target_mods = {"text"}
        if self.redact_alt_text:
            target_mods.add("image")
        row_mask = df["modality"].isin(target_mods) & df["text_content"].notna()
        if not row_mask.any():
            return df

        counts: Counter = Counter()

        def redact_one(text: str) -> str:
            for label, pat in patterns.items():
                new, n = pat.subn(f"[{label}]", text)
                if n:
                    counts[label] += n
                    text = new
            return text

        # Apply only to the masked rows; keeps the column dtype as object.
        if df["text_content"].dtype != object:
            df["text_content"] = df["text_content"].astype(object)
        df.loc[row_mask, "text_content"] = (
            df.loc[row_mask, "text_content"].astype(str).map(redact_one)
        )

        if counts:
            total = sum(counts.values())
            summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            logger.info(
                f"[{self.name}] redacted {total} PII match(es) over "
                f"{int(row_mask.sum())} text rows  ·  {summary}"
            )
        return df
