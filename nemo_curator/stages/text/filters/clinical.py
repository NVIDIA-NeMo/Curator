# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Quality filter for structural integrity of clinical report documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nemo_curator.stages.text.filters.doc_filter import DocumentFilter

# Standard clinical report section headers per language.
# Kept intentionally small: these are the structural headers shared
# by most reporting traditions (SOAP-like), not domain terminology.
_CLINICAL_SECTIONS: dict[str, tuple[str, ...]] = {
    "en": (
        "history",
        "chief complaint",
        "physical examination",
        "assessment",
        "diagnosis",
        "plan",
        "treatment",
        "findings",
        "impression",
    ),
    "it": (
        "anamnesi",
        "esame obiettivo",
        "diagnosi",
        "terapia",
        "conclusioni",
        "referto",
    ),
    "es": (
        "anamnesis",
        "exploración física",
        "diagnóstico",
        "tratamiento",
        "conclusiones",
        "antecedentes",
    ),
    "fr": (
        "anamnèse",
        "examen clinique",
        "diagnostic",
        "traitement",
        "conclusion",
        "antécédents",
    ),
    "de": (
        "anamnese",
        "körperliche untersuchung",
        "diagnose",
        "therapie",
        "beurteilung",
        "befund",
    ),
}

_MIN_SECTIONS_FLOOR = 1


@dataclass
class ClinicalSectionFilter(DocumentFilter):
    """Filter clinical documents by structural section integrity.

    Clinical reports follow a shared section structure (anamnesis,
    examination, diagnosis, treatment). Documents lacking a minimum
    number of distinct sections are typically fragmented, truncated,
    or not clinical reports at all, and are usually undesirable for
    training on clinical text.

    Duplicate mentions of the same section count once, so repetitive
    boilerplate does not inflate the score.

    Attributes:
        language: ISO-like language code selecting the section keyword
            set. One of "en", "it", "es", "fr", "de".
        min_sections: Minimum number of distinct sections required for
            a document to be kept.

    Example:
        Use with ScoreFilter to log scores as metadata for threshold
        analysis before committing to a cutoff::

            ScoreFilter(
                score_fn=ClinicalSectionFilter(language="it", min_sections=2),
                text_field="text",
                score_field="clinical_sections",
            )
    """

    language: str = "en"
    min_sections: int = 2
    _pattern: re.Pattern[str] = field(init=False, repr=False)
    _order: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and compile the section-matching pattern."""
        super().__init__()
        if self.language not in _CLINICAL_SECTIONS:
            msg = f"Unsupported language '{self.language}'. Supported: {sorted(_CLINICAL_SECTIONS)}"
            raise ValueError(msg)
        if self.min_sections < _MIN_SECTIONS_FLOOR:
            msg = f"min_sections must be >= {_MIN_SECTIONS_FLOOR}."
            raise ValueError(msg)

        self._name = "clinical_section"

        sections = _CLINICAL_SECTIONS[self.language]
        escaped = "|".join(re.escape(s) for s in sections)
        self._pattern = re.compile(rf"\b({escaped})\b", re.IGNORECASE)
        self._order = {s: i for i, s in enumerate(sections)}

    def score_document(self, text: str) -> int:
        """Count distinct clinical section headers present in the text.

        Args:
            text: The document text to score.

        Returns:
            The number of distinct section keywords matched (0 if none).
        """
        matches = {m.group(0).lower() for m in self._pattern.finditer(text)}
        return len(matches)

    def keep_document(self, score: int) -> bool:
        """Decide whether a document meets the minimum section count.

        Args:
            score: The value returned by `score_document`.

        Returns:
            True if `score` meets or exceeds `min_sections`.
        """
        return score >= self.min_sections
