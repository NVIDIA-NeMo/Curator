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

"""Acoustic distractor generation stage for contextual ASR (CPU-only).

Appends phonetically-similar words/phrases to the ``distractor_terms``
list produced by :class:`ContextualASRExtractionStage`.  Semantic
distractors from the LLM teach the model not to copy hints blindly;
acoustic distractors additionally teach the model to disambiguate
phonetically-confusable words.

For each entity in ``fine_context_terms``:

1. G2P the full phrase via :mod:`phonemizer` (espeak-ng backend) into an
   IPA phoneme token list.
2. Compute Normalized Phonetic Distance (NPD)
   ``editdistance(query, candidate) / len(query)`` against every entry in
   a precomputed phoneme vocabulary loaded at ``setup()``.
3. Filter by ``min_npd < NPD < max_npd`` to drop both trivially-identical
   matches and too-distant ones.
4. Exclude vocabulary entries already present in ``fine_context_terms``
   or the existing (semantic) ``distractor_terms``.
5. Take the top ``per_entity_top_k`` candidates by ascending NPD.

Candidates collected across all entities of a sample are merged,
deduplicated, sorted by NPD, capped at ``max_acoustic_distractors``, and
appended to ``distractor_terms``.  The combined list is then capped at
``max_total_distractors``.

The precomputed phoneme vocabulary is produced offline by
``scripts/build_phoneme_vocab.py``.  Build it once per target language.

This stage is CPU-only — no GPU or LLM is required at runtime.

Language handling
-----------------
The stage needs an espeak-ng language code to G2P entities.  In order
of precedence:

1. ``language`` (if set on the stage) — used for all samples.
2. ``source_lang`` from the manifest (display name like ``"English"``
   or ISO-639-1 code like ``"en"``) — mapped to an espeak code.
3. ``default_source_lang`` — used when neither of the above resolves.

When a sample's language cannot be mapped to a supported espeak code,
the stage records ``unsupported_language`` in the additional_notes and
leaves the existing ``distractor_terms`` untouched.

Dependencies
------------
- ``phonemizer`` (``pip install phonemizer``)
- ``espeak-ng`` system package (``apt install espeak-ng``)
- ``editdistance`` (already a transitive project dependency)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

from nemo_curator.stages.audio.pipeline_utils import LANG_CODE_TO_NAME, set_note
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

try:
    from phonemizer import phonemize as _phonemize
    from phonemizer.separator import Separator as _Separator

    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False
    _phonemize = None  # type: ignore[assignment]
    _Separator = None  # type: ignore[assignment]

try:
    import editdistance as _editdistance

    EDITDISTANCE_AVAILABLE = True
except ImportError:
    EDITDISTANCE_AVAILABLE = False
    _editdistance = None  # type: ignore[assignment]


# Display-name → espeak-ng language code.  Covers the 32 languages
# supported by the design (see ``plan/acoustic_distractors_plan.md``).
_LANG_TO_ESPEAK: dict[str, str] = {
    "Arabic": "ar",
    "Bulgarian": "bg",
    "Chinese": "cmn",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en-us",
    "Estonian": "et",
    "Finnish": "fi",
    "French": "fr-fr",
    "German": "de",
    "Greek": "el",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Maltese": "mt",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish": "es",
    "Swedish": "sv",
    "Thai": "th",
    "Ukrainian": "uk",
}

_WORD_BOUNDARY_MARKER = "|"


def _normalize_lang_to_espeak(value: Any) -> str | None:  # noqa: ANN401
    """Map a manifest ``source_lang`` value to an espeak-ng language code.

    Accepts either a display name (``"English"``) or an ISO-639-1 code
    (``"en"``).  Returns ``None`` when no mapping is known.
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in _LANG_TO_ESPEAK:
        return _LANG_TO_ESPEAK[text]
    title = text.title()
    if title in _LANG_TO_ESPEAK:
        return _LANG_TO_ESPEAK[title]
    display = LANG_CODE_TO_NAME.get(text.lower())
    if display and display in _LANG_TO_ESPEAK:
        return _LANG_TO_ESPEAK[display]
    return None


def _phonemize_one(text: str, language: str) -> list[str]:
    """G2P a single string into a flat list of IPA phoneme tokens."""
    if not text or _phonemize is None or _Separator is None:
        return []
    sep = _Separator(phone=" ", word=f" {_WORD_BOUNDARY_MARKER} ", syllable="")
    ipa = _phonemize(
        text,
        backend="espeak",
        language=language,
        separator=sep,
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
        njobs=1,
    )
    if isinstance(ipa, list):
        ipa = ipa[0] if ipa else ""
    return [tok for tok in ipa.split() if tok and tok != _WORD_BOUNDARY_MARKER]


def _npd(query: list[str], candidate: list[str]) -> float:
    if not query:
        return 1.0
    if _editdistance is None:
        return 1.0
    return _editdistance.eval(query, candidate) / len(query)


def _vocab_search(  # noqa: PLR0913
    query_phonemes: list[str],
    vocab_items: list[tuple[str, list[str]]],
    *,
    min_npd: float,
    max_npd: float,
    excluded_words: set[str],
    top_k: int,
) -> list[tuple[str, float]]:
    """Brute-force nearest-neighbor search in the precomputed phoneme vocab.

    Returns up to ``top_k`` ``(word, npd)`` pairs with the smallest NPD
    satisfying ``min_npd < npd < max_npd``.  Words in ``excluded_words``
    (case-insensitive match on the vocab key) are skipped.
    """
    if not query_phonemes or top_k <= 0:
        return []
    q_len = len(query_phonemes)
    len_lo = max(1, int(q_len * (1.0 - max_npd)))
    len_hi = max(1, int(q_len * (1.0 + max_npd)) + 1)

    hits: list[tuple[str, float]] = []
    for word, phonemes in vocab_items:
        if word in excluded_words:
            continue
        if not (len_lo <= len(phonemes) <= len_hi):
            continue
        d = _npd(query_phonemes, phonemes)
        if min_npd < d < max_npd:
            hits.append((word, d))

    hits.sort(key=lambda kv: kv[1])
    return hits[:top_k]


@dataclass
class AcousticDistractorStage(ProcessingStage[AudioTask, AudioTask]):
    """Append phonetically-similar distractors to ``context_asr.distractor_terms``.

    CPU-only stage.  Loads a precomputed phoneme vocabulary at
    ``setup()`` and, for each task, G2Ps every entity in
    ``fine_context_terms``, searches the vocab for phonetically similar
    words by Normalized Phonetic Distance (NPD), and appends the top
    candidates to ``distractor_terms`` (capped at
    ``max_total_distractors``).

    On samples with no extraction dict, an empty entity list, or an
    unsupported language, the stage is a no-op (existing
    ``distractor_terms`` are preserved).  A per-stage note is written
    via :func:`set_note`.

    Args:
        context_key: Manifest key holding the extraction dict produced
            by :class:`ContextualASRExtractionStage`.
        source_lang_key: Manifest key holding the per-sample source
            language.  Accepts display names (``"English"``) or ISO
            codes (``"en"``).
        default_source_lang: Fallback used when ``source_lang_key`` is
            missing or empty on a sample.
        language: Optional explicit espeak-ng code (e.g. ``"en-us"``).
            When set, used for all samples and the per-sample
            ``source_lang`` is ignored.  Use this when you know the
            entire dataset is a single language.
        phoneme_vocab_path: Path to the JSON file produced by
            ``scripts/build_phoneme_vocab.py``.  Required.
        max_acoustic_distractors: Maximum acoustic distractors appended
            per sample (combined across all entities of that sample).
        max_total_distractors: Cap on the combined ``distractor_terms``
            list (semantic + acoustic) after merging.
        per_entity_top_k: Top-K candidates retained per source entity
            before cross-entity merging.
        min_npd: Lower NPD bound — entries closer than this are
            considered too-similar (typically the entity itself or a
            near-duplicate).
        max_npd: Upper NPD bound — entries farther than this are
            considered acoustically unrelated.
        notes_key: Key holding the ``additional_notes`` dict that
            :func:`set_note` writes into.
        num_workers_override: Explicit worker count for Xenna.
    """

    name: str = "AcousticDistractor"
    context_key: str = "context_asr"
    source_lang_key: str = "source_lang"
    default_source_lang: str = "English"
    language: str | None = None
    phoneme_vocab_path: str = ""
    max_acoustic_distractors: int = 8
    max_total_distractors: int = 16
    per_entity_top_k: int = 3
    min_npd: float = 0.1
    max_npd: float = 0.5
    notes_key: str = "additional_notes"
    num_workers_override: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 256

    _vocab_items: list[tuple[str, list[str]]] = field(default_factory=list, init=False, repr=False)
    _g2p_cache: dict[tuple[str, str], list[str]] = field(default_factory=dict, init=False, repr=False)
    _n_processed: int = field(default=0, init=False, repr=False)
    _n_appended: int = field(default=0, init=False, repr=False)

    def num_workers(self) -> int | None:
        return self.num_workers_override

    def xenna_stage_spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if self.num_workers_override is not None:
            spec["num_workers"] = self.num_workers_override
        return spec

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        pass

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if not PHONEMIZER_AVAILABLE:
            msg = (
                "phonemizer is required for AcousticDistractorStage. "
                "Install it with `pip install phonemizer` and the espeak-ng system package."
            )
            raise ImportError(msg)
        if not EDITDISTANCE_AVAILABLE:
            msg = "editdistance is required for AcousticDistractorStage. `pip install editdistance`."
            raise ImportError(msg)
        if not self.phoneme_vocab_path:
            msg = "AcousticDistractorStage requires phoneme_vocab_path to be set."
            raise ValueError(msg)

        vocab_path = Path(self.phoneme_vocab_path)
        if not vocab_path.exists():
            msg = f"AcousticDistractorStage: phoneme vocab file not found: {vocab_path}"
            raise FileNotFoundError(msg)

        with vocab_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            msg = f"Phoneme vocab must be a JSON object mapping word→[phonemes]; got {type(raw).__name__}."
            raise TypeError(msg)

        self._vocab_items = [
            (str(word), [str(p) for p in phonemes])
            for word, phonemes in raw.items()
            if isinstance(phonemes, list) and phonemes
        ]
        logger.info(
            "%s: loaded %d phoneme vocab entries from %s (language=%s)",
            self.name,
            len(self._vocab_items),
            vocab_path,
            self.language or "(per-sample source_lang)",
        )

    def teardown(self) -> None:
        if self._n_processed:
            logger.info(
                "%s: processed %d samples, appended acoustic distractors to %d (%.1f%%)",
                self.name,
                self._n_processed,
                self._n_appended,
                100.0 * self._n_appended / self._n_processed,
            )

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.context_key, self.source_lang_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.context_key]

    def _resolve_language(self, task: AudioTask) -> str | None:
        if self.language:
            return self.language
        raw = task.data.get(self.source_lang_key) or self.default_source_lang
        return _normalize_lang_to_espeak(raw)

    def _g2p(self, text: str, language: str) -> list[str]:
        key = (language, text)
        cached = self._g2p_cache.get(key)
        if cached is not None:
            return cached
        try:
            phonemes = _phonemize_one(text, language)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: phonemize failed for %r (%s): %s", self.name, text, language, exc)
            phonemes = []
        self._g2p_cache[key] = phonemes
        return phonemes

    def _generate_acoustic_distractors(
        self,
        fine_terms: list[str],
        existing_distractors: list[str],
        language: str,
    ) -> list[str]:
        """Return up to ``max_acoustic_distractors`` words from the vocab."""
        if not fine_terms or not self._vocab_items:
            return []

        excluded = {w.lower() for w in fine_terms} | {w.lower() for w in existing_distractors}

        scored: dict[str, float] = {}
        for entity in fine_terms:
            query = self._g2p(entity, language)
            if not query:
                continue
            for word, dist in _vocab_search(
                query,
                self._vocab_items,
                min_npd=self.min_npd,
                max_npd=self.max_npd,
                excluded_words=excluded,
                top_k=self.per_entity_top_k,
            ):
                if word in excluded:
                    continue
                prev = scored.get(word)
                if prev is None or dist < prev:
                    scored[word] = dist

        ranked = sorted(scored.items(), key=lambda kv: kv[1])
        return [w for w, _ in ranked[: self.max_acoustic_distractors]]

    def _merge_distractors(self, existing: list[str], acoustic: list[str]) -> list[str]:
        seen: set[str] = {w.lower() for w in existing}
        merged: list[str] = list(existing)
        for word in acoustic:
            if word.lower() in seen:
                continue
            seen.add(word.lower())
            merged.append(word)
            if len(merged) >= self.max_total_distractors:
                break
        return merged[: self.max_total_distractors]

    def _process_one(self, task: AudioTask) -> None:
        extraction = task.data.get(self.context_key)
        if not isinstance(extraction, dict):
            set_note(task.data, self.name, "no_extraction", self.notes_key)
            return

        fine_terms = extraction.get("fine_context_terms") or []
        if not isinstance(fine_terms, list) or not fine_terms:
            set_note(task.data, self.name, "no_fine_terms", self.notes_key)
            return

        language = self._resolve_language(task)
        if not language:
            set_note(task.data, self.name, "unsupported_language", self.notes_key)
            return

        raw_existing = extraction.get("distractor_terms") or []
        existing = [str(t) for t in raw_existing] if isinstance(raw_existing, list) else []

        acoustic = self._generate_acoustic_distractors(
            [str(t) for t in fine_terms],
            existing,
            language,
        )

        self._n_processed += 1
        if not acoustic:
            set_note(task.data, self.name, "no_acoustic_candidates", self.notes_key)
            return

        extraction["distractor_terms"] = self._merge_distractors(existing, acoustic)
        self._n_appended += 1
        set_note(task.data, self.name, f"appended={len(acoustic)}", self.notes_key)

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        for task in tasks:
            self._process_one(task)
        logger.debug("%s: batch of %d tasks", self.name, len(tasks))
        return tasks
