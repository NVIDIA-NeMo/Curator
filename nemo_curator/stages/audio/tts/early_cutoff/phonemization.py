# ruff: noqa
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

from __future__ import annotations

import multiprocessing
import re
import unicodedata
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Sequence


MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
MODEL_REVISION = "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"
NORMALIZATION_VERSION = "nfc-lower-ws-v1"

PHONEMIZER_LANGUAGE_BY_ISO: dict[str, str] = {
    "bg": "bg",
    "cs": "cs",
    "da": "da",
    "de": "de",
    "el": "el",
    "en": "en-us",
    "es": "es",
    "et": "et",
    "fi": "fi",
    "fr": "fr-fr",
    "hr": "hr",
    "hu": "hu",
    "it": "it",
    "lt": "lt",
    "lv": "lv",
    "mt": "mt",
    "nl": "nl",
    "pl": "pl",
    "pt": "pt",
    "ro": "ro",
    "ru": "ru",
    "sk": "sk",
    "sl": "sl",
    "sv": "sv",
    "uk": "uk",
}

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Apply only deterministic normalization; eSpeak handles spoken forms."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).strip().lower())


def _isolated_phonemizer_worker(
    connection: Connection,
    backend_name: str,
    phonemizer_language: str,
) -> None:
    from phonemizer.backend import BACKENDS
    from phonemizer.separator import Separator

    backend = BACKENDS[backend_name](
        phonemizer_language,
        language_switch="remove-flags",
    )
    separator = Separator(phone=" ", word="", syllable="")
    while True:
        try:
            normalized_text = connection.recv()
        except EOFError:
            return
        try:
            output = backend.phonemize(
                [normalized_text],
                separator=separator,
            )[0].strip()
            per_word_output = backend.phonemize(
                normalized_text.split(),
                separator=separator,
            )
            connection.send(("ok", output, per_word_output))
        except Exception as exc:
            connection.send(("error", type(exc).__name__, str(exc)))


class _IsolatedPhonemizerBackend:
    def __init__(
        self,
        backend_name: str,
        phonemizer_language: str,
        timeout_sec: float,
    ) -> None:
        self.backend_name = backend_name
        self.phonemizer_language = phonemizer_language
        self.timeout_sec = timeout_sec
        self._connection: Connection | None = None
        self._process: multiprocessing.Process | None = None

    def _start(self) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=_isolated_phonemizer_worker,
            args=(child, self.backend_name, self.phonemizer_language),
            daemon=True,
        )
        process.start()
        child.close()
        self._connection = parent
        self._process = process

    def _stop(self) -> None:
        if self._connection is not None:
            self._connection.close()
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=5)
        self._connection = None
        self._process = None

    def phonemize(self, normalized_text: str) -> tuple[str, list[str]]:
        if self._process is None or not self._process.is_alive():
            self._stop()
            self._start()
        assert self._connection is not None
        assert self._process is not None
        try:
            self._connection.send(normalized_text)
            if not self._connection.poll(self.timeout_sec):
                raise TimeoutError(f"isolated phonemizer timed out after {self.timeout_sec:g} seconds")
            status, *payload = self._connection.recv()
        except (BrokenPipeError, EOFError, OSError, TimeoutError) as exc:
            exit_code = self._process.exitcode
            self._stop()
            raise RuntimeError(
                "isolated phonemizer process failed"
                + (f" with exit code {exit_code}" if exit_code is not None else "")
            ) from exc
        if status == "error":
            raise RuntimeError(f"{payload[0]}: {payload[1]}")
        return str(payload[0]), list(payload[1])


def _align_word_phones(
    full_phones: list[str],
    word_phone_groups: list[list[str]],
) -> list[int]:
    """Map continuous-context phones to words via edit-distance alignment."""
    flattened: list[str] = []
    labels: list[int] = []
    for word_idx, group in enumerate(word_phone_groups):
        flattened.extend(group)
        labels.extend([word_idx] * len(group))
    if not full_phones:
        return []
    if not flattened:
        return [0] * len(full_phones)

    rows = len(flattened) + 1
    cols = len(full_phones) + 1
    costs = [[0] * cols for _ in range(rows)]
    moves = [[""] * cols for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row
        moves[row][0] = "delete"
    for col in range(1, cols):
        costs[0][col] = col
        moves[0][col] = "insert"
    for row in range(1, rows):
        for col in range(1, cols):
            substitution = costs[row - 1][col - 1] + (0 if flattened[row - 1] == full_phones[col - 1] else 1)
            deletion = costs[row - 1][col] + 1
            insertion = costs[row][col - 1] + 1
            best = min(substitution, deletion, insertion)
            costs[row][col] = best
            moves[row][col] = "match" if best == substitution else "delete" if best == deletion else "insert"

    mapping: list[int | None] = [None] * len(full_phones)
    row, col = len(flattened), len(full_phones)
    while row > 0 or col > 0:
        move = moves[row][col]
        if move == "match":
            mapping[col - 1] = labels[row - 1]
            row -= 1
            col -= 1
        elif move == "delete":
            row -= 1
        else:
            neighbor = labels[row - 1] if row > 0 else 0
            mapping[col - 1] = neighbor
            col -= 1
    for idx in range(len(mapping)):
        if mapping[idx] is None:
            mapping[idx] = mapping[idx - 1] if idx > 0 and mapping[idx - 1] is not None else 0
    return [int(value) for value in mapping]


@dataclass(frozen=True)
class PhonemizedReference:
    normalized_text: str
    phones: tuple[str, ...]
    token_ids: tuple[int, ...]
    phone_to_word: tuple[int, ...]
    words: tuple[str, ...]
    phonemizer_language: str
    source: str
    oov_tokens: tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        return bool(self.token_ids) and not self.oov_tokens

    def text_spans(self, last_supported_phone: int | None) -> tuple[str | None, str | None]:
        if last_supported_phone is None or not self.words or not self.phone_to_word:
            return None, None
        if last_supported_phone < 0:
            return "", " ".join(self.words)
        phone_idx = min(last_supported_phone, len(self.phone_to_word) - 1)
        word_idx = self.phone_to_word[phone_idx]
        word_phone_indices = [idx for idx, mapped_word in enumerate(self.phone_to_word) if mapped_word == word_idx]
        word_complete = bool(word_phone_indices and phone_idx >= word_phone_indices[-1])
        split_word = word_idx + 1 if word_complete else word_idx
        supported = " ".join(self.words[:split_word])
        missing = " ".join(self.words[split_word:])
        return supported, missing


class ModelCompatiblePhonemizer:
    """Reproduce the Hugging Face tokenizer's eSpeak backend and separators."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        language_map: dict[str, str] | None = None,
        backend_name: str = "espeak",
        isolate_languages: Sequence[str] = (),
        isolated_timeout_sec: float = 60.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.vocab = dict(tokenizer.get_vocab())
        self.unk_token_id = int(tokenizer.unk_token_id)
        self.language_map = dict(PHONEMIZER_LANGUAGE_BY_ISO)
        self.language_map.update(language_map or {})
        self.backend_name = backend_name
        self.isolate_languages = {str(language).lower() for language in isolate_languages}
        self.isolated_timeout_sec = isolated_timeout_sec
        self._backends: dict[str, Any] = {}
        self._isolated_backends: dict[str, _IsolatedPhonemizerBackend] = {}

    def _segment_model_unit(self, unit: str) -> list[str] | None:
        """Split an OOV eSpeak phone into the fewest model vocabulary units."""
        if unit in self.vocab:
            return [unit]
        # eSpeak versions differ in a few IPA spellings that are acoustically
        # equivalent to tokens in the frozen model vocabulary. Canonicalize only
        # OOV units so already-supported model tokens remain byte-for-byte stable.
        unit = (
            unit.replace("ε", "ɛ")
            .replace("?", "ʔ")
            .replace("\u0361", "")
            .replace("\u035c", "")
            .replace("\u0329", "")
            .replace("\u032f", "")
        )
        while "ːː" in unit:
            unit = unit.replace("ːː", "ː")
        if unit in self.vocab:
            return [unit]
        solutions: list[list[str] | None] = [None] * (len(unit) + 1)
        solutions[len(unit)] = []
        for start in range(len(unit) - 1, -1, -1):
            best: list[str] | None = None
            # Longest-first makes equal-token-count segmentations deterministic.
            for end in range(len(unit), start, -1):
                token = unit[start:end]
                suffix = solutions[end]
                if token not in self.vocab or suffix is None:
                    continue
                candidate = [token, *suffix]
                if best is None or len(candidate) < len(best):
                    best = candidate
            solutions[start] = best
        return solutions[0]

    def _expand_model_units(
        self,
        units: list[str],
        unit_to_word: list[int],
    ) -> tuple[list[str], list[int], list[str]]:
        if len(units) != len(unit_to_word):
            raise ValueError("model units and word mapping must have equal length")
        expanded: list[str] = []
        expanded_to_word: list[int] = []
        unresolved: list[str] = []
        for unit, word_index in zip(units, unit_to_word):
            segments = self._segment_model_unit(unit)
            if segments is None:
                segments = [unit]
                unresolved.append(unit)
            expanded.extend(segments)
            expanded_to_word.extend([word_index] * len(segments))
        return expanded, expanded_to_word, unresolved

    def _backend(self, language: str) -> Any:
        phonemizer_language = self.language_map.get(language.lower(), language.lower())
        backend = self._backends.get(phonemizer_language)
        if backend is None:
            from phonemizer.backend import BACKENDS

            backend = BACKENDS[self.backend_name](
                phonemizer_language,
                language_switch="remove-flags",
            )
            self._backends[phonemizer_language] = backend
        return backend

    def _isolated_backend(self, language: str) -> _IsolatedPhonemizerBackend:
        phonemizer_language = self.language_map.get(language.lower(), language.lower())
        backend = self._isolated_backends.get(phonemizer_language)
        if backend is None:
            backend = _IsolatedPhonemizerBackend(
                self.backend_name,
                phonemizer_language,
                self.isolated_timeout_sec,
            )
            self._isolated_backends[phonemizer_language] = backend
        return backend

    def _encode_words(
        self,
        normalized_text: str,
        language: str,
    ) -> tuple[list[str], list[int], list[str], str]:
        phonemizer_language = self.language_map.get(language.lower(), language.lower())
        source_words = normalized_text.split()
        if language.lower() in self.isolate_languages:
            output, per_word_output = self._isolated_backend(language).phonemize(normalized_text)
        else:
            from phonemizer.separator import Separator

            backend = self._backend(language)
            separator = Separator(phone=" ", word="", syllable="")
            output = backend.phonemize(
                [normalized_text],
                separator=separator,
            )[0].strip()
            per_word_output = backend.phonemize(
                source_words,
                separator=separator,
            )
        phones = [phone for phone in output.split(" ") if phone.strip()]
        word_phone_groups = [
            [phone for phone in value.strip().split(" ") if phone.strip()] for value in per_word_output
        ]
        phone_to_word = _align_word_phones(phones, word_phone_groups)
        return phones, phone_to_word, per_word_output, phonemizer_language

    def phonemize(
        self,
        text: str,
        language: str,
        *,
        precomputed_ipa: str | None = None,
    ) -> PhonemizedReference:
        normalized = normalize_text(text)
        if not normalized:
            return PhonemizedReference(
                normalized_text="",
                phones=(),
                token_ids=(),
                phone_to_word=(),
                words=(),
                phonemizer_language=self.language_map.get(language, language),
                source="empty",
            )

        words = tuple(normalized.split())
        source = "model_phonemizer"
        phones: list[str]
        phone_to_word: list[int]
        phonemizer_language = self.language_map.get(language.lower(), language.lower())

        # Existing IPA is accepted only when it is already an unambiguous stream of
        # whitespace-delimited model tokens. Raw eSpeak IPA is otherwise re-created.
        cached = [token for token in (precomputed_ipa or "").split() if token]
        cached_mapping = [min(len(words) - 1, (idx * len(words)) // max(len(cached), 1)) for idx in range(len(cached))]
        cached_phones, cached_phone_to_word, cached_oov = self._expand_model_units(cached, cached_mapping)
        if cached and not cached_oov:
            phones = cached_phones
            source = "validated_manifest_ipa" if phones == cached else "segmented_manifest_ipa"
            # Existing IPA has no reliable word boundaries. A monotonic proportional
            # mapping is used only for human-readable text spans.
            phone_to_word = cached_phone_to_word
        else:
            phones, phone_to_word, _, phonemizer_language = self._encode_words(
                normalized,
                language,
            )
            original_phones = phones
            phones, phone_to_word, _ = self._expand_model_units(phones, phone_to_word)
            if phones != original_phones:
                source = "model_phonemizer_segmented"

        ids = [int(self.vocab.get(phone, self.unk_token_id)) for phone in phones]
        oov = tuple(dict.fromkeys(phone for phone, token_id in zip(phones, ids) if token_id == self.unk_token_id))
        return PhonemizedReference(
            normalized_text=normalized,
            phones=tuple(phones),
            token_ids=tuple(ids),
            phone_to_word=tuple(phone_to_word),
            words=words,
            phonemizer_language=phonemizer_language,
            source=source,
            oov_tokens=oov,
        )
