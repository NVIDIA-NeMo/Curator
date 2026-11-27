# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""fastText-format text language-ID adapters: the shared base and the official lid.176 model."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from loguru import logger


class FastTextFormatLIDAdapter(ABC):
    """Load a fastText-format checkpoint and return one language code per text (top-1, ``k=1``).

    No confidence threshold, short-text bypass, label proxy, lowercasing or probability clipping is applied.
    """

    def __init__(self, model_path: str) -> None:
        if not model_path:
            msg = f"model_path is required for {type(self).__name__}"
            raise ValueError(msg)
        self.model_path = model_path
        self._model: Any = None
        self._labels: frozenset[str] = frozenset()

    def setup(self) -> None:
        try:
            import fasttext
        except ImportError as exc:
            msg = (
                f"{type(self).__name__} requires fasttext. "
                "Install the tested CPU LID runtime with "
                "`pip install 'numpy==1.26.4' 'fasttext==0.9.3'`."
            )
            raise ImportError(msg) from exc

        resolved_path = Path(self.model_path).expanduser()
        if not resolved_path.is_file():
            msg = f"Language-ID model does not exist: {resolved_path}"
            raise ValueError(msg)

        self._model = fasttext.load_model(str(resolved_path))
        self._labels = frozenset(str(label).removeprefix("__label__") for label in self._model.get_labels())
        self._validate_labels(self._labels)
        logger.info("{}: loaded {} labels from {}", type(self).__name__, len(self._labels), resolved_path)

    def predict(self, texts: list[str]) -> list[str]:
        """Return the language code of each text."""
        labels, probabilities = self._model.predict([self._sanitize_text(text) for text in texts], k=1)
        if len(labels) != len(texts) or len(probabilities) != len(texts):
            msg = (
                f"{type(self).__name__} prediction count mismatch: "
                f"inputs={len(texts)}, labels={len(labels)}, probabilities={len(probabilities)}"
            )
            raise ValueError(msg)
        return [
            self._language(row_labels, row_probabilities)
            for row_labels, row_probabilities in zip(labels, probabilities, strict=True)
        ]

    def _language(self, row_labels: list[object], row_probabilities: list[object]) -> str:
        if len(row_labels) == 0 or len(row_probabilities) == 0:
            msg = f"{type(self).__name__} returned an empty top-1 prediction"
            raise ValueError(msg)
        raw_label = str(row_labels[0]).removeprefix("__label__")
        if raw_label not in self._labels:
            msg = f"{type(self).__name__} returned unknown label {raw_label!r}"
            raise ValueError(msg)
        probability = float(row_probabilities[0])
        if not math.isfinite(probability):
            msg = f"{type(self).__name__} returned non-finite probability {probability!r}"
            raise ValueError(msg)
        return self._map_label_to_language(raw_label)

    @abstractmethod
    def check_source_language(self, language: str) -> None:
        """Fail when the model cannot represent the row's expected language."""

    @abstractmethod
    def _validate_labels(self, labels: frozenset[str]) -> None:
        """Validate the loaded checkpoint's complete label inventory."""

    @abstractmethod
    def _sanitize_text(self, text: str) -> str:
        """Apply the model's text preprocessing."""

    @abstractmethod
    def _map_label_to_language(self, raw_label: str) -> str:
        """Map one stripped checkpoint label to the manifest language code."""


class FastTextLIDAdapter(FastTextFormatLIDAdapter):
    """The official fastText lid.176 model; input whitespace is collapsed with ``" ".join(text.split())``."""

    expected_label_count: ClassVar[int] = 176
    granary_exact_languages: ClassVar[frozenset[str]] = frozenset(
        ["as", "bn", "gu", "hi", "kn", "mai", "ml", "mr", "ne", "or", "pa", "sa", "sd", "ta", "te", "ur"]
    )
    # The other Granary Indic languages have no exact lid.176 label.
    granary_source_languages: ClassVar[frozenset[str]] = granary_exact_languages | frozenset(
        ["brx", "doi", "kok", "ks", "mni", "sat"]
    )

    def check_source_language(self, language: str) -> None:
        if language not in self._labels:
            msg = (
                f"FastText lid.176 has no exact label for source_lang={language!r}; "
                "route this language to IndicLID instead"
            )
            raise ValueError(msg)

    def _validate_labels(self, labels: frozenset[str]) -> None:
        if len(labels) != self.expected_label_count:
            msg = f"Expected {self.expected_label_count} lid.176 labels, found {len(labels)}"
            raise ValueError(msg)
        exact_languages = labels & self.granary_source_languages
        if exact_languages != self.granary_exact_languages or "gom" not in labels:
            msg = (
                "lid.176 Granary language inventory mismatch: "
                f"expected_exact={sorted(self.granary_exact_languages)}, "
                f"found_exact={sorted(exact_languages)}, has_konkani_proxy={'gom' in labels}"
            )
            raise ValueError(msg)

    def _sanitize_text(self, text: str) -> str:
        return " ".join(text.split())

    def _map_label_to_language(self, raw_label: str) -> str:
        return raw_label
