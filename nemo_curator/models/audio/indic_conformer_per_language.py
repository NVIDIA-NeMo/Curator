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

"""IndicConformer recovery with one checkpoint per language.

Each AI4Bharat ``indicconformer_stt_<lang>_hybrid_ctc_rnnt_large`` checkpoint is fine-tuned for its own language: its
language head is only usable with that checkpoint's weights, and the Hindi checkpoint returns unusable text for the other
languages. This adapter takes a ``model_id`` template with a ``{lang}`` placeholder, loads the checkpoint for each
language the first time a row in that language arrives, and sends each language's rows to
:class:`~nemo_curator.models.audio.indic_conformer_hybrid.IndicConformerHybridASR` loaded from that checkpoint.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from loguru import logger

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.models.audio.indic_conformer_hybrid import INDIC_CONFORMER_HYBRID_LANGS, IndicConformerHybridASR


class IndicConformerPerLanguageASR:
    """``ASRAdapter`` that routes each row to the IndicConformer checkpoint of its language."""

    def __init__(
        self,
        model_id: str,
        *,
        languages: list[str] | None = None,
        max_loaded_languages: int | None = None,
        **adapter_kwargs: Any,  # noqa: ANN401
    ):
        """Create the router.

        Args:
            model_id: Checkpoint template containing ``{lang}``, e.g.
                ``ai4bharat/indicconformer_stt_{lang}_hybrid_ctc_rnnt_large`` or a local ``.../{lang}.nemo`` path.
            languages: Languages whose weights ``download_weights_on_node`` prefetches (default: all 22).
            max_loaded_languages: Checkpoints kept loaded at once, least recently used first out (default: no limit).
            **adapter_kwargs: Passed to every ``IndicConformerHybridASR`` (decode mode, RNNT precision, ...).
        """
        if "{lang}" not in model_id:
            msg = f"IndicConformerPerLanguageASR.model_id must contain '{{lang}}', got: {model_id!r}"
            raise ValueError(msg)
        if max_loaded_languages is not None and max_loaded_languages < 1:
            msg = "max_loaded_languages must be at least 1"
            raise ValueError(msg)
        unknown = sorted(set(languages or []) - INDIC_CONFORMER_HYBRID_LANGS)
        if unknown:
            msg = f"Unsupported IndicConformer languages: {unknown}"
            raise ValueError(msg)
        self.model_id = model_id
        self.languages = sorted(languages) if languages else sorted(INDIC_CONFORMER_HYBRID_LANGS)
        self.max_loaded_languages = max_loaded_languages
        self._adapter_kwargs = adapter_kwargs
        self._num_gpus: int | None = None
        self._loaded: OrderedDict[str, IndicConformerHybridASR] = OrderedDict()

    def _new_adapter(self, language: str) -> IndicConformerHybridASR:
        return IndicConformerHybridASR(self.model_id.format(lang=language), **self._adapter_kwargs)

    def download_weights_on_node(self) -> None:
        for language in self.languages:
            self._new_adapter(language).download_weights_on_node()

    def load_model(self, *, num_gpus: int) -> None:
        """Remember the GPU count; each language's checkpoint loads when its first row arrives."""
        self._num_gpus = num_gpus

    def unload_model(self) -> None:
        for adapter in self._loaded.values():
            adapter.unload_model()
        self._loaded.clear()

    def _adapter_for(self, language: str) -> IndicConformerHybridASR:
        adapter = self._loaded.get(language)
        if adapter is not None:
            self._loaded.move_to_end(language)
            return adapter
        if self._num_gpus is None:
            msg = "Model not initialized. Call load_model() first."
            raise RuntimeError(msg)
        while self.max_loaded_languages is not None and len(self._loaded) >= self.max_loaded_languages:
            evicted, old = self._loaded.popitem(last=False)
            logger.info(f"IndicConformer: unloading {evicted} checkpoint")
            old.unload_model()
        adapter = self._new_adapter(language)
        adapter.load_model(num_gpus=self._num_gpus)
        self._loaded[language] = adapter
        return adapter

    def transcribe_batch(self, items: list[dict[str, Any]]) -> list[ASRResult]:
        results = [ASRResult(text="") for _ in items]
        by_language: dict[str, list[int]] = {}
        for index, item in enumerate(items):
            language = str(item.get("language_code") or "").strip().lower()
            if language in INDIC_CONFORMER_HYBRID_LANGS:
                by_language.setdefault(language, []).append(index)
            else:
                results[index] = ASRResult(text="", unsupported_language=language or None)
        for language, indices in by_language.items():
            for index, result in zip(
                indices, self._adapter_for(language).transcribe_batch([items[i] for i in indices]), strict=True
            ):
                results[index] = result
        return results
