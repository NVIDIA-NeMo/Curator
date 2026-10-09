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

"""Annotate TTS manifests with IPA from ``tn_raw`` (TTS Granary manifest IPA)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.stages.audio.tts.fields import MISSING, get_dotted, set_dotted
from nemo_curator.stages.audio.tts.ipa_espeak import (
    EspeakRunner,
    IPACache,
    find_espeak_binaries,
    language_to_espeak_voice,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_DEFAULT_TEXT_FALLBACKS = ("itn_text", "GranaryV2.tn_raw")


def empty_annotation(error: str) -> dict[str, Any]:
    return {"ipa": None, "error": error}


@dataclass
class ManifestIpaStage(ProcessingStage[AudioTask, AudioTask]):
    """Generate IPA for each utterance from the text-pipeline ``tn_raw`` field.

    Port of TTS Granary ``pipeline.stages.ipa.processor.ManifestIpaProcessor``.

    Reads ``text_key`` (default ``tn_raw``) from a flat Curator JSONL row or from
    each dict in ``segments``. Falls back to ``itn_text`` and nested
    ``GranaryV2.tn_raw`` so manifests from ``run_text_pipeline.py`` work without
    renaming fields.

    Writes a dict ``{"ipa": str | None, "error": str | None}`` at ``output_key``.
    Nested dotted paths (e.g. ``GranaryHifi.OrigAudioPipeline.IPA``) are supported.
    Existing non-empty IPA is preserved unless ``overwrite`` is true.

    Requires ``espeak-ng`` or ``espeak`` on PATH.
    """

    text_key: str = "tn_raw"
    output_key: str = "ipa"
    source_lang_key: str = "source_lang"
    fallback_text_keys: tuple[str, ...] = _DEFAULT_TEXT_FALLBACKS
    language: str | None = None
    overwrite: bool = False
    name: str = "ManifestIPA"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 32

    def __post_init__(self) -> None:
        super().__init__()
        self._runner: EspeakRunner | None = None
        self._cache = IPACache()
        self._setup_error: str | None = None
        self._failed_text_errors: dict[str, str] = {}

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.output_key.split(".")[0]]

    def setup(self, _worker_metadata: Any = None) -> None:  # noqa: ANN401
        voice = language_to_espeak_voice(self.language) if self.language else "en"
        try:
            binaries = find_espeak_binaries()
        except RuntimeError as exc:
            logger.error("Cannot initialize IPA stage: {}", exc)
            self._setup_error = "espeak_unavailable"
            return
        self._runner = EspeakRunner(
            exe=binaries[0],
            voice=voice,
            fallback_exe=binaries[1] if len(binaries) > 1 else None,
        )
        logger.info("ManifestIpaStage: espeak={} voice={}", binaries[0], voice)

    def _resolve_text(self, payload: dict[str, Any]) -> Any:  # noqa: ANN401
        for key in (self.text_key, *self.fallback_text_keys):
            value = get_dotted(payload, key, MISSING)
            if value is not MISSING:
                return value
        return MISSING

    def _resolve_voice(self, payload: dict[str, Any]) -> str:
        if self.language:
            return language_to_espeak_voice(self.language)
        lang = payload.get(self.source_lang_key) or payload.get("language") or payload.get("lang")
        return language_to_espeak_voice(str(lang) if lang else None)

    def _existing_ok(self, payload: dict[str, Any]) -> bool:
        if self.overwrite:
            return False
        existing = get_dotted(payload, self.output_key)
        return isinstance(existing, dict) and isinstance(existing.get("ipa"), str) and bool(existing["ipa"].strip())

    def _validated_text(self, payload: dict[str, Any]) -> tuple[str | None, str | None]:
        text = self._resolve_text(payload)
        if text is MISSING:
            return None, "missing_text"
        if not isinstance(text, str):
            return None, "invalid_text"
        text = text.strip()
        if not text:
            return None, "empty_text"
        if self._setup_error is not None or self._runner is None:
            return None, self._setup_error or "espeak_unavailable"
        return text, None

    def _ipa_for_text(self, text: str, voice: str, runner: EspeakRunner) -> dict[str, Any]:
        cache_key = f"{voice}\0{text}"
        prior_error = self._failed_text_errors.get(cache_key)
        if prior_error is not None:
            return empty_annotation(prior_error)

        ipa = self._cache.get(voice, text)
        if ipa is not None:
            return {"ipa": ipa, "error": None}
        try:
            ipa = runner.text_to_ipa(text)
        except Exception as exc:  # noqa: BLE001
            self._failed_text_errors[cache_key] = "ipa_failed"
            logger.warning("IPA conversion failed for voice={} text={!r}: {}", voice, text[:100], exc)
            return empty_annotation("ipa_failed")
        if not ipa:
            self._failed_text_errors[cache_key] = "empty_ipa"
            return empty_annotation("empty_ipa")
        self._cache.set(voice, text, ipa)
        return {"ipa": ipa, "error": None}

    def annotate_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._existing_ok(payload):
            return get_dotted(payload, self.output_key)

        text, error = self._validated_text(payload)
        if error is not None or text is None:
            return empty_annotation(error or "missing_text")

        voice = self._resolve_voice(payload)
        runner = self._runner
        if runner is None:
            return empty_annotation("espeak_unavailable")
        if runner.voice != voice:
            runner = EspeakRunner(exe=runner.exe, voice=voice, fallback_exe=runner.fallback_exe)
        return self._ipa_for_text(text, voice, runner)

    def _annotate_task_data(self, data: dict[str, Any]) -> None:
        segments = data.get("segments")
        if isinstance(segments, list):
            for segment in segments:
                if isinstance(segment, dict):
                    set_dotted(segment, self.output_key, self.annotate_payload(segment))
            return
        set_dotted(data, self.output_key, self.annotate_payload(data))

    def process(self, task: AudioTask) -> AudioTask:
        self._annotate_task_data(task.data)
        return task

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        for task in tasks:
            self._annotate_task_data(task.data)
        return tasks
