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

"""ASR transcript normalization for flat audio manifests."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_RESOURCE_ROOT = Path(__file__).parent / "langs"


@dataclass(frozen=True)
class NormalizationResult:
    text: str
    unknown_chars: dict[str, int]


class ResourceTranscriptNormalizer:
    """Resource-driven normalizer for one ASR transcript language."""

    def __init__(self, lang: str, *, remove_pnc_chars: bool = True) -> None:
        self.lang = resolve_lang(lang)
        lang_dir = _RESOURCE_ROOT / self.lang
        self.alphabet = _load_alphabet(lang_dir / "alphabet.txt")
        self.pretok_rules = _load_jsonl(lang_dir / "pretok.jsonl")
        remove_chars = _load_chars(lang_dir / "remove_chars.txt")
        pnc_chars = _load_chars(lang_dir / "pnc_chars.txt")
        if remove_pnc_chars:
            self.remove_chars = remove_chars + pnc_chars
        else:
            non_pnc_remove_chars = "".join(char for char in remove_chars if char not in pnc_chars)
            self.remove_chars = non_pnc_remove_chars

    def normalize(self, text: str) -> NormalizationResult:
        normalized = unicodedata.normalize("NFKC", text)
        # Language resources own punctuation normalization; rules may include
        # characters such as \u2014 (em dash) and \u2013 (en dash).
        for rule in self.pretok_rules:
            pattern = str(rule["pattern"])
            repl = str(rule.get("repl", ""))
            normalized = re.sub(pattern, repl, normalized)
            if rule.get("repeat"):
                while re.search(pattern, normalized):
                    normalized = re.sub(pattern, repl, normalized)
        if self.remove_chars:
            normalized = re.sub("[" + re.escape(self.remove_chars) + "]", " ", normalized)
        normalized = " ".join(normalized.split())
        unknown_chars = Counter(char for char in normalized if char not in self.alphabet and not char.isspace())
        return NormalizationResult(text=normalized, unknown_chars=dict(unknown_chars))


@dataclass
class TranscriptNormalizationStage(ProcessingStage[AudioTask, AudioTask]):
    """Normalize ASR transcript text and optionally drop rows with unknown chars."""

    name: str = "transcript_normalization"
    text_key: str = "text"
    lang_key: str = "lang"
    output_text_key: str = "text"
    output_original_text_key: str = "text_original"
    unknown_chars_key: str = "unknown_chars"
    transcript_error_key: str = "transcript_error"
    duration_key: str = "duration"
    remove_pnc_chars: bool = True
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        self._normalizers: dict[str, ResourceTranscriptNormalizer] = {}
        self._unknown_char_counts: Counter[str] = Counter()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.lang_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [
            self.output_text_key,
            self.output_original_text_key,
            self.unknown_chars_key,
            self.transcript_error_key,
        ]

    def process(self, task: AudioTask) -> AudioTask:
        start = time.perf_counter()
        original_text = str(task.data[self.text_key])
        lang = resolve_lang(str(task.data[self.lang_key]))
        result = self._normalizer(lang).normalize(original_text)
        unknown_chars = result.unknown_chars
        transcript_error = bool(unknown_chars)
        duration = float(task.data.get(self.duration_key, 0.0) or 0.0)

        self._unknown_char_counts.update(unknown_chars)
        metrics: dict[str, float | int] = {
            "input_tasks": 1,
            "emitted_tasks": 1,
            "unknown_duration_seconds": duration if transcript_error else 0.0,
            "process_time": time.perf_counter() - start,
        }
        self._log_metrics(metrics)
        if unknown_chars:
            logger.info(
                f"[{self.name}] unknown chars for lang={lang}: "
                f"{dict(sorted(unknown_chars.items(), key=lambda item: item[1], reverse=True))}; "
                f"running_total={dict(self._unknown_char_counts.most_common())}"
            )

        task.data[self.output_original_text_key] = original_text
        task.data[self.output_text_key] = result.text
        task.data[self.unknown_chars_key] = unknown_chars
        task.data[self.transcript_error_key] = transcript_error
        return task

    def _normalizer(self, lang: str) -> ResourceTranscriptNormalizer:
        if lang not in self._normalizers:
            self._normalizers[lang] = ResourceTranscriptNormalizer(lang, remove_pnc_chars=self.remove_pnc_chars)
        return self._normalizers[lang]


def resolve_lang(lang: str) -> str:
    normalized = lang.strip().lower()
    if (_RESOURCE_ROOT / normalized).is_dir():
        return normalized
    else:
        msg = f"Unsupported ASR normalization language: {lang!r}"
        raise ValueError(msg)


def _load_alphabet(path: Path) -> set[str]:
    chars = _load_chars(path)
    return set(chars)


def _load_chars(path: Path) -> str:
    if not path.exists():
        msg = f"Missing ASR normalization resource: {path}"
        raise FileNotFoundError(msg)
    chars = []
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            chars.append(line)
    return "".join(chars)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        msg = f"Missing ASR normalization resource: {path}"
        raise FileNotFoundError(msg)
    rules = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rules.append(json.loads(line))
    return rules
