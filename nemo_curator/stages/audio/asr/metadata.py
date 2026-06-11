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

"""Canonical metadata schema for ASR manifest entries.

``ASRMetadata`` is the typed contract every ASR dataset handler produces. It is
intentionally flattened into the plain ``dict`` carried by ``AudioTask.data`` via
:meth:`ASRMetadata.to_dict` so that all existing dict-based audio stages
(``GetAudioDurationStage``, ``AudioToDocumentStage``, ``JsonlWriter`` ...) keep
working without modification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class ASRMetadata:
    """A single ASR manifest entry.

    Core fields map directly to the JSONL manifest used for ASR training. The
    ``extra`` dict holds dataset-specific fields (e.g. ``snr``, ``gender``,
    ``collection_source``) which are spread into the top level on serialization.

    Args:
        audio_filepath: Path to the converted audio (WAV, 16 kHz, mono, PCM16).
        text: Ground-truth transcript.
        duration: Audio duration in seconds (after conversion).
        lang: Language identifier (e.g. ``"hindi"``).
        split_type: Dataset split this entry belongs to (``"train"``/``"dev"``/``"test"``).
        source: Source dataset name (e.g. ``"IndicVoices"``).
        sample_rate: Target sample rate of the converted audio.
        num_channels: Target channel count of the converted audio.
        orig_sample_rate: Source sample rate before conversion (if known).
        orig_num_channels: Source channel count before conversion (if known).
        extra: Dataset-specific extra fields, flattened on serialization.
    """

    audio_filepath: str
    text: str
    duration: float
    lang: str
    split_type: str
    source: str
    sample_rate: int = 16000
    num_channels: int = 1
    orig_sample_rate: int | None = None
    orig_num_channels: int | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Flatten to a plain dict suitable for ``AudioTask.data`` / a JSONL line.

        Core fields are emitted at the top level and ``extra`` is spread in.
        Core fields take precedence over any colliding key in ``extra``.
        """
        out = {k: v for k, v in asdict(self).items() if k != "extra"}
        for key, value in self.extra.items():
            out.setdefault(key, value)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> ASRMetadata:
        """Rebuild from a flattened dict; unknown keys are collected into ``extra``."""
        known = {f.name for f in fields(cls)} - {"extra"}
        core = {k: data[k] for k in known if k in data}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(**core, extra=extra)
