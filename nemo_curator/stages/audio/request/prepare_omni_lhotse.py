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

"""Build Qwen3-Omni (e.g. Qwen3-Omni-30B-A3B-Instruct) chat messages from Lhotse :class:`~lhotse.CutSet` data."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, _EmptyTask

if TYPE_CHECKING:
    from lhotse import Cut, CutSet

try:
    import soundfile as sf
except ModuleNotFoundError as exc:
    msg = "Install soundfile (e.g. pip install soundfile) to encode audio for Omni messages."
    raise RuntimeError(msg) from exc

try:
    from nemo.collections.common.data.lhotse.nemo_adapters import (
        LazyNeMoIterator,
        LazyNeMoTarredIterator,
    )
except ModuleNotFoundError as exc:
    msg = (
        "NeMo is required for lhotse_mode='nemo_tarred' or 'nemo_row'. "
        "Install nemo_toolkit (e.g. pip install nemo_toolkit[asr])."
    )
    raise RuntimeError(msg) from exc


def _audio_to_wav_bytes(audio: np.ndarray, sampling_rate: int) -> bytes:
    """Encode float audio array to WAV bytes (PCM16)."""

    buf = io.BytesIO()
    to_write = audio[:, np.newaxis] if audio.ndim == 1 else audio.T
    sf.write(buf, to_write, sampling_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@dataclass
class PrepareOmniLhotseStage(ProcessingStage[_EmptyTask, DocumentBatch]):
    """
    Initialize a Lhotse :class:`~lhotse.CutSet` from either NeMo tarred manifests or
    Lhotse Shar storage, iterate cuts, and populate a ``messages`` column suitable for
    `Qwen3-Omni-30B-A3B-Instruct` (and compatible APIs) via ``OmniLLMRequestStage``.

    **nemo_tarred** uses ``LazyNeMoTarredIterator`` (from NeMo) with ``input_manifest`` and
    ``input_tar``. Tar paths must be discoverable by NeMo's shard naming (e.g. ``audio_0.tar``,
    ``audio_1.tar``); see NeMo's tarred audio documentation.

    **lhotse_shar** uses :meth:`lhotse.CutSet.from_shar` with ``shar_in_dir`` (layout:
    ``cuts.*.jsonl.gz``, ``recording.*.tar``, etc.).

    **nemo_row** uses NeMo's ``LazyNeMoIterator`` to read a standard NeMo JSONL manifest
    (``audio_filepath``, ``duration``, ``text``, etc.) and yield Lhotse cuts. This is not the
    same as :meth:`lhotse.CutSet.from_jsonl`, which expects Lhotse-native JSON with a
    ``type`` field per line.

    Input rows on ``input_batch`` are ignored; the stage materializes rows from the Lhotse
    iterator. Downstream metadata (``dataset_name``, ``task_id``) is preserved from
    ``input_batch`` when present.

    Args:
        lhotse_mode: ``\"nemo_tarred\"`` or ``\"lhotse_shar\"`` or ``\"nemo_row``.
        input_manifest: NeMo JSON manifest path(s) for ``nemo_tarred`` or ``nemo_row`` (see
            NeMo lhotse adapters).
        input_tar: NeMo tarred audio path(s) for ``nemo_tarred`` (paired with manifest shards).
        shar_in_dir: Directory in Lhotse Shar format for ``lhotse_shar``.
        format: ``\"data_url\"`` for OpenAI-style ``audio_url`` / ``image_url`` payloads, or
            ``\"input_data\"`` for ``input_audio`` / ``input_image`` payloads.
        system_prompt: Optional fixed system message prepended to every sample.
        text_field: Supervision text field name (default ``text``).
        max_cuts: Optional cap on number of cuts (debugging).
    """

    name: str = "PrepareOmniLhotseStage"
    input_manifest: str = ""
    input_tar: str = ""
    shar_in_dir: str = ""
    lhotse_mode: Literal["nemo_tarred", "lhotse_shar", "nemo_row"] = "nemo_tarred"
    format: Literal["data_url", "input_data"] = "data_url"
    system_prompt: str | None = None
    user_prompt_key: str | None = None
    user_prompt: str = ""
    max_cuts: int | None = None

    def __post_init__(self) -> None:
        if self.user_prompt and self.user_prompt_key:
            msg = "user_prompt and user_prompt_key cannot be set at the same time"
            raise ValueError(msg)
        if not self.user_prompt and not self.user_prompt_key:
            msg = "user_prompt or user_prompt_key must be set"
            raise ValueError(msg)

    def _build_cutset(self) -> CutSet:
        try:
            from lhotse import CutSet
        except ModuleNotFoundError as exc:
            msg = "Install lhotse (e.g. pip install lhotse) to use PrepareOmniLhotseStage."
            raise RuntimeError(msg) from exc
        if self.lhotse_mode == "nemo_row":
            if not self.input_manifest.strip():
                msg = "nemo_row requires non-empty input_manifest (NeMo JSONL: audio_filepath, duration, text, ...)."
                raise ValueError(msg)
            return CutSet(LazyNeMoIterator(self.input_manifest.strip()))

        if self.lhotse_mode == "nemo_tarred":
            if not self.input_manifest.strip() or not self.input_tar.strip():
                msg = "nemo_tarred requires non-empty input_manifest and input_tar."
                raise ValueError(msg)
            return CutSet(
                LazyNeMoTarredIterator(
                    manifest_path=self.input_manifest.strip(),
                    tar_paths=self.input_tar.strip(),
                )
            )
        if self.lhotse_mode == "lhotse_shar":
            if not self.input_manifest.strip() or not self.input_tar.strip():
                if not self.shar_in_dir.strip():
                    msg = "lhotse_shar requires non-empty input_manifest and input_tar or shar_in_dir."
                    raise ValueError(msg)
                else:
                    return CutSet.from_shar(in_dir=self.shar_in_dir.strip())
            else:
                return CutSet.from_shar(
                    fields={"cuts": [self.input_manifest.strip()], "recording": [self.input_tar.strip()]}
                )

        msg = f"Unknown lhotse_mode: {self.lhotse_mode!r}"
        raise ValueError(msg)

    def _user_content_format(self, *, image: bool) -> str:
        """Images always use ``image_url``; audio depends on ``format`` setting."""
        if image:
            return "image_url"
        if self.format == "data_url":
            return "audio_url"
        if self.format == "input_data":
            return "input_audio"
        msg = f"Invalid format: {self.format!r}. Supported: 'data_url', 'input_data'."
        raise ValueError(msg)

    def _cut_to_messages(self, cut: Cut) -> list[dict]:
        """Build OpenAI-style messages with one user turn: optional system, audio, text."""
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        audio = cut.load_audio()
        sr = cut.sampling_rate
        wav_bytes = _audio_to_wav_bytes(audio, sr)
        b64 = base64.standard_b64encode(wav_bytes).decode("ascii")
        content_fmt = self._user_content_format(image=False)

        if content_fmt == "input_audio":
            part = {"type": content_fmt, content_fmt: {"data": b64, "format": "wav"}}
        else:
            mime = "audio/wav"
            part = {
                "type": content_fmt,
                content_fmt: {"url": f"data:{mime};base64,{b64}"},
            }

        text = ""
        if self.user_prompt:
            text = self.user_prompt
        elif self.user_prompt_key:
            if cut.supervisions:
                st = cut.supervisions[0].text
                text = st if st is not None else ""
            if not str(text).strip() and getattr(cut, "custom", None):
                text = str(cut.custom.get(self.user_prompt_key, "") or "")

        content_parts: list[dict] = [part]
        if str(text).strip():
            content_parts.append({"type": "text", "text": text})

        messages.append({"role": "user", "content": content_parts})
        return messages

    def process(self, _: _EmptyTask) -> DocumentBatch:
        cuts = self._build_cutset()
        rows: list[dict] = []
        for idx, cut in enumerate(cuts, start=1):
            row = {
                "cut_id": cut.id,
                "messages": self._cut_to_messages(cut),
            }
            if self.user_prompt_key and cut.supervisions and cut.supervisions[0].text is not None:
                row[self.user_prompt_key] = cut.supervisions[0].text
            rows.append(row)
            if self.max_cuts is not None and idx >= self.max_cuts:
                break

        df = pd.DataFrame(rows)
        return DocumentBatch(
            data=df,
            dataset_name="omni_lhotse",
            task_id="omni_lhotse",
        )
