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

"""Build Qwen3-Omni chat messages from Lhotse CutSet data (two-stage streaming).

Splits into:
  1. ``EmitLhotseShardRefsStage`` — emit lightweight shard/chunk references (no audio I/O)
  2. ``ProcessLhotseShardStage`` — load audio per shard, base64-encode, build messages

``PrepareOmniLhotseStage`` is a CompositeStage that auto-decomposes into both.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import DocumentBatch, _EmptyTask

if TYPE_CHECKING:
    from lhotse import Cut, CutSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audio_to_wav_bytes(audio: np.ndarray, sampling_rate: int) -> bytes:
    """Encode float audio array to WAV bytes (PCM16)."""
    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:
        msg = "Install soundfile (e.g. pip install soundfile) to encode audio for Omni messages."
        raise RuntimeError(msg) from exc

    buf = io.BytesIO()
    to_write = audio[:, np.newaxis] if audio.ndim == 1 else audio.T
    sf.write(buf, to_write, sampling_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _is_sharded_path(path: str) -> bool:
    """Check if a path uses NeMo shard notation like ``{0..99}`` or ``__OP_``."""
    return "{" in path or "__OP_" in path


def _expand_sharded(path: str) -> list[str]:
    """Expand brace-notation shard paths using NeMo's utility."""
    try:
        from nemo.collections.common.data.lhotse.nemo_adapters import expand_sharded_filepaths
    except ModuleNotFoundError as exc:
        msg = "NeMo is required to expand sharded paths. Install nemo_toolkit[asr]."
        raise RuntimeError(msg) from exc
    return list(expand_sharded_filepaths(path))


def _count_lines(path: str) -> int:
    """Count non-empty lines in a text file."""
    with open(path) as f:
        return sum(1 for line in f if line.strip())


# ---------------------------------------------------------------------------
# Stage 1: Emit shard references (no audio I/O)
# ---------------------------------------------------------------------------


@dataclass
class EmitLhotseShardRefsStage(ProcessingStage[_EmptyTask, DocumentBatch]):
    """Enumerate shards/chunks and emit one ``DocumentBatch`` per shard reference.

    This stage is fast — it performs no audio loading.  For ``nemo_tarred``
    with sharded manifests it simply expands brace paths.  For single manifests
    it reads the JSONL text (no audio) and groups by ``shard_id``.  For
    ``nemo_raw`` it counts manifest lines and chunks them.

    Returns ``list[DocumentBatch]`` so that Ray Data can distribute shard
    processing across workers via ``IS_FANOUT_STAGE``.
    """

    name: str = "EmitLhotseShardRefs"
    input_manifest: str = ""
    input_tar: str = ""
    shar_in_dir: str = ""
    lhotse_mode: Literal["nemo_tarred", "lhotse_shar", "nemo_raw"] = "nemo_tarred"
    format: Literal["data_url", "input_data"] = "data_url"
    system_prompt: str | None = None
    user_prompt_key: str | None = None
    user_prompt: str = ""
    max_cuts: int | None = None
    chunk_size: int = 1000

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_fanout_stage": True}

    def _make_shard_ref(self, shard_data: dict, batch_idx: int) -> DocumentBatch:
        """Create a 1-row DocumentBatch carrying shard reference + config."""
        common = {
            "lhotse_mode": self.lhotse_mode,
            "format": self.format,
            "system_prompt": self.system_prompt or "",
            "user_prompt": self.user_prompt,
            "user_prompt_key": self.user_prompt_key or "",
            "max_cuts": self.max_cuts if self.max_cuts is not None else -1,
        }
        common.update(shard_data)
        df = pd.DataFrame([common])
        return DocumentBatch(
            data=df,
            dataset_name="omni_lhotse",
            task_id=f"shard_ref_{batch_idx}",
        )

    # -- nemo_tarred --

    def _emit_nemo_tarred_shards(self) -> list[DocumentBatch]:
        manifest = self.input_manifest.strip()
        tar = self.input_tar.strip()
        if not manifest or not tar:
            msg = "nemo_tarred requires non-empty input_manifest and input_tar."
            raise ValueError(msg)

        if _is_sharded_path(manifest) or _is_sharded_path(tar):
            return self._emit_nemo_tarred_sharded(manifest, tar)
        return self._emit_nemo_tarred_single_manifest(manifest, tar)

    def _emit_nemo_tarred_sharded(self, manifest: str, tar: str) -> list[DocumentBatch]:
        """Sharded manifests: expand brace paths and pair them."""
        manifest_paths = _expand_sharded(manifest)
        tar_paths = _expand_sharded(tar)
        if len(manifest_paths) != len(tar_paths):
            msg = f"Manifest shard count ({len(manifest_paths)}) != tar shard count ({len(tar_paths)})"
            raise ValueError(msg)
        batches = []
        for i, (m, t) in enumerate(zip(manifest_paths, tar_paths, strict=True)):
            batches.append(self._make_shard_ref({"shard_manifest_path": m, "shard_tar_path": t}, i))
        logger.info("Emitted {} nemo_tarred shard refs (sharded manifests)", len(batches))
        return batches

    def _emit_nemo_tarred_single_manifest(self, manifest: str, tar: str) -> list[DocumentBatch]:
        """Single manifest: read JSONL, group by shard_id, pair with expanded tar paths."""
        tar_paths = _expand_sharded(tar) if _is_sharded_path(tar) else [tar]
        tar_by_id = {}
        for t in tar_paths:
            # Extract shard_id from filename stem: audio_0.tar → 0
            stem = Path(t).stem
            parts = stem.rsplit("_", 1)
            expected_parts = 2
            if len(parts) == expected_parts and parts[1].isdigit():
                tar_by_id[int(parts[1])] = t

        # Read manifest, group lines by shard_id
        shard_lines: dict[int, list[str]] = {}
        with open(manifest) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                sid = entry.get("shard_id")
                if sid is None:
                    # No shard_id — treat entire manifest as one shard
                    logger.warning("No shard_id in manifest; treating as single shard")
                    return [self._make_shard_ref({"shard_manifest_path": manifest, "shard_tar_path": tar}, 0)]
                shard_lines.setdefault(int(sid), []).append(line)

        batches = []
        for i, (sid, lines) in enumerate(sorted(shard_lines.items())):
            tar_path = tar_by_id.get(sid, tar)
            batches.append(
                self._make_shard_ref(
                    {
                        "shard_manifest_path": manifest,
                        "shard_tar_path": tar_path,
                        "shard_id": sid,
                        "shard_line_count": len(lines),
                    },
                    i,
                )
            )
        logger.info("Emitted {} nemo_tarred shard refs (single manifest, grouped by shard_id)", len(batches))
        return batches

    # -- lhotse_shar --

    def _emit_lhotse_shar_shards(self) -> list[DocumentBatch]:
        # Explicit paths provided — single shard
        if self.input_manifest.strip() and self.input_tar.strip():
            return [
                self._make_shard_ref(
                    {
                        "shard_cuts_path": self.input_manifest.strip(),
                        "shard_recording_path": self.input_tar.strip(),
                    },
                    0,
                )
            ]

        shar_dir = self.shar_in_dir.strip()
        if not shar_dir:
            msg = "lhotse_shar requires non-empty input_manifest and input_tar or shar_in_dir."
            raise ValueError(msg)

        shar_path = Path(shar_dir)
        cut_files = sorted(shar_path.glob("cuts.*.jsonl.gz"))
        rec_files = sorted(shar_path.glob("recording.*.tar"))
        if not cut_files:
            msg = f"No cuts.*.jsonl.gz files found in {shar_dir}"
            raise FileNotFoundError(msg)
        if len(cut_files) != len(rec_files):
            msg = f"Cut shard count ({len(cut_files)}) != recording shard count ({len(rec_files)})"
            raise ValueError(msg)

        batches = []
        for i, (c, r) in enumerate(zip(cut_files, rec_files, strict=True)):
            batches.append(
                self._make_shard_ref(
                    {
                        "shard_cuts_path": str(c),
                        "shard_recording_path": str(r),
                    },
                    i,
                )
            )
        logger.info("Emitted {} lhotse_shar shard refs", len(batches))
        return batches

    # -- nemo_raw --

    def _emit_nemo_raw_chunks(self) -> list[DocumentBatch]:
        manifest = self.input_manifest.strip()
        if not manifest:
            msg = "nemo_raw requires non-empty input_manifest."
            raise ValueError(msg)

        total = _count_lines(manifest)
        batches = []
        for i, start in enumerate(range(0, total, self.chunk_size)):
            end = min(start + self.chunk_size, total)
            batches.append(
                self._make_shard_ref(
                    {
                        "manifest_path": manifest,
                        "chunk_start": start,
                        "chunk_end": end,
                    },
                    i,
                )
            )
        logger.info("Emitted {} nemo_raw chunk refs ({} lines, chunk_size={})", len(batches), total, self.chunk_size)
        return batches

    # -- dispatch --

    def process(self, _: _EmptyTask) -> list[DocumentBatch]:
        if self.lhotse_mode == "nemo_tarred":
            return self._emit_nemo_tarred_shards()
        if self.lhotse_mode == "lhotse_shar":
            return self._emit_lhotse_shar_shards()
        if self.lhotse_mode == "nemo_raw":
            return self._emit_nemo_raw_chunks()
        msg = f"Unknown lhotse_mode: {self.lhotse_mode!r}"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Stage 2: Process one shard (loads audio, builds messages)
# ---------------------------------------------------------------------------


@dataclass
class ProcessLhotseShardStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Load audio for one shard and build OpenAI-style chat messages.

    Receives a 1-row ``DocumentBatch`` from ``EmitLhotseShardRefsStage``
    containing shard reference columns + config.  Reconstructs the Lhotse
    iterator for that shard, iterates cuts, loads audio, base64-encodes,
    and returns a multi-row ``DocumentBatch`` with ``cut_id``, ``messages``,
    ``duration``, ``text`` columns.
    """

    name: str = "ProcessLhotseShardStage"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], ["lhotse_mode"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], ["cut_id", "messages", "duration", "text"]

    def _cut_to_messages(
        self, cut: Cut, fmt: str, system_prompt: str, user_prompt: str, user_prompt_key: str
    ) -> list[dict]:
        """Build OpenAI-style messages with one user turn: optional system, audio, text."""
        from nemo_curator.stages.audio.request.prepare_omni_request import resolve_media_content_type

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        audio = cut.load_audio()
        sr = cut.sampling_rate
        wav_bytes = _audio_to_wav_bytes(audio, sr)
        b64 = base64.standard_b64encode(wav_bytes).decode("ascii")
        content_fmt = resolve_media_content_type(fmt, image=False)

        if content_fmt == "input_audio":
            part = {"type": content_fmt, content_fmt: {"data": b64, "format": "wav"}}
        else:
            part = {"type": content_fmt, content_fmt: {"url": f"data:audio/wav;base64,{b64}"}}

        text = ""
        if user_prompt:
            text = user_prompt
        elif user_prompt_key:
            if cut.supervisions:
                st = cut.supervisions[0].text
                text = st if st is not None else ""
            if not str(text).strip() and getattr(cut, "custom", None):
                text = str(cut.custom.get(user_prompt_key, "") or "")

        content_parts: list[dict] = [part]
        if str(text).strip():
            content_parts.append({"type": "text", "text": text})

        messages.append({"role": "user", "content": content_parts})
        return messages

    def _load_cuts(self, ref: dict) -> CutSet:
        """Reconstruct a CutSet for one shard from its reference columns."""
        mode = ref["lhotse_mode"]

        if mode == "nemo_tarred":
            return self._load_nemo_tarred(ref)
        if mode == "lhotse_shar":
            return self._load_lhotse_shar(ref)
        if mode == "nemo_raw":
            return self._load_nemo_raw(ref)
        msg = f"Unknown lhotse_mode: {mode!r}"
        raise ValueError(msg)

    def _load_nemo_tarred(self, ref: dict) -> CutSet:
        try:
            from lhotse import CutSet
            from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoTarredIterator
        except ModuleNotFoundError as exc:
            msg = "NeMo + lhotse required. Install nemo_toolkit[asr] and lhotse."
            raise RuntimeError(msg) from exc

        manifest = ref["shard_manifest_path"]
        tar_path = ref["shard_tar_path"]
        shard_id = ref.get("shard_id")

        if shard_id is not None and not pd.isna(shard_id):
            # Single-manifest case: filter by shard_id
            shard_id = int(shard_id)
            import tempfile

            filtered_lines = []
            with open(manifest) as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("shard_id") == shard_id:
                            filtered_lines.append(line)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                tmp.writelines(filtered_lines)
                tmp_path = tmp.name
            try:
                cuts = list(CutSet(LazyNeMoTarredIterator(manifest_path=tmp_path, tar_paths=tar_path)))
            finally:
                import os

                os.unlink(tmp_path)
            return CutSet.from_cuts(cuts)

        # Sharded manifest: direct load
        return CutSet(LazyNeMoTarredIterator(manifest_path=manifest, tar_paths=tar_path))

    def _load_lhotse_shar(self, ref: dict) -> CutSet:
        try:
            from lhotse import CutSet
        except ModuleNotFoundError as exc:
            msg = "Install lhotse to use lhotse_shar mode."
            raise RuntimeError(msg) from exc

        return CutSet.from_shar(
            fields={
                "cuts": [ref["shard_cuts_path"]],
                "recording": [ref["shard_recording_path"]],
            }
        )

    def _load_nemo_raw(self, ref: dict) -> CutSet:
        try:
            from lhotse import CutSet
            from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoIterator
        except ModuleNotFoundError as exc:
            msg = "NeMo + lhotse required. Install nemo_toolkit[asr] and lhotse."
            raise RuntimeError(msg) from exc

        manifest = ref["manifest_path"]
        chunk_start = int(ref["chunk_start"])
        chunk_end = int(ref["chunk_end"])
        full_iter = LazyNeMoIterator(manifest)
        chunk = list(islice(full_iter, chunk_start, chunk_end))
        return CutSet.from_cuts(chunk)

    def process(self, input_batch: DocumentBatch) -> DocumentBatch:
        ref = input_batch.to_pandas().iloc[0].to_dict()
        cuts = self._load_cuts(ref)

        fmt = ref.get("format", "data_url")
        system_prompt = ref.get("system_prompt", "")
        user_prompt = ref.get("user_prompt", "")
        user_prompt_key = ref.get("user_prompt_key", "")
        max_cuts_val = ref.get("max_cuts", -1)
        max_cuts = None if max_cuts_val == -1 else int(max_cuts_val)

        rows: list[dict] = []
        for idx, cut in enumerate(cuts, start=1):
            row = {
                "cut_id": cut.id,
                "messages": self._cut_to_messages(cut, fmt, system_prompt, user_prompt, user_prompt_key),
                "duration": cut.duration,
                "text": cut.supervisions[0].text if cut.supervisions else None,
            }
            if user_prompt_key and cut.supervisions and cut.supervisions[0].text is not None:
                row[user_prompt_key] = cut.supervisions[0].text
            rows.append(row)
            if max_cuts is not None and idx >= max_cuts:
                break

        if not rows:
            logger.warning("Shard {} produced no rows", ref.get("task_id", "unknown"))
            rows = [{"cut_id": "", "messages": [], "duration": 0.0, "text": None}]

        df = pd.DataFrame(rows)
        return DocumentBatch(
            data=df,
            dataset_name="omni_lhotse",
            task_id=input_batch.task_id,
        )


# ---------------------------------------------------------------------------
# CompositeStage: user-facing entry point (backward compatible)
# ---------------------------------------------------------------------------


@dataclass
class PrepareOmniLhotseStage(CompositeStage[_EmptyTask, DocumentBatch]):
    """
    Build Qwen3-Omni chat messages from Lhotse CutSet data (streaming).

    Decomposes into:
      1. ``EmitLhotseShardRefsStage`` — enumerate shards (no audio I/O)
      2. ``ProcessLhotseShardStage`` — load audio per shard, build messages

    This enables Ray to distribute shard processing across workers so GPUs
    can start inference as soon as the first shard is ready, instead of
    waiting for all audio to be loaded.

    **nemo_tarred** uses ``LazyNeMoTarredIterator`` (from NeMo) with ``input_manifest``
    and ``input_tar``.  Supports both sharded manifests (``manifest_{0..N}.json``)
    and single manifests with ``shard_id`` per line.

    **lhotse_shar** uses :meth:`lhotse.CutSet.from_shar` with ``shar_in_dir``.

    **nemo_raw** uses ``LazyNeMoIterator`` with a plain NeMo JSONL manifest,
    chunked by ``chunk_size`` lines.

    Args:
        lhotse_mode: ``"nemo_tarred"``, ``"lhotse_shar"``, or ``"nemo_raw"``.
        input_manifest: Manifest path(s) for ``nemo_tarred`` or ``nemo_raw``.
        input_tar: Tar path(s) for ``nemo_tarred``.
        shar_in_dir: Directory for ``lhotse_shar``.
        format: Media encoding strategy.
            ``"data_url"`` — audio uses ``audio_url`` with ``data:`` URI (vLLM).
            ``"input_data"`` — audio uses ``input_audio`` with raw base64 + format (OpenAI / NIM).
            Images always use ``image_url`` regardless (no ``input_image`` in current APIs).
        system_prompt: Optional fixed system message.
        user_prompt: Fixed user prompt text.
        user_prompt_key: Supervision field to use as prompt (mutually exclusive with user_prompt).
        max_cuts: Optional per-shard cap on cuts (debugging).
        chunk_size: Lines per chunk for ``nemo_raw`` mode (default 1000).
    """

    name: str = "PrepareOmniLhotseStage"
    input_manifest: str = ""
    input_tar: str = ""
    shar_in_dir: str = ""
    lhotse_mode: Literal["nemo_tarred", "lhotse_shar", "nemo_raw"] = "nemo_tarred"
    format: Literal["data_url", "input_data"] = "data_url"
    system_prompt: str | None = None
    user_prompt_key: str | None = None
    user_prompt: str = ""
    max_cuts: int | None = None
    chunk_size: int = 1000

    def __post_init__(self) -> None:
        super().__init__()
        if self.user_prompt and self.user_prompt_key:
            msg = "user_prompt and user_prompt_key cannot be set at the same time"
            raise ValueError(msg)
        if not self.user_prompt and not self.user_prompt_key:
            msg = "user_prompt or user_prompt_key must be set"
            raise ValueError(msg)

    def decompose(self) -> list[ProcessingStage]:
        return [
            EmitLhotseShardRefsStage(
                input_manifest=self.input_manifest,
                input_tar=self.input_tar,
                shar_in_dir=self.shar_in_dir,
                lhotse_mode=self.lhotse_mode,
                format=self.format,
                system_prompt=self.system_prompt,
                user_prompt_key=self.user_prompt_key,
                user_prompt=self.user_prompt,
                max_cuts=self.max_cuts,
                chunk_size=self.chunk_size,
            ),
            ProcessLhotseShardStage(),
        ]

    def process(self, _: _EmptyTask) -> DocumentBatch:
        msg = "PrepareOmniLhotseStage is a CompositeStage; use Pipeline to auto-decompose."
        raise NotImplementedError(msg)

    def get_description(self) -> str:
        return f"Prepare Omni Lhotse messages ({self.lhotse_mode}, chunk_size={self.chunk_size})"
