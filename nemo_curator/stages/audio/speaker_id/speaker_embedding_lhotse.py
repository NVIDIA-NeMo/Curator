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

"""Extract speaker embeddings from Lhotse :class:`~lhotse.CutSet` data.

Counterpart of :mod:`~nemo_curator.stages.audio.request.prepare_omni_lhotse`
which builds chat messages for an LLM.  This stage instead runs a NeMo
``EncDecSpeakerLabelModel`` locally and persists the resulting
speaker-embedding vectors to disk.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask, _EmptyTask


def _tqdm_enabled(default: bool = True) -> bool:
    """Decide whether tqdm progress bars should render.

    Order of precedence (highest first):
      1. ``CURATOR_TQDM=0|1`` -- explicit override.
      2. ``default`` argument from the caller (e.g. stage flag).
      3. Whether stderr is a TTY -- avoids thousands of carriage-returns
         in redirected log files.
    """
    env = os.environ.get("CURATOR_TQDM")
    if env is not None:
        return env.strip() not in ("0", "false", "False", "no", "off", "")
    if not default:
        return False
    return sys.stderr.isatty()


def _worker_tag() -> str:
    """Short tag identifying this worker for multi-GPU progress bars."""
    cv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cv:
        return f"gpu{cv.split(',')[0]}"
    return f"pid{os.getpid()}"

if TYPE_CHECKING:
    from lhotse import Cut, CutSet

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


def _expand_nemo_path(path: str) -> list[str]:
    """Expand NeMo-style brace patterns (``_OP_0..49_CL_``) to file list.

    NeMo uses ``_OP_`` for ``{`` and ``_CL_`` for ``}``, matching the
    convention in :func:`nemo.collections.asr.data.audio_to_text.expand_sharded_filepaths`.
    """
    for opener in ("(", "[", "<", "_OP_"):
        path = path.replace(opener, "{")
    for closer in (")", "]", ">", "_CL_"):
        path = path.replace(closer, "}")
    match = re.search(r"\{(\d+)\.\.(\d+)\}", path)
    if not match:
        return [path]
    start_idx, end_idx = int(match.group(1)), int(match.group(2))
    prefix, suffix = path[: match.start()], path[match.end() :]
    return [f"{prefix}{i}{suffix}" for i in range(start_idx, end_idx + 1)]


def _extract_shard_id(path: str) -> str:
    """Extract the numeric shard ID from a manifest/tar filename.

    ``manifest_25.json`` -> ``"25"``, ``audio_3.tar`` -> ``"3"``.
    """
    m = re.search(r"_(\d+)\.\w+$", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]


def merge_shard_embeddings(
    output_dir: str,
    merged_path: str | None = None,
    output_format: str = "npz",
) -> str:
    """Merge per-shard embedding files in *output_dir* into a single file.

    Returns the path to the merged file.
    """
    import glob

    ext = "npz" if output_format == "npz" else "pt"
    files = sorted(glob.glob(os.path.join(output_dir, f"embeddings_*.{ext}")))
    if not files:
        msg = f"No embeddings_*.{ext} files found in {output_dir}"
        raise FileNotFoundError(msg)

    all_ids: list[str] = []
    all_embs: list[np.ndarray] = []
    for f in files:
        if ext == "pt":
            data = torch.load(f, weights_only=False)
            all_ids.extend(data["cut_ids"])
            all_embs.append(data["embeddings"].numpy() if isinstance(data["embeddings"], torch.Tensor) else data["embeddings"])
        else:
            data = np.load(f, allow_pickle=True)
            all_ids.extend(data["cut_ids"])
            all_embs.append(data["embeddings"])

    merged_embs = np.concatenate(all_embs)
    if merged_path is None:
        merged_path = os.path.join(output_dir, f"embeddings_merged.{ext}")

    if ext == "pt":
        torch.save({"cut_ids": all_ids, "embeddings": torch.from_numpy(merged_embs)}, merged_path)
    else:
        np.savez(merged_path, cut_ids=np.array(all_ids, dtype=object), embeddings=merged_embs)

    logger.info(f"Merged {len(files)} shard files → {len(all_ids)} embeddings in {merged_path}")
    return merged_path


@dataclass
class SpeakerEmbeddingLhotseStage(ProcessingStage[_EmptyTask, _EmptyTask]):
    """Load audio from a Lhotse CutSet and extract speaker embeddings.

    Produces one embedding vector per utterance (cut).  Embeddings are saved
    **per-shard** into ``output_path`` (a directory).  For a manifest pattern
    that expands to *K* shards, *K* files named ``embeddings_{shard_id}.npz``
    (or ``.pt``) are written.  Use :func:`merge_shard_embeddings` to combine
    them into a single file if needed.

    Each shard file contains:

    * **cut_ids** -- list of cut identifier strings (length *N*).
    * **embeddings** -- ``(N, D)`` float32 array of speaker embeddings.

    Args:
        model_name: Pretrained NeMo speaker model name.
        input_manifest: NeMo JSON manifest path(s) (brace-expand pattern).
        input_tar: NeMo tarred audio path(s) for ``nemo_tarred``.
        shar_in_dir: Lhotse Shar directory for ``lhotse_shar``.
        lhotse_mode: ``"nemo_tarred"``, ``"lhotse_shar"``, or ``"nemo_row"``.
        target_sample_rate: Resample cuts to this rate (default model expects 16 kHz).
        output_path: Output **directory** for per-shard embedding files.
        output_format: ``"npz"`` or ``"pt"``.
        max_cuts: Optional cap on total number of cuts (for debugging).
        batch_size: Inference batch size (higher → more GPU utilization).
    """

    name: str = "SpeakerEmbeddingLhotseStage"
    model_name: str = "nvidia/speakerverification_en_titanet_large"
    cache_dir: str | None = None
    speaker_model: Any | None = field(default=None, repr=False)
    input_manifest: str = ""
    input_tar: str = ""
    shar_in_dir: str = ""
    lhotse_mode: Literal["nemo_tarred", "lhotse_shar", "nemo_row"] = "nemo_tarred"
    target_sample_rate: int = 16000
    output_path: str = "embeddings"
    output_format: Literal["npz", "pt"] = "npz"
    max_cuts: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=4.0))
    batch_size: int = 64
    show_progress: bool = True

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        if self.speaker_model is not None:
            return
        import nemo.collections.asr as nemo_asr

        try:
            kwargs: dict[str, Any] = {"model_name": self.model_name, "return_model_file": True}
            if self.cache_dir is not None:
                kwargs["cache_dir"] = self.cache_dir
            nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(**kwargs)
        except Exception:
            logger.info(f"Could not pre-cache {self.model_name}; will download on first use")

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self.speaker_model is not None:
            self.speaker_model.eval()
            return
        import nemo.collections.asr as nemo_asr

        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs: dict[str, Any] = {"model_name": self.model_name, "map_location": map_location}
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(**kwargs)
        self.speaker_model.eval()

    # ------------------------------------------------------------------
    # Audio loading & embedding extraction
    # ------------------------------------------------------------------

    def _load_audio(self, cut: Cut) -> np.ndarray:
        """Load mono float32 audio at ``target_sample_rate`` from a cut."""
        if cut.sampling_rate != self.target_sample_rate:
            cut = cut.resample(self.target_sample_rate)
        audio = cut.load_audio()
        if audio.ndim > 1:
            audio = audio[0]
        return audio.astype(np.float32)

    @torch.no_grad()
    def _extract_embeddings_batch(self, audio_list: list[np.ndarray]) -> np.ndarray:
        """Run the speaker model on a batch of waveforms, returning ``(B, D)`` embeddings."""
        device = self.speaker_model.device
        lengths = [a.shape[0] for a in audio_list]
        max_len = max(lengths)
        padded = np.zeros((len(audio_list), max_len), dtype=np.float32)
        for i, a in enumerate(audio_list):
            padded[i, : a.shape[0]] = a
        signal = torch.from_numpy(padded).to(device=device)
        signal_len = torch.tensor(lengths, device=device, dtype=torch.long)
        _, emb = self.speaker_model.forward(input_signal=signal, input_signal_length=signal_len)
        return emb.cpu().numpy()

    def _save_shard(self, path: str, cut_ids: list[str], embeddings: np.ndarray) -> None:
        """Persist one shard's embeddings to *path*."""
        if self.output_format == "pt":
            torch.save(
                {"cut_ids": cut_ids, "embeddings": torch.from_numpy(embeddings)},
                path,
            )
        else:
            np.savez(path, cut_ids=np.array(cut_ids, dtype=object), embeddings=embeddings)

    def _flush_batch(
        self,
        batch_audio: list[np.ndarray],
        batch_ids: list[str],
        cut_ids: list[str],
        embeddings: list[np.ndarray],
    ) -> None:
        """Forward one batch through the speaker model and append results."""
        if not batch_audio:
            return
        embs = self._extract_embeddings_batch(batch_audio)
        cut_ids.extend(batch_ids)
        embeddings.append(embs)

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------

    def _count_total_cuts(self) -> int | None:
        """Count total cuts by reading manifest line counts."""
        if not self.input_manifest.strip():
            return None
        try:
            total = 0
            for p in _expand_nemo_path(self.input_manifest.strip()):
                with open(p) as f:
                    total += sum(1 for _ in f)
            return total
        except (FileNotFoundError, OSError):
            return None

    # ------------------------------------------------------------------
    # Stage entry point
    # ------------------------------------------------------------------

    def _count_shard_cuts(self, manifest_path: str) -> int | None:
        """Count manifest lines for a single shard (used for the inner tqdm bar)."""
        try:
            with open(manifest_path) as f:
                return sum(1 for _ in f)
        except (FileNotFoundError, OSError):
            return None

    def _process_shard(
        self,
        manifest_path: str,
        tar_path: str,
        remaining: int | None,
        progress_position: int = 1,
    ) -> tuple[int, str]:
        """Extract embeddings for a single manifest/tar pair.

        Returns ``(num_processed, output_file_path)``.
        """
        from lhotse import CutSet

        shard_id = _extract_shard_id(manifest_path)
        ext = "pt" if self.output_format == "pt" else "npz"
        shard_output = os.path.join(self.output_path, f"embeddings_{shard_id}.{ext}")

        cuts = CutSet(
            LazyNeMoTarredIterator(manifest_path=manifest_path, tar_paths=tar_path)
        )

        cut_ids: list[str] = []
        embeddings: list[np.ndarray] = []
        batch_audio: list[np.ndarray] = []
        batch_ids: list[str] = []
        count = 0

        shard_total = self._count_shard_cuts(manifest_path)
        if remaining is not None and shard_total is not None:
            shard_total = min(shard_total, remaining)
        elif remaining is not None:
            shard_total = remaining

        bar_disable = not _tqdm_enabled(self.show_progress)
        is_tty = sys.stderr.isatty()
        with tqdm(
            total=shard_total,
            desc=f"  shard {shard_id} [{_worker_tag()}] cuts",
            unit="cut",
            leave=False,
            position=progress_position,
            dynamic_ncols=True,
            disable=bar_disable,
            mininterval=0.1 if is_tty else 5.0,
            miniters=1 if is_tty else 256,
        ) as pbar:
            for cut in cuts:
                try:
                    batch_audio.append(self._load_audio(cut))
                    batch_ids.append(cut.id)
                except Exception:
                    logger.warning(f"Failed to load cut {cut.id!r}, skipping")
                    continue

                if len(batch_audio) >= self.batch_size:
                    self._flush_batch(batch_audio, batch_ids, cut_ids, embeddings)
                    batch_audio, batch_ids = [], []

                count += 1
                pbar.update(1)
                if remaining is not None and count >= remaining:
                    break

            self._flush_batch(batch_audio, batch_ids, cut_ids, embeddings)

        emb_array = np.concatenate(embeddings) if embeddings else np.empty((0, 0), dtype=np.float32)
        self._save_shard(shard_output, cut_ids, emb_array)
        return count, shard_output

    def process(self, _: _EmptyTask) -> _EmptyTask:
        total_cuts = self._count_total_cuts()
        effective_total = total_cuts
        if self.max_cuts is not None and total_cuts is not None:
            effective_total = min(total_cuts, self.max_cuts)
        elif self.max_cuts is not None:
            effective_total = self.max_cuts

        manifest_paths = _expand_nemo_path(self.input_manifest.strip())
        tar_paths = _expand_nemo_path(self.input_tar.strip())
        num_shards = len(manifest_paths)

        logger.info(
            f"Starting extraction: {num_shards} shards, "
            f"~{effective_total or '?'} cuts  (batch_size={self.batch_size})"
        )
        os.makedirs(self.output_path, exist_ok=True)

        bar_disable = not _tqdm_enabled(self.show_progress)
        is_tty = sys.stderr.isatty()
        global_count = 0
        with tqdm(
            total=num_shards,
            desc=f"shards [{_worker_tag()}]",
            unit="shard",
            position=0,
            dynamic_ncols=True,
            disable=bar_disable,
            mininterval=0.1 if is_tty else 2.0,
        ) as shard_bar:
            for shard_idx, (mp, tp) in enumerate(zip(manifest_paths, tar_paths), start=1):
                remaining = None
                if self.max_cuts is not None:
                    remaining = self.max_cuts - global_count
                    if remaining <= 0:
                        break

                n, out_file = self._process_shard(
                    mp, tp, remaining, progress_position=1,
                )
                global_count += n
                pct = f" ({100 * global_count / effective_total:.1f}%)" if effective_total else ""
                logger.info(
                    f"Shard {shard_idx}/{num_shards}: {n} cuts → {os.path.basename(out_file)}  "
                    f"[total {global_count}{pct}]"
                )
                shard_bar.set_postfix(
                    cuts=global_count,
                    last=os.path.basename(out_file),
                )
                shard_bar.update(1)

        logger.info(f"Done. {global_count} embeddings across {num_shards} shard files in {self.output_path}")
        return EmptyTask
