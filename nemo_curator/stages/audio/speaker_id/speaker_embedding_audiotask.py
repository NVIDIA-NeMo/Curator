# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""AudioTask-native speaker embedding stage for AIS-streamed pipelines.

Counterpart of :class:`SpeakerEmbeddingRequestStage`, which expects a
:class:`DocumentBatch` (JSONL-backed dataframe) of audio file paths.
This stage instead consumes :class:`AudioTask` instances carrying an
in-memory waveform (``task.data["waveform"]``) and runs the same NeMo
speaker model on each.

Why a sister stage instead of a runtime branch:

* On multi-node Slurm allocations, downstream stages can't rely on
  ``audio_filepath`` written into per-shard JSONLs because each diarize
  actor produces a path on its own node-local NVMe.  Re-streaming from
  AIS gives a clean waveform on the actor that needs it.
* Keeps :class:`SpeakerEmbeddingRequestStage` API stable for the legacy
  JSONL flow.

Outputs:

* The ``embedding`` key is added to ``task.data``.
* On ``teardown()``, each actor writes its accumulated embeddings as
  ``output_dir/embeddings_<worker_tag>.npz`` with two arrays: ``cut_ids``
  (audio_filepath strings) and ``embeddings`` (float32 array).  The
  glob pattern matches what
  :func:`tutorials.audio.hifi_pipeline.run_pipeline_beta._gather_corpus_for_beta`
  already supports.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio.utils.audio_io import ensure_waveform
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class SpeakerEmbeddingAudioTaskStage(ProcessingStage[AudioTask, AudioTask]):
    """Run a NeMo speaker model on each task's in-memory waveform.

    Args:
        model_name: Pretrained NeMo speaker model name.
        cache_dir: Optional directory for model download cache.
        target_sample_rate: Resample to this rate before extracting (TitaNet wants 16 kHz).
        output_dir: If set, write per-actor ``embeddings_<tag>.npz`` files
            on ``teardown()``.  Each NPZ has ``cut_ids`` and ``embeddings``
            arrays.
        audio_filepath_key: Key used as ``cut_id`` in the saved NPZ.
            Should be a stable identifier (tar member name) so the diarize
            manifest's ``audio_filepath`` field matches.
        embedding_key: Key under which the numpy embedding is added to
            ``task.data``.
    """

    name: str = "SpeakerEmbeddingAudioTaskStage"
    model_name: str = "nvidia/speakerverification_en_titanet_large"
    cache_dir: str | None = None
    speaker_model: Any | None = field(default=None, repr=False)
    target_sample_rate: int = 16000
    output_dir: str = ""
    audio_filepath_key: str = "audio_filepath"
    embedding_key: str = "embedding"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=4.0))
    batch_size: int = 64

    _accumulated_ids: list[str] = field(default_factory=list, init=False, repr=False)
    _accumulated_embs: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _flush_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_name and self.speaker_model is None:
            msg = "Either model_name or speaker_model is required."
            raise ValueError(msg)

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
        except Exception:  # noqa: BLE001
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

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.embedding_key]

    @torch.no_grad()
    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        device = self.speaker_model.device
        signal = torch.tensor(audio[np.newaxis, :], device=device, dtype=torch.float32)
        signal_len = torch.tensor([audio.shape[0]], device=device)
        _, emb = self.speaker_model.forward(input_signal=signal, input_signal_length=signal_len)
        return emb.squeeze().detach().cpu().numpy().astype(np.float32)

    def process(self, task: AudioTask) -> AudioTask:
        audio = ensure_waveform(task, target_sr=self.target_sample_rate)
        emb = self._extract_embedding(audio)

        # Use audio_filepath as the cut identifier — stable across the
        # AIS-streamed pipeline.  NemoTarShardReader sets it to the tar
        # member name and the in-memory-waveform Sortformer does not
        # mutate it, so the diarize manifest's ``audio_filepath`` field
        # is the natural key for cluster_scotch's _gather_corpus_for_beta
        # to join embeddings against.  Falls back to task_id only when
        # audio_filepath is missing.
        cut_id = str(task.data.get(self.audio_filepath_key) or task.task_id)
        self._accumulated_ids.append(cut_id)
        self._accumulated_embs.append(emb)

        out_data = dict(task.data)
        out_data[self.embedding_key] = emb
        return AudioTask(
            task_id=f"{task.task_id}_emb",
            dataset_name=task.dataset_name,
            filepath_key=task.filepath_key or self.audio_filepath_key,
            data=out_data,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Process a batch of tasks, then flush accumulated embeddings to NPZ.

        Flushing here (rather than in :meth:`teardown`) is what actually
        runs across Xenna actors — teardown is unreliable in the
        streaming executor.  Each actor's process_batch flush produces
        an ``embeddings_<tag>_part<N>.npz`` that cluster_scotch can
        glob with the existing ``embeddings_*.npz`` pattern.
        """
        results = [self.process(t) for t in tasks]
        self._flush_npz()
        return results

    def teardown(self) -> None:
        # Backup flush in case some path hits teardown before process_batch
        # completes (e.g. driver-side cleanup); usually no-op since
        # process_batch already drained the buffer.
        self._flush_npz()

    def _flush_npz(self) -> None:
        if not self.output_dir or not self._accumulated_embs:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        # Tag the NPZ with something stable per actor + a per-flush
        # counter so concurrent writes never collide.  CUDA_VISIBLE_DEVICES
        # + pid is unique within a job.
        self._flush_count += 1
        tag = (
            os.environ.get("CUDA_VISIBLE_DEVICES", "x").split(",")[0]
            + "_"
            + str(os.getpid())
        )
        path = os.path.join(
            self.output_dir, f"embeddings_{tag}_part{self._flush_count}.npz"
        )
        np.savez(
            path,
            cut_ids=np.array(self._accumulated_ids, dtype=object),
            embeddings=np.stack(self._accumulated_embs, axis=0).astype(np.float32),
        )
        logger.info(f"Saved {len(self._accumulated_ids)} embeddings to {path}")
        self._accumulated_ids.clear()
        self._accumulated_embs.clear()
