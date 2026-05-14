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

"""NeMo ASR model wrapper + ``ProcessingStage`` shell.

Two classes live here side by side because they share the same
``nemo.collections.asr.models.ASRModel`` lifecycle:

- :class:`NemoASRModel` — owns download / load / transcribe / teardown for
  a single NeMo ASR checkpoint (e.g. Parakeet-TDT v3
  ``nvidia/parakeet-tdt-0.6b-v3``). Exposes both file-path
  (:meth:`~NemoASRModel.transcribe_files`) and in-memory waveform
  (:meth:`~NemoASRModel.transcribe_waveforms`) inference, so it can be
  reused by other stages (e.g. ``InferenceLanguageRoutedAsrStage``)
  without writing temp ``.wav`` files.
- :class:`InferenceAsrNemoStage` — thin ``ProcessingStage`` over
  :class:`NemoASRModel` that takes ``audio_filepath`` inputs and writes
  ``pred_text``. Use this directly as a top-level pipeline stage when you
  just want to run a NeMo ASR model on a list of files.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any

import nemo.collections.asr as nemo_asr
import numpy as np
import torch
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.base import ModelInterface
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_DEFAULT_TARGET_SR = 16000


class NemoASRModel(ModelInterface):
    """Wrapper for a single NeMo ``ASRModel`` checkpoint."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str | None = None,
        inference_batch_size: int = 16,
        preloaded_model: Any | None = None,
        target_sample_rate: int = _DEFAULT_TARGET_SR,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.inference_batch_size = max(1, int(inference_batch_size))
        self.preloaded_model = preloaded_model
        self.target_sample_rate = int(target_sample_rate)
        self.asr_model: Any | None = preloaded_model

    @property
    def model_id_names(self) -> list[str]:
        return [self.model_name]

    def setup_on_node(self) -> None:
        """Pre-download checkpoint into the NeMo cache (no-op when ``preloaded_model`` was given)."""
        if self.preloaded_model is not None:
            return
        kwargs: dict[str, Any] = {"model_name": self.model_name, "return_model_file": True}
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        try:
            nemo_asr.models.ASRModel.from_pretrained(**kwargs)
        except Exception as e:
            msg = f"Failed to download {self.model_name}"
            raise RuntimeError(msg) from e

    def setup(self, device: torch.device | str | None = None) -> None:
        """Load the checkpoint onto ``device`` (no-op if already loaded or ``preloaded_model`` is set)."""
        if self.asr_model is not None:
            return
        kwargs: dict[str, Any] = {"model_name": self.model_name}
        if device is not None:
            kwargs["map_location"] = device
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        logger.info(f"Loading NeMo ASR model={self.model_name} device={device}")
        try:
            self.asr_model = nemo_asr.models.ASRModel.from_pretrained(**kwargs)
        except Exception as e:
            msg = f"Failed to load {self.model_name}"
            raise RuntimeError(msg) from e
        try:
            self.asr_model.eval()
        except AttributeError:
            pass
        self._disable_cuda_graphs()
        logger.info(f"NeMo ASR model loaded: {self.model_name}")

    def _disable_cuda_graphs(self) -> None:
        # CUDA graphs in NeMo's RNN-T / TDT greedy decoders crash with
        # cudaErrorIllegalAddress when the GPU memory pool layout changes
        # between capture and replay — which happens whenever a co-tenant
        # (vLLM) on the same device allocates/frees.
        if self.asr_model is None:
            return
        from omegaconf import OmegaConf, open_dict

        decoding_cfg = getattr(self.asr_model.cfg, "decoding", None)
        if decoding_cfg is None:
            return
        with open_dict(self.asr_model.cfg):
            greedy_cfg = OmegaConf.select(self.asr_model.cfg, "decoding.greedy")
            if greedy_cfg is None:
                self.asr_model.cfg.decoding.greedy = OmegaConf.create({})
            self.asr_model.cfg.decoding.greedy.use_cuda_graph_decoder = False
            self.asr_model.cfg.decoding.greedy.allow_cuda_graphs = False
        try:
            self.asr_model.change_decoding_strategy(self.asr_model.cfg.decoding)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"change_decoding_strategy failed when disabling CUDA graphs: {e}")

    def teardown(self) -> None:
        """Release the model and clear CUDA cache."""
        if self.asr_model is None:
            return
        del self.asr_model
        self.asr_model = None
        self.preloaded_model = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110
            pass

    def transcribe_files(self, files: list[str]) -> list[str]:
        """Transcribe a list of audio file paths via NeMo ``ASRModel.transcribe``."""
        if self.asr_model is None:
            msg = "NeMo ASR model not loaded; call setup() first."
            raise RuntimeError(msg)
        outputs = self.asr_model.transcribe(files)
        return self._extract_texts(outputs)

    def transcribe_waveforms(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
    ) -> list[str]:
        """Transcribe in-memory waveforms; resamples each to ``target_sample_rate`` and mixes to mono.

        Returns one string per input. Empty / zero-length waveforms produce ``""``.
        """
        if self.asr_model is None:
            msg = "NeMo ASR model not loaded; call setup() first."
            raise RuntimeError(msg)
        if not waveforms:
            return []

        prepared, empty_mask = self._prepare_waveforms(waveforms, sample_rates)

        with torch.inference_mode():
            outputs = self.asr_model.transcribe(
                prepared,
                batch_size=self.inference_batch_size,
                verbose=False,
            )

        texts = self._extract_texts(outputs)
        return ["" if is_empty else t for t, is_empty in zip(texts, empty_mask, strict=True)]

    def _prepare_waveforms(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
    ) -> tuple[list[np.ndarray], list[bool]]:
        """Resample each clip to ``target_sample_rate``, mix down to mono, and substitute silence for empty inputs."""
        import torchaudio.functional as F_ta

        target = self.target_sample_rate
        prepared: list[np.ndarray] = []
        empty_mask: list[bool] = []
        for w, sr in zip(waveforms, sample_rates, strict=True):
            arr = np.asarray(w)
            if arr.size == 0:
                empty_mask.append(True)
                # Pad with a small silent buffer so transcribe() can still process the slot;
                # we'll overwrite the resulting text with "" via empty_mask after inference.
                prepared.append(np.zeros(target // 10, dtype=np.float32))
                continue
            empty_mask.append(False)
            wav = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
            if wav.ndim > 1:
                wav = wav.mean(dim=-1)
            if int(sr) != target:
                wav = F_ta.resample(wav, orig_freq=int(sr), new_freq=target)
            prepared.append(wav.contiguous().numpy())
        return prepared, empty_mask

    @staticmethod
    def _extract_texts(outputs: Any) -> list[str]:
        """Normalise NeMo ``transcribe`` outputs to a flat ``list[str]``.

        Handles the common return shapes: ``(hyps, _)`` tuples, ``list[list[Hypothesis]]``,
        ``list[list[str]]``, ``list[Hypothesis]``, and bare ``list[str]``.
        """
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        texts: list[str] = []
        for out in outputs:
            if isinstance(out, list):
                inner = out[0] if out else ""
                texts.append(inner.text if hasattr(inner, "text") else str(inner))
            elif hasattr(out, "text"):
                texts.append(out.text)
            else:
                texts.append(str(out))
        return texts


@dataclass
class InferenceAsrNemoStage(ProcessingStage[AudioTask, AudioTask]):
    """Speech recognition inference using a NeMo ASR model.

    Thin ``ProcessingStage`` over :class:`NemoASRModel` (defined just above).
    Reads an ``audio_filepath`` per ``AudioTask``, calls
    ``NemoASRModel.transcribe_files``, and writes ``pred_text``.

    Args:
        model_name: Pretrained NeMo ASR model name.
            See full list at https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/all_chkpt.html
        cache_dir: Optional directory for model download cache.
            When set, NeMo stores/loads the pretrained checkpoint here
            instead of the default cache location.
        asr_model: Optional pre-loaded NeMo ``ASRModel`` (or test double).
            When provided, ``setup_on_node`` / ``setup`` are no-ops and the
            stage transcribes through it directly.
        filepath_key: Key in the entry dict pointing to the audio file.
        pred_text_key: Key where the predicted transcription is stored.
    """

    name: str = "ASR_inference"
    model_name: str = ""
    cache_dir: str | None = None
    asr_model: Any | None = field(default=None, repr=False)
    filepath_key: str = "audio_filepath"
    pred_text_key: str = "pred_text"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 16
    _wrapper: NemoASRModel | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_name and not self.asr_model:
            msg = "Either model_name or asr_model is required for InferenceAsrNemoStage"
            raise ValueError(msg)
        self._wrapper = NemoASRModel(
            model_name=self.model_name or "preloaded",
            cache_dir=self.cache_dir,
            inference_batch_size=self.batch_size,
            preloaded_model=self.asr_model,
        )

    def check_cuda(self) -> torch.device:
        return torch.device("cuda") if self.resources.gpus > 0 else torch.device("cpu")

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        assert self._wrapper is not None
        self._wrapper.setup_on_node()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        assert self._wrapper is not None
        self._wrapper.setup(device=self.check_cuda())
        self.asr_model = self._wrapper.asr_model

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.filepath_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.filepath_key, self.pred_text_key]

    def transcribe(self, files: list[str]) -> list[str]:
        assert self._wrapper is not None
        if self._wrapper.asr_model is None and self.asr_model is not None:
            self._wrapper.asr_model = self.asr_model
        return self._wrapper.transcribe_files(files)

    def process(self, task: AudioTask) -> AudioTask:
        msg = "InferenceAsrNemoStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if len(tasks) == 0:
            return []
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task.task_id} missing required columns for {type(self).__name__}: {self.inputs()}"
                raise ValueError(msg)
        files = [t.data[self.filepath_key] for t in tasks]
        texts = self.transcribe(files)
        for task, text in zip(tasks, texts, strict=True):
            task.data[self.pred_text_key] = text
        return tasks
