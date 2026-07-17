# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Atomic FastConformer dispatch owner with last-moment CPU decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd, isclose
from numbers import Real
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.global_bucketing import (
    SEGMENT_DESCRIPTOR_KEY,
    AudioSegmentDecoder,
    AudioSegmentDescriptor,
    SoundFileSegmentDecoder,
)
from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask, DispatchBatchTask

if TYPE_CHECKING:
    import numpy as np

    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

_WAVEFORM_2D_NDIM = 2


@dataclass
class FastConformerDispatchStage(ProcessingStage[DispatchBatchTask, DispatchBatchTask]):
    """Validate and execute each planner envelope as one FastConformer call."""

    model_name: str = "nvidia/stt_en_fastconformer_ctc_large"
    pred_text_key: str = "pred_text"
    batch_policy: BatchPolicy = field(default_factory=BatchPolicy)
    segment_decoder: AudioSegmentDecoder | None = None
    asr_model: Any | None = field(default=None, repr=False)
    cache_dir: str | None = None
    target_sample_rate: int | None = None
    num_workers_per_call: int = 0
    name: str = "FastConformerDispatchStage"
    resources: Resources = field(default_factory=lambda: Resources(cpus=2.0, gpu_memory_gb=16.0))
    batch_size: int = 4

    def __post_init__(self) -> None:
        if not self.batch_policy.enabled:
            msg = "FastConformerDispatchStage requires BatchPolicy.enabled=true"
            raise ValueError(msg)
        if self.num_workers_per_call < 0:
            msg = "num_workers_per_call must be non-negative"
            raise ValueError(msg)
        if self.target_sample_rate is not None and self.target_sample_rate <= 0:
            msg = "target_sample_rate must be positive when set"
            raise ValueError(msg)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        if self.asr_model is not None:
            return
        try:
            import nemo.collections.asr as nemo_asr

            kwargs: dict[str, Any] = {"model_name": self.model_name, "return_model_file": True}
            if self.cache_dir is not None:
                kwargs["cache_dir"] = self.cache_dir
            nemo_asr.models.ASRModel.from_pretrained(**kwargs)
        except Exception as exc:
            msg = f"Failed to download FastConformer model {self.model_name!r}"
            raise RuntimeError(msg) from exc

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self.segment_decoder is None:
            self.segment_decoder = SoundFileSegmentDecoder()
        if self.asr_model is not None:
            return
        try:
            import nemo.collections.asr as nemo_asr
            import torch

            device = torch.device("cuda" if self.resources.requires_gpu and torch.cuda.is_available() else "cpu")
            kwargs: dict[str, Any] = {"model_name": self.model_name, "map_location": device}
            if self.cache_dir is not None:
                kwargs["cache_dir"] = self.cache_dir
            self.asr_model = nemo_asr.models.ASRModel.from_pretrained(**kwargs)
        except Exception as exc:
            msg = f"Failed to load FastConformer model {self.model_name!r}"
            raise RuntimeError(msg) from exc

    def process(self, task: DispatchBatchTask) -> DispatchBatchTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[DispatchBatchTask]) -> list[DispatchBatchTask]:
        if self.asr_model is None:
            msg = "FastConformerDispatchStage is not initialized; call setup() first"
            raise RuntimeError(msg)
        if self.segment_decoder is None:
            self.segment_decoder = SoundFileSegmentDecoder()

        results: list[DispatchBatchTask] = []
        for batch in tasks:
            children = self._validate_dispatch(batch)
            descriptors = [self._descriptor(child) for child in children]
            # This is intentionally the first payload-producing operation and
            # immediately precedes the sole model call for this envelope.
            waveforms = [
                self._waveform_for_model(child, descriptor)
                for child, descriptor in zip(children, descriptors, strict=True)
            ]
            outputs = self.asr_model.transcribe(
                audio=waveforms,
                batch_size=len(waveforms),
                return_hypotheses=False,
                num_workers=self.num_workers_per_call,
                verbose=False,
            )
            texts = self._normalize_outputs(outputs)
            if len(texts) != len(children):
                msg = f"FastConformer returned {len(texts)} results for {len(children)} dispatch items"
                raise RuntimeError(msg)
            for child, text in zip(children, texts, strict=True):
                child.data[self.pred_text_key] = text
            results.append(batch.with_items(children))
        return results

    def _validate_dispatch(self, batch: DispatchBatchTask) -> list[AudioTask]:
        if not isinstance(batch, DispatchBatchTask) or not batch.validate():
            msg = "FastConformerDispatchStage received an invalid dispatch envelope"
            raise TypeError(msg)
        if batch.owner_stage != self.name:
            msg = f"Dispatch batch {batch.batch_id!r} belongs to {batch.owner_stage!r}, not {self.name!r}"
            raise ValueError(msg)
        if batch.policy_signature != self.batch_policy.dispatch_signature(cost_unit=batch.cost_unit):
            msg = f"Dispatch batch {batch.batch_id!r} policy does not match its owner"
            raise ValueError(msg)
        children = list(batch.items)
        if not all(isinstance(child, AudioTask) for child in children):
            msg = f"Dispatch batch {batch.batch_id!r} contains non-audio tasks"
            raise TypeError(msg)
        observed_costs = [self._descriptor(child).duration_s for child in children]
        if any(
            not isclose(observed, planned, rel_tol=1e-7, abs_tol=1e-6)
            for observed, planned in zip(observed_costs, batch.item_costs, strict=True)
        ):
            msg = f"Dispatch batch {batch.batch_id!r} descriptor costs changed after planning"
            raise ValueError(msg)
        if not isclose(sum(observed_costs), batch.total_cost, rel_tol=1e-7, abs_tol=1e-6):
            msg = f"Dispatch batch {batch.batch_id!r} has an inconsistent total cost"
            raise ValueError(msg)
        if any(self.batch_policy.bucket_for(cost) != batch.bucket_index for cost in observed_costs):
            msg = f"Dispatch batch {batch.batch_id!r} mixes duration buckets"
            raise ValueError(msg)
        item_cap = self.batch_policy.max_items_per_batch_by_bucket[batch.bucket_index]
        if len(children) > item_cap:
            msg = f"Dispatch batch {batch.batch_id!r} exceeds its item cap"
            raise ValueError(msg)
        cost_cap = self.batch_policy.max_audio_sec_per_batch
        if cost_cap is not None and len(children) > 1 and batch.total_cost > cost_cap:
            msg = f"Dispatch batch {batch.batch_id!r} exceeds its total duration cap"
            raise ValueError(msg)
        return children

    def _waveform_for_model(self, task: AudioTask, descriptor: AudioSegmentDescriptor) -> np.ndarray:
        if task.data.get("waveform") is not None:
            sample_rate = task.data.get("sample_rate")
            if isinstance(sample_rate, bool) or not isinstance(sample_rate, Real) or sample_rate <= 0:
                msg = "Preloaded waveform requires a positive numeric sample_rate"
                raise ValueError(msg)
            waveform = self._to_mono_numpy_1d(task.data["waveform"])
            return self._normalize_for_model(waveform, int(sample_rate))
        return self._decode_for_model(descriptor)

    def _decode_for_model(self, descriptor: AudioSegmentDescriptor) -> np.ndarray:
        decoder = self.segment_decoder
        if decoder is None:
            msg = "FastConformerDispatchStage has no segment decoder"
            raise RuntimeError(msg)
        waveform, source_rate = decoder.decode(descriptor)
        return self._normalize_for_model(self._to_mono_numpy_1d(waveform), source_rate)

    def _normalize_for_model(self, waveform: np.ndarray, source_rate: int) -> np.ndarray:
        import numpy as np

        target_rate = self._model_sample_rate()
        if source_rate == target_rate:
            return np.ascontiguousarray(waveform, dtype=np.float32)
        if source_rate <= 0:
            msg = f"Audio input has invalid sample rate {source_rate}"
            raise ValueError(msg)

        from scipy.signal import resample_poly

        divisor = gcd(int(source_rate), target_rate)
        resampled = resample_poly(waveform, target_rate // divisor, int(source_rate) // divisor)
        return np.ascontiguousarray(resampled, dtype=np.float32)

    @staticmethod
    def _to_mono_numpy_1d(waveform: object) -> np.ndarray:
        import numpy as np

        if hasattr(waveform, "detach"):
            waveform = waveform.detach().cpu().numpy()
        samples = np.asarray(waveform, dtype=np.float32)
        if samples.size == 0:
            return samples.reshape(0)
        if samples.ndim == 0:
            return samples.reshape(1)
        if samples.ndim == 1:
            return np.ascontiguousarray(samples)

        squeezed = np.squeeze(samples)
        if squeezed.ndim == 1:
            return np.ascontiguousarray(squeezed.astype(np.float32, copy=False))
        if squeezed.ndim == _WAVEFORM_2D_NDIM:
            channel_axis = 0 if squeezed.shape[0] <= squeezed.shape[1] else 1
            mono = squeezed.mean(axis=channel_axis)
            return np.ascontiguousarray(mono.astype(np.float32, copy=False))
        msg = f"Expected a 1-D or 2-D waveform, got shape {samples.shape}"
        raise ValueError(msg)

    def _model_sample_rate(self) -> int:
        if self.target_sample_rate is not None:
            return self.target_sample_rate
        preprocessor = getattr(self.asr_model, "preprocessor", None)
        sample_rate = getattr(preprocessor, "_sample_rate", None)
        if isinstance(sample_rate, (int, float)) and sample_rate > 0:
            return int(sample_rate)
        config = getattr(self.asr_model, "cfg", None)
        if config is not None and hasattr(config, "get"):
            sample_rate = config.get("sample_rate")
            if isinstance(sample_rate, (int, float)) and sample_rate > 0:
                return int(sample_rate)
        return 16_000

    @staticmethod
    def _descriptor(task: AudioTask) -> AudioSegmentDescriptor:
        descriptor = task.data.get(SEGMENT_DESCRIPTOR_KEY)
        if not isinstance(descriptor, AudioSegmentDescriptor):
            msg = f"Audio task is missing an {AudioSegmentDescriptor.__name__}"
            raise TypeError(msg)
        return descriptor

    @staticmethod
    def _normalize_outputs(outputs: object) -> list[str]:
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if not isinstance(outputs, list):
            msg = f"Unsupported FastConformer output type: {type(outputs).__name__}"
            raise TypeError(msg)
        texts: list[str] = []
        for output in outputs:
            primary = output[0] if isinstance(output, list) and output else output
            text = getattr(primary, "text", primary)
            if not isinstance(text, str):
                msg = f"Unsupported FastConformer result type: {type(primary).__name__}"
                raise TypeError(msg)
            texts.append(text)
        return texts
