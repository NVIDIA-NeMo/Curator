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

import numpy as np
import pytest

from nemo_curator.stages.audio.global_bucketing import SEGMENT_DESCRIPTOR_KEY, AudioSegmentDescriptor
from nemo_curator.stages.audio.inference.asr.global_fastconformer import FastConformerDispatchStage
from nemo_curator.stages.audio.inference.batch_policy import BatchPolicy
from nemo_curator.tasks import AudioTask, DispatchBatchTask


class _Decoder:
    def __init__(self, sample_rate: int = 10) -> None:
        self.sample_rate = sample_rate
        self.decoded: list[AudioSegmentDescriptor] = []

    def decode(self, descriptor: AudioSegmentDescriptor) -> tuple[np.ndarray, int]:
        self.decoded.append(descriptor)
        return (
            np.zeros(round(descriptor.duration_s * self.sample_rate), dtype=np.float32),
            self.sample_rate,
        )


class _FastConformer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, **kwargs: object) -> list[str]:
        self.calls.append(kwargs)
        return [f"text-{index}" for index in range(len(kwargs["audio"]))]  # type: ignore[arg-type]


def _policy() -> BatchPolicy:
    return BatchPolicy(
        buckets_sec=[0.0, 2.0],
        max_items_per_batch_by_bucket=[4, 2],
        max_audio_sec_per_batch=6.0,
    )


def _batch(owner: str = "FastConformerDispatchStage") -> DispatchBatchTask:
    policy = _policy()
    descriptors = [
        AudioSegmentDescriptor("/audio/a.wav", 0.0, 2.0),
        AudioSegmentDescriptor("/audio/b.wav", 1.0, 3.0),
    ]
    return DispatchBatchTask(
        dataset_name="audio",
        data=[AudioTask(data={SEGMENT_DESCRIPTOR_KEY: descriptor}) for descriptor in descriptors],
        batch_id="batch-0",
        owner_stage=owner,
        sequence_index=0,
        bucket_index=1,
        total_cost=5.0,
        item_costs=(2.0, 3.0),
        cost_unit="seconds",
        policy_signature=policy.dispatch_signature(),
    )


def test_dispatch_owner_decodes_lazily_and_makes_one_call_per_envelope() -> None:
    decoder = _Decoder()
    model = _FastConformer()
    stage = FastConformerDispatchStage(
        batch_policy=_policy(),
        segment_decoder=decoder,
        asr_model=model,
        target_sample_rate=10,
    )

    result = stage.process(_batch())

    assert len(model.calls) == 1
    assert [waveform.shape[0] for waveform in model.calls[0]["audio"]] == [20, 30]
    assert len(decoder.decoded) == 2
    assert [item.data["pred_text"] for item in result.items] == ["text-0", "text-1"]
    assert all("waveform" not in item.data for item in result.items)


def test_dispatch_owner_rejects_envelope_for_another_stage() -> None:
    stage = FastConformerDispatchStage(
        batch_policy=_policy(),
        segment_decoder=_Decoder(),
        asr_model=_FastConformer(),
        target_sample_rate=10,
    )

    with pytest.raises(ValueError, match="belongs to"):
        stage.process(_batch(owner="other-stage"))


def test_dispatch_owner_resamples_bounded_segments_to_model_rate() -> None:
    pytest.importorskip("scipy")
    model = _FastConformer()
    stage = FastConformerDispatchStage(
        batch_policy=_policy(),
        segment_decoder=_Decoder(sample_rate=8),
        asr_model=model,
        target_sample_rate=16,
    )

    stage.process(_batch())

    assert [waveform.shape[0] for waveform in model.calls[0]["audio"]] == [32, 48]


def test_preloaded_waveforms_bypass_decoder_and_preserve_envelope_items() -> None:
    pytest.importorskip("scipy")
    decoder = _Decoder(sample_rate=99)
    model = _FastConformer()
    batch = _batch()
    original_items = list(batch.items)
    batch.items[0].data.update(
        {
            "waveform": np.stack([np.zeros(16, dtype=np.float32), np.ones(16, dtype=np.float32)]),
            "sample_rate": 8,
        }
    )
    batch.items[1].data.update(
        {
            "waveform": np.stack([np.zeros(24, dtype=np.float32), np.ones(24, dtype=np.float32)], axis=1),
            "sample_rate": 8,
        }
    )
    stage = FastConformerDispatchStage(
        batch_policy=_policy(),
        segment_decoder=decoder,
        asr_model=model,
        target_sample_rate=16,
    )

    result = stage.process(batch)

    assert decoder.decoded == []
    assert result.items == original_items
    assert all(
        result_item is original_item for result_item, original_item in zip(result.items, original_items, strict=True)
    )
    assert [waveform.shape for waveform in model.calls[0]["audio"]] == [(32,), (48,)]
    assert all(waveform.dtype == np.float32 for waveform in model.calls[0]["audio"])
    assert all("waveform" in item.data and "sample_rate" in item.data for item in result.items)
