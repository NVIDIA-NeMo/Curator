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

from __future__ import annotations

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
torch = pytest.importorskip("torch")

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, StageContract  # noqa: E402
from nemo_curator.stages.audio._agent._residency import (  # noqa: E402
    produce_audio_filepath,
    resolve_audio,
    resolve_audio_path,
)
from nemo_curator.stages.audio.alm.alm_data_builder import ALMDataBuilderStage  # noqa: E402
from nemo_curator.stages.audio.alm.alm_data_overlap import ALMDataOverlapStage  # noqa: E402
from nemo_curator.stages.audio.common import (  # noqa: E402
    GetAudioDurationStage,
    ManifestReaderStage,
    PreserveByValueStage,
)
from nemo_curator.stages.audio.io.convert import AudioToDocumentStage  # noqa: E402
from nemo_curator.stages.audio.postprocessing.timestamp_mapper import TimestampMapperStage  # noqa: E402
from nemo_curator.stages.audio.preprocessing.concatenation import SegmentConcatenationStage  # noqa: E402
from nemo_curator.stages.audio.preprocessing.mono_conversion import MonoConversionStage  # noqa: E402
from nemo_curator.stages.audio.tagging.merge_alignment_diarization import MergeAlignmentDiarizationStage  # noqa: E402
from nemo_curator.stages.audio.tagging.prepare_module_segments import PrepareModuleSegmentsStage  # noqa: E402
from nemo_curator.stages.audio.tagging.split import (  # noqa: E402
    JoinSplitAudioMetadataStage,
    SplitASRAlignJoinStage,
    SplitLongAudioStage,
)
from nemo_curator.tasks import AudioTask  # noqa: E402


def test_residency_helpers_accept_file_waveform_and_custom_keys(tmp_path) -> None:  # noqa: ANN001
    audio_path = tmp_path / "input.wav"
    data = np.stack(
        [
            np.linspace(-0.5, 0.5, 32, dtype=np.float32),
            np.linspace(0.5, -0.5, 32, dtype=np.float32),
        ],
        axis=1,
    )
    sf.write(audio_path, data, 16000)

    resolved = resolve_audio({"path": str(audio_path)}, audio_filepath_key="path", mono=False)
    assert resolved is not None
    waveform, sample_rate = resolved
    assert sample_rate == 16000
    assert tuple(waveform.shape) == (2, 32)

    item = {
        "wf": torch.stack([torch.ones(16), torch.zeros(16)]),
        "sr": 8000,
    }
    temp_path = resolve_audio_path(
        item, residency="waveform", waveform_key="wf", sample_rate_key="sr", temp_dir=str(tmp_path)
    )
    assert temp_path is not None
    info = sf.info(temp_path)
    assert info.samplerate == 8000
    assert info.channels == 2

    produce_audio_filepath(item, "next.wav", key="path", original_key="old_path")
    assert item["path"] == "next.wav"
    assert "old_path" not in item
    produce_audio_filepath(item, "final.wav", key="path", original_key="old_path")
    assert item["old_path"] == "next.wav"
    assert item["path"] == "final.wav"


def test_lightweight_audio_stages_expose_agent_contracts(tmp_path) -> None:  # noqa: ANN001, ARG001
    stages = [
        MonoConversionStage(),
        SegmentConcatenationStage(),
        GetAudioDurationStage(),
        PreserveByValueStage("keep", True),
        ManifestReaderStage(),
        AudioToDocumentStage(keep_keys=["text"]),
        TimestampMapperStage(),
        MergeAlignmentDiarizationStage(),
        PrepareModuleSegmentsStage(),
        SplitLongAudioStage(),
        JoinSplitAudioMetadataStage(),
        SplitASRAlignJoinStage(),
        ALMDataBuilderStage(),
        ALMDataOverlapStage(),
    ]

    for stage in stages:
        assert isinstance(stage, AgentReady)
        assert isinstance(stage.describe(), StageContract)

    extraction_contract = SegmentConcatenationStage().describe()
    assert extraction_contract.metadata_writes == ["segment_mappings"]
    assert extraction_contract.cardinality == "N:1"

    doc_stage = AudioToDocumentStage(batch_size=2, keep_keys=["text"])
    assert doc_stage.batch_size == 2
    assert doc_stage.keep_keys == ["text"]


def test_segment_concatenation_preserves_metadata_and_stage_perf() -> None:
    stage = SegmentConcatenationStage(silence_duration_sec=0.0)
    parent = AudioTask(
        dataset_name="ds",
        data={
            "segments": [
                {
                    "waveform": torch.ones(1, 8),
                    "sample_rate": 8,
                    "start_ms": 0,
                    "end_ms": 1000,
                    "segment_num": 0,
                    "original_file": "source.wav",
                },
                {
                    "waveform": torch.zeros(1, 8),
                    "sample_rate": 8,
                    "start_ms": 1000,
                    "end_ms": 2000,
                    "segment_num": 1,
                    "original_file": "source.wav",
                },
            ]
        },
        _metadata={"upstream": {"kept": True}},
        _stage_perf=["perf-entry"],  # type: ignore[list-item]
    )

    result = stage.process(parent)

    assert isinstance(result, AudioTask)
    assert result.dataset_name == "ds"
    assert result._metadata["upstream"] == {"kept": True}
    assert "segment_mappings" in result._metadata
    assert result._stage_perf == ["perf-entry"]


def test_timestamp_mapper_contract_declares_data_replacement() -> None:
    contract = TimestampMapperStage().describe()

    assert contract.preserves_upstream_keys is False
    assert "original_file" in contract.writes.data_keys
