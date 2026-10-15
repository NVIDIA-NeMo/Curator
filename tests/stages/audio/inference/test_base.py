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

"""Architecture tests for shared adapter-backed audio inference behavior."""

from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.stages.audio.inference.base import AdapterInferenceStage
from nemo_curator.stages.audio.inference.sed.stage import SEDInferenceStage
from nemo_curator.stages.audio.inference.speaker_diarization.stage import InferenceSortformerStage
from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage


def test_audio_inference_stages_inherit_one_adapter_stage_base() -> None:
    assert issubclass(ASRStage, AdapterInferenceStage)
    assert issubclass(SEDInferenceStage, AdapterInferenceStage)
    assert issubclass(InferenceSortformerStage, AdapterInferenceStage)
    assert issubclass(VADSegmentationStage, AdapterInferenceStage)


def test_common_adapter_infrastructure_is_not_reimplemented() -> None:
    common_methods = {
        "_adapter_class",
        "_adapter_gpu_count",
        "setup_on_node",
        "setup",
        "teardown",
    }
    for stage_type in (ASRStage, SEDInferenceStage, InferenceSortformerStage, VADSegmentationStage):
        assert common_methods.isdisjoint(stage_type.__dict__)

    # VAD accepts either an in-memory waveform or a file and therefore owns
    # its disjunctive input declaration; the other adapter stages use the base.
    for stage_type in (ASRStage, SEDInferenceStage, InferenceSortformerStage):
        assert "inputs" not in stage_type.__dict__


def test_worker_sizing_uses_the_processing_stage_override() -> None:
    sed = SEDInferenceStage(adapter_target="package.Adapter", checkpoint_path="/checkpoint.pth")
    asr = ASRStage(
        adapter_target="package.Adapter",
        model_id="model",
        max_audio_sec_per_actor=2400.0,
    )
    sortformer = InferenceSortformerStage()
    vad = VADSegmentationStage()

    assert "num_workers_override" not in SEDInferenceStage.__dataclass_fields__
    assert "num_workers_override" not in InferenceSortformerStage.__dataclass_fields__
    assert "num_workers_override" not in VADSegmentationStage.__dataclass_fields__
    assert sed.num_workers() is None
    assert asr.num_workers() is None
    assert sortformer.num_workers() is None
    assert vad.num_workers() is None
    assert sed.with_(num_workers=3).num_workers() == 3
    assert asr.with_(num_workers=3).num_workers() == 3
    assert sortformer.with_(num_workers=3).num_workers() == 3
    assert vad.with_(num_workers=3).num_workers() == 3
