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


def test_asr_and_sed_inherit_one_adapter_stage_base() -> None:
    assert issubclass(ASRStage, AdapterInferenceStage)
    assert issubclass(SEDInferenceStage, AdapterInferenceStage)


def test_common_adapter_infrastructure_is_not_reimplemented() -> None:
    common_methods = {
        "_adapter_class",
        "_adapter_gpu_count",
        "inputs",
        "setup_on_node",
        "setup",
        "teardown",
    }
    assert common_methods.isdisjoint(ASRStage.__dict__)
    assert common_methods.isdisjoint(SEDInferenceStage.__dict__)


def test_worker_sizing_uses_the_processing_stage_override() -> None:
    sed = SEDInferenceStage(adapter_target="package.Adapter", checkpoint_path="/checkpoint.pth")
    asr = ASRStage(adapter_target="package.Adapter", model_id="model")

    assert "num_workers_override" not in SEDInferenceStage.__dataclass_fields__
    assert sed.num_workers() is None
    assert asr.num_workers() is None
    assert sed.with_(num_workers=3).num_workers() == 3
    assert asr.with_(num_workers=3).num_workers() == 3
