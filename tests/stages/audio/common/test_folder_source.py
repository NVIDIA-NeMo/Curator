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


from nemo_curator.stages.audio.common import (
    CreateInitialManifestAudioFolderStage,
)


def test_bounded_audio_folder_source_is_not_row_independent() -> None:
    from nemo_curator.stages.audio._agent._agent_registry import static_contract

    bounded = CreateInitialManifestAudioFolderStage(data_dir="/tmp/x", max_samples=10)  # noqa: S108
    unbounded = CreateInitialManifestAudioFolderStage(data_dir="/tmp/x")  # noqa: S108

    assert static_contract(CreateInitialManifestAudioFolderStage).gates.per_row_independent is False
    assert bounded.describe().gates.per_row_independent is False
    assert unbounded.describe().gates.per_row_independent is True
