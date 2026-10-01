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

from nemo_curator.models.audio.vad.base import VADAdapter, VADResult, VADSegment


class _CompleteAdapter:
    def download_weights_on_node(self) -> None:
        pass

    def load_model(self, *, num_gpus: int) -> None:
        pass

    def unload_model(self) -> None:
        pass

    def detect_batch(self, items: list[dict]) -> list[VADResult]:
        return [VADResult([]) for _ in items]


class _MissingLifecycle:
    def detect_batch(self, items: list[dict]) -> list[VADResult]:
        return [VADResult([]) for _ in items]


def test_result_schema_is_typed_and_ordered() -> None:
    result = VADResult([VADSegment(start=0.25, end=1.5)])
    assert result.segments == [VADSegment(0.25, 1.5)]


def test_runtime_protocol_accepts_a_complete_structural_adapter() -> None:
    assert isinstance(_CompleteAdapter(), VADAdapter)


def test_runtime_protocol_rejects_an_incomplete_adapter() -> None:
    assert not isinstance(_MissingLifecycle(), VADAdapter)
