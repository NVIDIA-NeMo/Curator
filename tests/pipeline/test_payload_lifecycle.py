# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.pipeline.payload_lifecycle import payload_lifecycle_enabled
from nemo_curator.stages.audio.common import ManifestReaderStage
from nemo_curator.stages.audio.inference.asr.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.payload_lifecycle import (
    AudioPayloadMaterializeStage,
    PayloadReleaseStage,
    PayloadResolvingStage,
)
from nemo_curator.stages.resources import Resources


@dataclass
class _StructuralSource(ProcessingStage[object, object]):
    name: str = "structural_source"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: object) -> object:
        return task


@dataclass
class _FastConformerEnvelopeConsumer(ProcessingStage[object, object]):
    name: str = "fastconformer_owner"
    resources: Resources = field(default_factory=lambda: Resources(cpus=3.0))
    batch_size: int = 7

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: object) -> object:
        return task

    def num_workers(self) -> int | None:
        return 2


def test_pipeline_inserts_audio_payload_helpers_for_local_batching() -> None:
    reader = ManifestReaderStage()
    consumer = InferenceAsrNemoStage(model_name="fake", asr_model=MagicMock(), batch_size=4)
    pipeline = Pipeline(
        name="payload-local",
        stages=[reader, consumer],
        config={
            "payload_lifecycle": {
                "enabled": True,
                "materialize_after": "manifest_reader_stage",
                "release_after": "ASR_inference",
                "consumers": ["ASR_inference"],
                "max_node_payload_bytes": "1g",
            }
        },
    )

    pipeline.build()

    assert [type(stage) for stage in pipeline.stages] == [
        ManifestReaderStage,
        AudioPayloadMaterializeStage,
        InferenceAsrNemoStage,
        PayloadReleaseStage,
    ]
    assert consumer.batch_size == 4
    assert consumer._curator_tracks_payload_refs is True


def test_pipeline_payload_expansion_is_idempotent() -> None:
    pipeline = Pipeline(
        name="payload-local",
        stages=[ManifestReaderStage(), InferenceAsrNemoStage(model_name="fake", asr_model=MagicMock())],
        config={
            "payload_lifecycle": {
                "enabled": True,
                "materialize_after": "ManifestReaderStage",
                "release_after": "InferenceAsrNemoStage",
                "consumers": ["InferenceAsrNemoStage"],
            }
        },
    )

    pipeline.build()
    first_types = [type(stage) for stage in pipeline.stages]
    pipeline.build()

    assert [type(stage) for stage in pipeline.stages] == first_types


def test_pipeline_uses_default_audio_materializer_and_wraps_structural_consumer() -> None:
    source = _StructuralSource()
    consumer = _FastConformerEnvelopeConsumer()
    pipeline = Pipeline(
        name="structural-payload",
        stages=[source, consumer],
        config={
            "payload_lifecycle": {
                "enabled": True,
                "materialize_after": source.name,
                "release_after": consumer.name,
                "consumers": [consumer.name],
            }
        },
    )

    pipeline.build()

    assert [type(stage) for stage in pipeline.stages] == [
        _StructuralSource,
        AudioPayloadMaterializeStage,
        PayloadResolvingStage,
        PayloadReleaseStage,
    ]
    wrapper = pipeline.stages[2]
    assert wrapper.wrapped_stage is consumer
    assert wrapper.name == consumer.name
    assert wrapper.resources is consumer.resources
    assert wrapper.batch_size == consumer.batch_size
    assert wrapper.num_workers() == consumer.num_workers()


def test_pipeline_accepts_configurable_dotted_materializer_target() -> None:
    source = _StructuralSource()
    consumer = _FastConformerEnvelopeConsumer()
    pipeline = Pipeline(
        name="configured-materializer",
        stages=[source, consumer],
        config={
            "payload_lifecycle": {
                "enabled": True,
                "materialize_after": source.name,
                "release_after": consumer.name,
                "consumers": [consumer.name],
                "materializer_target": "nemo_curator.stages.payload_lifecycle:AudioPayloadMaterializeStage",
            }
        },
    )

    pipeline.build()

    assert isinstance(pipeline.stages[1], AudioPayloadMaterializeStage)


@pytest.mark.parametrize("execution_mode", ["streaming", "batch"])
def test_payload_lifecycle_run_rejects_xenna_executor(execution_mode: str) -> None:
    source = _StructuralSource()
    consumer = _FastConformerEnvelopeConsumer()
    pipeline = Pipeline(
        name="xenna-rejected",
        stages=[source, consumer],
        config={
            "payload_lifecycle": {
                "enabled": True,
                "materialize_after": source.name,
                "release_after": consumer.name,
                "consumers": [consumer.name],
            }
        },
    )

    with pytest.raises(RuntimeError, match="not supported on XennaExecutor"):
        pipeline.run(XennaExecutor(config={"execution_mode": execution_mode}))


def test_payload_lifecycle_enabled_reads_the_config_gate() -> None:
    assert payload_lifecycle_enabled({"payload_lifecycle": {"enabled": True}})
    assert not payload_lifecycle_enabled({"payload_lifecycle": {"enabled": False}})
    assert not payload_lifecycle_enabled({"payload_lifecycle": {}})
    assert not payload_lifecycle_enabled({})
    with pytest.raises(TypeError, match="must be a mapping"):
        payload_lifecycle_enabled({"payload_lifecycle": ["enabled"]})
