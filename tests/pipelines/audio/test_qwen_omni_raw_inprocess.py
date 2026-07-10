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
# ruff: noqa: ANN001, ANN202, S108

import pytest
from omegaconf import OmegaConf

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import ManifestReader, ManifestReaderStage, ManifestWriterStage
from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.payload_lifecycle import AudioPayloadMaterializeStage, PayloadReleaseStage


def _cfg(*, consumers: list[str] | None = None, release_after: str = "qwen_omni"):
    return OmegaConf.create(
        {
            "audio_reader_skip_on_read_error": True,
            "payload_lifecycle": {
                "enabled": True,
                "materialize_after": "manifest_reader",
                "source_key": "audio_filepath",
                "ref_key": "audio_ref",
                "consumers": consumers or ["qwen_omni"],
                "release_after": release_after,
                "target_sample_rate": 16000,
                "target_nchannels": 1,
                "node_memory_fraction": 0.55,
                "max_node_payload_bytes": "10g",
                "admission_actor_name": "test_payload_admission",
                "admission_poll_interval_s": 0.5,
            },
        }
    )


def _logical_stages():
    reader = ManifestReader(
        manifest_path="/tmp/input.jsonl",
    )
    asr = ASRStage(
        adapter_target="nemo_curator.models.asr.QwenOmniASRAdapter",
        model_id="test/model",
        name="qwen_omni",
        waveform_key="audio",
        waveform_ref_key="audio_ref",
        sample_rate_key="sr",
        pred_text_key="prediction",
        disfluency_text_key="raw_prediction",
        skip_me_key="skip_me",
        max_inference_duration_s=120,
        adapter_batch_size=1,
    )
    writer = ManifestWriterStage(output_path="/tmp/output.jsonl")
    return reader, asr, writer


def _asr_stage(name: str, *, pred_text_key: str = "prediction"):
    return ASRStage(
        adapter_target="nemo_curator.models.asr.QwenOmniASRAdapter",
        model_id="test/model",
        name=name,
        waveform_key="audio",
        waveform_ref_key="audio_ref",
        sample_rate_key="sr",
        pred_text_key=pred_text_key,
        disfluency_text_key=None,
        skip_me_key="skip_me",
        max_inference_duration_s=120,
        adapter_batch_size=1,
    )


def _expanded(stages, cfg=None):
    cfg = cfg or _cfg()
    pipeline = Pipeline(
        name="test_pipeline",
        stages=list(stages),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    pipeline.build()
    return pipeline.stages


def test_payload_lifecycle_build_is_idempotent() -> None:
    pipeline = Pipeline(
        name="test_pipeline",
        stages=list(_logical_stages()),
        config=OmegaConf.to_container(_cfg(), resolve=True),
    )

    pipeline.build()
    first_names = [stage.name for stage in pipeline.stages]
    pipeline.build()

    assert [stage.name for stage in pipeline.stages] == first_names


def test_payload_lifecycle_helpers_use_fresh_pipeline_run_id_without_mutating_config() -> None:
    cfg = OmegaConf.to_container(_cfg(), resolve=True)
    cfg["_curator_pipeline_run_id"] = "stale-run"
    first_pipeline = Pipeline(
        name="test_pipeline",
        stages=list(_logical_stages()),
        config=cfg,
    )
    second_pipeline = Pipeline(
        name="test_pipeline",
        stages=list(_logical_stages()),
        config=cfg,
    )

    first_pipeline.build()
    second_pipeline.build()
    first_materialize = next(
        stage for stage in first_pipeline.stages if isinstance(stage, AudioPayloadMaterializeStage)
    )
    second_materialize = next(
        stage for stage in second_pipeline.stages if isinstance(stage, AudioPayloadMaterializeStage)
    )

    assert first_materialize.run_id != "stale-run"
    assert second_materialize.run_id != "stale-run"
    assert first_materialize.run_id != second_materialize.run_id
    assert cfg["_curator_pipeline_run_id"] == "stale-run"


def test_payload_lifecycle_add_stage_after_build_replans_from_logical_graph() -> None:
    reader, asr, writer = _logical_stages()
    pipeline = Pipeline(
        name="test_pipeline",
        stages=[reader, asr],
        config=OmegaConf.to_container(_cfg(), resolve=True),
    )

    pipeline.build()
    assert [type(stage) for stage in pipeline.stages] == [
        FilePartitioningStage,
        ManifestReaderStage,
        AudioPayloadMaterializeStage,
        ASRStage,
        PayloadReleaseStage,
    ]

    pipeline.add_stage(writer)
    pipeline.build()

    assert [type(stage) for stage in pipeline.stages] == [
        FilePartitioningStage,
        ManifestReaderStage,
        AudioPayloadMaterializeStage,
        ASRStage,
        PayloadReleaseStage,
        ManifestWriterStage,
    ]
    assert [stage.name for stage in pipeline.stages if stage.is_source_stage] == ["file_partitioning"]
    assert [stage.name for stage in pipeline.stages if stage.is_sink_stage] == ["manifest_writer"]


def test_logical_graph_expands_to_payload_lifecycle() -> None:
    expanded = _expanded(_logical_stages())

    assert [type(stage) for stage in expanded] == [
        FilePartitioningStage,
        ManifestReaderStage,
        AudioPayloadMaterializeStage,
        ASRStage,
        PayloadReleaseStage,
        ManifestWriterStage,
    ]

    materialize = expanded[2]
    assert materialize.target_sample_rate == 16000
    assert materialize.target_nchannels == 1
    assert materialize.duration_key == "duration"
    assert materialize.num_samples_key == "num_samples"
    assert materialize.waveform_key == "audio"
    assert materialize.waveform_ref_key == "audio_ref"
    assert materialize.sample_rate_key == "sr"
    assert materialize.skip_on_read_error is True
    assert materialize.node_memory_fraction == 0.55
    assert materialize.max_node_payload_bytes == "10g"
    assert materialize.admission_actor_name == "test_payload_admission"
    assert materialize.admission_poll_interval_s == 0.5
    assert materialize.run_id

    release = expanded[4]
    assert release.payload_ref_key == "audio_ref"
    assert release.waveform_key == "audio"
    assert expanded[3]._curator_payload_ref_key == "audio_ref"


def test_payload_lifecycle_can_span_multiple_backend_visible_consumers() -> None:
    reader = ManifestReader(manifest_path="/tmp/input.jsonl")
    first = _asr_stage("gpu_stage_1", pred_text_key="stage_1_text")
    second = _asr_stage("gpu_stage_2", pred_text_key="stage_2_text")
    writer = ManifestWriterStage(output_path="/tmp/output.jsonl")

    expanded = _expanded(
        [reader, first, second, writer],
        _cfg(consumers=["gpu_stage_1", "gpu_stage_2"], release_after="gpu_stage_2"),
    )

    assert [type(stage) for stage in expanded] == [
        FilePartitioningStage,
        ManifestReaderStage,
        AudioPayloadMaterializeStage,
        ASRStage,
        ASRStage,
        PayloadReleaseStage,
        ManifestWriterStage,
    ]
    assert [stage.name for stage in expanded] == [
        "file_partitioning",
        "manifest_reader_stage",
        "audio_payload_materialize",
        "gpu_stage_1",
        "gpu_stage_2",
        "payload_release",
        "manifest_writer",
    ]
    assert [stage._curator_payload_ref_key for stage in expanded[3:5]] == ["audio_ref", "audio_ref"]


def test_payload_lifecycle_rejects_plural_payload_source_config() -> None:
    reader, asr, writer = _logical_stages()
    cfg = _cfg()
    cfg.payload_lifecycle.payload_keys = ["audio_filepath", "reference_audio_filepath"]

    with pytest.raises(ValueError, match="supports exactly one payload source"):
        _expanded([reader, asr, writer], cfg)


@pytest.mark.parametrize("key", ["lease_ttl_s", "materialized_lease_ttl_s"])
def test_payload_lifecycle_rejects_removed_lease_config(key: str) -> None:
    reader, asr, writer = _logical_stages()
    cfg = _cfg()
    cfg.payload_lifecycle[key] = 3600

    with pytest.raises(ValueError, match="internal lifecycle policy"):
        _expanded([reader, asr, writer], cfg)


def test_raw_qwen_config_rejects_explicit_helper_stages() -> None:
    reader, asr, writer = _logical_stages()

    with pytest.raises(ValueError, match="logical stages only"):
        _expanded([reader, AudioPayloadMaterializeStage(), asr, writer])

    with pytest.raises(ValueError, match="logical stages only"):
        _expanded([reader, asr, PayloadReleaseStage(), writer])
