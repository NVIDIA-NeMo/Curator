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

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import (
    GlobalSegmentAssemblerStage,
    ManifestReader,
    ManifestReaderStage,
    ManifestWriterStage,
    _ManifestReaderGlobalBucketingStage,
)
from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.payload_lifecycle import AudioPayloadMaterializeStage, PayloadReleaseStage


class _DualPayloadConsumer(ASRStage):
    def payload_bindings(self) -> list[dict[str, str]]:
        return [
            {
                "ref_key": "audio_ref",
                "waveform_key": "audio",
                "sample_rate_key": "sr",
                "num_samples_key": "samples",
            },
            {
                "ref_key": "reference_ref",
                "waveform_key": "reference_audio",
                "sample_rate_key": "reference_sr",
                "num_samples_key": "reference_samples",
            },
        ]


def _cfg(*, consumers: list[str] | None = None, release_after: str = "qwen_omni"):
    return OmegaConf.create(
        {
            "audio_reader_skip_on_read_error": True,
            "payload_lifecycle": {
                "enabled": True,
                "materialize_after": "manifest_reader",
                "payload_keys": ["audio_filepath"],
                "ref_key": "audio_ref",
                "consumers": consumers or ["qwen_omni"],
                "release_after": release_after,
                "target_sample_rate": 8000,
                "target_nchannels": 2,
                "node_memory_fraction": 0.55,
                "max_node_payload_bytes": "10g",
                "lease_ttl_s": 1234,
                "admission_actor_name": "test_payload_admission",
                "admission_poll_interval_s": 0.5,
            },
        }
    )


def _logical_stages(*, enable_global_bucketing: bool = False):
    reader = ManifestReader(
        manifest_path="/tmp/input.jsonl",
        enable_global_bucketing=enable_global_bucketing,
        owner_stage="qwen_omni" if enable_global_bucketing else None,
        target_sample_rate=8000,
        target_nchannels=2,
        duration_key="dur",
        num_samples_key="samples",
        max_inference_duration_s=120,
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
    writer = ManifestWriterStage(output_path="/tmp/output.jsonl", duration_key="dur")
    return reader, asr, writer


def _asr_stage(name: str, *, pred_text_key: str = "prediction", max_inference_duration_s: float = 120):
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
        max_inference_duration_s=max_inference_duration_s,
        adapter_batch_size=1,
    )


def _dual_stage(name: str = "dual_gpu_stage"):
    return _DualPayloadConsumer(
        adapter_target="nemo_curator.models.asr.QwenOmniASRAdapter",
        model_id="test/model",
        name=name,
        waveform_key="audio",
        waveform_ref_key="audio_ref",
        sample_rate_key="sr",
        pred_text_key="prediction",
        disfluency_text_key=None,
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
        stages=list(_logical_stages(enable_global_bucketing=False)),
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
        stages=list(_logical_stages(enable_global_bucketing=False)),
        config=cfg,
    )
    second_pipeline = Pipeline(
        name="test_pipeline",
        stages=list(_logical_stages(enable_global_bucketing=False)),
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
    reader, asr, writer = _logical_stages(enable_global_bucketing=False)
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


def test_global_segment_assembler_is_single_worker_actor_stage() -> None:
    assembler = GlobalSegmentAssemblerStage()

    assert assembler.num_workers() == 1
    assert assembler.ray_stage_spec() == {RayStageSpecKeys.IS_ACTOR_STAGE: True}
    assert assembler.xenna_stage_spec() == {}


def test_bucket_off_logical_graph_expands_to_payload_lifecycle() -> None:
    expanded = _expanded(_logical_stages(enable_global_bucketing=False))

    assert [type(stage) for stage in expanded] == [
        FilePartitioningStage,
        ManifestReaderStage,
        AudioPayloadMaterializeStage,
        ASRStage,
        PayloadReleaseStage,
        ManifestWriterStage,
    ]

    materialize = expanded[2]
    assert materialize.target_sample_rate == 8000
    assert materialize.target_nchannels == 2
    assert materialize.duration_key == "dur"
    assert materialize.num_samples_key == "samples"
    assert materialize.waveform_key == "audio"
    assert materialize.waveform_ref_key == "audio_ref"
    assert materialize.sample_rate_key == "sr"
    assert materialize.skip_on_read_error is True
    assert materialize.node_memory_fraction == 0.55
    assert materialize.max_node_payload_bytes == "10g"
    assert materialize.lease_ttl_s == 1234
    assert materialize.admission_actor_name == "test_payload_admission"
    assert materialize.admission_poll_interval_s == 0.5
    assert materialize.run_id

    release = expanded[4]
    assert release.payload_ref_key == "audio_ref"
    assert release.waveform_key == "audio"


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


def test_payload_lifecycle_supports_multiple_source_keys() -> None:
    reader = ManifestReader(manifest_path="/tmp/input.jsonl")
    consumer = _dual_stage()
    writer = ManifestWriterStage(output_path="/tmp/output.jsonl")
    cfg = OmegaConf.create(
        {
            "payload_lifecycle": {
                "enabled": True,
                "materialize_after": "manifest_reader",
                "payloads": [
                    {
                        "source_key": "audio_filepath",
                        "ref_key": "audio_ref",
                        "waveform_key": "audio",
                        "sample_rate_key": "sr",
                        "num_samples_key": "samples",
                    },
                    {
                        "source_key": "reference_audio_filepath",
                        "ref_key": "reference_ref",
                        "waveform_key": "reference_audio",
                        "sample_rate_key": "reference_sr",
                        "num_samples_key": "reference_samples",
                        "duration_key": "reference_duration",
                    },
                ],
                "consumers": ["dual_gpu_stage"],
                "release_after": "dual_gpu_stage",
            }
        }
    )

    expanded = _expanded([reader, consumer, writer], cfg)

    assert [type(stage) for stage in expanded] == [
        FilePartitioningStage,
        ManifestReaderStage,
        AudioPayloadMaterializeStage,
        AudioPayloadMaterializeStage,
        _DualPayloadConsumer,
        PayloadReleaseStage,
        ManifestWriterStage,
    ]
    assert expanded[2].audio_filepath_key == "audio_filepath"
    assert expanded[2].waveform_ref_key == "audio_ref"
    assert expanded[3].audio_filepath_key == "reference_audio_filepath"
    assert expanded[3].waveform_ref_key == "reference_ref"
    assert expanded[3].duration_key == "reference_duration"


def test_global_bucket_on_accepts_one_owner_among_multiple_payload_consumers() -> None:
    reader = ManifestReader(
        manifest_path="/tmp/input.jsonl",
        enable_global_bucketing=True,
        owner_stage="gpu_stage_1",
        max_inference_duration_s=120,
    )
    first = _asr_stage("gpu_stage_1", pred_text_key="stage_1_text")
    second = _asr_stage("gpu_stage_2", pred_text_key="stage_2_text")
    writer = ManifestWriterStage(output_path="/tmp/output.jsonl")

    expanded = _expanded(
        [reader, first, second, writer],
        _cfg(consumers=["gpu_stage_1", "gpu_stage_2"], release_after="gpu_stage_2"),
    )

    assert [type(stage) for stage in expanded] == [
        _ManifestReaderGlobalBucketingStage,
        AudioPayloadMaterializeStage,
        ASRStage,
        ASRStage,
        PayloadReleaseStage,
        GlobalSegmentAssemblerStage,
        ManifestWriterStage,
    ]


def test_global_bucket_on_accepts_segment_inputs_when_owner_has_largest_window() -> None:
    reader = ManifestReader(
        manifest_path="/tmp/input.jsonl",
        enable_global_bucketing=True,
        owner_stage="gpu_stage_1",
        max_inference_duration_s=240,
    )
    first = _asr_stage("gpu_stage_1", pred_text_key="stage_1_text", max_inference_duration_s=240)
    second = _asr_stage("gpu_stage_2", pred_text_key="stage_2_text", max_inference_duration_s=120)
    writer = ManifestWriterStage(output_path="/tmp/output.jsonl")
    cfg = _cfg(consumers=["gpu_stage_1", "gpu_stage_2"], release_after="gpu_stage_2")
    cfg.global_audio_scheduler = {
        "segment_input_keys": ["audio_filepath", "source_lang"],
    }

    expanded = _expanded([reader, first, second, writer], cfg)

    global_reader = expanded[0]
    assert isinstance(global_reader, _ManifestReaderGlobalBucketingStage)
    assert global_reader.segment_input_keys == ["audio_filepath", "source_lang"]


def test_global_bucket_on_propagates_parent_key_overwrite_config() -> None:
    reader = ManifestReader(
        manifest_path="/tmp/input.jsonl",
        enable_global_bucketing=True,
        owner_stage="gpu_stage_1",
        max_inference_duration_s=120,
    )
    first = _asr_stage("gpu_stage_1", pred_text_key="stage_1_text")
    writer = ManifestWriterStage(output_path="/tmp/output.jsonl")
    cfg = _cfg(consumers=["gpu_stage_1"], release_after="gpu_stage_1")
    cfg.global_segment_assembler = {"overwrite": ["speaker_id"]}

    expanded = _expanded([reader, first, writer], cfg)

    assembler = next(stage for stage in expanded if isinstance(stage, GlobalSegmentAssemblerStage))
    assert assembler.overwrite == ["speaker_id"]
    assert assembler.field_merge_strategies["speaker_id"] == "overwrite"


def test_global_bucket_on_rejects_owner_that_is_not_largest_inference_window() -> None:
    reader = ManifestReader(
        manifest_path="/tmp/input.jsonl",
        enable_global_bucketing=True,
        owner_stage="gpu_stage_1",
        max_inference_duration_s=120,
    )
    first = _asr_stage("gpu_stage_1")
    second = _asr_stage("gpu_stage_2", max_inference_duration_s=240)
    writer = ManifestWriterStage(output_path="/tmp/output.jsonl")
    cfg = _cfg(consumers=["gpu_stage_1", "gpu_stage_2"], release_after="gpu_stage_2")

    with pytest.raises(ValueError, match="largest max_inference_duration_s"):
        _expanded([reader, first, second, writer], cfg)


def test_global_bucket_on_rejects_reader_window_that_differs_from_owner() -> None:
    reader = ManifestReader(
        manifest_path="/tmp/input.jsonl",
        enable_global_bucketing=True,
        owner_stage="gpu_stage_1",
        max_inference_duration_s=120,
    )
    first = _asr_stage("gpu_stage_1", max_inference_duration_s=240)
    second = _asr_stage("gpu_stage_2")
    writer = ManifestWriterStage(output_path="/tmp/output.jsonl")

    with pytest.raises(ValueError, match="Reader has 120s"):
        _expanded(
            [reader, first, second, writer],
            _cfg(consumers=["gpu_stage_1", "gpu_stage_2"], release_after="gpu_stage_2"),
        )


def test_global_bucket_on_rejects_reader_window_below_owner() -> None:
    reader = ManifestReader(
        manifest_path="/tmp/input.jsonl",
        enable_global_bucketing=True,
        owner_stage="gpu_stage_1",
        max_inference_duration_s=60,
    )
    first = _asr_stage("gpu_stage_1")
    second = _asr_stage("gpu_stage_2")
    writer = ManifestWriterStage(output_path="/tmp/output.jsonl")

    with pytest.raises(ValueError, match="max_inference_duration_s must match"):
        _expanded(
            [reader, first, second, writer],
            _cfg(consumers=["gpu_stage_1", "gpu_stage_2"], release_after="gpu_stage_2"),
        )


def test_global_bucket_on_rejects_owner_outside_payload_consumers() -> None:
    reader = ManifestReader(
        manifest_path="/tmp/input.jsonl",
        enable_global_bucketing=True,
        owner_stage="gpu_stage_3",
        max_inference_duration_s=120,
    )
    first = _asr_stage("gpu_stage_1")
    second = _asr_stage("gpu_stage_2")
    writer = ManifestWriterStage(output_path="/tmp/output.jsonl")

    with pytest.raises(ValueError, match="owner_stage must select exactly one stage listed"):
        _expanded(
            [reader, first, second, writer],
            _cfg(consumers=["gpu_stage_1", "gpu_stage_2"], release_after="gpu_stage_2"),
        )


def test_global_bucket_on_requires_single_owner_stage() -> None:
    with pytest.raises(ValueError, match="requires owner_stage"):
        ManifestReader(manifest_path="/tmp/input.jsonl", enable_global_bucketing=True)

    with pytest.raises(ValueError, match="exactly one stage selector"):
        ManifestReader(
            manifest_path="/tmp/input.jsonl",
            enable_global_bucketing=True,
            owner_stage=["gpu_stage_1", "gpu_stage_2"],  # type: ignore[arg-type]
        )


def test_global_bucket_on_logical_graph_adds_segment_assembler() -> None:
    expanded = _expanded(_logical_stages(enable_global_bucketing=True))

    assert [type(stage) for stage in expanded] == [
        _ManifestReaderGlobalBucketingStage,
        AudioPayloadMaterializeStage,
        ASRStage,
        PayloadReleaseStage,
        GlobalSegmentAssemblerStage,
        ManifestWriterStage,
    ]

    assembler = expanded[4]
    assert assembler.text_keys_to_join == ["prediction", "raw_prediction"]
    assert assembler.skip_me_key == "skip_me"
    assert assembler.waveform_key == "audio"
    assert assembler.waveform_ref_key == "audio_ref"
    assert assembler.duration_key == "dur"
    assert assembler.num_samples_key == "samples"


def test_raw_qwen_config_rejects_explicit_helper_stages() -> None:
    reader, asr, writer = _logical_stages(enable_global_bucketing=True)

    with pytest.raises(ValueError, match="logical stages only"):
        _expanded([reader, AudioPayloadMaterializeStage(), asr, writer])

    with pytest.raises(ValueError, match="logical stages only"):
        _expanded([reader, asr, PayloadReleaseStage(), writer])

    with pytest.raises(ValueError, match="logical stages only"):
        _expanded([reader, asr, GlobalSegmentAssemblerStage(), writer])
