# modality: audio
#
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

import json
from pathlib import Path

import fsspec
import numpy as np
import torch
from omegaconf import OmegaConf

from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.checkpointing.audio.serialization import serialize_audio_task
from nemo_curator.stages.audio.io.materialize import CleanupTemporaryAudioStage
from nemo_curator.stages.audio.io.tarred import MaterializeTarredAudioStage, TarredAudioManifestReader
from nemo_curator.stages.audio.speaker_id import BuildUploadKeyStage, SpeakerEmbeddingAudioTaskStage
from nemo_curator.stages.file_io import UploadFilesStage
from nemo_curator.tasks import AudioTask


class _FakeSpeakerModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def eval(self) -> "_FakeSpeakerModel":
        return self

    def forward(self, *, input_signal: torch.Tensor, input_signal_length: torch.Tensor) -> tuple[None, torch.Tensor]:
        _ = input_signal_length
        batch_size = int(input_signal.shape[0])
        embeddings = torch.ones((batch_size, 3), dtype=torch.float32, device=input_signal.device)
        return None, embeddings


def _make_task() -> AudioTask:
    return AudioTask(
        task_id="sample-a",
        dataset_name="dataset",
        sample_key="sample-a",
        data={
            "audio_filepath": "sample-a.wav",
            "output_key": "diarization/sample-a",
            "diarized_segments": [
                {"start_time": 0.0, "end_time": 0.25, "speaker": "speaker_0"},
                {"start_time": 0.25, "end_time": 0.5, "speaker": "speaker_1"},
            ],
            "waveform": torch.zeros(1, 8000, dtype=torch.float32),
            "sample_rate": 16000,
        },
        _metadata={"checkpoint_shard_id": "shard_0"},
    )


def test_speaker_embedding_stage_writes_npz_and_returns_json_safe_task(tmp_path: Path) -> None:
    stage = SpeakerEmbeddingAudioTaskStage(
        model_name="",
        speaker_model=_FakeSpeakerModel(),
        output_dir=str(tmp_path / "npz"),
    )
    result = stage.process(_make_task())

    output_path = Path(result.data["output_filepath"])
    assert result.sample_key == "sample-a"
    assert result.data["embedding_count"] == 2
    assert result._metadata["checkpoint_shard_id"] == "shard_0"
    assert output_path.exists()
    assert "waveform" not in result.data

    npz = np.load(output_path, allow_pickle=True)
    assert npz["embeddings"].shape == (2, 3)
    assert list(npz["speaker_ids"]) == ["speaker_0", "speaker_1"]
    json.dumps(serialize_audio_task(result), sort_keys=True)


def test_speaker_embedding_stage_upload_pipeline_fields_work_with_upload_files_stage(tmp_path: Path) -> None:
    embedding_stage = SpeakerEmbeddingAudioTaskStage(
        model_name="",
        speaker_model=_FakeSpeakerModel(),
        output_dir=str(tmp_path / "npz"),
    )
    task = embedding_stage.process(_make_task())

    key_stage = BuildUploadKeyStage(key_prefix="speaker_embeddings")
    task = key_stage.process(task)

    upload_stage = UploadFilesStage(
        source_field_path="output_filepath",
        output_field_path="uploaded_output_filepath",
        bucket="tts_granary",
        protocol="memory",
        key_field_path="speaker_embedding_upload_key",
    )
    task = upload_stage.process(task)

    uploaded_path = task.data["uploaded_output_filepath"]
    assert uploaded_path == "memory://tts_granary/speaker_embeddings/diarization/sample-a.npz"
    with fsspec.open(uploaded_path, "rb") as fin:
        assert fin.read()


def test_speaker_workflow_yaml_instantiates_expected_stages(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    config_path = (
        repo_root
        / "tutorials"
        / "audio"
        / "hifi_pipeline"
        / "speaker_workflow"
        / "0_embeddings"
        / "embeddings_workflow.yaml"
    )
    cfg = OmegaConf.load(config_path)
    cfg.manifest_paths = "s3://bucket/manifests/manifest__OP_0..1_CL_.json"
    cfg.tar_paths = "s3://bucket/audio/audio__OP_0..1_CL_.tar"
    cfg.output_dir = str(tmp_path / "output")
    cfg.upload_bucket = "tts_granary"

    pipeline = create_pipeline_from_yaml(cfg)

    assert len(pipeline.stages) == 6
    assert isinstance(pipeline.stages[0], TarredAudioManifestReader)
    assert isinstance(pipeline.stages[1], MaterializeTarredAudioStage)
    assert isinstance(pipeline.stages[2], SpeakerEmbeddingAudioTaskStage)
    assert isinstance(pipeline.stages[3], BuildUploadKeyStage)
    assert isinstance(pipeline.stages[4], UploadFilesStage)
    assert isinstance(pipeline.stages[5], CleanupTemporaryAudioStage)
