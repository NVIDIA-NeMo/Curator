# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import hydra
import pytest
from omegaconf import DictConfig

from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.tasks import AudioTask

TUTORIAL_DIR = Path(__file__).resolve().parents[2] / "tutorials/audio/granary_v2/full_english_pipeline"


@pytest.fixture(scope="session")
def shared_ray_cluster() -> None:
    """These tests construct stages without starting workers or loading models."""


@pytest.fixture
def pipeline_config() -> DictConfig:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(TUTORIAL_DIR)):
        return hydra.compose(
            config_name="pipeline",
            overrides=[
                "manifest_path=/data/recordings.jsonl",
                "model_dir=/models",
                "output_dir=/output",
                "endpoint=http://text-host:8000/v1",
                "cache_root=/cache",
            ],
        )


def test_segment_handoff_and_english_selection(pipeline_config: DictConfig):
    stages = create_pipeline_from_yaml(pipeline_config, log_config=False).stages
    by_name = {stage.name: stage for stage in stages}
    assert len(by_name) == len(stages)
    names = list(by_name)
    assert names.index("AudioSegmentWriter") < names.index("MetadataManifestWriter")
    assert names.index("MetadataManifestWriter") < names.index("PreserveByValueStage")
    assert names.index("PreserveByValueStage") < names.index("ResampleEnglishClip") < names.index("ASR_inference")
    assert by_name["ResampleEnglishClip"].audio_filepath_key == "audio_filepath"
    assert by_name["ResampleEnglishClip"].resampled_audio_filepath_key == by_name["ASR_inference"].audio_filepath_key
    assert by_name["MetadataManifestWriter"].output_path == "/output/metadata.jsonl"
    assert stages[-1].output_path == "/output/english.jsonl"

    english = AudioTask(data={"source_lang": "en", "audio_filepath": "/output/audio/clip.opus"})
    rejected = [
        AudioTask(data={"source_lang": "hi"}),
        AudioTask(data={"source_lang": "skipped", "_skipme": "LID disagreement"}),
        AudioTask(data={"source_lang": "", "vad_empty": True}),
        AudioTask(data={"vad_empty": True}),
    ]
    assert by_name["PreserveByValueStage"].process_batch([english, *rejected]) == [english]


def test_hallucination_does_not_recover_parent_transcript(pipeline_config: DictConfig):
    stages = create_pipeline_from_yaml(pipeline_config, log_config=False).stages
    by_name = {stage.name: stage for stage in stages}
    selector = by_name["SelectBestPrediction"]
    assert selector.reference_text_key is None
    task = AudioTask(
        data={
            "text": "A reference for the entire recording, including other segments.",
            "primary_model_prediction": "suspicious clip prediction",
            "_skipme": "Hallucination detected",
        }
    )
    result = selector.process(task)
    assert result.data["best_prediction"] == "suspicious clip prediction"
    assert result.data["_skipme"] == "Hallucination detected"
    assert by_name["RecoverEntities"].ground_truth_key == "primary_model_prediction"
    assert by_name["ASR_inference"].preserve_existing_skip


def test_text_operations_preserve_source_prompts(pipeline_config: DictConfig):
    pipeline = create_pipeline_from_yaml(pipeline_config, log_config=False)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(TUTORIAL_DIR.parent / "text_pipeline")):
        source = hydra.compose(
            config_name="text_pipeline",
            overrides=[
                "manifest_path=/data/clips.jsonl",
                "output_path=/output/text.jsonl",
                "endpoint=http://text-host:8000/v1",
                "cache_root=/cache",
            ],
        )
    reference = create_pipeline_from_yaml(source, log_config=False)
    combined_operations = [operation for stage in pipeline.stages[-8:-1] for operation in stage.operation_stages()]
    reference_operations = [operation for stage in reference.stages[1:-1] for operation in stage.operation_stages()]
    assert len(combined_operations) == len(reference_operations) == 9
    for actual, expected in zip(combined_operations, reference_operations, strict=True):
        assert (actual.name, actual.prompt, actual.text_key, actual.output_text_key) == (
            expected.name,
            expected.prompt,
            expected.text_key,
            expected.output_text_key,
        )
