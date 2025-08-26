import argparse
import os
import tempfile
from operator import le
from pathlib import Path
from typing import ClassVar

import pytest
from omegaconf import OmegaConf

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.examples.audio.fleurs.pipeline import create_audio_pipeline
from ray_curator.examples.audio.run import create_pipeline_from_yaml
from ray_curator.pipeline import Pipeline
from ray_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from ray_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from ray_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from ray_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from ray_curator.stages.text.io.reader import JsonlReader


def get_test_data_path() -> str:
    """Returns path to test data"""
    current_root = os.path.abspath(__file__)
    return os.path.abspath(os.path.join(current_root, "../../../data/audio"))


def get_examples_audio_path() -> str:
    """Returns path to test data"""
    current_root = os.path.abspath(__file__)
    return os.path.abspath(os.path.join(current_root, "../../../../ray_curator/examples/audio"))


class TestFleursWer:
    """Test suite for TestAsrInference."""

    test_data_root: ClassVar[str] = get_test_data_path()
    examples_audio_path: ClassVar[str] = get_examples_audio_path()
    executor: ClassVar[XennaExecutor] = XennaExecutor()
    drop_fields: ClassVar[list[str]] = ["audio_filepath"]

    def read_json(self, file_paths: str | list[str]) -> Pipeline:
        p = Pipeline(name="read", description="Read json")
        p.add_stage(JsonlReader(file_paths=file_paths))
        return p.run(self.executor)

    @pytest.mark.skip("Import NeMo without apex")
    @pytest.mark.gpu
    def test_py_run_pipeline(self):
        # General arguments
        cfg = argparse.Namespace(
            raw_data_dir=os.path.join(self.test_data_root, "armenian/fleurs"),
            model_name="nvidia/stt_hy_fastconformer_hybrid_large_pc",
            lang="hy_am",
            split="dev",
            wer_threshold=5.5,
        )

        # Act: Create the pipeline
        pipeline = create_audio_pipeline(cfg)

        # Assert: Check pipeline object
        assert isinstance(pipeline, Pipeline), "Should return a Pipeline instance"
        assert pipeline.name == "audio_inference"
        assert "Inference audio" in pipeline.description

        # Check number of stages
        assert len(pipeline.stages) == 7  # We know there should be exactly 7 stages

        # Check individual stages are of the expected type
        assert isinstance(pipeline.stages[0], CreateInitialManifestFleursStage)
        assert pipeline.stages[0].lang == "hy_am"
        assert pipeline.stages[0].split == "dev"
        assert pipeline.stages[0].raw_data_dir.split("/")[-1] == "fleurs"

        assert isinstance(pipeline.stages[1], InferenceAsrNemoStage)
        assert pipeline.stages[1].model_name == "nvidia/stt_hy_fastconformer_hybrid_large_pc"

        assert isinstance(pipeline.stages[2], GetPairwiseWerStage)
        assert pipeline.stages[2].wer_key == "wer"

        assert isinstance(pipeline.stages[3], GetAudioDurationStage)
        assert pipeline.stages[3].audio_filepath_key == "audio_filepath"

        assert isinstance(pipeline.stages[4], PreserveByValueStage)
        assert pipeline.stages[4].target_value == 5.5
        assert pipeline.stages[4].operator == le

        write_result = pipeline.run(self.executor)
        assert len(write_result) == 1

        predict = self.read_json(os.path.join(cfg.raw_data_dir, "result"))
        assert len(predict) == 1
        assert len(predict[0].data) == 1
        row = predict[0].data.iloc[0]
        assert row["wer"] < 5.5
        assert row["audio_filepath"].split("/")[-1] == "18278756351935270941.wav"
        assert row["duration"] == 10.74
        assert len(row["text"]) == 137
        assert len(row["pred_text"]) == 137

        target = self.read_json(os.path.join(cfg.raw_data_dir, "test_data_reference.json"))
        assert len(target) == 1
        assert len(target[0].data) == 1
        row = target[0].data.iloc[0]
        assert row["wer"] < 5.5
        assert row["audio_filepath"].split("/")[-1] == "18278756351935270941.wav"
        assert row["duration"] == 10.74
        assert len(row["text"]) == 137
        assert len(row["pred_text"]) == 137

        assert predict[0].data.drop(self.drop_fields, axis=1).equals(target[0].data.drop(self.drop_fields, axis=1))

    @pytest.mark.skip("Import NeMo without apex")
    @pytest.mark.gpu
    def test_yaml_run_pipeline(self):
        conf_path = os.path.join(self.examples_audio_path, "fleurs/pipeline.yaml")
        cfg = OmegaConf.load(Path(conf_path))

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg["raw_data_dir"] = os.path.join(self.test_data_root, "armenian/fleurs")
            cfg["output_dir"] = tmpdir
            cfg["processors"][4]["target_value"] = 5.5

            # Act: Create the pipeline
            pipeline = create_pipeline_from_yaml(cfg)

            # Assert: Check pipeline object
            assert isinstance(pipeline, Pipeline), "Should return a Pipeline instance"
            assert pipeline.name == "yaml_pipeline"
            assert "yaml config file" in pipeline.description

            # Check number of stages
            assert len(pipeline.stages) == 7  # We know there should be exactly 7 stages

            # Check individual stages are of the expected type
            assert isinstance(pipeline.stages[0], CreateInitialManifestFleursStage)
            assert pipeline.stages[0].lang == "hy_am"
            assert pipeline.stages[0].split == "dev"

            assert isinstance(pipeline.stages[1], InferenceAsrNemoStage)
            assert pipeline.stages[1].model_name == "nvidia/stt_hy_fastconformer_hybrid_large_pc"

            assert isinstance(pipeline.stages[2], GetPairwiseWerStage)
            assert pipeline.stages[2].wer_key == "wer"

            assert isinstance(pipeline.stages[3], GetAudioDurationStage)
            assert pipeline.stages[3].audio_filepath_key == "audio_filepath"

            assert isinstance(pipeline.stages[4], PreserveByValueStage)
            assert pipeline.stages[4].target_value == 5.5
            assert pipeline.stages[4].operator == le

            write_result = pipeline.run(self.executor)
            assert len(write_result) == 1

            predict = self.read_json(os.path.join(cfg.raw_data_dir, "result"))
            assert len(predict) == 1
            assert len(predict[0].data) == 1
            row = predict[0].data.iloc[0]
            assert row["wer"] < 5.5
            assert row["audio_filepath"].split("/")[-1] == "18278756351935270941.wav"
            assert row["duration"] == 10.74
            assert len(row["text"]) == 137
            assert len(row["pred_text"]) == 137

            target = self.read_json(os.path.join(cfg.raw_data_dir, "test_data_reference.json"))
            assert len(target) == 1
            assert len(target[0].data) == 1
            row = target[0].data.iloc[0]
            assert row["wer"] < 5.5
            assert row["audio_filepath"].split("/")[-1] == "18278756351935270941.wav"
            assert row["duration"] == 10.74
            assert len(row["text"]) == 137
            assert len(row["pred_text"]) == 137

            assert predict[0].data.drop(self.drop_fields, axis=1).equals(target[0].data.drop(self.drop_fields, axis=1))
