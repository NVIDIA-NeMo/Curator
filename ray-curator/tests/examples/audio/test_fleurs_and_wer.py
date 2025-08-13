import os
from operator import le

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.examples.audio.fleurs_and_wer_example import TranscriptionConfig, create_audio_pipeline
from ray_curator.pipeline import Pipeline
from ray_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from ray_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from ray_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from ray_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage


def get_e2e_test_data_path() -> str:
    """Returns path to e2e test data"""
    test_data_root = os.getenv("TEST_DATA_ROOT")
    if test_data_root:  # assume it's present locally
        return test_data_root
    else:
        raise ValueError


class TestFleursWer:
    """Test suite for TestAsrInference."""

    test_data_root = get_e2e_test_data_path()

    def test_main_runs_pipeline(self):
        cfg = TranscriptionConfig(
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
        assert len(pipeline.stages) == 5  # We know there should be exactly 5 stages

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

        executor = XennaExecutor()
        result = pipeline.run(executor)

        assert len(result) == 1
        assert result[0].data["wer"] < 5.5
        assert result[0].data["audio_filepath"].split("/")[-1] == "18278756351935270941.wav"
        assert result[0].data["duration"] == 10.74
        assert len(result[0].data["text"]) == 137
        assert len(result[0].data["pred_text"]) == 137
