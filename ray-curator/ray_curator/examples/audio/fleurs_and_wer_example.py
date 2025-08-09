from dataclasses import dataclass

from loguru import logger
from nemo.core.config import hydra_runner
from omegaconf import OmegaConf

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from ray_curator.stages.audio.io.asr_inference import AsrNemoInferenceStage
from ray_curator.stages.resources import Resources


@dataclass
class TranscriptionConfig:
    """
    Transcription Configuration for audio to text transcription.
    """

    # Required configs
    model_name: str  # NeMo model_name
    raw_data_dir: str
    lang: str | None = "hy_am"
    split: str | None = "dev"


def create_audio_pipeline(args: TranscriptionConfig) -> Pipeline:
    # Define pipeline
    pipeline = Pipeline(name="audio_inference", description="Inference audio and filter by WER threshold.")

    # Add stages
    # Add the composite stage that combines reading and downloading
    pipeline.add_stage(
        CreateInitialManifestFleursStage(
            lang=args.lang,
            split=args.split,
            raw_data_dir=args.raw_data_dir,
        )
    )
    pipeline.add_stage(
        AsrNemoInferenceStage(model_name=args.model_name, batch_size=16, cuda="cuda", _resources=Resources(gpus=1.0))
    )

    return pipeline


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> None:
    """
    Prepare FLEURS dataset, run ASR inference and filer by WER threshold.
    """
    logger.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
    pipeline = create_audio_pipeline(cfg)

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Create executor
    executor = XennaExecutor()

    # Execute pipeline
    print("Starting pipeline execution...")
    pipeline.run(executor)

    # Print results
    print("\nPipeline completed!")


if __name__ == "__main__":
    main()
