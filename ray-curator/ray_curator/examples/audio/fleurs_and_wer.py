from dataclasses import dataclass

from loguru import logger
from nemo.core.config import hydra_runner
from omegaconf import OmegaConf

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from ray_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from ray_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from ray_curator.stages.audio.io.write_jsonl import WriteJsonlStage
from ray_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from ray_curator.stages.resources import Resources


@dataclass
class TranscriptionConfig:
    """
    FLEURS audio data pretaration, transcription and filtration by WER threshold.
    """

    # Required configs
    raw_data_dir: str  # path to store processed data
    output_manifest_file: str | None = None  # path to output jsonl file
    model_name: str = "nvidia/stt_hy_fastconformer_hybrid_large_pc"  # NeMo model name
    lang: str = "hy_am"
    split: str = "dev"
    wer_threshold: float = 75.0


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
        InferenceAsrNemoStage(model_name=args.model_name).with_(batch_size=16, resources=Resources(gpus=1.0))
    )
    pipeline.add_stage(GetPairwiseWerStage(text_key="text", pred_text_key="pred_text", wer_key="wer"))
    pipeline.add_stage(GetAudioDurationStage(audio_filepath_key="audio_filepath", duration_key="duration"))
    pipeline.add_stage(PreserveByValueStage(input_value_key="wer", target_value=args.wer_threshold, operator="le"))
    if args.output_manifest_file is not None:
        pipeline.add_stage(WriteJsonlStage(output_manifest_file=args.output_manifest_file))
    return pipeline


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> None:
    """
    Prepare FLEURS dataset, run ASR inference and filer by WER threshold.
    """
    logger.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")
    pipeline = create_audio_pipeline(cfg)

    # Print pipeline description
    logger.info(pipeline.describe())
    logger.info("\n" + "=" * 50 + "\n")

    # Create executor
    executor = XennaExecutor()

    # Execute pipeline
    logger.info("Starting pipeline execution...")
    pipeline.run(executor)

    # Print results
    logger.info("\nPipeline completed!")


if __name__ == "__main__":
    main()
