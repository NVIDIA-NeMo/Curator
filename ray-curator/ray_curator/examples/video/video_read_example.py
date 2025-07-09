import argparse

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.video.io.video_download import VideoDownloadStage


def create_video_reading_pipeline(args: argparse.Namespace) -> Pipeline:

    # Define pipeline
    pipeline = Pipeline(name="video_reading", description="Read videos from a folder and extract metadata on video level.")

    # Add stages
    pipeline.add_stage(VideoDownloadStage(folder_path=args.video_folder, debug=args.debug))

    # TODO: Add Writer stage in the following PR

    return pipeline


def main(args: argparse.Namespace) -> None:

    pipeline = create_video_reading_pipeline(args)

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
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode")
    parser.add_argument("--video-folder", type=str, required=True, help="Path to the video folder")
    args = parser.parse_args()
    main(args)
