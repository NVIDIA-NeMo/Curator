# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import argparse

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.video.io.video_reader import VideoReader


def create_video_reading_pipeline(args: argparse.Namespace) -> Pipeline:
    # Define pipeline
    pipeline = Pipeline(
        name="video_reading", description="Read videos from a folder and extract metadata on video level."
    )

    # Add stages
    # Add the composite stage that combines reading and downloading
    pipeline.add_stage(
        VideoReader(input_video_path=args.video_folder, video_limit=args.video_limit, verbose=args.verbose)
    )

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
    parser.add_argument("--video-folder", type=str, required=True, help="Path to the video folder")
    parser.add_argument("--video-limit", type=int, default=-1, help="Limit the number of videos to read")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose output")
    args = parser.parse_args()
    main(args)
