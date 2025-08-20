import argparse

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.video.clipping.clip_extraction_stages import ClipTranscodingStage, FixedStrideExtractorStage
from ray_curator.stages.video.io.video_reader import VideoReader


def create_video_splitting_pipeline(args: argparse.Namespace) -> Pipeline:
    # Define pipeline
    pipeline = Pipeline(name="video_splitting", description="Split videos into clips")

    # Add stages
    pipeline.add_stage(
        VideoReader(input_video_path=args.video_folder, video_limit=args.video_limit, verbose=args.verbose)
    )

    if args.splitting_algorithm == "fixed_stride":
        pipeline.add_stage(
            FixedStrideExtractorStage(
                clip_len_s=args.fixed_stride_split_duration,
                clip_stride_s=args.fixed_stride_split_duration,
                min_clip_length_s=args.fixed_stride_min_clip_length_s,
                limit_clips=args.limit_clips,
            )
        )
    else:
        msg = f"Splitting algorithm {args.splitting_algorithm} not supported"
        raise ValueError(msg)

    pipeline.add_stage(
        ClipTranscodingStage(
            num_cpus_per_worker=args.transcode_cpus_per_worker,
            encoder=args.transcode_encoder,
            encoder_threads=args.transcode_encoder_threads,
            encode_batch_size=args.transcode_ffmpeg_batch_size,
            use_hwaccel=args.transcode_use_hwaccel,
            use_input_bit_rate=args.transcode_use_input_video_bit_rate,
            num_clips_per_chunk=args.clip_re_chunk_size,
            verbose=args.verbose,
        )
    )

    return pipeline


def main(args: argparse.Namespace) -> None:
    pipeline = create_video_splitting_pipeline(args)

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
    parser.add_argument("--video-folder", type=str, default="/home/aot/Videos")
    parser.add_argument("--video-limit", type=int, default=-1, help="Limit the number of videos to read")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--output-clip-path", type=str, help="Path to output clips", required=True)
    parser.add_argument(
        "--no-upload-clips",
        dest="upload_clips",
        action="store_false",
        default=True,
        help="Whether to upload clips to output path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="If set only write minimum metadata",
    )

    # Splitting parameters
    parser.add_argument(
        "--splitting-algorithm",
        type=str,
        default="fixed_stride",
        choices=["fixed_stride", "transnetv2"],
        help="Splitting algorithm to use",
    )
    parser.add_argument(
        "--fixed-stride-split-duration",
        type=float,
        default=10.0,
        help="Duration of clips (in seconds) generated from the fixed stride splitting stage.",
    )
    parser.add_argument(
        "--fixed-stride-min-clip-length-s",
        type=float,
        default=2.0,
        help="Minimum length of clips (in seconds) for fixed stride splitting stage.",
    )
    parser.add_argument(
        "--limit-clips",
        type=int,
        default=0,
        help="limit number of clips from each input video to process. 0 means no limit.",
    )
    parser.add_argument(
        "--transnetv2-frame-decoder-mode",
        type=str,
        default="pynvc",
        choices=["pynvc", "ffmpeg_gpu", "ffmpeg_cpu"],
        help="Choose between ffmpeg on CPU or GPU or PyNvVideoCodec for video decode.",
    )

    # Transcoding arguments
    parser.add_argument(
        "--transcode-cpus-per-worker",
        type=float,
        default=6.0,
        help="Number of CPU threads per worker. The stage uses a batched ffmpeg "
        "commandline with batch_size (-transcode-ffmpeg-batch-size) of ~64 and per-batch thread count of 1.",
    )
    parser.add_argument(
        "--transcode-encoder",
        type=str,
        default="libopenh264",
        choices=["libopenh264", "h264_nvenc", "libx264"],
        help="Codec for transcoding clips; None to skip transocding.",
    )
    parser.add_argument(
        "--transcode-encoder-threads",
        type=int,
        default=1,
        help="Number of threads per ffmpeg encoding sub-command for transcoding clips.",
    )
    parser.add_argument(
        "--transcode-ffmpeg-batch-size",
        type=int,
        default=16,
        help="FFMPEG batchsize for transcoding clips. Each clip/sub-command in "
        "the batch uses --transcode-encoder-threads number of CPU threads",
    )
    parser.add_argument(
        "--transcode-use-hwaccel",
        action="store_true",
        default=False,
        help="Whether to use cuda acceleration for decoding in transcoding stage.",
    )
    parser.add_argument(
        "--transcode-use-input-video-bit-rate",
        action="store_true",
        default=False,
        help="Whether to use input video's bit rate for encoding clips.",
    )
    parser.add_argument(
        "--clip-re-chunk-size",
        type=int,
        default=32,
        help="Number of clips per chunk after transcoding stage.",
    )

    # Motion vector decoding arguments
    parser.add_argument(
        "--motion-filter",
        choices=["disable", "enable", "score-only"],
        default="disable",
        help=(
            "Control motion filtering behavior:\n"
            "  - disable: No filtering or scoring.\n"
            "  - enable: Automatically filter clips based on motion thresholds.\n"
            "      (controlled by --motion-global-mean-threshold and --motion-per-patch-min-256-threshold).\n"
            "  - score-only: Calculate motion scores without filtering clips."
        ),
    )
    parser.add_argument(
        "--motion-global-mean-threshold",
        type=float,
        default=0.00098,
        help=(
            "Threshold for global average motion magnitude. "
            "Clips with global motion below this value may be flagged as low-motion. "
            "Only applies when --motion-filter is set to 'enable' or 'score-only'."
        ),
    )
    parser.add_argument(
        "--motion-per-patch-min-256-threshold",
        type=float,
        default=0.000001,
        help=(
            "Threshold for minimal average motion magnitude in any 256x256-pixel patch. "
            "Clips containing patches below this threshold may be flagged as low-motion. "
            "Only applies when --motion-filter is set to 'enable' or 'score-only'."
        ),
    )
    parser.add_argument(
        "--motion-decode-target-fps",
        type=float,
        default=2.0,
        help="Target frames per second to sample for motion vector decoding.",
    )
    parser.add_argument(
        "--motion-decode-target-duration-ratio",
        type=float,
        default=0.5,
        help="Target ratio of video duration to sample for motion vector decoding (0.5 = 50%%).",
    )
    parser.add_argument(
        "--motion-decode-cpus-per-worker",
        type=float,
        default=4.0,
        help="Number of CPUs per worker allocated to motion vector decoding.",
    )
    parser.add_argument(
        "--motion-score-batch-size",
        type=int,
        default=64,
        help="Batch size for motion score computation.",
    )
    parser.add_argument(
        "--motion-score-gpus-per-worker",
        type=float,
        default=0.5,
        help="Number of GPUs per worker allocated to motion score computation. Set to 0 to use CPU instead of GPU.",
    )
    parser.add_argument(
        "--clip-extraction-target-res",
        type=int,
        default=-1,
        help="Target resolution for clip extraction as (height, width). A value of -1 implies disables resize",
    )

    # Aesthetic arguments
    parser.add_argument(
        "--aesthetic-threshold",
        type=float,
        default=None,
        help="If specified (e.g. 3.5), filter out clips with an aesthetic score below this threshold.",
    )
    # Embedding arguments
    parser.add_argument(
        "--embedding-algorithm",
        type=str,
        default="internvideo2",
        choices=["cosmos-embed1", "internvideo2"],
        help="Embedding algorithm to use.",
    )
    parser.add_argument(
        "--embedding-gpus-per-worker",
        type=float,
        default=1.0,
        help="Number of GPUs per worker for InternVideo2 or Cosmos-Embed1 embedding stage.",
    )
    parser.add_argument(
        "--no-generate-embeddings",
        dest="generate_embeddings",
        action="store_false",
        default=True,
        help="Whether to generate embeddings for clips.",
    )
    args = parser.parse_args()
    main(args)
