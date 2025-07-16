import argparse
import time

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.video.clipping.clip_extraction_stages import ClipTranscodingStage, FixedStrideExtractorStage
from ray_curator.stages.video.clipping.clip_frame_extraction import ClipFrameExtractionStage
from ray_curator.stages.video.clipping.transnetv2_extraction import TransNetV2ClipExtractionStage
from ray_curator.stages.video.clipping.video_frame_extraction import VideoFrameExtractionStage
from ray_curator.stages.video.filtering.motion_filter import MotionFilterStage, MotionVectorDecodeStage
from ray_curator.stages.video.io.clip_writer import ClipWriterStage
from ray_curator.stages.video.io.video_download import VideoDownloadStage
from ray_curator.stages.video.io.video_reader import VideoReaderStage
from ray_curator.utils.decoder_utils import FrameExtractionPolicy


def create_video_splitting_pipeline(args: argparse.Namespace) -> Pipeline:

    # Define pipeline
    pipeline = Pipeline(name="video_splitting", description="Split videos into clips")

    # Add stages
    pipeline.add_stage(VideoReaderStage(input_video_path=args.video_folder, video_limit=args.video_limit))
    pipeline.add_stage(VideoDownloadStage(verbose=args.verbose))

    if args.splitting_algorithm == "fixed_stride":
        pipeline.add_stage(
            FixedStrideExtractorStage(
                clip_len_s=args.fixed_stride_split_duration,
                clip_stride_s=args.fixed_stride_split_duration,
                min_clip_length_s=args.fixed_stride_min_clip_length_s,
                limit_clips=args.limit_clips,
            )
        )
    elif args.splitting_algorithm == "transnetv2":
        pipeline.add_stage(
            VideoFrameExtractionStage(
                decoder_mode=args.transnetv2_frame_decoder_mode,
                verbose=args.verbose,
            )
        )
        pipeline.add_stage(
            TransNetV2ClipExtractionStage(
                threshold=args.transnetv2_threshold,
                min_length_s=args.transnetv2_min_length_s,
                max_length_s=args.transnetv2_max_length_s,
                max_length_mode=args.transnetv2_max_length_mode,
                crop_s=args.transnetv2_crop_s,
                gpu_memory_gb=args.transnetv2_gpu_memory_gb,
                limit_clips=args.limit_clips,
                verbose=args.verbose,
            )
        )

    else:
        msg = f"Splitting algorithm {args.splitting_algorithm} not supported"
        raise ValueError(msg)

    pipeline.add_stage(ClipTranscodingStage(
        num_cpus_per_worker=args.transcode_cpus_per_worker,
        encoder=args.transcode_encoder,
        encoder_threads=args.transcode_encoder_threads,
        encode_batch_size=args.transcode_ffmpeg_batch_size,
        use_hwaccel=args.transcode_use_hwaccel,
        use_input_bit_rate=args.transcode_use_input_video_bit_rate,
        num_clips_per_chunk=args.clip_re_chunk_size,
        verbose=args.verbose,
    ))

    if args.motion_filter != "disable":
        pipeline.add_stage(MotionVectorDecodeStage(
            num_cpus_per_worker=args.motion_decode_cpus_per_worker,
            verbose=args.verbose,
            target_fps=args.motion_decode_target_fps,
            target_duration_ratio=args.motion_decode_target_duration_ratio,
        ))
        pipeline.add_stage(MotionFilterStage(
            score_only=args.motion_filter == "score-only",
            global_mean_threshold=args.motion_global_mean_threshold,
            per_patch_min_256_threshold=args.motion_per_patch_min_256_threshold,
            gpu_memory_gb=args.motion_score_gpu_memory_gb,
            batch_size=args.motion_score_batch_size,
            verbose=args.verbose,
        ))


    has_embeddings = args.generate_embeddings
    has_aesthetics = args.aesthetic_threshold is not None
    # If both aesthetics AND embeddings are needed: [1, 2] - extract frames at both 1 FPS and 2 FPS
    # If only aesthetics is needed: [1] - extract frames at 1 FPS
    # If only embeddings is needed: [2] - extract frames at 2 FPS
    target_fps: list[float | int] = (
        [1, 2] if has_aesthetics and has_embeddings else [1] if has_aesthetics else [2] if has_embeddings else []
    )

    # TODO: add a check once we have embeddings / aesthetics stages
    target_fps = [2]
    pipeline.add_stage(ClipFrameExtractionStage(
        extraction_policies=(FrameExtractionPolicy.sequence,),
        target_fps=target_fps,
        target_res=(
            args.clip_extraction_target_res,
            args.clip_extraction_target_res,
        ),
        verbose=args.verbose,
    ))

    pipeline.add_stage(
        ClipWriterStage(
            output_path=args.output_clip_path,
            input_path=args.video_folder,
            upload_clips=args.upload_clips,
            dry_run=args.dry_run,
            generate_embeddings=False, # TODO: Change this once we have an embedding stage
            generate_previews=False, # TODO: Change this once we have a preview stage
            generate_captions=False, # TODO: Change this once we have a caption stage
            embedding_algorithm=args.embedding_algorithm,
            caption_models=None, # TODO: Change this once we have a caption stage
            enhanced_caption_models=None, # TODO: Change this once we have a caption stage
            verbose=args.verbose,
        )
    )

    return pipeline


def main(args: argparse.Namespace) -> None:

    print("Starting pipeline execution...")
    start_time = time.time()
    pipeline = create_video_splitting_pipeline(args)

    # Print pipeline description
    print(pipeline.describe())
    print("\n" + "=" * 50 + "\n")

    # Create executor
    executor = XennaExecutor()

    # Execute pipeline
    pipeline.run(executor)
    end_time = time.time()

    # Calculate and print execution time
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\nPipeline completed!")
    print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--video-folder", type=str, default="/home/aot/Videos")
    parser.add_argument("--video-limit", type=int, default=-1, help="Limit the number of videos to read")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--output-clip-path", type=str, help="Path to output clips")
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
    parser.add_argument(
        "--transnetv2-threshold",
        type=float,
        default=0.4,
        help=(
            "TransNetV2 probability threshold above which a frame is classified as a shot transition. "
            "Default is 0.4, which prioritizes recall over precision."
        ),
    )
    parser.add_argument(
        "--transnetv2-min-length-s",
        type=float,
        default=2.0,
        help=(
            "Minimum length of clips (in seconds) for TransNetV2 splitting stage. "
            "If specified, will remove any scenes below this length."
        ),
    )
    parser.add_argument(
        "--transnetv2-max-length-s",
        type=float,
        default=60.0,
        help=(
            "Maximum length of clips (in seconds) for TransNetV2 splitting stage. "
            "If specified, will deal with the scene by the `max_length_mode` specified."
        ),
    )
    parser.add_argument(
        "--transnetv2-max-length-mode",
        type=str,
        default="stride",
        choices=["truncate", "stride"],
        help=(
            "Maximum length mode for TransNetV2 splitting stage. "
            "If `truncate`, will truncate the scene to `max_length_s`. "
            "If `stride`, will generate a number of max_length_s scenes until the end of the scene. "
            "If the end scene is less than `min_length_s`, it will drop the last scene."
        ),
    )
    parser.add_argument(
        "--transnetv2-crop-s",
        type=float,
        default=0.5,
        help=(
            "Crop size for TransNetV2 splitting stage. If specified, will crop each scene at start and end. "
            "E.g. 0.25 will crop ~250ms from start, and ~250ms from end frame (reducing all clips by ~0.5 seconds). "
            "If cropped scenes result in zero-length scenes, these will be filtered."
        ),
    )
    parser.add_argument(
        "--transnetv2-gpu-memory-gb",
        type=float,
        default=10,
        help="GPU memory in GB per worker for TransNetV2 splitting stage.",
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
        "--motion-score-gpu-memory-gb",
        type=float,
        default=20,
        help="GPU memory in GB per worker allocated to motion score computation.",
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
