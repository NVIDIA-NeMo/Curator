import argparse

from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.video.caption.caption_enhancement import CaptionEnhancementStage
from ray_curator.stages.video.caption.caption_generation import CaptionGenerationStage
from ray_curator.stages.video.caption.caption_preparation import CaptionPreparationStage
from ray_curator.stages.video.clipping.clip_extraction_stages import ClipTranscodingStage, FixedStrideExtractorStage
from ray_curator.stages.video.clipping.clip_frame_extraction import ClipFrameExtractionStage
from ray_curator.stages.video.clipping.transnetv2_extraction import TransNetV2ClipExtractionStage
from ray_curator.stages.video.clipping.video_frame_extraction import VideoFrameExtractionStage
from ray_curator.stages.video.embedding.cosmos_embed1 import CosmosEmbed1EmbeddingStage, CosmosEmbed1FrameCreationStage
from ray_curator.stages.video.embedding.internvideo2 import InternVideo2EmbeddingStage, InternVideo2FrameCreationStage
from ray_curator.stages.video.filtering.clip_aesthetic_filter import ClipAestheticFilterStage
from ray_curator.stages.video.filtering.motion_filter import MotionFilterStage, MotionVectorDecodeStage
from ray_curator.stages.video.io.clip_writer import ClipWriterStage
from ray_curator.stages.video.io.video_reader import VideoReader
from ray_curator.stages.video.preview.preview import PreviewStage
from ray_curator.utils.decoder_utils import FrameExtractionPolicy, FramePurpose


def create_video_splitting_pipeline(args: argparse.Namespace) -> Pipeline:  # noqa: PLR0912, C901
    # Define pipeline
    pipeline = Pipeline(name="video_splitting", description="Split videos into clips")

    # Add stages
    pipeline.add_stage(
        VideoReader(input_video_path=args.video_dir, video_limit=args.video_limit, verbose=args.verbose)
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
    elif args.splitting_algorithm == "transnetv2":
        pipeline.add_stage(
            VideoFrameExtractionStage(
                decoder_mode=args.transnetv2_frame_decoder_mode,
                verbose=args.verbose,
            )
        )
        pipeline.add_stage(
            TransNetV2ClipExtractionStage(
                model_dir=args.model_dir,
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

    if args.motion_filter != "disable":
        pipeline.add_stage(
            MotionVectorDecodeStage(
                target_fps=args.motion_decode_target_fps,
                target_duration_ratio=args.motion_decode_target_duration_ratio,
                num_cpus_per_worker=args.motion_decode_cpus_per_worker,
            )
        )
        pipeline.add_stage(
            MotionFilterStage(
                score_only=args.motion_filter == "score-only",
                global_mean_threshold=args.motion_global_mean_threshold,
                per_patch_min_256_threshold=args.motion_per_patch_min_256_threshold,
                num_gpus_per_worker=args.motion_score_gpus_per_worker,
                motion_filter_batch_size=args.motion_score_batch_size,
                verbose=args.verbose,
            )
        )

    has_embeddings = args.generate_embeddings
    has_aesthetics = args.aesthetic_threshold is not None
    purposes = []
    if has_aesthetics:
        purposes.append(FramePurpose.AESTHETICS)
    if has_embeddings:
        purposes.append(FramePurpose.EMBEDDINGS)
    if len(purposes) != 0:
        pipeline.add_stage(
            ClipFrameExtractionStage(
                extraction_policies=(FrameExtractionPolicy.sequence,),
                extract_purposes=purposes,
                target_res=(
                    args.clip_extraction_target_res,
                    args.clip_extraction_target_res,
                ),
                verbose=args.verbose,
            )
        )
    if args.aesthetic_threshold is not None:
        pipeline.add_stage(
            ClipAestheticFilterStage(
                model_dir=args.model_dir,
                score_threshold=args.aesthetic_threshold,
                reduction=args.aesthetic_reduction,
                num_gpus_per_worker=args.aesthetic_gpus_per_worker,
                verbose=args.verbose,
            )
        )
    if args.generate_embeddings:
        if args.embedding_algorithm.startswith("cosmos-embed1"):
            variant = args.embedding_algorithm.split("-")[-1]
            pipeline.add_stage(
                CosmosEmbed1FrameCreationStage(
                    model_dir=args.model_dir,
                    variant=variant,
                    target_fps=2.0,
                    verbose=args.verbose,
                )
            )
            pipeline.add_stage(
                CosmosEmbed1EmbeddingStage(
                    model_dir=args.model_dir,
                    variant=variant,
                    gpu_memory_gb=args.embedding_gpu_memory_gb,
                    verbose=args.verbose,
                )
            )
        elif args.embedding_algorithm.startswith("internvideo2"):
            pipeline.add_stage(
                InternVideo2FrameCreationStage(
                    model_dir=args.model_dir,
                    target_fps=2.0,
                    verbose=args.verbose,
                )
            )
            pipeline.add_stage(
                InternVideo2EmbeddingStage(
                    model_dir=args.model_dir,
                    gpu_memory_gb=args.embedding_gpu_memory_gb,
                    verbose=args.verbose,
                )
            )
        else:
            msg = f"Embedding algorithm {args.embedding_algorithm} not supported"
            raise ValueError(msg)

    if args.generate_captions:
        pipeline.add_stage(
            CaptionPreparationStage(
                model_variant=args.captioning_algorithm,
                prompt_variant=args.captioning_prompt_variant,
                prompt_text=args.captioning_prompt_text,
                sampling_fps=args.captioning_sampling_fps,
                window_size=args.captioning_window_size,
                remainder_threshold=args.captioning_remainder_threshold,
                preprocess_dtype=args.captioning_preprocess_dtype,
                model_does_preprocess=args.captioning_model_does_preprocess,
                generate_previews=args.generate_previews,
                verbose=args.verbose,
            )
        )
        if args.generate_previews:
            pipeline.add_stage(
                PreviewStage(
                    target_fps=args.preview_target_fps,
                    target_height=args.preview_target_height,
                    verbose=args.verbose,
                )
            )

        pipeline.add_stage(
            CaptionGenerationStage(
                model_dir=args.model_dir,
                model_variant=args.captioning_algorithm,
                caption_batch_size=args.captioning_batch_size,
                fp8=args.captioning_use_fp8_weights,
                max_output_tokens=args.captioning_max_output_tokens,
                model_does_preprocess=args.captioning_model_does_preprocess,
                generate_stage2_caption=args.captioning_stage2_caption,
                stage2_prompt_text=args.captioning_stage2_prompt_text,
                disable_mmcache=not args.captioning_use_vllm_mmcache,
            )
        )

        if args.enhance_captions:
            pipeline.add_stage(
                CaptionEnhancementStage(
                    model_dir=args.model_dir,
                    model_variant=args.enhance_captions_algorithm,
                    prompt_variant=args.enhance_captioning_prompt_variant,
                    prompt_text=args.enhance_captions_prompt_text,
                    model_batch_size=args.enhance_captions_batch_size,
                    fp8=args.enhance_captions_use_fp8_weights,
                    max_output_tokens=args.enhance_captions_max_output_tokens,
                    verbose=args.verbose,
                )
            )

    pipeline.add_stage(
        ClipWriterStage(
            output_path=args.output_clip_path,
            input_path=args.video_dir,
            upload_clips=args.upload_clips,
            dry_run=args.dry_run,
            generate_embeddings=args.generate_embeddings,
            generate_previews=args.generate_previews,
            generate_captions=args.generate_captions,
            embedding_algorithm=args.embedding_algorithm,
            caption_models=[args.captioning_algorithm],
            enhanced_caption_models=[args.enhanced_caption_models],
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
    parser.add_argument("--video-dir", type=str, required=True, help="Path to input video directory")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model directory")
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
    parser.add_argument(
        "--transnetv2-threshold",
        type=float,
        default=0.4,
        help="Threshold for transnetv2 clip extraction stage.",
    )
    parser.add_argument(
        "--transnetv2-min-length-s",
        type=float,
        default=2.0,
        help="Minimum length of clips (in seconds) for transnetv2 splitting stage.",
    )
    parser.add_argument(
        "--transnetv2-max-length-s",
        type=float,
        default=10.0,
        help="Maximum length of clips (in seconds) for transnetv2 splitting stage.",
    )
    parser.add_argument(
        "--transnetv2-max-length-mode",
        type=str,
        default="stride",
        choices=["truncate", "stride"],
        help="Mode for handling clips longer than max_length_s in transnetv2 splitting stage.",
    )
    parser.add_argument(
        "--transnetv2-crop-s",
        type=float,
        default=0.5,
        help="Crop length (in seconds) for transnetv2 splitting stage.",
    )
    parser.add_argument(
        "--transnetv2-gpu-memory-gb",
        type=float,
        default=10.0,
        help="GPU memory (in GB) for transnetv2 splitting stage.",
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
    parser.add_argument(
        "--aesthetic-reduction",
        choices=[
            "mean",
            "min",
        ],
        default="min",
        help="Method to reduce the frame-level aesthetic scores.",
    )
    parser.add_argument(
        "--aesthetic-gpus-per-worker",
        type=float,
        default=0.25,
        help="Number of GPUs per worker allocated to aesthetic filter.",
    )
    # Embedding arguments
    parser.add_argument(
        "--embedding-algorithm",
        type=str,
        default="cosmos-embed1-224p",
        choices=["cosmos-embed1-224p", "cosmos-embed1-336p", "cosmos-embed1-448p", "internvideo2"],
        help="Embedding algorithm to use.",
    )
    parser.add_argument(
        "--embedding-gpu-memory-gb",
        type=float,
        default=20.0,
        help="GPU memory in GB per worker for Cosmos-Embed1 embedding stage.",
    )
    parser.add_argument(
        "--no-generate-embeddings",
        dest="generate_embeddings",
        action="store_false",
        default=True,
        help="Whether to generate embeddings for clips.",
    )
    parser.add_argument(
        "--generate-previews",
        dest="generate_previews",
        action="store_true",
        default=False,
        help="Whether to generate previews for clip windows.",
    )
    parser.add_argument(
        "--preview-target-fps",
        type=int,
        default=1,
        help="Target FPS for preview generation.",
    )
    parser.add_argument(
        "--preview-target-height",
        type=int,
        default=240,
        help="Target height for preview generation.",
    )
    parser.add_argument(
        "--generate-captions",
        dest="generate_captions",
        action="store_true",
        default=False,
        help="Whether to generate captions for clips.",
    )
    parser.add_argument(
        "--captioning-algorithm",
        type=str,
        default="qwen",
        choices=["qwen"],
        help="Captioning algorithm to use in annotation pipeline.",
    )
    parser.add_argument(
        "--captioning-window-size",
        type=int,
        default=256,
        help="Window size for captioning algorithm.",
    )
    parser.add_argument(
        "--captioning-remainder-threshold",
        type=int,
        default=128,
        help="Remainder threshold for captioning algorithm.",
    )
    parser.add_argument(
        "--captioning-prompt-variant",
        type=str,
        default="default",
        choices=[
            "default",
            "av",
            "av-surveillance",
        ],
        help="Prompt variant for captioning algorithm.",
    )
    parser.add_argument(
        "--captioning-prompt-text",
        type=str,
        default=None,
        help="Prompt text for captioning algorithm.",
    )
    parser.add_argument(
        "--captioning-sampling-fps",
        type=float,
        default=2.0,
        help="Controls number of frames sampled per second from input clip for captioning model",
    )
    parser.add_argument(
        "--captioning-preprocess-dtype",
        type=str,
        default="float16",
        choices=[
            "float32",
            "float16",
            "bfloat16",
            "uint8",
        ],
        help="Precision for tensor preprocess operations in QwenInputPreparationStage.",
    )
    parser.add_argument(
        "--captioning-model-does-preprocess",
        dest="captioning_model_does_preprocess",
        action="store_true",
        default=False,
        help="If set, Captioning model will handle preprocessing (resize, rescale, normalize) instead of our code.",
    )
    parser.add_argument(
        "--captioning-stage2-caption",
        dest="captioning_stage2_caption",
        action="store_true",
        default=False,
        help="If set, generated captions are used as input prompts again into QwenVL to refine them",
    )
    parser.add_argument(
        "--captioning-stage2-prompt-text",
        type=str,
        default=None,
        help="Specify the input prompt used to generate stage2 Qwen captions",
    )
    parser.add_argument(
        "--captioning-batch-size",
        type=int,
        default=8,
        help="Batch size for Qwen captioning stage.",
    )
    parser.add_argument(
        "--captioning-use-fp8-weights",
        action="store_true",
        default=False,
        help="Whether to use fp8 weights for Qwen VL model or not.",
    )
    parser.add_argument(
        "--captioning-max-output-tokens",
        type=int,
        default=512,
        help="Max number of output tokens requested from captioning model",
    )
    parser.add_argument(
        "--captioning-use-vllm-mmcache",
        action="store_true",
        default=False,
        help="vLLM MultiModal Cache Usage, default disabled for better performance and GPU Utilization",
    )
    # Caption enhancement arguments
    parser.add_argument(
        "--enhance-captions",
        dest="enhance_captions",
        action="store_true",
        default=False,
        help="Whether to enhance captions for clips.",
    )
    parser.add_argument(
        "--enhance-captions-algorithm",
        type=str,
        default="qwen",
        choices=["qwen"],
        help="Caption enhancement algorithm to use.",
    )
    parser.add_argument(
        "--enhance-captions-batch-size",
        type=int,
        default=128,
        help="Batch size for caption enhancement.",
    )
    parser.add_argument(
        "--enhance-captions-use-fp8-weights",
        action="store_true",
        default=False,
        help="Whether to use fp8 weights for caption enhancement.",
    )
    parser.add_argument(
        "--enhance-captions-max-output-tokens",
        type=int,
        default=512,
        help="Max number of output tokens requested from caption enhancement model",
    )
    parser.add_argument(
        "--enhance-captioning-prompt-variant",
        type=str,
        default="default",
        choices=[
            "default",
            "av",
            "av-surveillance",
        ],
        help="Prompt variant for enhanced captioning algorithm.",
    )
    parser.add_argument(
        "--enhance-captions-prompt-text",
        type=str,
        default=None,
        help="Prompt text for further enhancing captions using EnhanceCaptionStage w/ Qwen-LM.",
    )
    parser.add_argument(
        "--enhanced-caption-models",
        type=str,
        default="qwen_lm",
        choices=["qwen_lm"],
        help="Enhanced LLM models to use to improve captions",
    )
    args = parser.parse_args()
    main(args)
