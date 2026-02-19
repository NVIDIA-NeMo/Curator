# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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


def create_video_splitting_argparser() -> argparse.ArgumentParser:  # noqa: PLR0915
    """Create and return the argument parser for video splitting pipeline.

    This function is extracted to allow reuse by other scripts (e.g., benchmarks).
    """
    parser = argparse.ArgumentParser(
        description="Split videos into clips with optional embeddings, captions, and filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # General arguments
    parser.add_argument("--video-dir", type=str, required=True, help="Path to input video directory")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models",
        help=(
            "Path to model directory containing required model weights. "
            "Models will be automatically downloaded on first use if not present. "
            "Required models depend on selected algorithms:\n"
            "  - TransNetV2: For scene detection (--splitting-algorithm transnetv2)\n"
            "  - InternVideo2: For embeddings (--embedding-algorithm internvideo2)\n"
            "  - Cosmos-Embed1: For embeddings (--embedding-algorithm cosmos-embed1-*)\n"
            "  - Qwen: For captioning (--generate-captions)\n"
            "  - Aesthetic models: For filtering (--aesthetic-threshold)\n"
            "Default: ./models\n"
            "Example: --model-dir /path/to/models or --model-dir ./models"
        ),
    )
    parser.add_argument("--video-limit", type=int, default=None, help="Limit the number of videos to read")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--output-path", type=str, help="Path to output clips", required=True)

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
        help="If set, only write minimum metadata",
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
        help="Limit number of clips from each input video to process. 0 means no limit.",
    )
    parser.add_argument(
        "--transnetv2-frame-decoder-mode",
        type=str,
        default="pynvc",
        choices=["pynvc", "ffmpeg_gpu", "ffmpeg_cpu"],
        help="Choose between FFmpeg on CPU or GPU or PyNvVideoCodec for video decode.",
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
        help="Number of CPU threads per worker. The stage uses a batched FFmpeg "
        "commandline with batch_size (-transcode-ffmpeg-batch-size) of ~64 and per-batch thread count of 1.",
    )
    parser.add_argument(
        "--transcode-encoder",
        type=str,
        default="libopenh264",
        choices=["libopenh264", "h264_nvenc", "libx264"],
        help="Codec for transcoding clips; None to skip transcoding.",
    )
    parser.add_argument(
        "--transcode-encoder-threads",
        type=int,
        default=1,
        help="Number of threads per FFmpeg encoding sub-command for transcoding clips.",
    )
    parser.add_argument(
        "--transcode-ffmpeg-batch-size",
        type=int,
        default=16,
        help="FFmpeg batchsize for transcoding clips. Each clip/sub-command in "
        "the batch uses --transcode-encoder-threads number of CPU threads",
    )
    parser.add_argument(
        "--transcode-use-hwaccel",
        action="store_true",
        default=False,
        help="Whether to use CUDA acceleration for decoding in transcoding stage.",
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
        help="Target resolution for clip extraction as a square (height=width). A value of -1 disables resize",
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
        help="If set, captioning model will handle preprocessing (resize, rescale, normalize) instead of our code.",
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
        help="vLLM MultiModal Cache Usage, default disabled for better performance and GPU utilization",
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
        help="Prompt text for further enhancing captions using EnhanceCaptionStage with Qwen-LM.",
    )
    parser.add_argument(
        "--enhanced-caption-models",
        type=str,
        default="qwen_lm",
        choices=["qwen_lm"],
        help="Enhanced LLM models to use to improve captions",
    )
    return parser
