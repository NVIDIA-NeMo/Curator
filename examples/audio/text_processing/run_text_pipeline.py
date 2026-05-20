# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Text post-processing pipeline for ASR output.

Reads JSONL manifests produced by the audio ASR pipeline and applies
text-only LLM stages (ITN, disfluency correction) using a single
shared vLLM model.

Architecture:
    ALMManifestReader (CPU)
        → reads per-shard JSONL output from the ASR pipeline
    [if --enable_itn] TextLLMStage: ITN (GPU)
        → inverse text normalization, preserves disfluencies
        → writes itn_text
    [if --enable_correction] TextLLMStage: Correction (GPU)
        → ITN + disfluency removal (fillers, repetitions)
        → writes corrected_text
    ShardedManifestWriterStage (CPU)
        → writes per-shard JSONL output with .done markers

Both GPU stages share the same vLLM model instance (loaded once).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm.alm_manifest_reader import ALMManifestReader
from nemo_curator.stages.audio.alm.sharded_manifest_writer import ShardedManifestWriterStage
from nemo_curator.stages.audio.text_filtering.text_llm_stage import TextLLMStage
from nemo_curator.stages.resources import Resources

_PROMPT_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "nemo_curator"
    / "stages"
    / "audio"
    / "text_filtering"
    / "prompts"
)
_ITN_PROMPT = _PROMPT_DIR / "itn_prompt.md"
_CORRECTION_PROMPT = _PROMPT_DIR / "correction_prompt.md"
_CAPTIONING_PROMPT = _PROMPT_DIR / "captioning_prompt.md"
_PNC_PROMPT = _PROMPT_DIR / "pnc_prompt.md"


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Text post-processing pipeline for ASR output")

    ap.add_argument(
        "--input_manifest",
        type=str,
        required=True,
        help="Path to JSONL manifest(s) from the ASR pipeline output. Can be a directory or a glob pattern.",
    )
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory for processed manifests.")

    ap.add_argument(
        "--enable_itn",
        action="store_true",
        default=False,
        help="Enable ITN stage: spoken→written form, preserves disfluencies. Output key: itn_text",
    )
    ap.add_argument(
        "--enable_itn_no-disfluencies",
        action="store_true",
        default=False,
        help="Enable ITN + disfluency removal stage: removes fillers/repetitions + ITN. Output key: itn_no-disfluencies_text",
    )
    ap.add_argument(
        "--enable_captioning",
        action="store_true",
        default=False,
        help="Enable captioning stage: summarizes transcript into a short caption. Output key: captioning_text",
    )
    ap.add_argument(
        "--enable_pnc",
        action="store_true",
        default=False,
        help="Enable PnC stage: restores punctuation and capitalization. Output key: pnc_text",
    )

    ap.add_argument("--text_key", type=str, default="pnc_text", help="Input text field from ASR pipeline output.")
    ap.add_argument("--itn_output_key", type=str, default="itn_text", help="Output field for ITN result.")
    ap.add_argument(
        "--itn_no_disfluencies_output_key",
        type=str,
        default="itn_no-disfluencies_text",
        help="Output field for ITN + disfluency removal result.",
    )
    ap.add_argument(
        "--captioning_output_key", type=str, default="captioning_text", help="Output field for captioning result."
    )
    ap.add_argument("--pnc_output_key", type=str, default="pnc_text", help="Output field for PnC result.")

    ap.add_argument(
        "--model_id", type=str, default="Qwen/Qwen3.5-35B-A3B-FP8", help="HuggingFace model ID for the text LLM."
    )
    ap.add_argument(
        "--itn_prompt_file", type=str, default=None, help="Path to ITN prompt file. Defaults to bundled itn_prompt.md."
    )
    ap.add_argument(
        "--itn_no_disfluencies_prompt_file",
        type=str,
        default=None,
        help="Path to ITN + disfluency removal prompt file. Defaults to bundled correction_prompt.md.",
    )
    ap.add_argument(
        "--captioning_prompt_file",
        type=str,
        default=None,
        help="Path to captioning prompt file. Defaults to bundled captioning_prompt.md.",
    )
    ap.add_argument(
        "--pnc_prompt_file", type=str, default=None, help="Path to PnC prompt file. Defaults to bundled pnc_prompt.md."
    )

    ap.add_argument(
        "--tensor_parallel_size", type=int, default=None, help="GPUs for tensor parallelism (default: auto-detect)."
    )
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--max_num_seqs", type=int, default=64)
    ap.add_argument("--max_output_tokens", type=int, default=512)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    ap.add_argument("--kv_cache_dtype", type=str, default="fp8")
    ap.add_argument("--num_workers", type=int, default=None, help="Explicit GPU worker count for Xenna.")
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--execution_mode", type=str, default="streaming", choices=["streaming", "batch"])

    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()

    if (
        not args.enable_itn
        and not args.enable_itn_no_disfluencies
        and not args.enable_captioning
        and not args.enable_pnc
    ):
        logger.warning(
            "No stages enabled. Use --enable_pnc, --enable_itn, --enable_itn_no-disfluencies, or --enable_captioning."
        )
        return

    itn_prompt = args.itn_prompt_file or str(_ITN_PROMPT)
    itn_no_disfl_prompt = args.itn_no_disfluencies_prompt_file or str(_CORRECTION_PROMPT)
    captioning_prompt = args.captioning_prompt_file or str(_CAPTIONING_PROMPT)
    pnc_prompt = args.pnc_prompt_file or str(_PNC_PROMPT)

    shared_model_kwargs = {
        "model_id": args.model_id,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_output_tokens": args.max_output_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "num_workers_override": args.num_workers,
        "batch_size": args.batch_size,
    }

    stages = [
        ALMManifestReader(manifest_path=args.input_manifest),
    ]

    if args.enable_pnc:
        pnc_input_key = "abbreviated_text" if args.text_key == "pnc_text" else args.text_key
        stages.append(
            TextLLMStage(
                name="PnCRestoration",
                prompt_file=pnc_prompt,
                text_key=pnc_input_key,
                output_text_key=args.pnc_output_key,
                resources=Resources(gpus=1.0),
                **shared_model_kwargs,
            )
        )
        logger.info(f"PnC stage enabled: {pnc_input_key} → {args.pnc_output_key}")

    if args.enable_itn:
        stages.append(
            TextLLMStage(
                name="ITNRestoration",
                prompt_file=itn_prompt,
                text_key=args.text_key,
                output_text_key=args.itn_output_key,
                resources=Resources(gpus=1.0),
                **shared_model_kwargs,
            )
        )
        logger.info(f"ITN stage enabled: {args.text_key} → {args.itn_output_key}")

    if args.enable_itn_no_disfluencies:
        if not args.enable_itn:
            logger.warning(
                "--enable_itn_no-disfluencies requires --enable_itn (needs itn_text as input). Enabling ITN automatically."
            )
            stages.append(
                TextLLMStage(
                    name="ITNRestoration",
                    prompt_file=itn_prompt,
                    text_key=args.text_key,
                    output_text_key=args.itn_output_key,
                    resources=Resources(gpus=1.0),
                    **shared_model_kwargs,
                )
            )

        stages.append(
            TextLLMStage(
                name="DisfluencyRemoval",
                prompt_file=itn_no_disfl_prompt,
                text_key=args.itn_output_key,
                output_text_key=args.itn_no_disfluencies_output_key,
                max_deletion_ratio=0.5,
                resources=Resources(gpus=1.0),
                **shared_model_kwargs,
            )
        )
        logger.info(f"DisfluencyRemoval stage enabled: {args.itn_output_key} → {args.itn_no_disfluencies_output_key}")

    if args.enable_captioning:
        stages.append(
            TextLLMStage(
                name="Captioning",
                prompt_file=captioning_prompt,
                text_key=args.text_key,
                output_text_key=args.captioning_output_key,
                enable_validation=False,
                resources=Resources(gpus=1.0),
                **shared_model_kwargs,
            )
        )
        logger.info(f"Captioning stage enabled: {args.text_key} → {args.captioning_output_key}")

    stages.append(ShardedManifestWriterStage(output_dir=args.output_dir))

    pipeline = Pipeline(name="text_post_processing", stages=stages)

    from nemo_curator.backends.ray_data import RayDataExecutor

    executor = RayDataExecutor()

    logger.info(f"Running text pipeline: {len(stages)} stages, mode={args.execution_mode}")
    pipeline.run(executor=executor)
    logger.info("Text pipeline complete.")


if __name__ == "__main__":
    main()
