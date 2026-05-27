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
text-only LLM stages (PnC, ITN, disfluency correction, captioning,
contextual ASR entity extraction, code-switching, speechQA).
Each stage runs as a separate Ray Data actor with its own vLLM engine.
All stages use the same model ID but load independently per actor.


Architecture:
    ALMManifestReader (CPU)
        → reads per-shard JSONL output from the ASR pipeline
    [if --enable_pnc] TextLLMStage: PnC (GPU)
        → restores punctuation/capitalisation, writes pnc_text
    [if --enable_itn] TextLLMStage: ITN (GPU)
        → inverse text normalization, preserves disfluencies
        → writes itn_text
    [if --enable_itn_no-disfluencies] TextLLMStage: DisfluencyRemoval (GPU)
        → ITN + disfluency removal (fillers, repetitions)
        → writes itn_no-disfluencies_text
    [if --enable_captioning] TextLLMStage: Captioning (GPU)
        → summarises transcript into a short caption
        → writes captioning_text
    [if --enable_context_asr] ContextualASRExtractionStage (GPU)
        → extracts domain, named-entity buckets, distractors
        → writes context_asr dict
    [if --enable_acoustic_distractor] AcousticDistractorStage (CPU)
        → G2Ps fine_context_terms and appends phonetically-similar
          words from the precomputed phoneme vocab to distractor_terms
    [if --enable_context_asr] ContextualASRPromptVariantStage (CPU)
        → appends five prompt variants to the context_asr dict
    ShardedManifestWriterStage (CPU)
        → writes per-shard JSONL output with .done markers
        → writes corrected_text
    [if --enable_code_switching] TextLLMStage: CodeSwitching (GPU)
        → restores English Latin spelling for English loanwords
          transliterated into the native script
        → writes code_switched_text
    [if --enable_speech_qa] TextLLMStage: SpeechQA (GPU)
        → either "SKIP" when the transcript is not suitable, or two lines
          "Q: <question>\nA: <answer>" grounded in the transcript
        → writes speech_qa_text
    ShardedManifestWriterStage (CPU)
        → writes per-shard JSONL output with .done markers

All GPU stages share the same vLLM model instance (loaded once).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm.alm_manifest_reader import ALMManifestReader
from nemo_curator.stages.audio.alm.sharded_manifest_writer import ShardedManifestWriterStage
from nemo_curator.stages.audio.text_filtering.acoustic_distractor import AcousticDistractorStage
from nemo_curator.stages.audio.text_filtering.contextual_asr_extraction import ContextualASRExtractionStage
from nemo_curator.stages.audio.text_filtering.contextual_asr_prompt_variant import ContextualASRPromptVariantStage
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
_CONTEXT_ASR_PROMPT = _PROMPT_DIR / "contextual_asr_prompt.md"

# Minimum recommended max_model_len when --enable_context_asr is used.
# The extraction prompt + output JSON can exceed shorter contexts.
_CONTEXT_ASR_MIN_MAX_MODEL_LEN = 4096
_CODE_SWITCHING_PROMPT = _PROMPT_DIR / "code_switching_prompt.md"
_SPEECH_QA_PROMPT = _PROMPT_DIR / "speech_qa_prompt.md"


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
    ap.add_argument(
        "--enable_context_asr",
        action="store_true",
        default=False,
        help=(
            "Enable contextual ASR stages: extract domain / named entities / distractors via LLM, "
            "then append five prompt variants. Output key: context_asr (nested dict)."
        ),
    )
    ap.add_argument(
        "--enable_acoustic_distractor",
        action="store_true",
        default=False,
        help=(
            "Enable acoustic distractor stage (CPU-only): G2Ps fine_context_terms via phonemizer "
            "and appends phonetically-similar words from --phoneme_vocab_path to context_asr.distractor_terms. "
            "Requires --enable_context_asr and --phoneme_vocab_path."
        ),
    )
    ap.add_argument(
        "--enable_code_switching",
        action="store_true",
        default=False,
        help="Enable code-switching stage: restores English Latin spelling for English loanwords "
        "transliterated into the native script. Output key: code_switched_text",
    )
    ap.add_argument(
        "--enable_speech_qa",
        action="store_true",
        default=False,
        help="Enable SpeechQA stage: generates a (Q, A) pair grounded in the transcript, or the "
        "single token SKIP when the transcript is not suitable. Output key: speech_qa_text",
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
        "--code_switching_output_key",
        type=str,
        default="code_switched_text",
        help="Output field for code-switching restoration result.",
    )
    ap.add_argument(
        "--speech_qa_output_key",
        type=str,
        default="speech_qa_text",
        help="Output field for SpeechQA result. Either 'SKIP' (single line) or two lines 'Q: ...' + 'A: ...'.",
    )

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
        "--code_switching_prompt_file",
        type=str,
        default=None,
        help="Path to code-switching prompt file. Defaults to bundled code_switching_prompt.md.",
    )
    ap.add_argument(
        "--speech_qa_prompt_file",
        type=str,
        default=None,
        help="Path to SpeechQA prompt file. Defaults to bundled speech_qa_prompt.md.",
    )

    ctx = ap.add_argument_group("contextual ASR")
    ctx.add_argument(
        "--context_asr_prompt_file",
        type=str,
        default=None,
        help="Path to context ASR extraction prompt file. Defaults to bundled contextual_asr_prompt.md.",
    )
    ctx.add_argument(
        "--context_asr_text_key",
        type=str,
        default="pnc_text",
        help="Input text field for context ASR extraction.",
    )
    ctx.add_argument(
        "--context_asr_output_key",
        type=str,
        default="context_asr",
        help="Output field (nested dict) for context ASR extraction + prompt variants.",
    )
    ctx.add_argument(
        "--context_asr_max_output_tokens",
        type=int,
        default=2048,
        help=(
            "Max tokens generated per sample for context ASR extraction. The shared vLLM engine's "
            "max_model_len must be ≥ this value plus prompt length — pass --max_model_len 8192 (or higher) "
            "when enabling context ASR."
        ),
    )
    ctx.add_argument("--context_asr_seed", type=int, default=42, help="Base RNG seed for prompt-variant generation.")
    ctx.add_argument(
        "--context_asr_partial_keep_lo",
        type=float,
        default=0.5,
        help="Lower bound of random keep fraction for the partial-context variant.",
    )
    ctx.add_argument(
        "--context_asr_partial_keep_hi",
        type=float,
        default=0.8,
        help="Upper bound of random keep fraction for the partial-context variant.",
    )

    ad = ap.add_argument_group("acoustic distractors")
    ad.add_argument(
        "--phoneme_vocab_path",
        type=str,
        default=None,
        help=(
            "Path to a precomputed phoneme vocabulary JSON (produced by scripts/build_phoneme_vocab.py). "
            "Required when --enable_acoustic_distractor is set."
        ),
    )
    ad.add_argument(
        "--phoneme_vocab_language",
        type=str,
        default=None,
        help=(
            "Optional espeak-ng language code (e.g. en-us, fr, de, cmn). When set, used for all samples; "
            "otherwise the per-sample source_lang is mapped to an espeak code."
        ),
    )
    ad.add_argument(
        "--max_acoustic_distractors",
        type=int,
        default=8,
        help="Maximum acoustic distractors appended per sample.",
    )
    ad.add_argument(
        "--max_total_distractors",
        type=int,
        default=16,
        help="Cap on the combined (semantic + acoustic) distractor_terms list.",
    )
    ad.add_argument(
        "--acoustic_per_entity_top_k",
        type=int,
        default=3,
        help="Top-K acoustic candidates retained per source entity before cross-entity merging.",
    )
    ad.add_argument(
        "--min_npd",
        type=float,
        default=0.1,
        help="Lower NPD bound — entries closer than this are dropped (near-duplicates).",
    )
    ad.add_argument(
        "--max_npd",
        type=float,
        default=0.5,
        help="Upper NPD bound — entries farther than this are dropped (acoustically unrelated).",
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


def main() -> None:  # noqa: C901, PLR0915
    args = _build_arg_parser().parse_args()

    if (
        not args.enable_itn
        and not args.enable_itn_no_disfluencies
        and not args.enable_captioning
        and not args.enable_pnc
        and not args.enable_context_asr
        and not args.enable_code_switching
        and not args.enable_speech_qa
    ):
        logger.warning(
            "No stages enabled. Use --enable_pnc, --enable_itn, --enable_itn_no-disfluencies, "
            "--enable_captioning, --enable_context_asr, --enable_code_switching, or --enable_speech_qa."
        )
        return

    itn_prompt = args.itn_prompt_file or str(_ITN_PROMPT)
    itn_no_disfl_prompt = args.itn_no_disfluencies_prompt_file or str(_CORRECTION_PROMPT)
    captioning_prompt = args.captioning_prompt_file or str(_CAPTIONING_PROMPT)
    pnc_prompt = args.pnc_prompt_file or str(_PNC_PROMPT)
    context_asr_prompt = args.context_asr_prompt_file or str(_CONTEXT_ASR_PROMPT)
    code_switching_prompt = args.code_switching_prompt_file or str(_CODE_SWITCHING_PROMPT)
    speech_qa_prompt = args.speech_qa_prompt_file or str(_SPEECH_QA_PROMPT)

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
        ALMManifestReader(manifest_path=args.input_manifest, output_dir=args.output_dir),
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

    if args.enable_context_asr:
        context_asr_text_key = args.context_asr_text_key
        # Engine-level kwargs are shared with the other TextLLMStage instances via the
        # class-level vLLM cache in text_llm_stage.py.  Per-stage knobs (prompt file,
        # max_output_tokens, batch size) stay local.
        stages.append(
            ContextualASRExtractionStage(
                model_id=args.model_id,
                prompt_file=context_asr_prompt,
                text_key=context_asr_text_key,
                source_lang_key="source_lang",
                output_key=args.context_asr_output_key,
                tensor_parallel_size=args.tensor_parallel_size,
                max_output_tokens=args.context_asr_max_output_tokens,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
                kv_cache_dtype=args.kv_cache_dtype,
                num_workers_override=args.num_workers,
                batch_size=args.batch_size,
                resources=Resources(gpus=1.0),
            )
        )
        if args.enable_acoustic_distractor:
            if not args.phoneme_vocab_path:
                msg = "--enable_acoustic_distractor requires --phoneme_vocab_path."
                raise ValueError(msg)
            stages.append(
                AcousticDistractorStage(
                    context_key=args.context_asr_output_key,
                    source_lang_key="source_lang",
                    language=args.phoneme_vocab_language,
                    phoneme_vocab_path=args.phoneme_vocab_path,
                    max_acoustic_distractors=args.max_acoustic_distractors,
                    max_total_distractors=args.max_total_distractors,
                    per_entity_top_k=args.acoustic_per_entity_top_k,
                    min_npd=args.min_npd,
                    max_npd=args.max_npd,
                )
            )
            logger.info(
                f"AcousticDistractor stage enabled: vocab={args.phoneme_vocab_path} "
                f"(language={args.phoneme_vocab_language or 'per-sample'}, "
                f"max_acoustic={args.max_acoustic_distractors}, total_cap={args.max_total_distractors})"
            )
        stages.append(
            ContextualASRPromptVariantStage(
                context_key=args.context_asr_output_key,
                source_lang_key="source_lang",
                seed=args.context_asr_seed,
                partial_keep_lo=args.context_asr_partial_keep_lo,
                partial_keep_hi=args.context_asr_partial_keep_hi,
            )
        )
        logger.info(
            f"Context ASR stages enabled: {context_asr_text_key} → {args.context_asr_output_key} "
            f"(extraction + 5 prompt variants)"
        )
        if args.max_model_len < _CONTEXT_ASR_MIN_MAX_MODEL_LEN:
            logger.warning(
                f"--max_model_len={args.max_model_len} may be too small for context ASR extraction. "
                f"Consider --max_model_len 8192 or higher when --enable_context_asr is used."
            )
    if args.enable_code_switching:
        stages.append(
            TextLLMStage(
                name="CodeSwitching",
                prompt_file=code_switching_prompt,
                text_key=args.text_key,
                output_text_key=args.code_switching_output_key,
                enable_validation=False,
                resources=Resources(gpus=1.0),
                **shared_model_kwargs,
            )
        )
        logger.info(f"CodeSwitching stage enabled: {args.text_key} → {args.code_switching_output_key}")

    if args.enable_speech_qa:
        stages.append(
            TextLLMStage(
                name="SpeechQA",
                prompt_file=speech_qa_prompt,
                text_key=args.text_key,
                output_text_key=args.speech_qa_output_key,
                enable_validation=False,
                resources=Resources(gpus=1.0),
                **shared_model_kwargs,
            )
        )
        logger.info(f"SpeechQA stage enabled: {args.text_key} → {args.speech_qa_output_key}")

    stages.append(ShardedManifestWriterStage(output_dir=args.output_dir))

    pipeline = Pipeline(name="text_post_processing", stages=stages)

    from nemo_curator.backends.ray_data import RayDataExecutor

    executor = RayDataExecutor()

    logger.info(f"Running text pipeline: {len(stages)} stages, mode={args.execution_mode}")
    pipeline.run(executor=executor)
    logger.info("Text pipeline complete.")


if __name__ == "__main__":
    main()
