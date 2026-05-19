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

"""Qwen3-Omni audio transcription + text filtering pipeline.

Runs Qwen3-Omni directly inside the Curator pipeline with optional
QwenASR hallucination recovery (``--asr_model_id``) and LLM-based
punctuation/capitalisation restoration (``--enable_pnc`` to enable).

Model selection:
    Pass ``--language <code>`` to auto-select both models, or set them explicitly:

    ``--primary_model``   (required unless --language is given)
        qwen_omni  → InferenceQwenOmniStage       (Qwen3-Omni vLLM, ``--model_id``)
                     recommended: en de es fr it pt ru nl ur
        qwen_asr   → InferenceQwenASRStage         (Qwen3-ASR, ``--asr_model_id``)
                     recommended: pl cs ro hu el fi da sv hi
        whisper    → InferenceFasterWhisperStage   (Whisper Large V3, ``--whisper_model_size_or_path``)
                     recommended: lt lv hr et bg sk sl mt uk

    ``--recovery_model``  (optional; auto-derived from primary when omitted)
        qwen_asr       → InferenceQwenASRStage        (default recovery for qwen_omni primary)
        whisper        → InferenceFasterWhisperStage   (default recovery for qwen_asr primary)
        parakeet       → InferenceParakeetStage        (default recovery for whisper primary)
        indic_conformer→ InferenceIndicConformerStage  (auto-set for hi/ur via --language)

Architecture:
    NemoTarredAudioReader (CPU)
        → streams NeMo-tarred shards, decodes audio in memory
    InitializeFieldsStage (CPU)
        → sets _skipme = "", renames text → granary_v1_prediction
    [optional] SEDInferenceStage (GPU)
        → runs PANNs CNN14 on each audio, produces framewise probabilities
    [optional] SEDPostprocessingStage (CPU)
        → labels entries with detected sound events (speech, music, noise, etc.)
    Primary ASR stage (GPU) — model chosen via --primary_model or --language
        → outputs primary_prediction (and primary_prediction_s2 for Omni with followup)
    [optional] DisfluencyWerGuardStage (CPU)
        → compares Turn 1 vs Turn 2 WER (Omni + followup_prompt only)
    WhisperHallucinationStage (CPU)
        → flags hallucination patterns on primary output, sets _skipme
    Recovery ASR stage (GPU) — model chosen via --recovery_model or auto-derived
        → always runs after primary; primary prediction wins unless flagged as hallucinated
    WhisperHallucinationStage (CPU)
        → checks recovery output; recovers or confirms hallucination
    SelectBestPredictionStage (CPU)
        → picks recovery prediction if primary was hallucinated, else primary
    FastTextLIDStage (CPU)
        → flags wrong language / low confidence
    RegexSubstitutionStage (CPU)
        → applies regex rules, writes cleaned_text
    AbbreviationConcatStage (CPU)
        → re-joins split abbreviations, writes abbreviated_text
    [optional] PnCRestorationStage (GPU, text-only LLM)
        → restores punctuation/capitalisation, writes pnc_text
    [optional] PnCContentGuardStage (CPU)
        → reverts pnc_text when the LLM changed words
    [optional] ITNRestorationStage (GPU, text-only LLM)
        → converts spoken-form to written form (numbers, dates, symbols)
        → validates output, falls back to input on hallucination
        → writes itn_text
    ShardedManifestWriterStage (CPU)
        → writes per-shard JSONL output with .done markers
"""

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

import argparse
import time

from loguru import logger

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline

QWEN_ASR_DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
INDIC_CONFORMER_DEFAULT_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
WHISPER_DEFAULT_MODEL = "large-v3"
PARAKEET_DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"

# Language codes that define which primary inference model is used per invocation.
OMNI_PRIMARY_LANGUAGE_CODES     = {"en", "de", "es", "fr", "it", "pt", "ru", "nl", "ur"}
QWEN_ASR_PRIMARY_LANGUAGE_CODES = {"pl", "cs", "ro", "hu", "el", "fi", "da", "sv", "hi"}
WHISPER_PRIMARY_LANGUAGE_CODES  = {"lt", "lv", "hr", "et", "bg", "sk", "sl", "mt", "uk"}

# Default recovery model paired with each primary (when --recovery_model is not set):
#   qwen_omni → qwen_asr;  qwen_asr → whisper;  whisper → parakeet
# Indic languages override recovery to indic_conformer automatically via --language.
INDIC_LANGUAGE_CODES = {"hi", "ur"}
from nemo_curator.stages.audio.alm.sharded_manifest_writer import ShardedManifestWriterStage
from nemo_curator.stages.audio.inference.faster_whisper import InferenceFasterWhisperStage
from nemo_curator.stages.audio.inference.indic_conformer import InferenceIndicConformerStage
from nemo_curator.stages.audio.inference.parakeet import InferenceParakeetStage
from nemo_curator.stages.audio.inference.qwen_asr import InferenceQwenASRStage
from nemo_curator.stages.audio.inference.qwen_omni import InferenceQwenOmniStage
from nemo_curator.stages.audio.io.nemo_tarred_reader import NemoTarredAudioReader
from nemo_curator.stages.audio.io.unified_reader import UnifiedAudioReader
from nemo_curator.stages.audio.text_filtering import (
    AbbreviationConcatStage,
    DisfluencyWerGuardStage,
    FastTextLIDStage,
    InitializeFieldsStage,
    ITNRestorationStage,
    PnCContentGuardStage,
    PnCRestorationStage,
    RegexSubstitutionStage,
    WhisperHallucinationStage,
)
from nemo_curator.stages.audio.text_filtering.select_best_prediction import SelectBestPredictionStage
from nemo_curator.stages.resources import Resources


def _build_arg_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description="QwenOmni in-process vLLM pipeline")
    ap.add_argument("--data_config", type=str, required=True, help="Granary YAML data config.")
    ap.add_argument("--corpus", type=str, nargs="*", default=None, help="Process only these corpora.")
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory for per-shard manifests.")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--ml_prompt", type=str, default="Transcribe the audio.", help="Multilingual prompt text. Supports {language} placeholder resolved per-sample from source_lang.")
    ap.add_argument("--ml_prompt_file", type=str, default=None, help="Read multilingual prompt from file. Overrides --ml_prompt.")
    ap.add_argument("--en_prompt_file", type=str, default=None, help="English-specific prompt file. Used for en samples; --ml_prompt_file is used for all other languages.")
    ap.add_argument("--followup_prompt", type=str, default=None, help="Turn 2 follow-up prompt text.")
    ap.add_argument("--followup_prompt_file", type=str, default=None, help="Read Turn 2 follow-up prompt from file.")
    ap.add_argument("--system_prompt", type=str, default=None, help="System prompt text or path to file.")
    ap.add_argument("--tensor_parallel_size", type=int, default=None,
                    help="Tensor parallel size for the primary vLLM model. "
                         "Defaults to 2 for qwen_omni, 1 for qwen_asr and whisper.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_output_tokens", type=int, default=256)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--max_num_seqs", type=int, default=16)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    ap.add_argument("--prep_workers", type=int, default=16, help="Thread pool size for audio preprocessing.")
    ap.add_argument("--source_lang_key", type=str, default="source_lang",
                    help="Manifest key holding per-sample language code. "
                         "Used for prompt interpolation ({language} placeholder) and per-sample LID filtering.")
    ap.add_argument("--s3_endpoint_url", type=str, default=None)
    ap.add_argument("--use_unified_reader", action="store_true", default=False,
                    help="Use UnifiedAudioReader (NeMo lhotse adapters for tarred + non-tarred data).")
    ap.add_argument(
        "--execution_mode", type=str, default="streaming", choices=["streaming", "batch"], help="Xenna execution mode."
    )
    ap.add_argument(
        "--autoscale_interval_s", type=int, default=180,
        help="Seconds between Xenna streaming autoscaler checks. Lower values ramp up GPU actors faster on multi-node."
    )

    primary = ap.add_argument_group(
        "primary inference model",
        "Pass --language <code> to auto-select both models, or set --primary_model and --recovery_model "
        "explicitly. --primary_model takes precedence over --language for the primary choice; "
        "--recovery_model takes precedence over --language for the recovery choice.",
    )
    primary.add_argument(
        "--language",
        type=str,
        default=None,
        metavar="LANG_CODE",
        help=(
            "ISO 639-1 language code for this pipeline invocation. "
            "Auto-selects --primary_model and --recovery_model when they are not set explicitly: "
            f"{sorted(OMNI_PRIMARY_LANGUAGE_CODES)} → primary=qwen_omni, recovery=qwen_asr; "
            f"{sorted(QWEN_ASR_PRIMARY_LANGUAGE_CODES)} → primary=qwen_asr, recovery=whisper; "
            f"{sorted(WHISPER_PRIMARY_LANGUAGE_CODES)} → primary=whisper, recovery=parakeet. "
            f"Indic languages {sorted(INDIC_LANGUAGE_CODES)} set recovery=indic_conformer "
            "(unless --recovery_model is set explicitly)."
        ),
    )
    primary.add_argument(
        "--primary_model",
        type=str,
        choices=["qwen_omni", "qwen_asr", "whisper"],
        default=None,
        help=(
            "Primary ASR model placed after SED. "
            "qwen_omni: Qwen3-Omni vLLM (--model_id); "
            "qwen_asr: Qwen3-ASR (--asr_model_id); "
            "whisper: Faster-Whisper Large V3 (--whisper_model_size_or_path). "
            "Set automatically by --language when omitted."
        ),
    )

    tf = ap.add_argument_group("text filtering")
    tf.add_argument("--hall_phrases", type=str, required=True, help="Path to hallucination phrases text file.")
    tf.add_argument(
        "--fasttext_model",
        type=str,
        default="facebook/fasttext-language-identification",
        help="FastText LID model: HuggingFace repo ID, local path, or known name (lid.176.bin / lid.176.ftz).",
    )
    tf.add_argument("--regex_yaml", type=str, required=True, help="Path to regex substitution rules YAML.")

    tf.add_argument(
        "--min_lang_prob", type=float, default=0.8, help="Minimum FastText language probability to keep an entry."
    )
    tf.add_argument(
        "--unique_words_threshold",
        type=float,
        default=0.4,
        help="Unique-word ratio threshold for repeated n-gram hallucination detection.",
    )
    tf.add_argument(
        "--long_word_threshold",
        type=int,
        default=25,
        help="Absolute character length above which a word is flagged as abnormally long.",
    )
    tf.add_argument(
        "--long_word_rel_threshold",
        type=float,
        default=3.0,
        help="Relative length ratio for long-word hallucination detection.",
    )
    tf.add_argument(
        "--max_char_rate",
        type=float,
        default=40.0,
        help="Min chars/s above which text is considered impossibly dense.",
    )

    pnc = ap.add_argument_group("PnC restoration")
    pnc.add_argument("--pnc_model_id", type=str, default="Qwen/Qwen3.5-35B-A3B-FP8",
                     help="Model ID for PnC restoration LLM.")
    pnc.add_argument("--pnc_prompt", type=str, default=None,
                     help="PnC restoration prompt (use {text} placeholder). Read from --pnc_prompt_file if set.")
    pnc.add_argument("--pnc_prompt_file", type=str, default=None,
                     help="Read PnC restoration prompt from file.")
    pnc.add_argument("--completeness_prompt", type=str, default=None,
                     help="Completeness check prompt (use {text} placeholder).")
    pnc.add_argument("--pnc_tensor_parallel_size", type=int, default=2,
                     help="Tensor parallel size for PnC model.")
    pnc.add_argument("--pnc_batch_size", type=int, default=64,
                     help="Batch size for PnC restoration stage.")
    pnc.add_argument("--pnc_max_model_len", type=int, default=8192,
                     help="Max model length for PnC model.")
    pnc.add_argument("--pnc_max_num_seqs", type=int, default=64,
                     help="Max concurrent sequences for PnC model.")
    pnc.add_argument("--pnc_prep_workers", type=int, default=8,
                     help="Thread pool size for PnC prompt preprocessing.")
    pnc.add_argument("--pnc_gpu_memory_utilization", type=float, default=0.95,
                     help="Fraction of GPU memory for PnC vLLM engine.")
    pnc.add_argument("--pnc_source_lang_key", type=str, default="source_lang",
                     help="Task data key holding per-sample language name for PnC prompt {language} placeholder.")
    pnc.add_argument("--enable_pnc", action="store_true", default=False,
                     help="Enable PnC restoration stage.")

    itn = ap.add_argument_group("ITN (inverse text normalization)")
    itn.add_argument("--enable_itn", action="store_true", help="Enable ITN stage after PnC restoration.")
    itn.add_argument("--itn_model_id", type=str, default="Qwen/Qwen3.5-35B-A3B-FP8", help="Model for ITN inference.")
    itn.add_argument("--itn_prompt_file", type=str, default=None,
                     help="ITN system prompt file. Uses bundled default if not set.")
    itn.add_argument("--itn_text_key", type=str, default=None,
                     help="Input key for ITN (default: pnc_text if PnC enabled, abbreviated_text otherwise).")
    itn.add_argument("--itn_output_key", type=str, default="itn_text", help="Output key for ITN result.")
    itn.add_argument("--itn_batch_size", type=int, default=64, help="Batch size for ITN inference.")
    itn.add_argument("--itn_tensor_parallel_size", type=int, default=None,
                     help="TP size for ITN model (None = auto-detect).")
    itn.add_argument("--itn_max_output_tokens", type=int, default=512,
                     help="Max tokens to generate per ITN sample.")
    itn.add_argument("--itn_max_model_len", type=int, default=4096,
                     help="Max context length for ITN vLLM engine.")
    itn.add_argument("--itn_max_num_seqs", type=int, default=16,
                     help="Max concurrent sequences for ITN vLLM engine.")
    itn.add_argument("--itn_gpu_memory_utilization", type=float, default=0.95,
                     help="Fraction of GPU memory for ITN vLLM engine.")
    itn.add_argument("--itn_no_validation", action="store_true", help="Disable ITN output validation.")

    sed = ap.add_argument_group("SED (sound event detection)")
    sed.add_argument("--sed_checkpoint", type=str, default=None,
                     help="Path to PANNs CNN14 .pth checkpoint. Enables SED stages when set.")
    sed.add_argument("--sed_model_type", type=str, default="Cnn14_DecisionLevelMax",
                     help="CNN14 variant name (see sed_models.MODEL_REGISTRY).")
    sed.add_argument("--sed_speech_threshold", type=float, default=0.4,
                     help="Probability threshold for event detection (applied per-class).")
    sed.add_argument("--sed_min_duration", type=float, default=0.3,
                     help="Minimum speech event duration in seconds.")
    sed.add_argument("--sed_merge_gap", type=float, default=0.0,
                     help="Merge events with gaps smaller than this (seconds, 0 = disabled).")
    sed.add_argument("--sed_batch_size", type=int, default=32,
                     help="Batch size for SED GPU inference.")
    sed.add_argument("--sed_gpu_memory_gb", type=float, default=4.0,
                     help="GPU memory in GB for SED inference stage.")
    sed.add_argument("--sed_superclasses", action="store_true", default=False,
                     help="Emit aggregated superclass events (noisy-or per group) instead of per-class subcategory events.")
    sed.add_argument("--sed_num_workers", type=int, default=None,
                     help="Fixed actor count for SED stage.")

    asr = ap.add_argument_group(
        "recovery ASR (secondary inference)",
        "Use --recovery_model to pick the fallback model explicitly, or let --language set it automatically. "
        "Default auto-pairing (when neither --recovery_model nor --language is given): "
        "primary=qwen_omni → qwen_asr; primary=qwen_asr → whisper; primary=whisper → parakeet.",
    )
    asr.add_argument(
        "--recovery_model",
        type=str,
        choices=["qwen_asr", "whisper", "parakeet", "indic_conformer"],
        default=None,
        help=(
            "Recovery (fallback) ASR model. "
            "qwen_asr: Qwen3-ASR (--asr_model_id); "
            "whisper: Faster-Whisper Large V3 (--whisper_model_size_or_path); "
            "parakeet: Parakeet-TDT v3 (--parakeet_model_id); "
            "indic_conformer: AI4Bharat Indic Conformer (--indic_conformer_model_id). "
            "Auto-derived from --primary_model or --language when omitted."
        ),
    )
    asr.add_argument(
        "--asr_model_id",
        type=str,
        default=QWEN_ASR_DEFAULT_MODEL_ID,
        help=(
            f"Qwen3-ASR model ID or local path (default: {QWEN_ASR_DEFAULT_MODEL_ID}). "
            "Used when --primary_model qwen_asr or --recovery_model qwen_asr."
        ),
    )
    asr.add_argument(
        "--indic_conformer_model_id",
        type=str,
        default=INDIC_CONFORMER_DEFAULT_MODEL_ID,
        metavar="HF_REPO_OR_PATH",
        help=(
            f"AI4Bharat Indic Conformer model ID (default: {INDIC_CONFORMER_DEFAULT_MODEL_ID}). "
            "Used when --recovery_model indic_conformer. "
            "The HF repo is gated — set HF_TOKEN in the environment."
        ),
    )
    asr.add_argument(
        "--indic_conformer_decode",
        type=str,
        default="rnnt",
        choices=["ctc", "rnnt"],
        help="Decoding mode for Indic Conformer (model card).",
    )
    asr.add_argument(
        "--whisper_model_size_or_path",
        type=str,
        default=WHISPER_DEFAULT_MODEL,
        metavar="MODEL",
        help=(
            f"Faster-Whisper model name or path (default: {WHISPER_DEFAULT_MODEL}). "
            "Used when --primary_model whisper or --recovery_model whisper."
        ),
    )
    asr.add_argument(
        "--whisper_device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Device for faster-whisper (cuda/cpu/auto).",
    )
    asr.add_argument(
        "--whisper_compute_type",
        type=str,
        default="float16",
        help="faster-whisper compute_type (e.g. float16, int8_float16, int8).",
    )
    asr.add_argument(
        "--whisper_download_root",
        type=str,
        default=None,
        help="Optional cache directory for faster-whisper downloads.",
    )
    asr.add_argument(
        "--parakeet_model_id",
        type=str,
        default=PARAKEET_DEFAULT_MODEL_ID,
        metavar="HF_REPO_OR_PATH",
        help=(
            f"NVIDIA Parakeet-TDT v3 model ID (default: {PARAKEET_DEFAULT_MODEL_ID}). "
            "Used when --recovery_model parakeet."
        ),
    )
    asr.add_argument(
        "--parakeet_cache_dir",
        type=str,
        default=None,
        help="Optional NeMo cache directory for Parakeet downloads.",
    )
    asr.add_argument(
        "--parakeet_inference_batch_size",
        type=int,
        default=16,
        help="Batch size passed to Parakeet ASRModel.transcribe().",
    )
    asr.add_argument("--asr_batch_size", type=int, default=128)
    asr.add_argument("--asr_gpu_memory_utilization", type=float, default=0.95)
    asr.add_argument("--asr_max_new_tokens", type=int, default=4096)

    scaling = ap.add_argument_group("multi-node scaling")
    scaling.add_argument("--omni_num_workers", type=int, default=None,
                         help="Fixed actor count for QwenOmni stage. Default: autoscaler decides.")
    scaling.add_argument("--asr_num_workers", type=int, default=None,
                         help="Fixed actor count for ASR recovery (QwenASR or language-routed).")
    scaling.add_argument("--pnc_num_workers", type=int, default=None,
                         help="Fixed actor count for PnC stage.")
    scaling.add_argument("--itn_num_workers", type=int, default=None,
                         help="Fixed actor count for ITN stage.")
    return ap


def _resolve_language_flags(args: argparse.Namespace) -> None:
    """Derive primary_model and recovery_model from --language.

    --primary_model and --recovery_model take priority when set explicitly;
    --language fills in only what is missing.
    """
    if args.language is None:
        return
    lang = args.language.lower().strip()
    all_known = OMNI_PRIMARY_LANGUAGE_CODES | QWEN_ASR_PRIMARY_LANGUAGE_CODES | WHISPER_PRIMARY_LANGUAGE_CODES
    if lang not in all_known:
        raise SystemExit(
            f"Unknown --language '{lang}'. Supported codes: {sorted(all_known)}. "
            "Or omit --language and set --primary_model / --recovery_model manually."
        )
    primary_was_explicit = args.primary_model is not None
    if not primary_was_explicit:
        if lang in OMNI_PRIMARY_LANGUAGE_CODES:
            args.primary_model = "qwen_omni"
        elif lang in QWEN_ASR_PRIMARY_LANGUAGE_CODES:
            args.primary_model = "qwen_asr"
        else:
            args.primary_model = "whisper"
    if args.recovery_model is None and lang in INDIC_LANGUAGE_CODES:
        args.recovery_model = "indic_conformer"
    logger.info(
        "Language '{}' → primary_model={} ({}), recovery_model={}",
        lang,
        args.primary_model,
        "explicit" if primary_was_explicit else "auto",
        args.recovery_model or "(auto from primary)",
    )


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    args = _build_arg_parser().parse_args()
    _resolve_language_flags(args)

    prompt = args.ml_prompt
    if args.ml_prompt_file:
        with open(args.ml_prompt_file, encoding="utf-8") as f:
            prompt = f.read().strip()

    en_prompt: str | None = None
    if args.en_prompt_file:
        with open(args.en_prompt_file, encoding="utf-8") as f:
            en_prompt = f.read().strip()

    followup_prompt = args.followup_prompt
    if args.followup_prompt_file:
        with open(args.followup_prompt_file, encoding="utf-8") as f:
            followup_prompt = f.read().strip()

    system_prompt = None
    if args.system_prompt:
        if os.path.isfile(args.system_prompt):
            with open(args.system_prompt, encoding="utf-8") as f:
                system_prompt = f.read().strip()
        else:
            system_prompt = args.system_prompt

    pnc_prompt_text = args.pnc_prompt
    if args.pnc_prompt_file:
        with open(args.pnc_prompt_file, encoding="utf-8") as f:
            pnc_prompt_text = f.read().strip()

    itn_prompt_text = None
    if args.itn_prompt_file:
        with open(args.itn_prompt_file, encoding="utf-8") as f:
            itn_prompt_text = f.read().strip()

    # Auto-derive recovery_model from primary when not set by --recovery_model or --language.
    _PRIMARY_TO_RECOVERY = {"qwen_omni": "qwen_asr", "qwen_asr": "whisper", "whisper": "parakeet"}
    if args.primary_model is None:
        raise SystemExit(
            "Either --language <code> or --primary_model {qwen_omni,qwen_asr,whisper} is required."
        )
    if args.recovery_model is None:
        args.recovery_model = _PRIMARY_TO_RECOVERY[args.primary_model]
    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = 2 if args.primary_model == "qwen_omni" else 1
    logger.info(
        "primary_model={}, recovery_model={}, tensor_parallel_size={}",
        args.primary_model, args.recovery_model, args.tensor_parallel_size,
    )

    # Key that holds the primary model's transcription used by all downstream stages.
    # Omni with a disfluency follow-up produces two turns; we use the second (s2).
    # All other primary models produce a single output written to "primary_prediction".
    primary_text_key = "primary_prediction_s2" if (args.primary_model == "qwen_omni" and followup_prompt) else "primary_prediction"

    stages = [
        UnifiedAudioReader(
            yaml_path=args.data_config,
            corpus_filter=args.corpus,
            output_dir=args.output_dir,
        )
        if args.use_unified_reader else
        NemoTarredAudioReader(
            yaml_path=args.data_config,
            corpus_filter=args.corpus,
            s3_endpoint_url=args.s3_endpoint_url,
            output_dir=args.output_dir,
        ).with_({"nemo_tar_shard_reader": {"resources": Resources(cpus=4.0)}}),
        InitializeFieldsStage(
            pipeline_notes={
                "primary_model": args.primary_model,
                "recovery_model": args.recovery_model,
            },
        ),
    ]

    if args.sed_checkpoint:
        from nemo_curator.stages.audio.inference.sed import SEDInferenceStage
        from nemo_curator.stages.audio.postprocessing.sed_postprocessing import SEDPostprocessingStage

        stages.extend([
            SEDInferenceStage(
                checkpoint_path=args.sed_checkpoint,
                model_type=args.sed_model_type,
                batch_size=args.sed_batch_size,
                num_workers_override=args.sed_num_workers,
                resources=Resources(cpus=1.0, gpu_memory_gb=args.sed_gpu_memory_gb),
            ),
            SEDPostprocessingStage(
                threshold=args.sed_speech_threshold,
                min_duration_sec=args.sed_min_duration,
                merge_gap_sec=args.sed_merge_gap,
                emit_superclasses=args.sed_superclasses,
            ),
        ])

    if args.primary_model == "qwen_omni":
        stages.append(InferenceQwenOmniStage(
            model_id=args.model_id,
            prompt_text=prompt,
            en_prompt_text=en_prompt,
            followup_prompt=followup_prompt,
            system_prompt=system_prompt,
            tensor_parallel_size=args.tensor_parallel_size,
            batch_size=args.batch_size,
            max_output_tokens=args.max_output_tokens,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            prep_workers=args.prep_workers,
            source_lang_key=args.source_lang_key,
            pred_text_key="primary_prediction",
            disfluency_text_key="primary_prediction_s2",
            keep_waveform=True,
            num_workers_override=args.omni_num_workers,
        ))

    elif args.primary_model == "qwen_asr":
        stages.append(InferenceQwenASRStage(
            name="QwenASR_primary",
            model_id=args.asr_model_id,
            pred_text_key="primary_prediction",
            keep_waveform=True,
            source_lang_key=args.source_lang_key,
            batch_size=args.asr_batch_size,
            gpu_memory_utilization=args.asr_gpu_memory_utilization,
            max_new_tokens=args.asr_max_new_tokens,
            max_inference_batch_size=args.asr_batch_size,
            num_workers_override=args.omni_num_workers,
        ))

    elif args.primary_model == "whisper":
        stages.append(InferenceFasterWhisperStage(
            name="Whisper_primary",
            model_size_or_path=args.whisper_model_size_or_path,
            device=args.whisper_device,
            compute_type=args.whisper_compute_type,
            download_root=args.whisper_download_root,
            pred_text_key="primary_prediction",
            keep_waveform=True,
            source_lang_key=args.source_lang_key,
            batch_size=args.asr_batch_size,
            num_workers_override=args.omni_num_workers,
        ))

    if args.primary_model == "qwen_omni" and followup_prompt:
        stages.append(DisfluencyWerGuardStage(
            ref_text_key="primary_prediction",
            hyp_text_key="primary_prediction_s2",
            max_wer_pct=50.0,
        ))

    stages.append(WhisperHallucinationStage(
        name="WhisperHallucination_primary",
        common_hall_file=args.hall_phrases,
        text_key=primary_text_key,
        unique_words_threshold=args.unique_words_threshold,
        long_word_threshold=args.long_word_threshold,
        long_word_rel_threshold=args.long_word_rel_threshold,
        max_char_rate=args.max_char_rate,
    ))
    if args.recovery_model == "indic_conformer":
        recovery_stage = InferenceIndicConformerStage(
            name="IndicConformer_recovery",
            model_id=args.indic_conformer_model_id,
            decode_mode=args.indic_conformer_decode,
            source_lang_key=args.source_lang_key,
            batch_size=args.asr_batch_size,
            num_workers_override=args.asr_num_workers,
        )
    elif args.recovery_model == "qwen_asr":
        recovery_stage = InferenceQwenASRStage(
            name="QwenASR_recovery",
            model_id=args.asr_model_id,
            source_lang_key=args.source_lang_key,
            batch_size=args.asr_batch_size,
            gpu_memory_utilization=args.asr_gpu_memory_utilization,
            max_new_tokens=args.asr_max_new_tokens,
            max_inference_batch_size=args.asr_batch_size,
            num_workers_override=args.asr_num_workers,
        )
    elif args.recovery_model == "whisper":
        recovery_stage = InferenceFasterWhisperStage(
            name="Whisper_recovery",
            model_size_or_path=args.whisper_model_size_or_path,
            device=args.whisper_device,
            compute_type=args.whisper_compute_type,
            download_root=args.whisper_download_root,
            source_lang_key=args.source_lang_key,
            batch_size=args.asr_batch_size,
            num_workers_override=args.asr_num_workers,
        )
    elif args.recovery_model == "parakeet":
        recovery_stage = InferenceParakeetStage(
            name="Parakeet_recovery",
            model_id=args.parakeet_model_id,
            cache_dir=args.parakeet_cache_dir,
            inference_batch_size=args.parakeet_inference_batch_size,
            source_lang_key=args.source_lang_key,
            batch_size=args.asr_batch_size,
            num_workers_override=args.asr_num_workers,
        )

    stages.extend([
        recovery_stage,
        WhisperHallucinationStage(
            name="WhisperHallucination_asr",
            common_hall_file=args.hall_phrases,
            text_key="asr_prediction",
            overwrite=True,
            recovery_value="Recovered:ASR",
            unique_words_threshold=args.unique_words_threshold,
            long_word_threshold=args.long_word_threshold,
            long_word_rel_threshold=args.long_word_rel_threshold,
            max_char_rate=args.max_char_rate,
        ),
    ])

    stages.append(SelectBestPredictionStage(
        primary_text_key=primary_text_key,
        asr_text_key="asr_prediction",
    ))

    stages.extend([
        FastTextLIDStage(
            model_path=args.fasttext_model,
            text_key="best_prediction",
            source_lang_key=args.source_lang_key,
            min_lang_prob=args.min_lang_prob,
        ),
        RegexSubstitutionStage(
            regex_params_yaml=args.regex_yaml,
            text_key="best_prediction",
            output_text_key="cleaned_text",
        ),
        AbbreviationConcatStage(
            text_key="cleaned_text",
            output_text_key="abbreviated_text",
            source_lang_key=args.source_lang_key,
        ),
    ])

    if args.enable_pnc:
        stages.extend([
            PnCRestorationStage(
                model_id=args.pnc_model_id,
                text_key="abbreviated_text",
                output_text_key="pnc_text",
                tensor_parallel_size=args.pnc_tensor_parallel_size,
                batch_size=args.pnc_batch_size,
                max_model_len=args.pnc_max_model_len,
                max_num_seqs=args.pnc_max_num_seqs,
                gpu_memory_utilization=args.pnc_gpu_memory_utilization,
                prep_workers=args.pnc_prep_workers,
                num_workers_override=args.pnc_num_workers,
                **({"pnc_prompt": pnc_prompt_text} if pnc_prompt_text else {}),
                **({"completeness_prompt": args.completeness_prompt} if args.completeness_prompt else {}),
                source_lang_key=args.pnc_source_lang_key,
            ),
            PnCContentGuardStage(
                text_key="abbreviated_text",
                pnc_text_key="pnc_text",
                rejected_text_key="rejected_pnc_text",
            ),
        ])

    if args.enable_itn:
        stages.append(ITNRestorationStage(
            model_id=args.itn_model_id,
            prompt_text=itn_prompt_text,
            text_key=args.itn_text_key or ("pnc_text" if args.enable_pnc else "abbreviated_text"),
            output_text_key=args.itn_output_key,
            tensor_parallel_size=args.itn_tensor_parallel_size,
            max_output_tokens=args.itn_max_output_tokens,
            max_model_len=args.itn_max_model_len,
            max_num_seqs=args.itn_max_num_seqs,
            gpu_memory_utilization=args.itn_gpu_memory_utilization,
            batch_size=args.itn_batch_size,
            enable_validation=not args.itn_no_validation,
            num_workers_override=args.itn_num_workers,
        ))

    stages.append(ShardedManifestWriterStage(output_dir=args.output_dir))

    pipeline = Pipeline(
        name="qwen_omni_inference",
        stages=stages,
    )

    logger.info(f"Pipeline: {pipeline.describe()}")

    executor = RayDataExecutor()

    t0 = time.time()
    pipeline.run(executor=executor)
    elapsed = time.time() - t0
    logger.info(f"Pipeline finished in {elapsed / 60:.1f} min. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
