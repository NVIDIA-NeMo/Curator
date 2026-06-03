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
QwenASR hallucination recovery (``--asr_model_id``).

Model selection:
    Pass ``--language <code>`` to auto-select both models, or set them explicitly:

    ``--primary_model``   (required unless --language is given)
        qwen_omni  → InferenceQwenOmniStage       (Qwen3-Omni vLLM, ``--model_id``)
                     recommended: en de es fr it pt ru nl
        qwen_asr   → InferenceQwenASRStage         (Qwen3-ASR, ``--asr_model_id``)
                     recommended: pl cs ro hu el fi da sv
        whisper    → InferenceFasterWhisperStage   (Whisper Large V3, ``--whisper_model_size_or_path``)
                     recommended: lt lv hr et bg sk sl mt uk
        parakeet_riva→ InferenceParakeetStage      (local Riva Parakeet ``.nemo``, ``--parakeet_riva_model_id``)
                     recommended: hi ta bn (fallback: indic_hybrid)
        indic_hybrid→ InferenceIndicConformerHybridStage (AI4Bharat per-language hybrid CTC+RNNT
                     ``.nemo``, ``--indic_hybrid_model_id``)
                     recommended: other Indic ISO codes (no fallback)

    ``--recovery_model``  (optional; auto-derived from primary when omitted)
        qwen_asr       → InferenceQwenASRStage        (default recovery for qwen_omni primary)
        whisper        → InferenceFasterWhisperStage   (default recovery for qwen_asr primary)
        parakeet_v3    → InferenceParakeetStage        (Parakeet-TDT v3; default recovery for whisper primary)
        indic_hybrid   → InferenceIndicConformerHybridStage (default recovery for parakeet_riva primary on hi/ta/bn)
        none           → skip recovery (default for indic_hybrid primary)

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
        → outputs primary_model_prediction (and primary_model_prediction_s2 for Omni with followup)
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

from nemo_curator.stages.audio.alm.sharded_manifest_writer import ShardedManifestWriterStage
from nemo_curator.stages.audio.inference.faster_whisper import InferenceFasterWhisperStage
from nemo_curator.stages.audio.inference.indic_conformer_hybrid import InferenceIndicConformerHybridStage
from nemo_curator.stages.audio.inference.parakeet import InferenceParakeetStage
from nemo_curator.stages.audio.inference.qwen_asr import InferenceQwenASRStage
from nemo_curator.stages.audio.inference.qwen_omni import InferenceQwenOmniStage
from nemo_curator.stages.audio.io.nemo_speech_reader import NeMoSpeechAudioReader
from nemo_curator.stages.audio.pipeline_utils import INDIC_CONFORMER_LANGUAGE_CODES
from nemo_curator.stages.audio.text_filtering import (
    AbbreviationConcatStage,
    DisfluencyWerGuardStage,
    FastTextLIDStage,
    InitializeFieldsStage,
    RegexSubstitutionStage,
    WhisperHallucinationStage,
)
from nemo_curator.stages.audio.text_filtering.select_best_prediction import SelectBestPredictionStage
from nemo_curator.stages.resources import Resources

QWEN_ASR_DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
# Per-language AI4Bharat IndicConformer hybrid (CTC+RNNT) checkpoints. {lang} is filled from --language.
INDIC_CONFORMER_HYBRID_DEFAULT_MODEL_ID = "ai4bharat/indicconformer_stt_{lang}_hybrid_ctc_rnnt_large"
# faster-whisper alias; equivalent HF openai/whisper-large-v3 in Transformers format.
WHISPER_DEFAULT_MODEL = "large-v3"
PARAKEET_V3_DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
PARAKEET_RIVA_DEFAULT_MODEL_ID = (
    "/lustre/fsw/portfolios/llmservice/users/ntadevosyan/projects/granary-v2-asr/"
    "checkpoints/parakeet_1.1b_indic_multilingual_v1.0_epoch_37.nemo"
)

# Recommended language codes for each primary inference model (used for auto-selection via --language).
QWEN_OMNI_RECOMMENDED_LANGS     = {"en", "de", "es", "fr", "it", "pt", "ru", "nl"}
QWEN_ASR_RECOMMENDED_LANGS      = {"pl", "cs", "ro", "hu", "el", "fi", "da", "sv"}
WHISPER_RECOMMENDED_LANGS       = {"lt", "lv", "hr", "et", "bg", "sk", "sl", "mt", "uk"}
PARAKEET_RIVA_PRIMARY_LANGS     = frozenset({"hi", "ta", "bn"})  # languages covered by the local Riva Parakeet .nemo
# Indic languages handled directly by a per-language IndicConformer hybrid monolingual
# model (every Indic language except the hi/ta/bn covered by the Riva Parakeet primary).
INDIC_HYBRID_PRIMARY_LANGS      = INDIC_CONFORMER_LANGUAGE_CODES - PARAKEET_RIVA_PRIMARY_LANGS

# Default recovery model paired with each primary (when --recovery_model is not set):
#   qwen_omni → qwen_asr;  qwen_asr → whisper;  whisper → parakeet_v3
#   parakeet_riva (hi/ta/bn) → indic_hybrid (matching monolingual);  indic_hybrid → none


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="QwenOmni in-process vLLM pipeline")
    ap.add_argument("--data_config", type=str, required=True, help="Granary YAML data config.")
    ap.add_argument("--corpus", type=str, nargs="*", default=None, help="Process only these corpora.")
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory for per-shard manifests.")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument(
        "--ml_prompt",
        type=str,
        default="Transcribe the audio.",
        help="Multilingual prompt text. Supports {language} placeholder resolved per-sample from source_lang.",
    )
    ap.add_argument(
        "--ml_prompt_file", type=str, default=None, help="Read multilingual prompt from file. Overrides --ml_prompt."
    )
    ap.add_argument(
        "--en_prompt_file",
        type=str,
        default=None,
        help="English-specific prompt file. Used for en samples; --ml_prompt_file is used for all other languages.",
    )
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
    ap.add_argument(
        "--execution_mode", type=str, default="streaming", choices=["streaming", "batch"], help="Xenna execution mode."
    )
    ap.add_argument(
        "--autoscale_interval_s",
        type=int,
        default=180,
        help="Seconds between Xenna streaming autoscaler checks. Lower values ramp up GPU actors faster on multi-node.",
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
            f"{sorted(PARAKEET_RIVA_PRIMARY_LANGS)} → primary=parakeet_riva, recovery=indic_hybrid; "
            f"{sorted(INDIC_HYBRID_PRIMARY_LANGS)} → primary=indic_hybrid, recovery=none; "
            f"{sorted(QWEN_OMNI_RECOMMENDED_LANGS)} → primary=qwen_omni, recovery=qwen_asr; "
            f"{sorted(QWEN_ASR_RECOMMENDED_LANGS)} → primary=qwen_asr, recovery=whisper; "
            f"{sorted(WHISPER_RECOMMENDED_LANGS)} → primary=whisper, recovery=parakeet_v3."
        ),
    )
    primary.add_argument(
        "--primary_model",
        type=str,
        choices=["qwen_omni", "qwen_asr", "whisper", "parakeet_riva", "indic_hybrid"],
        default=None,
        help=(
            "Primary ASR model placed after SED. "
            "qwen_omni: Qwen3-Omni vLLM (--model_id); "
            "qwen_asr: Qwen3-ASR (--asr_model_id); "
            "whisper: Faster-Whisper Large V3 (--whisper_model_size_or_path); "
            "parakeet_riva: local Indic Riva Parakeet .nemo (--parakeet_riva_model_id), used for hi/ta/bn; "
            "indic_hybrid: AI4Bharat IndicConformer hybrid CTC+RNNT per-language .nemo (--indic_hybrid_model_id), "
            "used for the other Indic languages. "
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
    sed.add_argument("--sed_gpu_memory_gb", type=float, default=2.0,
                     help="GPU memory in GB for SED inference stage.")
    sed.add_argument("--sed_superclasses", action="store_true", default=False,
                     help="Emit aggregated superclass events (noisy-or per group) instead of per-class subcategory events.")
    sed.add_argument("--sed_num_workers", type=int, default=None,
                     help="Fixed actor count for SED stage.")

    asr = ap.add_argument_group(
        "recovery ASR (secondary inference)",
        "Use --recovery_model to pick the fallback model explicitly, or let --language set it automatically. "
        "Default auto-pairing (when neither --recovery_model nor --language is given): "
        "primary=qwen_omni → qwen_asr; primary=qwen_asr → whisper; primary=whisper → parakeet_v3; "
        "primary=parakeet_riva → indic_hybrid; primary=indic_hybrid → none.",
    )
    asr.add_argument(
        "--recovery_model",
        type=str,
        choices=["qwen_asr", "whisper", "parakeet_v3", "indic_hybrid", "none"],
        default=None,
        help=(
            "Recovery (fallback) ASR model. "
            "qwen_asr: Qwen3-ASR (--asr_model_id); "
            "whisper: Faster-Whisper Large V3 (--whisper_model_size_or_path); "
            "parakeet_v3: Parakeet-TDT v3 (--parakeet_v3_model_id); "
            "indic_hybrid: AI4Bharat IndicConformer hybrid per-language .nemo (--indic_hybrid_model_id), "
            "fallback for the hi/ta/bn Parakeet primary; "
            "none: skip recovery entirely. "
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
        "--indic_hybrid_model_id",
        type=str,
        default=INDIC_CONFORMER_HYBRID_DEFAULT_MODEL_ID,
        metavar="HF_REPO_OR_PATH",
        help=(
            f"AI4Bharat IndicConformer hybrid per-language .nemo (default: {INDIC_CONFORMER_HYBRID_DEFAULT_MODEL_ID}). "
            "A literal '{lang}' is replaced by --language. Local .nemo path or gated HF repo (set HF_TOKEN). "
            "Used when --primary_model/--recovery_model is indic_hybrid."
        ),
    )
    asr.add_argument(
        "--indic_hybrid_decode",
        type=str,
        default="rnnt",
        choices=["ctc", "rnnt"],
        help="Decoding mode for the IndicConformer hybrid model (model card recommends rnnt).",
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
        "--parakeet_v3_model_id",
        type=str,
        default=PARAKEET_V3_DEFAULT_MODEL_ID,
        metavar="HF_REPO_OR_PATH",
        help=(
            f"NVIDIA Parakeet-TDT v3 model ID (default: {PARAKEET_V3_DEFAULT_MODEL_ID}). "
            "Multilingual European model used as the recovery model for --recovery_model parakeet "
            "(whisper-primary European languages)."
        ),
    )
    asr.add_argument(
        "--parakeet_riva_model_id",
        type=str,
        default=PARAKEET_RIVA_DEFAULT_MODEL_ID,
        metavar="NEMO_OR_PATH",
        help=(
            f"Local Indic Riva Parakeet NeMo checkpoint (default: {PARAKEET_RIVA_DEFAULT_MODEL_ID}). "
            "Not on HuggingFace — must be a reachable ``.nemo`` path on the compute nodes. "
            "Indic hi/ta/bn model used as the primary for --primary_model parakeet. "
            "Distinct model from --parakeet_v3_model_id."
        ),
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
    scaling.add_argument("--primary_num_workers", type=int, default=None,
                         help="Fixed actor count for primary ASR stage. Default: autoscaler decides.")
    scaling.add_argument("--fallback_num_workers", type=int, default=None,
                         help="Fixed actor count for fallback/recovery ASR stage. Default: autoscaler decides.")
    return ap


def _resolve_language_flags(args: argparse.Namespace) -> None:
    """Derive primary_model and recovery_model from --language.

    --primary_model and --recovery_model take priority when set explicitly;
    --language fills in only what is missing.
    """
    if args.language is None:
        return
    lang = args.language.lower().strip()
    all_known = (
        QWEN_OMNI_RECOMMENDED_LANGS
        | QWEN_ASR_RECOMMENDED_LANGS
        | WHISPER_RECOMMENDED_LANGS
        | PARAKEET_RIVA_PRIMARY_LANGS
        | INDIC_HYBRID_PRIMARY_LANGS
    )
    if lang not in all_known:
        raise SystemExit(
            f"Unknown --language '{lang}'. Supported codes: {sorted(all_known)}. "
            "Or omit --language and set --primary_model / --recovery_model manually."
        )
    primary_was_explicit = args.primary_model is not None
    recovery_was_explicit = args.recovery_model is not None
    if not primary_was_explicit:
        if lang in PARAKEET_RIVA_PRIMARY_LANGS:
            args.primary_model = "parakeet_riva"
        elif lang in INDIC_HYBRID_PRIMARY_LANGS:
            args.primary_model = "indic_hybrid"
        elif lang in QWEN_OMNI_RECOMMENDED_LANGS:
            args.primary_model = "qwen_omni"
        elif lang in QWEN_ASR_RECOMMENDED_LANGS:
            args.primary_model = "qwen_asr"
        else:
            args.primary_model = "whisper"
    if not recovery_was_explicit:
        if lang in PARAKEET_RIVA_PRIMARY_LANGS:
            args.recovery_model = "indic_hybrid"
        elif lang in INDIC_HYBRID_PRIMARY_LANGS:
            args.recovery_model = "none"
    logger.info(
        "Language '{}' → primary_model={} ({}), recovery_model={} ({})",
        lang,
        args.primary_model,
        "explicit" if primary_was_explicit else "auto",
        args.recovery_model or "(auto from primary)",
        "explicit" if recovery_was_explicit else "auto",
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

    # Auto-derive recovery_model from primary when not set by --recovery_model or --language.
    _PRIMARY_TO_RECOVERY = {
        "qwen_omni": "qwen_asr",
        "qwen_asr": "whisper",
        "whisper": "parakeet_v3",
        "parakeet_riva": "indic_hybrid",
        "indic_hybrid": "none",
    }

    def _resolve_indic_hybrid_model_id() -> str:
        mid = args.indic_hybrid_model_id
        if "{lang}" in mid:
            if not args.language:
                raise SystemExit("--indic_hybrid_model_id uses '{lang}' but --language is not set.")
            mid = mid.format(lang=args.language.lower().strip())
        return mid
    if args.primary_model is None:
        raise SystemExit(
            "Either --language <code> or --primary_model "
            "{qwen_omni,qwen_asr,whisper,parakeet_riva,indic_hybrid} is required."
        )
    if args.recovery_model is None:
        args.recovery_model = _PRIMARY_TO_RECOVERY[args.primary_model]
    has_recovery = args.recovery_model != "none"
    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = 2 if args.primary_model == "qwen_omni" else 1
    logger.info(
        "primary_model={}, recovery_model={}, has_recovery={}, tensor_parallel_size={}",
        args.primary_model, args.recovery_model, has_recovery, args.tensor_parallel_size,
    )

    # Key that holds the primary model's transcription used by all downstream stages.
    # Omni with a disfluency follow-up produces two turns; we use the second (s2).
    # All other primary models produce a single output written to "primary_model_prediction".
    primary_text_key = "primary_model_prediction_s2" if (args.primary_model == "qwen_omni" and followup_prompt) else "primary_model_prediction"

    language_filter = [args.language] if args.language else None

    stages = [
        NeMoSpeechAudioReader(
            yaml_path=args.data_config,
            corpus_filter=args.corpus,
            language_filter=language_filter,
            output_dir=args.output_dir),
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
            pred_text_key="primary_model_prediction",
            disfluency_text_key="primary_model_prediction_s2",
            keep_waveform=True,
            num_workers_override=args.primary_num_workers,
        ))

    elif args.primary_model == "qwen_asr":
        stages.append(InferenceQwenASRStage(
            name="QwenASR_primary",
            model_id=args.asr_model_id,
            pred_text_key="primary_model_prediction",
            keep_waveform=True,
            source_lang_key=args.source_lang_key,
            batch_size=args.asr_batch_size,
            gpu_memory_utilization=args.asr_gpu_memory_utilization,
            max_new_tokens=args.asr_max_new_tokens,
            max_inference_batch_size=args.asr_batch_size,
            num_workers_override=args.primary_num_workers,
        ))

    elif args.primary_model == "whisper":
        stages.append(InferenceFasterWhisperStage(
            name="Whisper_primary",
            model_size_or_path=args.whisper_model_size_or_path,
            device=args.whisper_device,
            compute_type=args.whisper_compute_type,
            pred_text_key="primary_model_prediction",
            keep_waveform=True,
            source_lang_key=args.source_lang_key,
            batch_size=args.asr_batch_size,
            num_workers_override=args.primary_num_workers,
        ))

    elif args.primary_model == "parakeet_riva":
        stages.append(InferenceParakeetStage(
            name="ParakeetRiva_primary",
            model_id=args.parakeet_riva_model_id,
            supported_langs=PARAKEET_RIVA_PRIMARY_LANGS,
            inference_batch_size=args.parakeet_inference_batch_size,
            source_lang_key=args.source_lang_key,
            pred_text_key="primary_model_prediction",
            keep_waveform=True,
            batch_size=args.asr_batch_size,
            num_workers_override=args.primary_num_workers,
        ))

    elif args.primary_model == "indic_hybrid":
        stages.append(InferenceIndicConformerHybridStage(
            name="IndicConformerHybrid_primary",
            model_id=_resolve_indic_hybrid_model_id(),
            decode_mode=args.indic_hybrid_decode,
            source_lang_key=args.source_lang_key,
            pred_text_key="primary_model_prediction",
            keep_waveform=True,
            batch_size=args.asr_batch_size,
            num_workers_override=args.primary_num_workers,
        ))

    if args.primary_model == "qwen_omni" and followup_prompt:
        stages.append(DisfluencyWerGuardStage(
            ref_text_key="primary_model_prediction",
            hyp_text_key="primary_model_prediction_s2",
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
    if has_recovery:
        if args.recovery_model == "indic_hybrid":
            recovery_stage = InferenceIndicConformerHybridStage(
                name="IndicConformerHybrid_recovery",
                model_id=_resolve_indic_hybrid_model_id(),
                decode_mode=args.indic_hybrid_decode,
                source_lang_key=args.source_lang_key,
                pred_text_key="fallback_model_prediction",
                batch_size=args.asr_batch_size,
                num_workers_override=args.fallback_num_workers,
            )
        elif args.recovery_model == "qwen_asr":
            recovery_stage = InferenceQwenASRStage(
                name="QwenASR_recovery",
                model_id=args.asr_model_id,
                source_lang_key=args.source_lang_key,
                pred_text_key="fallback_model_prediction",
                batch_size=args.asr_batch_size,
                gpu_memory_utilization=args.asr_gpu_memory_utilization,
                max_new_tokens=args.asr_max_new_tokens,
                max_inference_batch_size=args.asr_batch_size,
                num_workers_override=args.fallback_num_workers,
            )
        elif args.recovery_model == "whisper":
            recovery_stage = InferenceFasterWhisperStage(
                name="Whisper_recovery",
                model_size_or_path=args.whisper_model_size_or_path,
                device=args.whisper_device,
                compute_type=args.whisper_compute_type,
                source_lang_key=args.source_lang_key,
                pred_text_key="fallback_model_prediction",
                batch_size=args.asr_batch_size,
                num_workers_override=args.fallback_num_workers,
            )
        elif args.recovery_model == "parakeet_v3":
            recovery_stage = InferenceParakeetStage(
                name="ParakeetV3_recovery",
                model_id=args.parakeet_v3_model_id,
                inference_batch_size=args.parakeet_inference_batch_size,
                source_lang_key=args.source_lang_key,
                pred_text_key="fallback_model_prediction",
                batch_size=args.asr_batch_size,
                num_workers_override=args.fallback_num_workers,
            )

        stages.extend([
            recovery_stage,
            WhisperHallucinationStage(
                name="WhisperHallucination_asr",
                common_hall_file=args.hall_phrases,
                text_key="fallback_model_prediction",
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
        asr_text_key="fallback_model_prediction",
        primary_source_label="primary",
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
