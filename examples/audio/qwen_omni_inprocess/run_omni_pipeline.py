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

"""Qwen3-Omni-only pipeline for benchmark / reference-improvement datasets.

Runs a slim subset of :file:`run_pipeline.py`:

    NeMoSpeechAudioReader
        → InitializeFieldsStage (text → granary_v1_prediction)
    InferenceQwenOmniStage
        → [optional] DisfluencyWerGuardStage
    WhisperHallucinationStage
    SelectBestPredictionStage
        → with optional reference fallback on hallucination
    RegexSubstitutionStage
    AbbreviationConcatStage
    ShardedManifestWriterStage

No SED, no recovery ASR.  Use ``--reference_text_key granary_v1_prediction``
with a prompt that contains ``{transcript}`` for reference-improvement runs.
"""

from __future__ import annotations

import argparse
import os
import time

from loguru import logger

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm.sharded_manifest_writer import ShardedManifestWriterStage
from nemo_curator.stages.audio.inference.qwen_omni import InferenceQwenOmniStage
from nemo_curator.stages.audio.io.nemo_speech_reader import NeMoSpeechAudioReader
from nemo_curator.stages.audio.text_filtering import (
    AbbreviationConcatStage,
    DisfluencyWerGuardStage,
    InitializeFieldsStage,
    RegexSubstitutionStage,
    WhisperHallucinationStage,
)
from nemo_curator.stages.audio.text_filtering.select_best_prediction import SelectBestPredictionStage


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Qwen3-Omni-only reference-improvement pipeline")
    ap.add_argument("--data_config", type=str, required=True, help="Granary YAML data config.")
    ap.add_argument("--corpus", type=str, nargs="*", default=None, help="Process only these corpora.")
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory for per-shard manifests.")
    ap.add_argument("--language", type=str, default=None, help="ISO 639-1 language code filter.")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument(
        "--ml_prompt",
        type=str,
        default="Transcribe the audio.",
        help="Multilingual prompt text. Supports {language} and {transcript} placeholders.",
    )
    ap.add_argument("--ml_prompt_file", type=str, default=None, help="Read multilingual prompt from file.")
    ap.add_argument(
        "--en_prompt_file",
        type=str,
        default=None,
        help="English-specific prompt file (e.g. reference-improvement prompt with {transcript}).",
    )
    ap.add_argument("--followup_prompt", type=str, default=None, help="Turn 2 follow-up prompt text.")
    ap.add_argument("--followup_prompt_file", type=str, default=None, help="Read Turn 2 follow-up prompt from file.")
    ap.add_argument("--system_prompt", type=str, default=None, help="System prompt text or path to file.")
    ap.add_argument(
        "--reference_text_key",
        type=str,
        default="granary_v1_prediction",
        help="Manifest key for the dataset reference transcript. "
             "Bound to {transcript} in prompts and used as hallucination fallback when enabled.",
    )
    ap.add_argument(
        "--use_reference_on_hallucination",
        action="store_true",
        help="When Omni output is flagged as hallucinated, keep the reference_text_key text instead.",
    )
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_output_tokens", type=int, default=256)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--max_num_seqs", type=int, default=16)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    ap.add_argument("--prep_workers", type=int, default=16)
    ap.add_argument("--source_lang_key", type=str, default="source_lang")
    ap.add_argument("--primary_num_workers", type=int, default=None)

    tf = ap.add_argument_group("text filtering")
    tf.add_argument("--hall_phrases", type=str, required=True, help="Path to hallucination phrases text file.")
    tf.add_argument("--regex_yaml", type=str, required=True, help="Path to regex substitution rules YAML.")
    tf.add_argument("--unique_words_threshold", type=float, default=0.4)
    tf.add_argument("--long_word_threshold", type=int, default=25)
    tf.add_argument("--long_word_rel_threshold", type=float, default=3.0)
    tf.add_argument("--max_char_rate", type=float, default=40.0)
    return ap


def _load_prompt_file(path: str | None) -> str | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def main() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    args = _build_arg_parser().parse_args()

    prompt = args.ml_prompt
    if args.ml_prompt_file:
        prompt = _load_prompt_file(args.ml_prompt_file) or prompt

    en_prompt = _load_prompt_file(args.en_prompt_file)
    followup_prompt = args.followup_prompt
    if args.followup_prompt_file:
        followup_prompt = _load_prompt_file(args.followup_prompt_file)

    system_prompt = None
    if args.system_prompt:
        if os.path.isfile(args.system_prompt):
            system_prompt = _load_prompt_file(args.system_prompt)
        else:
            system_prompt = args.system_prompt

    primary_text_key = (
        "primary_model_prediction_s2"
        if followup_prompt
        else "primary_model_prediction"
    )

    language_filter = [args.language.lower().strip()] if args.language else None

    stages = [
        NeMoSpeechAudioReader(
            yaml_path=args.data_config,
            corpus_filter=args.corpus,
            language_filter=language_filter,
            output_dir=args.output_dir,
        ),
        InitializeFieldsStage(
            pipeline_notes={
                "primary_model": "qwen_omni",
                "recovery_model": "none",
            },
        ),
        InferenceQwenOmniStage(
            model_id=args.model_id,
            prompt_text=prompt,
            en_prompt_text=en_prompt,
            followup_prompt=followup_prompt,
            system_prompt=system_prompt,
            reference_text_key=args.reference_text_key,
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
            num_workers_override=args.primary_num_workers,
        ),
    ]

    if followup_prompt:
        stages.append(DisfluencyWerGuardStage(
            ref_text_key="primary_model_prediction",
            hyp_text_key="primary_model_prediction_s2",
            max_wer_pct=50.0,
        ))

    stages.append(WhisperHallucinationStage(
        name="WhisperHallucination_primary",
        common_hall_file=args.hall_phrases,
        text_key=primary_text_key,
        language_key=args.source_lang_key,
        unique_words_threshold=args.unique_words_threshold,
        long_word_threshold=args.long_word_threshold,
        long_word_rel_threshold=args.long_word_rel_threshold,
        max_char_rate=args.max_char_rate,
    ))

    stages.append(SelectBestPredictionStage(
        primary_text_key=primary_text_key,
        reference_text_key=args.reference_text_key,
        use_reference_on_hallucination=args.use_reference_on_hallucination,
        primary_source_label="primary",
    ))

    stages.extend([
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

    pipeline = Pipeline(name="qwen_omni_reference_pipeline", stages=stages)
    logger.info(f"Pipeline: {pipeline.describe()}")
    logger.info(
        "language_filter={}, reference_text_key={}, use_reference_on_hallucination={}, followup={}",
        language_filter,
        args.reference_text_key,
        args.use_reference_on_hallucination,
        bool(followup_prompt),
    )

    t0 = time.time()
    pipeline.run(executor=RayDataExecutor())
    logger.info(f"Pipeline finished in {(time.time() - t0) / 60:.1f} min. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
