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
punctuation/capitalisation restoration (``--skip_pnc`` to disable).

Architecture:
    NemoTarredAudioReader (CPU)
        → streams NeMo-tarred shards, decodes audio in memory
    InitializeFieldsStage (CPU)
        → sets _skip_me = "", renames text → granary_v1_prediction
    InferenceQwenOmniStage (GPU, TP=2)
        → batched vLLM inference, outputs qwen3_prediction_s1/s2
    [optional] DisfluencyWerGuardStage (CPU)
        → compares Turn 1 vs Turn 2 WER
    WhisperHallucinationStage (CPU)
        → flags hallucination patterns, sets _skip_me
    [optional] InferenceQwenASRStage (GPU)
        → re-transcribes only hallucinated samples with Qwen3-ASR
    [optional] WhisperHallucinationStage (CPU)
        → checks QwenASR output; recovers or confirms hallucination
    SelectBestPredictionStage (CPU)
        → picks ASR prediction if recovered, else omni prediction
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
    [optional] ITNRestorationStage (GPU, text-only LLM, --enable_itn)
        → spoken→written text normalization, writes itn_text
    ShardedManifestWriterStage (CPU)
        → writes per-shard JSONL output with .done markers
"""

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

import argparse
import time

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm.sharded_manifest_writer import ShardedManifestWriterStage
from nemo_curator.stages.audio.inference.qwen_asr import InferenceQwenASRStage
from nemo_curator.stages.audio.inference.qwen_omni import InferenceQwenOmniStage
from nemo_curator.stages.audio.io.nemo_tarred_reader import NemoTarredAudioReader
from nemo_curator.stages.audio.text_filtering import (
    AbbreviationConcatStage,
    DisfluencyWerGuardStage,
    FastTextLIDStage,
    InitializeFieldsStage,
    PnCContentGuardStage,
    PnCRestorationStage,
    RegexSubstitutionStage,
    WhisperHallucinationStage,
)
from nemo_curator.stages.audio.text_filtering.itn_restoration import ITNRestorationStage
from nemo_curator.stages.audio.text_filtering.select_best_prediction import SelectBestPredictionStage
from nemo_curator.stages.resources import Resources


def _build_arg_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description="QwenOmni in-process vLLM pipeline")
    ap.add_argument("--data_config", type=str, required=True, help="Granary YAML data config.")
    ap.add_argument("--corpus", type=str, nargs="*", default=None, help="Process only these corpora.")
    ap.add_argument("--output_dir", type=str, required=True, help="Output directory for per-shard manifests.")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--prompt", type=str, default="Transcribe the audio.")
    ap.add_argument("--prompt_file", type=str, default=None, help="Read prompt from file.")
    ap.add_argument("--followup_prompt", type=str, default=None, help="Turn 2 follow-up prompt text.")
    ap.add_argument("--followup_prompt_file", type=str, default=None, help="Read Turn 2 follow-up prompt from file.")
    ap.add_argument("--system_prompt", type=str, default=None, help="System prompt text or path to file.")
    ap.add_argument("--tensor_parallel_size", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_output_tokens", type=int, default=256)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--max_num_seqs", type=int, default=16)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    ap.add_argument("--prep_workers", type=int, default=16, help="Thread pool size for audio preprocessing.")
    ap.add_argument("--s3_endpoint_url", type=str, default=None)
    ap.add_argument("--execution_mode", type=str, default="streaming",
                    choices=["streaming", "batch"], help="Xenna execution mode.")
    ap.add_argument("--autoscale_interval_s", type=int, default=180,
                    help="Seconds between Xenna streaming autoscaler checks. "
                         "Lower values ramp up GPU actors faster on multi-node.")

    tf = ap.add_argument_group("text filtering")
    tf.add_argument("--hall_phrases", type=str, required=True,
                    help="Path to hallucination phrases text file.")
    tf.add_argument("--fasttext_model", type=str, default="lid.176.ftz",
                    help="FastText LID model: HuggingFace repo ID, local path, or known name (lid.176.bin / lid.176.ftz).")
    tf.add_argument("--regex_yaml", type=str, required=True,
                    help="Path to regex substitution rules YAML.")
    tf.add_argument("--target_lang", type=str, default="en",
                    help="Expected language code for LID filtering.")
    tf.add_argument("--source_lang_key", type=str, default="",
                    help="Per-sample language key in manifest for {language} prompt substitution. "
                         "Empty (default) disables per-sample language threading.")
    tf.add_argument("--min_lang_prob", type=float, default=0.8,
                    help="Minimum FastText language probability to keep an entry.")
    tf.add_argument("--unique_words_threshold", type=float, default=0.4,
                    help="Unique-word ratio threshold for repeated n-gram hallucination detection.")
    tf.add_argument("--long_word_threshold", type=int, default=25,
                    help="Absolute character length above which a word is flagged as abnormally long.")
    tf.add_argument("--long_word_rel_threshold", type=float, default=3.0,
                    help="Relative length ratio for long-word hallucination detection.")
    tf.add_argument("--max_char_rate", type=float, default=40.0,
                    help="Min chars/s above which text is considered impossibly dense.")

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
    pnc.add_argument("--skip_pnc", action="store_true", default=False,
                     help="Skip PnC restoration stage entirely.")

    asr = ap.add_argument_group("QwenASR hallucination recovery")
    asr.add_argument("--asr_model_id", type=str, default=None,
                     help="QwenASR model ID or local path. If set, enables hallucination recovery.")
    asr.add_argument("--asr_language", type=str, default=None, help="Language hint for QwenASR.")
    asr.add_argument("--asr_batch_size", type=int, default=128)
    asr.add_argument("--asr_gpu_memory_utilization", type=float, default=0.7)
    asr.add_argument("--asr_max_new_tokens", type=int, default=4096)

    diar = ap.add_argument_group("Sortformer speaker diarization")
    diar.add_argument("--sortformer_model", type=str, default=None,
                      help="Sortformer HF model id or local .nemo path. If set, enables diarization.")
    diar.add_argument("--sortformer_cache_dir", type=str, default=None,
                      help="Cache directory for Sortformer model weights.")
    diar.add_argument("--sortformer_batch_size", type=int, default=8,
                      help="Batch size for Sortformer diarization.")
    diar.add_argument("--sortformer_gpu_memory_gb", type=float, default=8.0,
                      help="GPU memory (GB) reserved for Sortformer.")

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
    itn.add_argument("--itn_max_output_tokens", type=int, default=4096,
                     help="Max tokens to generate per ITN sample.")
    itn.add_argument("--itn_max_model_len", type=int, default=4096,
                     help="Max context length for ITN vLLM engine.")
    itn.add_argument("--itn_max_num_seqs", type=int, default=16,
                     help="Max concurrent sequences for ITN vLLM engine.")
    itn.add_argument("--itn_gpu_memory_utilization", type=float, default=0.95,
                     help="Fraction of GPU memory for ITN vLLM engine.")
    itn.add_argument("--itn_no_validation", action="store_true", help="Disable ITN output validation.")

    scaling = ap.add_argument_group("multi-node scaling")
    scaling.add_argument("--omni_num_workers", type=int, default=None,
                         help="Fixed actor count for QwenOmni stage. Default: autoscaler decides.")
    scaling.add_argument("--asr_num_workers", type=int, default=None,
                         help="Fixed actor count for QwenASR stage.")
    scaling.add_argument("--pnc_num_workers", type=int, default=None,
                         help="Fixed actor count for PnC stage.")
    scaling.add_argument("--itn_num_workers", type=int, default=None,
                         help="Fixed actor count for ITN stage.")
    return ap


def main() -> None:  # noqa: C901
    args = _build_arg_parser().parse_args()

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read().strip()

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

    omni_text_key = "qwen3_prediction_s2" if followup_prompt else "qwen3_prediction_s1"

    stages = [
        NemoTarredAudioReader(
            yaml_path=args.data_config,
            corpus_filter=args.corpus,
            s3_endpoint_url=args.s3_endpoint_url,
            output_dir=args.output_dir,
        ).with_({"nemo_tar_shard_reader": {"resources": Resources(cpus=4.0)}}),
        InitializeFieldsStage(),
    ]

    if args.sortformer_model:
        from nemo_curator.stages.audio.inference.sortformer import InferenceSortformerStage
        stages.append(InferenceSortformerStage(
            model_name=args.sortformer_model,
            cache_dir=args.sortformer_cache_dir,
            batch_size=args.sortformer_batch_size,
            store_segments=False,
            resources=Resources(gpu_memory_gb=args.sortformer_gpu_memory_gb),
        ))

    stages.append(InferenceQwenOmniStage(
        model_id=args.model_id,
        prompt_text=prompt,
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
        pred_text_key="qwen3_prediction_s1",
        disfluency_text_key="qwen3_prediction_s2",
        keep_waveform=bool(args.asr_model_id),
        num_workers=args.omni_num_workers,
    ))

    if followup_prompt:
        stages.append(DisfluencyWerGuardStage(
            ref_text_key="qwen3_prediction_s1",
            hyp_text_key="qwen3_prediction_s2",
            max_wer_pct=50.0,
        ))

    stages.append(WhisperHallucinationStage(
        name="WhisperHallucination_omni",
        common_hall_file=args.hall_phrases,
        text_key=omni_text_key,
        unique_words_threshold=args.unique_words_threshold,
        long_word_threshold=args.long_word_threshold,
        long_word_rel_threshold=args.long_word_rel_threshold,
        max_char_rate=args.max_char_rate,
    ))

    if args.asr_model_id:
        stages.extend([
            InferenceQwenASRStage(
                model_id=args.asr_model_id,
                language=args.asr_language,
                batch_size=args.asr_batch_size,
                gpu_memory_utilization=args.asr_gpu_memory_utilization,
                max_new_tokens=args.asr_max_new_tokens,
                num_workers=args.asr_num_workers,
            ),
            WhisperHallucinationStage(
                name="WhisperHallucination_asr",
                common_hall_file=args.hall_phrases,
                text_key="qwen3_asr_prediction",
                overwrite=True,
                recovery_value="Recovered:QwenASR",
                unique_words_threshold=args.unique_words_threshold,
                long_word_threshold=args.long_word_threshold,
                long_word_rel_threshold=args.long_word_rel_threshold,
                max_char_rate=args.max_char_rate,
            ),
        ])

    stages.append(SelectBestPredictionStage(
        primary_text_key=omni_text_key,
        asr_text_key="qwen3_asr_prediction",
    ))

    stages.extend([
        FastTextLIDStage(
            model_path=args.fasttext_model,
            text_key="best_prediction",
            target_lang=args.target_lang,
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
            language=args.target_lang,
        ),
    ])

    if not args.skip_pnc:
        stages.extend([
            PnCRestorationStage(
                model_id=args.pnc_model_id,
                text_key="abbreviated_text",
                output_text_key="pnc_text",
                tensor_parallel_size=args.pnc_tensor_parallel_size,
                batch_size=args.pnc_batch_size,
                max_model_len=args.pnc_max_model_len,
                max_num_seqs=args.pnc_max_num_seqs,
                prep_workers=args.pnc_prep_workers,
                num_workers=args.pnc_num_workers,
                **({"pnc_prompt": pnc_prompt_text} if pnc_prompt_text else {}),
                **({"completeness_prompt": args.completeness_prompt} if args.completeness_prompt else {}),
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
            text_key=args.itn_text_key or ("pnc_text" if not args.skip_pnc else "abbreviated_text"),
            output_text_key=args.itn_output_key,
            tensor_parallel_size=args.itn_tensor_parallel_size,
            max_output_tokens=args.itn_max_output_tokens,
            max_model_len=args.itn_max_model_len,
            max_num_seqs=args.itn_max_num_seqs,
            gpu_memory_utilization=args.itn_gpu_memory_utilization,
            batch_size=args.itn_batch_size,
            enable_validation=not args.itn_no_validation,
            num_workers=args.itn_num_workers,
        ))

    stages.append(ShardedManifestWriterStage(output_dir=args.output_dir))

    pipeline = Pipeline(
        name="qwen_omni_inference",
        stages=stages,
    )

    logger.info(f"Pipeline: {pipeline.describe()}")

    executor = XennaExecutor(config={
        "execution_mode": args.execution_mode,
        "autoscale_interval_s": args.autoscale_interval_s,
    })

    t0 = time.time()
    pipeline.run(executor=executor)
    elapsed = time.time() - t0
    logger.info(f"Pipeline finished in {elapsed / 60:.1f} min. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
