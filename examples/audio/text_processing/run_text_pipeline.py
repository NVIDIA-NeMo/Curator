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
    [if --enable_language_id] TextLLMStage: LanguageID (GPU)
        → asks the LLM for the primary language plus all languages present
        → writes llm_language_prediction
        → LLMLanguageVerificationStage (CPU): compares to source_lang; keeps
          code-switched samples containing source_lang, else sets _skipme to
          "Wrong language:LLMLanguageVerification"
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
    [if --enable_instruction_packer] InstructionPackerStage (CPU)
        → collects every enabled text variant into a single
          'preference_instructions' field (list of {prompt, target} pairs) ready
          for instruction-tuning training
    ShardedManifestWriterStage (CPU)
        → writes per-shard JSONL output with .done markers

Each GPU stage runs as its own Ray Data actor (one GPU each) and loads its own
vLLM engine, so each stage may use its own max_model_len: the lightweight stages
use --max_model_len (2048) while contextual ASR uses --context_asr_max_model_len
(8192). They are NOT one shared engine.
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
from nemo_curator.stages.audio.text_filtering.instruction_packer import InstructionPackerStage
from nemo_curator.stages.audio.text_filtering.llm_language_verification import LLMLanguageVerificationStage
from nemo_curator.stages.audio.text_filtering.remote_contextual_asr_extraction import (
    RemoteContextualASRExtractionStage,
)
from nemo_curator.stages.audio.text_filtering.fused_remote_text_llm_stage import FusedRemoteTextLLMStage
from nemo_curator.stages.audio.text_filtering.remote_text_llm_stage import RemoteTextLLMStage
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
_LANGUAGE_ID_PROMPT = _PROMPT_DIR / "language_id_prompt.md"


def _build_arg_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
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
        "--enable_language_id",
        action="store_true",
        default=False,
        help=(
            "Enable language-ID stage (GPU) plus verification (CPU): LLM writes "
            "llm_language_prediction (primary + all languages), then compares to source_lang. "
            "Code-switched samples containing source_lang are kept; otherwise _skipme is set "
            "to 'Wrong language:LLMLanguageVerification'."
        ),
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
    ap.add_argument(
        "--enable_instruction_packer",
        action="store_true",
        default=False,
        help="Pack all enabled text outputs (pnc/itn/itn_no-disfluencies/captioning/code-switched/"
        "speech_qa/context_asr) into a single 'preference_instructions' field — a list of "
        "{prompt, target} pairs the trainer samples from one-per-epoch.",
    )
    ap.add_argument(
        "--instructions_output_key",
        type=str,
        default="preference_instructions",
        help="Output field name for the packed instruction-target list.",
    )
    ap.add_argument(
        "--instruction_packer_seed",
        type=int,
        default=42,
        help="Base RNG seed for per-sample prompt-template sampling in the instruction packer.",
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
        "--language_id_prompt_file",
        type=str,
        default=None,
        help="Path to language-ID prompt file. Defaults to bundled language_id_prompt.md.",
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
            "Max tokens generated per sample for context ASR extraction. This stage's vLLM engine "
            "max_model_len (--context_asr_max_model_len) must be ≥ this value plus prompt length."
        ),
    )
    ctx.add_argument(
        "--context_asr_max_model_len",
        type=int,
        default=8192,
        help=(
            "vLLM max_model_len for the contextual-ASR extraction stage ONLY. This stage runs as its "
            "own Ray Data actor (one GPU, its own vLLM engine), so it can use a larger context window "
            "than the lightweight stages without inflating their KV cache. The global --max_model_len "
            "stays small (2048) for PnC/ITN/etc. Must exceed the extraction prompt + "
            "--context_asr_max_output_tokens."
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
    ap.add_argument(
        "--context_asr_num_workers",
        type=int,
        default=None,
        help="Explicit Ray actor count for the ContextualASR stage. Overrides --num_workers for that stage only.",
    )
    ap.add_argument(
        "--context_asr_max_concurrent_requests",
        type=int,
        default=None,
        help="Max concurrent requests for the ContextualASR stage. Overrides --inference_max_concurrent_requests for that stage only.",
    )
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--execution_mode", type=str, default="streaming", choices=["streaming", "batch"])

    # ── Remote inference server (optional) ───────────────────────────
    # When enabled, the LLM stages send OpenAI-compatible requests to one
    # shared NVIDIA Dynamo (vLLM) server instead of each loading its own engine.
    srv = ap.add_argument_group("remote inference server")
    srv.add_argument(
        "--use_inference_server",
        action="store_true",
        default=False,
        help="Start a local RayClient + NVIDIA Dynamo InferenceServer and route all LLM "
        "stages to it as CPU-only HTTP clients. Without this flag, stages run in-process vLLM.",
    )
    srv.add_argument(
        "--inference_served_model_name",
        type=str,
        default=None,
        help="Model name sent as 'model=' in requests. Defaults to --model_id.",
    )
    srv.add_argument("--inference_api_key", type=str, default="EMPTY", help="API key forwarded to the server.")
    srv.add_argument(
        "--inference_max_replicas", type=int, default=1, help="Number of Dynamo vLLM worker replicas (num_replicas)."
    )
    srv.add_argument(
        "--inference_server_tp",
        type=int,
        default=None,
        help="tensor_parallel_size for the server engine (default: auto = visible GPU count).",
    )
    srv.add_argument("--inference_port", type=int, default=8000, help="Server HTTP port.")
    srv.add_argument(
        "--inference_max_concurrent_requests",
        type=int,
        default=64,
        help="Max in-flight requests per stage actor (async client semaphore bound).",
    )
    srv.add_argument("--inference_request_timeout", type=int, default=120, help="Per-request timeout (seconds).")
    srv.add_argument(
        "--inference_health_timeout",
        type=int,
        default=300,
        help="Seconds to wait for the inference server to become healthy before failing (default: 300).",
    )
    srv.add_argument(
        "--fuse_stages",
        action="store_true",
        default=False,
        help="Fuse independent LLM stages (LanguageID, ITN, Captioning, CodeSwitching, SpeechQA) "
        "into a single actor that fires all their prompts in parallel via asyncio.gather. "
        "Only active with --use_inference_server. Reduces sequential stage overhead and keeps "
        "the Dynamo server continuously saturated.",
    )

    return ap


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    args = _build_arg_parser().parse_args()

    if (
        not args.enable_itn
        and not args.enable_itn_no_disfluencies
        and not args.enable_captioning
        and not args.enable_pnc
        and not args.enable_language_id
        and not args.enable_context_asr
        and not args.enable_code_switching
        and not args.enable_speech_qa
    ):
        logger.warning(
            "No stages enabled. Use --enable_pnc, --enable_language_id, --enable_itn, --enable_itn_no-disfluencies, "
            "--enable_captioning, --enable_context_asr, --enable_code_switching, or --enable_speech_qa."
        )
        return

    # ── Optional Dynamo inference server ─────────────────────────────
    # Two modes:
    #   --use_inference_server -> start a local RayClient + NVIDIA Dynamo server
    #                             and route all LLM stages to it as HTTP clients
    #   (default)              -> in-process vLLM, one engine per stage
    # Resolved BEFORE building stages so the server is healthy at pipeline start.
    inference_server = None
    ray_client = None
    remote_base_url = None
    remote_model_name = None
    if args.use_inference_server:
        import torch

        from nemo_curator.core.client import RayClient
        from nemo_curator.core.serve import DynamoServerConfig, DynamoVLLMModelConfig, InferenceServer

        # Count GPUs without Ray (ray.available_resources() requires a running
        # cluster). torch.cuda honours CUDA_VISIBLE_DEVICES. Start the cluster
        # exposing those GPUs BEFORE the backend deploys; the server is torn down
        # before the client at teardown (see finally).
        n_gpus = torch.cuda.device_count()
        ray_client = RayClient(num_gpus=n_gpus)
        ray_client.start()

        server_tp = args.inference_server_tp or n_gpus
        # One engine serves every stage, so its max_model_len must cover the
        # largest requirement (context_asr 8192 vs the text stages' 2048).
        server_max_len = args.max_model_len
        if args.enable_context_asr:
            server_max_len = max(server_max_len, args.context_asr_max_model_len)
        engine_kwargs = {
            "tensor_parallel_size": server_tp,
            "max_model_len": server_max_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_cache_dtype": args.kv_cache_dtype,
            "trust_remote_code": True,
            "max_num_seqs": args.max_num_seqs,
        }
        if server_tp == 1:
            # TP=1: force the uniprocessor executor so each replica skips
            # torch.distributed init (no TCPStore / rendezvous port). For TP>1 leave
            # it unset — vLLM picks 'mp' for single-node TP, and Dynamo drives
            # multi-node TP via --nnodes/--node-rank/--master-addr; forcing 'mp'
            # would break the multi-node case.
            engine_kwargs["distributed_executor_backend"] = "uni"

        # Dynamo uses a fixed replica count (no autoscaling). engine_kwargs are
        # forwarded to `python -m dynamo.vllm` as CLI flags. Needs the `dynamo`
        # package plus `etcd` + `nats-server` on PATH.
        model_cfg = DynamoVLLMModelConfig(
            model_identifier=args.model_id,
            model_name=args.inference_served_model_name,
            engine_kwargs=engine_kwargs,
            num_replicas=args.inference_max_replicas,
        )
        backend_cfg = DynamoServerConfig()

        inference_server = InferenceServer(
            models=[model_cfg],
            backend=backend_cfg,
            port=args.inference_port,
            health_check_timeout_s=args.inference_health_timeout,
        )
        inference_server.start()
        # endpoint/port are finalized during start() (Dynamo binds to the infra
        # node IP and may reallocate the port), so read them after.
        remote_base_url = inference_server.endpoint
        remote_model_name = model_cfg.resolved_model_name

    remote_kwargs: dict = {}
    if remote_base_url:
        remote_kwargs = {
            "inference_base_url": remote_base_url,
            "served_model_name": remote_model_name,
            "inference_api_key": args.inference_api_key,
            "max_concurrent_requests": args.inference_max_concurrent_requests,
            "request_timeout": args.inference_request_timeout,
        }
        logger.info(f"Routing LLM stages to remote inference server at {remote_base_url} (model={remote_model_name})")
    # Pick in-process vs remote stage classes. The Remote* classes subclass
    # the in-process ones, inherit (and ignore) the GPU engine kwargs, and
    # force CPU-only resources in their __post_init__.
    text_stage_cls = RemoteTextLLMStage if remote_base_url else TextLLMStage
    ctx_stage_cls = RemoteContextualASRExtractionStage if remote_base_url else ContextualASRExtractionStage

    itn_prompt = args.itn_prompt_file or str(_ITN_PROMPT)
    itn_no_disfl_prompt = args.itn_no_disfluencies_prompt_file or str(_CORRECTION_PROMPT)
    captioning_prompt = args.captioning_prompt_file or str(_CAPTIONING_PROMPT)
    pnc_prompt = args.pnc_prompt_file or str(_PNC_PROMPT)
    language_id_prompt = args.language_id_prompt_file or str(_LANGUAGE_ID_PROMPT)
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
        **remote_kwargs,
    }

    # When --fuse_stages is active, independent stages (LanguageID, ITN, Captioning,
    # CodeSwitching, SpeechQA) are collected here instead of appended directly.
    # They are later wrapped in one FusedRemoteTextLLMStage actor. Only has effect
    # with --use_inference_server; falls back to normal behaviour otherwise.
    use_fusing = bool(args.fuse_stages and remote_base_url)
    fuseable_sub_stages: list[RemoteTextLLMStage] = []

    stages = [
        ALMManifestReader(manifest_path=args.input_manifest, output_dir=args.output_dir, fanout=False),
    ]

    if args.enable_pnc:
        pnc_input_key = "abbreviated_text" if args.text_key == "pnc_text" else args.text_key
        stages.append(
            text_stage_cls(
                name="PnCRestoration",
                prompt_file=pnc_prompt,
                text_key=pnc_input_key,
                output_text_key=args.pnc_output_key,
                **shared_model_kwargs,
            )
        )
        logger.info(f"PnC stage enabled: {pnc_input_key} → {args.pnc_output_key}")

    # post_fused_stages: stages that depend on fused output (itn_text, llm_language_prediction)
    # and must be inserted after the fused stage when use_fusing is active.
    post_fused_stages: list = []

    if args.enable_language_id:
        _language_id_stage = text_stage_cls(
            name="LanguageID",
            prompt_file=language_id_prompt,
            text_key="pnc_text",
            output_text_key="llm_language_prediction",
            enable_validation=False,
            **shared_model_kwargs,
        )
        if use_fusing:
            fuseable_sub_stages.append(_language_id_stage)
            # LLMLanguageVerification reads llm_language_prediction written by the fused stage;
            # must come after it — collected here and inserted at assembly time below.
            post_fused_stages.append(LLMLanguageVerificationStage())
        else:
            stages.append(_language_id_stage)
            stages.append(LLMLanguageVerificationStage())
        logger.info(
            "LanguageID + LLMLanguageVerification stages enabled: pnc_text → llm_language_prediction "
            "→ keep code-switch w/ source_lang, else _skipme='Wrong language:LLMLanguageVerification'"
        )

    if args.enable_itn:
        _itn_stage = text_stage_cls(
            name="ITNRestoration",
            prompt_file=itn_prompt,
            text_key=args.text_key,
            output_text_key=args.itn_output_key,
            **shared_model_kwargs,
        )
        if use_fusing:
            fuseable_sub_stages.append(_itn_stage)
        else:
            stages.append(_itn_stage)
        logger.info(f"ITN stage enabled: {args.text_key} → {args.itn_output_key}")

    if args.enable_itn_no_disfluencies:
        if not args.enable_itn:
            logger.warning(
                "--enable_itn_no-disfluencies requires --enable_itn (needs itn_text as input). Enabling ITN automatically."
            )
            _itn_auto_stage = text_stage_cls(
                name="ITNRestoration",
                prompt_file=itn_prompt,
                text_key=args.text_key,
                output_text_key=args.itn_output_key,
                **shared_model_kwargs,
            )
            if use_fusing:
                fuseable_sub_stages.append(_itn_auto_stage)
            else:
                stages.append(_itn_auto_stage)

        _disfl_stage = text_stage_cls(
            name="DisfluencyRemoval",
            prompt_file=itn_no_disfl_prompt,
            text_key=args.itn_output_key,
            output_text_key=args.itn_no_disfluencies_output_key,
            max_deletion_ratio=0.5,
            **shared_model_kwargs,
        )
        # DisfluencyRemoval reads itn_text from the fused stage — must follow it.
        if use_fusing:
            post_fused_stages.append(_disfl_stage)
        else:
            stages.append(_disfl_stage)
        logger.info(f"DisfluencyRemoval stage enabled: {args.itn_output_key} → {args.itn_no_disfluencies_output_key}")

    if args.enable_captioning:
        _captioning_stage = text_stage_cls(
            name="Captioning",
            prompt_file=captioning_prompt,
            text_key=args.text_key,
            output_text_key=args.captioning_output_key,
            enable_validation=False,
            **shared_model_kwargs,
        )
        if use_fusing:
            fuseable_sub_stages.append(_captioning_stage)
        else:
            stages.append(_captioning_stage)
        logger.info(f"Captioning stage enabled: {args.text_key} → {args.captioning_output_key}")

    if args.enable_context_asr:
        context_asr_text_key = args.context_asr_text_key
        # This stage runs as its own Ray Data actor with its own vLLM engine, so it
        # uses its own engine kwargs — note context_asr_max_model_len (8192), larger
        # than the global --max_model_len (2048) used by the lightweight stages.
        stages.append(
            ctx_stage_cls(
                model_id=args.model_id,
                prompt_file=context_asr_prompt,
                text_key=context_asr_text_key,
                source_lang_key="source_lang",
                output_key=args.context_asr_output_key,
                tensor_parallel_size=args.tensor_parallel_size,
                max_output_tokens=args.context_asr_max_output_tokens,
                max_model_len=args.context_asr_max_model_len,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
                kv_cache_dtype=args.kv_cache_dtype,
                num_workers_override=args.context_asr_num_workers if args.context_asr_num_workers is not None else args.num_workers,
                batch_size=args.batch_size,
                **{
                    **remote_kwargs,
                    **({"max_concurrent_requests": args.context_asr_max_concurrent_requests}
                       if args.context_asr_max_concurrent_requests is not None and remote_kwargs else {}),
                },
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
        if args.context_asr_max_model_len < _CONTEXT_ASR_MIN_MAX_MODEL_LEN:
            logger.warning(
                f"--context_asr_max_model_len={args.context_asr_max_model_len} may be too small for context "
                f"ASR extraction. Consider 8192 or higher when --enable_context_asr is used."
            )
    if args.enable_code_switching:
        _code_switching_stage = text_stage_cls(
            name="CodeSwitching",
            prompt_file=code_switching_prompt,
            text_key=args.text_key,
            output_text_key=args.code_switching_output_key,
            enable_validation=False,
            **shared_model_kwargs,
        )
        if use_fusing:
            fuseable_sub_stages.append(_code_switching_stage)
        else:
            stages.append(_code_switching_stage)
        logger.info(f"CodeSwitching stage enabled: {args.text_key} → {args.code_switching_output_key}")

    if args.enable_speech_qa:
        _speech_qa_stage = text_stage_cls(
            name="SpeechQA",
            prompt_file=speech_qa_prompt,
            text_key=args.text_key,
            output_text_key=args.speech_qa_output_key,
            enable_validation=False,
            **shared_model_kwargs,
        )
        if use_fusing:
            fuseable_sub_stages.append(_speech_qa_stage)
        else:
            stages.append(_speech_qa_stage)
        logger.info(f"SpeechQA stage enabled: {args.text_key} → {args.speech_qa_output_key}")

    # Fused stage assembly — one actor fires all collected sub-stage prompts in parallel.
    if use_fusing and fuseable_sub_stages:
        stages.append(
            FusedRemoteTextLLMStage(
                sub_stages=fuseable_sub_stages,
                inference_base_url=remote_base_url,
                inference_api_key=args.inference_api_key,
                served_model_name=remote_model_name,
                max_concurrent_requests=args.inference_max_concurrent_requests,
                request_timeout=args.inference_request_timeout,
                batch_size=args.batch_size,
            )
        )
        logger.info(
            "FusedRemoteTextLLMStage assembled: %s sub-stages firing in parallel",
            [s.name for s in fuseable_sub_stages],
        )
        # Post-fused stages depend on outputs written by the fused actor
        # (llm_language_prediction → LLMLanguageVerification, itn_text → DisfluencyRemoval).
        stages.extend(post_fused_stages)

    if args.enable_instruction_packer:
        stages.append(
            InstructionPackerStage(
                output_key=args.instructions_output_key,
                pnc_key=args.pnc_output_key,
                itn_key=args.itn_output_key,
                itn_no_disfluencies_key=args.itn_no_disfluencies_output_key,
                captioning_key=args.captioning_output_key,
                code_switched_key=args.code_switching_output_key,
                speech_qa_key=args.speech_qa_output_key,
                transcription_target_key=args.pnc_output_key,
                seed=args.instruction_packer_seed,
            )
        )
        logger.info(f"InstructionPacker stage enabled → {args.instructions_output_key}")

    stages.append(ShardedManifestWriterStage(output_dir=args.output_dir))

    pipeline = Pipeline(name="text_post_processing", stages=stages)

    from nemo_curator.backends.ray_data import RayDataExecutor

    executor = RayDataExecutor(ignore_head_node=bool(remote_base_url))

    logger.info(f"Running text pipeline: {len(stages)} stages, mode={args.execution_mode}")
    try:
        pipeline.run(executor=executor)
    finally:
        if inference_server is not None:
            inference_server.stop()
        if ray_client is not None:
            ray_client.stop()
    logger.info("Text pipeline complete.")


if __name__ == "__main__":
    main()
