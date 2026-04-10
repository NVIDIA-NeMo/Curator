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

"""Qwen3-Omni audio transcription pipeline using in-process vLLM.

Runs Qwen3-Omni-30B-A3B-Instruct directly inside the Curator pipeline
(no external HTTP server). Each GPU worker loads its own vLLM engine
for maximum throughput with zero network overhead.

Audio is read from NeMo-tarred datasets via lhotse/soundfile and
decoded entirely in memory — no temporary files are written to disk.

Architecture:
    NemoTarredAudioReader (CPU, parallel)
        → streams NeMo-tarred shards from S3/local via lhotse
        → decodes audio in memory, emits AudioTask with waveform arrays
    InferenceQwenOmniStage (GPU, TP=2 → 4 workers on 8 GPUs)
        → resamples to 16 kHz, batched vLLM inference
    ALMManifestWriterStage (CPU, single)
        → writes JSONL output

Usage:
    python run_pipeline.py \\
        --data_config /path/to/granary_config.yaml \\
        --corpus yodas \\
        --output /path/to/output.jsonl

    # With custom prompt
    python run_pipeline.py \\
        --data_config /path/to/config.yaml \\
        --corpus yodas \\
        --prompt_file /path/to/prompt.md \\
        --output results.jsonl
"""

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

import argparse
import time

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm.alm_manifest_writer import ALMManifestWriterStage
from nemo_curator.stages.audio.inference.qwen_omni import InferenceQwenOmniStage
from nemo_curator.stages.audio.io.nemo_tarred_reader import NemoTarredAudioReader
from nemo_curator.stages.resources import Resources


def main():
    ap = argparse.ArgumentParser(description="QwenOmni in-process vLLM pipeline")
    ap.add_argument("--data_config", type=str, required=True, help="Granary YAML data config.")
    ap.add_argument("--corpus", type=str, nargs="*", default=None, help="Process only these corpora.")
    ap.add_argument("--output", type=str, required=True, help="Output JSONL path.")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--prompt", type=str, default="Transcribe the audio.")
    ap.add_argument("--prompt_file", type=str, default=None, help="Read prompt from file.")
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
    args = ap.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read().strip()

    system_prompt = None
    if args.system_prompt:
        if os.path.isfile(args.system_prompt):
            with open(args.system_prompt, encoding="utf-8") as f:
                system_prompt = f.read().strip()
        else:
            system_prompt = args.system_prompt

    pipeline = Pipeline(
        name="qwen_omni_inference",
        stages=[
            NemoTarredAudioReader(
                yaml_path=args.data_config,
                corpus_filter=args.corpus,
                s3_endpoint_url=args.s3_endpoint_url,
            ).with_({"nemo_tar_shard_reader": {"resources": Resources(cpus=4.0)}}),
            InferenceQwenOmniStage(
                model_id=args.model_id,
                prompt_text=prompt,
                system_prompt=system_prompt,
                tensor_parallel_size=args.tensor_parallel_size,
                batch_size=args.batch_size,
                max_output_tokens=args.max_output_tokens,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_memory_utilization,
                prep_workers=args.prep_workers,
            ),
            ALMManifestWriterStage(
                output_path=args.output,
            ),
        ],
    )

    logger.info("Pipeline: %s", pipeline.describe())

    executor = XennaExecutor(config={
        "execution_mode": args.execution_mode,
    })

    t0 = time.time()
    pipeline.run(executor=executor)
    elapsed = time.time() - t0
    logger.info("Pipeline finished in %.1f min. Output: %s", elapsed / 60, args.output)


if __name__ == "__main__":
    main()
