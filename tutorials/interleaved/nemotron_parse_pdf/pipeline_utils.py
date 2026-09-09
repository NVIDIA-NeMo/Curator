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

"""Shared argument and pipeline builders for the Nemotron-Parse PDF tutorial."""

from __future__ import annotations

import argparse

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse import NemotronParsePDFReader
from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import DEFAULT_MAX_TOKENS


def create_nemotron_parse_pdf_argparser() -> argparse.ArgumentParser:
    """Create the argument parser for the Nemotron-Parse PDF pipeline."""
    parser = argparse.ArgumentParser(description="Process PDFs through Nemotron-Parse into interleaved parquet")

    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest listing PDFs")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf-dir", help="Directory containing PDF files")
    source.add_argument("--zip-base-dir", help="Root of CC-MAIN zip archive hierarchy")
    source.add_argument("--jsonl-base-dir", help="Root of JSONL-based PDF dataset (e.g. GitHub PDFs)")

    parser.add_argument("--output-dir", required=True, help="Output directory for parquet files")
    parser.add_argument("--dataset-name", default="pdf_dataset", help="Dataset name for output tasks")

    parser.add_argument(
        "--model-path",
        default="nvidia/NVIDIA-Nemotron-Parse-v1.2",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--pdfs-per-task", type=int, default=10, help="PDFs per processing task")
    parser.add_argument("--max-pdfs", type=int, default=None, help="Limit total PDFs (for testing)")
    parser.add_argument("--dpi", type=int, default=300, help="PDF rendering resolution")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages per PDF")
    parser.add_argument("--min-crop-size", type=int, default=10, help="Min pixel dimension for image crops")
    parser.add_argument(
        "--text-in-pic",
        action="store_true",
        help="Predict text inside pictures (v1.2+ only). Default: no text in pictures.",
    )

    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=4,
        help="Pages per HF GPU pass or maximum concurrent inference-server page requests",
    )
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Max concurrent sequences (vLLM only)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum output tokens per page")
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable vLLM CUDA graph capture (enforce_eager=True). Eliminates ~35min compilation "
        "idle at startup; slight throughput reduction. Recommended on clusters with GPU "
        "utilization enforcement.",
    )

    parser.add_argument(
        "--execution-mode",
        default="streaming",
        choices=["streaming", "batch"],
        help="XennaExecutor execution mode",
    )

    parser.add_argument("--file-name-field", default="file_name", help="JSONL field for single PDF filename")
    parser.add_argument(
        "--file-names-field", default="cc_pdf_file_names", help="JSONL field for list of PDF filenames"
    )
    parser.add_argument("--url-field", default="url", help="JSONL field for source URL")

    return parser


def create_nemotron_parse_pdf_pipeline(
    args: argparse.Namespace,
    *,
    inprocess_backend: str = "vllm",
    inference_server_endpoint: str | None = None,
    inference_server_model_name: str | None = None,
    inference_server_client_num_workers: int = 4,
) -> Pipeline:
    """Build the PDF pipeline, optionally using an inference server.

    For an inference-server deployment, use four HTTP client workers per
    serving GPU and start with ``args.inference_batch_size=32``. Tune request
    concurrency on the target hardware and corpus.
    """
    pipeline = Pipeline(
        name="nemotron_parse_pdf",
        description="PDF -> Nemotron-Parse -> Interleaved Parquet",
    )
    pipeline.add_stage(
        NemotronParsePDFReader(
            manifest_path=args.manifest,
            zip_base_dir=args.zip_base_dir,
            pdf_dir=args.pdf_dir,
            jsonl_base_dir=args.jsonl_base_dir,
            model_path=args.model_path,
            backend=inprocess_backend,
            pdfs_per_task=args.pdfs_per_task,
            max_pdfs=args.max_pdfs,
            dpi=args.dpi,
            max_pages=args.max_pages,
            inference_batch_size=args.inference_batch_size,
            max_num_seqs=args.max_num_seqs,
            max_tokens=args.max_tokens,
            text_in_pic=args.text_in_pic,
            enforce_eager=args.enforce_eager,
            min_crop_px=args.min_crop_size,
            dataset_name=args.dataset_name,
            file_name_field=args.file_name_field,
            file_names_field=args.file_names_field,
            url_field=args.url_field,
            inference_server_endpoint=inference_server_endpoint,
            inference_server_model_name=inference_server_model_name,
            inference_server_client_num_workers=inference_server_client_num_workers,
            inference_server_request_timeout_s=getattr(args, "inference_server_request_timeout_s", 300.0),
            inference_server_max_retries=getattr(args, "inference_server_max_retries", 3),
        )
    )
    pipeline.add_stage(
        InterleavedParquetWriterStage(
            path=args.output_dir,
            materialize_on_write=False,
        )
    )
    return pipeline
