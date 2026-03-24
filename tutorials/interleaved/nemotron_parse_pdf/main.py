# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tutorial: Process PDFs through Nemotron-Parse into interleaved parquet.

This pipeline reads PDFs (from a directory or CC-MAIN-style zip archives),
renders each page to an image, runs Nemotron-Parse for structured extraction
(text, tables, images), and writes interleaved parquet output.

Pipeline stages::

    1. PDFPartitioningStage           (_EmptyTask -> FileGroupTask)   [CPU]
       Reads a JSONL manifest and packs PDF entries into FileGroupTasks.

    2. PDFPreprocessStage             (FileGroupTask -> InterleavedBatch) [CPU]
       Extracts PDF bytes (from directory or zip), renders pages to images.

    3. NemotronParseInferenceStage    (InterleavedBatch -> InterleavedBatch) [GPU]
       Runs Nemotron-Parse model inference on page images.

    4. NemotronParsePostprocessStage  (InterleavedBatch -> InterleavedBatch) [CPU]
       Parses model output, aligns images/captions, crops, builds rows.

    5. InterleavedParquetWriterStage  (InterleavedBatch -> FileGroupTask)
       Writes final interleaved parquet output.

Supported data sources:

- **PDF directory**: Set ``--pdf-dir`` to a directory containing ``.pdf`` files.
  Create a simple manifest with::

      for f in /path/to/pdfs/*.pdf; do
          echo "{\"file_name\": \"$(basename $f)\"}" >> manifest.jsonl
      done

- **CC-MAIN zip archives**: Set ``--zip-base-dir`` to the root of the
  CC-MAIN-2021-31-PDF-UNTRUNCATED zip hierarchy. The manifest should use
  ``cc_pdf_file_names`` (list) or ``file_name`` fields.
  See: https://github.com/tballison/CC-MAIN-2021-31-PDF-UNTRUNCATED

Usage::

    # From a PDF directory (3 PDFs for testing)
    python main.py --pdf-dir /path/to/pdfs --manifest manifest.jsonl \\
        --output-dir ./output --max-pdfs 3

    # From CC-MAIN zip archives
    python main.py --zip-base-dir /path/to/zipfiles --manifest manifest.jsonl \\
        --output-dir ./output

    # With vLLM backend (recommended for throughput)
    python main.py --pdf-dir /path/to/pdfs --manifest manifest.jsonl \\
        --output-dir ./output --backend vllm

    # Full run with resume support
    python main.py --pdf-dir /path/to/pdfs --manifest manifest.jsonl \\
        --output-dir ./output --resume
"""

from __future__ import annotations

import argparse
import os
import time

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.nemotron_parse import NemotronParsePDFReader
from nemo_curator.stages.interleaved.nemotron_parse.utils import load_completed_sample_ids


def create_nemotron_parse_pdf_argparser() -> argparse.ArgumentParser:
    """Create the argument parser for the Nemotron-Parse PDF pipeline."""
    parser = argparse.ArgumentParser(description="Process PDFs through Nemotron-Parse into interleaved parquet")

    # Data source
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest listing PDFs")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf-dir", help="Directory containing PDF files")
    source.add_argument("--zip-base-dir", help="Root of CC-MAIN zip archive hierarchy")

    # Output
    parser.add_argument("--output-dir", required=True, help="Output directory for parquet files")
    parser.add_argument("--dataset-name", default="pdf_dataset", help="Dataset name for output tasks")

    # Model
    parser.add_argument(
        "--model-path",
        default="nvidia/NVIDIA-Nemotron-Parse-v1.2",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--backend", default="vllm", choices=["hf", "vllm"], help="Inference backend")

    # Processing
    parser.add_argument("--pdfs-per-task", type=int, default=10, help="PDFs per processing task")
    parser.add_argument("--max-pdfs", type=int, default=None, help="Limit total PDFs (for testing)")
    parser.add_argument("--dpi", type=int, default=300, help="PDF rendering resolution")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages per PDF")
    parser.add_argument("--min-crop-size", type=int, default=10, help="Min pixel dimension for image crops")

    # Inference
    parser.add_argument("--inference-batch-size", type=int, default=4, help="Pages per GPU pass (HF only)")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Max concurrent sequences (vLLM only)")

    # Executor
    parser.add_argument(
        "--execution-mode",
        default="streaming",
        choices=["streaming", "batch"],
        help="XennaExecutor execution mode",
    )

    # Resume
    parser.add_argument("--resume", action="store_true", help="Skip already-processed PDFs")

    return parser


def create_nemotron_parse_pdf_pipeline(args: argparse.Namespace) -> Pipeline:
    """Build the Nemotron-Parse PDF processing pipeline from parsed arguments."""
    completed_ids: set[str] = set()
    if args.resume and os.path.isdir(args.output_dir):
        completed_ids = load_completed_sample_ids(args.output_dir)
        if completed_ids:
            logger.info(f"Resume: {len(completed_ids)} PDFs already processed, will skip them")

    pipeline = Pipeline(
        name="nemotron_parse_pdf",
        description="PDF -> Nemotron-Parse -> Interleaved Parquet",
    )
    pipeline.add_stage(
        NemotronParsePDFReader(
            manifest_path=args.manifest,
            zip_base_dir=args.zip_base_dir,
            pdf_dir=args.pdf_dir,
            model_path=args.model_path,
            backend=args.backend,
            pdfs_per_task=args.pdfs_per_task,
            max_pdfs=args.max_pdfs,
            dpi=args.dpi,
            max_pages=args.max_pages,
            inference_batch_size=args.inference_batch_size,
            max_num_seqs=args.max_num_seqs,
            min_crop_px=args.min_crop_size,
            completed_ids=completed_ids,
            dataset_name=args.dataset_name,
        )
    )
    pipeline.add_stage(
        InterleavedParquetWriterStage(
            path=args.output_dir,
            materialize_on_write=False,
        )
    )
    return pipeline


def main() -> None:
    parser = create_nemotron_parse_pdf_argparser()
    args = parser.parse_args()

    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    pipeline = create_nemotron_parse_pdf_pipeline(args)
    logger.info(f"\n{pipeline.describe()}")

    executor = XennaExecutor(
        config={
            "execution_mode": args.execution_mode,
            "ignore_failures": True,
        }
    )

    t0 = time.perf_counter()
    results = pipeline.run(executor=executor)
    wall_time = time.perf_counter() - t0

    logger.info(f"Pipeline finished in {wall_time:.1f}s, {len(results)} output tasks")
    for task in results:
        for perf in task._stage_perf:
            logger.info(f"  Stage '{perf.stage_name}': process={perf.process_time:.2f}s")


if __name__ == "__main__":
    main()
