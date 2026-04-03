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
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse import NemotronParsePDFReader
from nemo_curator.tasks import FileGroupTask


@dataclass
class PerfLoggingStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Append per-task stage perf stats to a JSONL file as each task completes.

    Placed after the writer stage so perf data is flushed to disk
    incrementally — survives job kills from Slurm time limits.
    """

    output_dir: str
    name: str = "perf_logging"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: FileGroupTask) -> FileGroupTask:
        perf_path = os.path.join(self.output_dir, f"_perf_stats_{os.getpid()}.jsonl")
        record = {
            "task_id": task.task_id,
            "stages": [
                {
                    "stage_name": p.stage_name,
                    "process_time_s": p.process_time,
                    "actor_idle_time_s": p.actor_idle_time,
                    "num_items_processed": p.num_items_processed,
                    **{f"custom_{k}": v for k, v in p.custom_metrics.items()},
                }
                for p in task._stage_perf
            ],
        }
        with open(perf_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return task


def create_nemotron_parse_pdf_argparser() -> argparse.ArgumentParser:
    """Create the argument parser for the Nemotron-Parse PDF pipeline."""
    parser = argparse.ArgumentParser(description="Process PDFs through Nemotron-Parse into interleaved parquet")

    # Data source
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest listing PDFs")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf-dir", help="Directory containing PDF files")
    source.add_argument("--zip-base-dir", help="Root of CC-MAIN zip archive hierarchy")
    source.add_argument("--jsonl-base-dir", help="Root of JSONL-based PDF dataset (e.g. GitHub PDFs)")

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
    parser.add_argument(
        "--text-in-pic",
        action="store_true",
        help="Predict text inside pictures (v1.2+ only). Default: no text in pictures.",
    )

    # Inference
    parser.add_argument("--inference-batch-size", type=int, default=4, help="Pages per GPU pass (HF only)")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Max concurrent sequences (vLLM only)")
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable vLLM CUDA graph capture (enforce_eager=True). Eliminates ~35min compilation "
             "idle at startup; slight throughput reduction. Recommended on clusters with GPU "
             "utilization enforcement.",
    )

    # Executor
    parser.add_argument(
        "--execution-mode",
        default="streaming",
        choices=["streaming", "batch"],
        help="XennaExecutor execution mode",
    )

    # Manifest field names
    parser.add_argument("--file-name-field", default="file_name", help="JSONL field for single PDF filename")
    parser.add_argument("--file-names-field", default="cc_pdf_file_names", help="JSONL field for list of PDF filenames")
    parser.add_argument("--url-field", default="url", help="JSONL field for source URL")

    return parser


def create_nemotron_parse_pdf_pipeline(args: argparse.Namespace) -> Pipeline:
    """Build the Nemotron-Parse PDF processing pipeline from parsed arguments."""
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
            backend=args.backend,
            pdfs_per_task=args.pdfs_per_task,
            max_pdfs=args.max_pdfs,
            dpi=args.dpi,
            max_pages=args.max_pages,
            inference_batch_size=args.inference_batch_size,
            max_num_seqs=args.max_num_seqs,
            text_in_pic=args.text_in_pic,
            enforce_eager=args.enforce_eager,
            min_crop_px=args.min_crop_size,
            dataset_name=args.dataset_name,
            file_name_field=args.file_name_field,
            file_names_field=args.file_names_field,
            url_field=args.url_field,
        )
    )
    pipeline.add_stage(
        InterleavedParquetWriterStage(
            path=args.output_dir,
            materialize_on_write=False,
        )
    )
    return pipeline


def _write_perf_summary(results: list, output_dir: str, wall_time: float) -> None:
    """Write per-task stage timings to a parquet file and log aggregate stats."""
    import pandas as pd

    valid_results = [r for r in results if r is not None] if results else []
    if not valid_results:
        logger.warning("No results to write perf summary for")
        return

    if len(valid_results) < len(results):
        logger.warning(f"{len(results) - len(valid_results)} tasks returned None (failed)")

    rows = []
    for task in valid_results:
        for perf in task._stage_perf:
            row = {
                "task_id": task.task_id,
                "stage_name": perf.stage_name,
                "process_time_s": perf.process_time,
                "actor_idle_time_s": perf.actor_idle_time,
                "num_items_processed": perf.num_items_processed,
            }
            for k, v in perf.custom_metrics.items():
                row[f"custom_{k}"] = v
            rows.append(row)

    df = pd.DataFrame(rows)
    job_id = os.environ.get("SLURM_JOB_ID", f"local_{int(time.time())}")
    perf_path = os.path.join(output_dir, f"_perf_stats_{job_id}.parquet")
    df.to_parquet(perf_path, index=False)
    logger.info(f"Wrote {len(df)} perf records ({len(valid_results)} tasks) to {perf_path}")

    n_tasks = len(valid_results)
    logger.info(f"\n{'='*70}\n  PERFORMANCE SUMMARY  (wall_time={wall_time:.1f}s, tasks={n_tasks})\n{'='*70}")
    for stage_name, group in df.groupby("stage_name", sort=False):
        avg_t = group["process_time_s"].mean()
        sum_t = group["process_time_s"].sum()
        p50 = group["process_time_s"].median()
        p95 = group["process_time_s"].quantile(0.95)
        total_items = group["num_items_processed"].sum()
        logger.info(
            f"  {stage_name:40s}  avg={avg_t:8.2f}s  p50={p50:8.2f}s  p95={p95:8.2f}s  "
            f"sum={sum_t:10.1f}s  items={total_items}"
        )
    logger.info(f"{'='*70}\n")


def main() -> None:
    parser = create_nemotron_parse_pdf_argparser()
    args = parser.parse_args()

    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if os.environ.get("SLURM_JOB_ID"):
        from nemo_curator.core.client import SlurmRayClient
        ray_client = SlurmRayClient()
    else:
        ray_client = RayClient()
    ray_client.start()

    try:
        pipeline = create_nemotron_parse_pdf_pipeline(args)
        logger.info(f"\n{pipeline.describe()}")

        executor = XennaExecutor(
            config={
                "execution_mode": args.execution_mode,
                "ignore_failures": True,
                "failures_return_nones": True,
                "reset_workers_on_failure": True,
            }
        )

        t0 = time.perf_counter()
        results = pipeline.run(executor=executor)
        wall_time = time.perf_counter() - t0

        n_valid = sum(1 for r in results if r is not None)
        n_failed = len(results) - n_valid
        logger.info(f"Pipeline finished in {wall_time:.1f}s, {n_valid} output tasks ({n_failed} failed)")
        _write_perf_summary(results, args.output_dir, wall_time)
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
