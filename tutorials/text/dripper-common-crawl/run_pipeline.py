#!/usr/bin/env python3
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

"""Single-command Dripper pipeline: input parquet(s) → output parquet with extracted content.

Usage (recommended — layout clustering for host-chunked input):

    python run_pipeline.py \\
        --input  /data/host_pages.parquet \\
        --output /data/output/ \\
        --server-url http://localhost:8000/v1

Usage (standalone — no clustering, every page gets its own LLM call):

    python run_pipeline.py --input /data/pages.parquet --output /data/output/ \\
        --server-url http://localhost:8000/v1 --no-clustering

Input parquet must have: url, html  (url_host_name recommended for clustering)
Output adds:             dripper_content, dripper_html, dripper_error

Pipeline stages:
  With clustering (default): Preprocess → LayoutTemplate (cluster + LLM reps + propagate siblings)
  Without clustering:         Preprocess → Inference → Postprocess
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger


def _load_input(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {path}")
        return pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    return pq.read_table(str(p)).to_pandas()


def run(args: argparse.Namespace) -> int:
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
    from nemo_curator.models.client.openai_client import OpenAIClient
    from nemo_curator.stages.text.experimental.dripper import DripperHTMLWorkflow
    from nemo_curator.tasks import DocumentBatch

    t0 = time.perf_counter()
    df = _load_input(args.input)
    logger.info("Loaded {:,} pages from {}", len(df), args.input)

    missing = {"url", "html"} - set(df.columns)
    if missing:
        logger.error("Input missing required columns: {}", sorted(missing))
        return 1

    client = OpenAIClient(model=args.model_name, base_url=args.server_url, api_key="EMPTY")
    workflow = DripperHTMLWorkflow(
        client=client,
        model_name=args.model_name,
        html_col=args.html_col,
        url_col=args.url_col,
        output_col=args.output_col,
        perform_layout_clustering=not args.no_clustering,
        layout_cluster_threshold=args.cluster_threshold,
        fallback=args.fallback,
        output_format=args.output_format,
        max_concurrent_requests=args.max_concurrent_requests,
        health_check=not args.no_health_check,
    )

    chunk = max(1, len(df) // max(1, args.workers))
    tasks = [
        DocumentBatch(dataset_name="dripper", data=df.iloc[i : i + chunk].reset_index(drop=True))
        for i in range(0, len(df), chunk)
    ]
    result = workflow.run(executor=RayActorPoolExecutor(), initial_tasks=tasks)
    output_tasks = result.pipeline_tasks.get("dripper_html_extraction", [])
    if not output_tasks:
        logger.error("Pipeline returned no output — check server and logs")
        return 1

    out_df = pd.concat([t.to_pandas() for t in output_tasks], ignore_index=True)

    # Summary
    n = len(out_df)
    ok = int(out_df.get(args.output_col, pd.Series()).astype(str).str.len().gt(10).sum())
    elapsed = time.perf_counter() - t0
    logger.info(
        "Done — pages={:,} content_ok={} ({:.0f}%) elapsed={:.1f}s ({:.0f} p/s)",
        n,
        ok,
        100 * ok / max(1, n),
        elapsed,
        n / max(elapsed, 0.001),
    )

    # Write output
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem if not Path(args.input).is_dir() else "output"
    out_path = out_dir / f"{stem}.parquet"
    tmp = out_path.with_suffix(f".tmp_{os.getpid()}.parquet")
    out_df.to_parquet(str(tmp), index=False, compression="snappy")
    tmp.rename(out_path)
    logger.info("Output → {}", out_path)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dripper HTML extraction: input parquet → output parquet with extracted content",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Input parquet file or directory (url, html required)")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--server-url", default="http://localhost:8000/v1", help="OpenAI-compatible server URL")
    p.add_argument("--model-name", default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact")
    p.add_argument("--no-clustering", action="store_true", help="Standalone extraction (no layout clustering)")
    p.add_argument("--cluster-threshold", type=float, default=0.95, help="DOM similarity threshold")
    p.add_argument("--fallback", default="trafilatura", choices=["trafilatura", "bypass", "empty"])
    p.add_argument("--output-format", default="mm_md")
    p.add_argument("--output-col", default="dripper_content", help="Name of output content column")
    p.add_argument("--html-col", default="html")
    p.add_argument("--url-col", default="url")
    p.add_argument("--max-concurrent-requests", type=int, default=64)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--no-health-check", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
