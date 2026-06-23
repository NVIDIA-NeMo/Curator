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

"""Stage 1c: CPU preprocessing for Stage 2 GPU inference (thin wrapper around DripperHTMLPreprocessStage)."""

import argparse
import glob as _g
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.experimental.dripper import DripperHTMLPreprocessStage
from nemo_curator.tasks import DocumentBatch

OUTPUT_COLS = [
    "url",
    "url_host_name",
    "cluster_id",
    "cluster_role",
    "dripper_simplified_html",
    "dripper_mapped_html",
    "_dripper_prompt",
    "_dripper_needs_llm",
    "dripper_item_count",
    "html",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
]


def run(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    if inp.is_dir():
        files = sorted(_g.glob(str(inp / f"shard_{args.shard_index:04d}.parquet")))
        if not files:
            files = sorted(_g.glob(str(inp / "shard_*.parquet")))
        inp = Path(files[0]) if files else inp

    df = pq.ParquetFile(str(inp)).read().to_pandas()

    # Filter to representatives and singletons only
    if "cluster_role" in df.columns:
        mask = df["cluster_role"].isin(["representative", "singleton"])
    elif "is_representative" in df.columns:
        mask = df["is_representative"].astype(bool)
    else:
        mask = pd.Series(True, index=df.index)
    df = df[mask].reset_index(drop=True)

    logger.info("{:,} representative/singleton pages to preprocess", len(df))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / (f"shard_{args.shard_index:04d}.parquet" if args.num_shards > 1 else "shard_0000.parquet")

    if len(df) == 0:
        pd.DataFrame(columns=OUTPUT_COLS).to_parquet(str(out_path), index=False)
        return

    n_workers = args.workers
    chunk = max(1, len(df) // n_workers)
    tasks = [
        DocumentBatch(dataset_name="stage1c", data=df.iloc[i : i + chunk].reset_index(drop=True))
        for i in range(0, len(df), chunk)
    ]

    # Simple Curator pattern: construct library stage, build pipeline, call run()
    stage = DripperHTMLPreprocessStage(
        html_col="html",
        url_col="url",
        worker_count=n_workers,
    )
    pipeline = Pipeline(name="stage1c")
    pipeline.add_stage(stage)
    result_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=tasks) or []

    result_df = pd.concat([t.to_pandas() for t in result_tasks], ignore_index=True) if result_tasks else df

    tmp = out_path.with_suffix(".parquet.tmp")
    result_df.to_parquet(str(tmp), index=False, compression="snappy")
    tmp.rename(out_path)

    # Count prompts successfully built (non-empty _dripper_prompt for rows that need LLM)
    if "_dripper_prompt" in result_df.columns:
        ok = int((result_df["_dripper_prompt"].astype(str).str.len() > 10).sum())
    else:
        ok = 0
    logger.info("prompts_ok={}/{}  output -> {}", ok, len(result_df), out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Stage 1b output dir or parquet")
    p.add_argument("--output", required=True, help="Output dir")
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    run(p.parse_args())


if __name__ == "__main__":
    main()
