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

"""Stage 2b: CPU postprocessing from LLM responses (thin wrapper around DripperHTMLPostprocessStage)."""

import argparse
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.experimental.dripper import DripperHTMLPostprocessStage
from nemo_curator.tasks import DocumentBatch

_MIN_NONEMPTY_LEN: int = 5
_MIN_ERROR_LEN: int = 2


def run(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    if inp.is_dir():
        files = sorted(inp.glob(f"shard_{args.shard_index:04d}.parquet")) or sorted(inp.glob("*.parquet"))
        inp = files[0] if files else inp

    df = pq.ParquetFile(str(inp)).read().to_pandas()
    logger.info("{:,} pages to postprocess ({} workers)", len(df), args.workers)

    n_workers = args.workers
    chunk = max(1, len(df) // n_workers)
    tasks = [
        DocumentBatch(dataset_name="stage2b", data=df.iloc[i : i + chunk].reset_index(drop=True))
        for i in range(0, len(df), chunk)
    ]

    # Simple Curator pattern: construct library stage, build pipeline, call run()
    stage = DripperHTMLPostprocessStage(
        html_col="html",
        url_col="url",
        fallback="trafilatura",
        output_format="mm_md",
        worker_count=n_workers,
    )
    pipeline = Pipeline(name="stage2b")
    pipeline.add_stage(stage)
    result_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=tasks) or []

    result_df = pd.concat([t.to_pandas() for t in result_tasks], ignore_index=True) if result_tasks else df

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / (
        f"shard_{args.shard_index:04d}.parquet" if args.num_shards > 1 else "postprocess_results.parquet"
    )
    tmp = out_path.with_suffix(".parquet.tmp")
    result_df.to_parquet(str(tmp), index=False, compression="snappy")
    tmp.rename(out_path)

    content_ok = int(
        (result_df["dripper_content"].astype(str).str.len() > _MIN_NONEMPTY_LEN).sum()
        if "dripper_content" in result_df.columns
        else 0
    )
    errors = int(
        (result_df["dripper_error"].astype(str).str.len() > _MIN_ERROR_LEN).sum()
        if "dripper_error" in result_df.columns
        else 0
    )
    logger.info(
        "content_ok={}/{}  errors={}  output -> {}",
        content_ok,
        len(result_df),
        errors,
        out_path,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Stage 2 output dir")
    p.add_argument("--output", required=True, help="Output dir")
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    run(p.parse_args())


if __name__ == "__main__":
    main()
