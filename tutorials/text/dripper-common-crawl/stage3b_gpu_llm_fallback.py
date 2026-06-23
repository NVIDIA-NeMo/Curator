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

"""Stage 3b: GPU LLM fallback for siblings where Stage 3 propagation failed.

Without this stage, F1 is ~0.84. With it, F1 reaches ~0.92 (above the 0.90 target).

Siblings where DripperHTMLLayoutPropagationStage returned propagation_success=False
(content ratio too high/low, no template, etc.) are re-run through the full LLM
extraction pipeline (DripperHTMLPreprocessStage -> GPU inference -> PostprocessStage).

INPUT:  Stage 3 propagation results (shard_*.parquet)
        Stage 1b cluster manifest (for html column)
OUTPUT: Updated shard with failed siblings replaced by LLM extraction results
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

_DEFAULT_SHARD_INDEX = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
_DEFAULT_NUM_SHARDS = 80


def _load_failed_siblings(
    propagation_dir: Path,
    manifest_dir: Path,
    shard_index: int,
    num_shards: int,
) -> pd.DataFrame:
    """Load siblings where propagation failed and attach their html for LLM re-inference."""
    prop_files = sorted(propagation_dir.glob("shard_*.parquet")) or sorted(propagation_dir.glob("*.parquet"))
    if not prop_files:
        raise FileNotFoundError(f"No propagation result files in {propagation_dir}")

    n = len(prop_files)
    my_files = prop_files[n * shard_index // num_shards : n * (shard_index + 1) // num_shards]
    if not my_files:
        logger.info("shard {}: no propagation files — nothing to do", shard_index)
        return pd.DataFrame()

    prop_df = pd.concat([pq.read_table(f).to_pandas() for f in my_files], ignore_index=True)

    # Select only siblings where propagation failed
    failed_mask = ~prop_df.get("propagation_success", pd.Series(True, index=prop_df.index)).fillna(True).astype(
        bool
    ) & (prop_df.get("cluster_role", pd.Series("singleton", index=prop_df.index)) == "sibling")
    failed_df = prop_df[failed_mask].copy()
    if failed_df.empty:
        logger.info("shard {}: no failed siblings — all propagation succeeded", shard_index)
        return pd.DataFrame()

    logger.info("shard {}: {:,} / {:,} siblings need LLM fallback", shard_index, len(failed_df), len(prop_df))

    # Load html from manifest for the failed siblings
    manifest_files = sorted(manifest_dir.glob("shard_*.parquet")) or sorted(manifest_dir.glob("*.parquet"))
    if not manifest_files:
        raise FileNotFoundError(f"No manifest files in {manifest_dir}")

    failed_urls = set(failed_df["url"].astype(str))
    html_parts = []
    for mf in manifest_files:
        schema = pq.read_schema(str(mf)).names
        if "html" not in schema:
            continue
        cols = [c for c in ["url", "html"] if c in schema]
        mdf = pq.read_table(str(mf), columns=cols).to_pandas()
        matched = mdf[mdf["url"].astype(str).isin(failed_urls)]
        if not matched.empty:
            html_parts.append(matched)

    if not html_parts:
        logger.warning("No html found for failed siblings — cannot run LLM fallback")
        return pd.DataFrame()

    html_df = pd.concat(html_parts, ignore_index=True).drop_duplicates("url", keep="first")
    failed_df = failed_df.merge(html_df[["url", "html"]], on="url", how="inner")
    logger.info("shard {}: {:,} siblings with html for LLM fallback", shard_index, len(failed_df))
    return failed_df


def run_llm_fallback(
    failed_df: pd.DataFrame,
    model_name: str,
    server_url: str,
    max_concurrent_requests: int,
    num_workers: int,
) -> pd.DataFrame:
    """Run LLM extraction on failed siblings using library stages."""
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
    from nemo_curator.models.client.openai_client import OpenAIClient
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.stages.text.experimental.dripper import (
        DripperHTMLPostprocessStage,
        DripperHTMLPreprocessStage,
    )
    from nemo_curator.stages.text.experimental.dripper._base_stages import DripperHTMLInferenceStage
    from nemo_curator.tasks import DocumentBatch

    client = OpenAIClient(model=model_name, base_url=server_url, api_key="EMPTY")

    preprocess = DripperHTMLPreprocessStage(html_col="html", url_col="url", worker_count=num_workers)
    inference = DripperHTMLInferenceStage(
        client=client,
        model_name=model_name,
        max_concurrent_requests=max_concurrent_requests,
        health_check=False,
    )
    postprocess = DripperHTMLPostprocessStage(
        html_col="html",
        url_col="url",
        fallback="trafilatura",
        output_format="mm_md",
        worker_count=num_workers,
    )

    pipeline = Pipeline(name="stage3b_llm_fallback")
    pipeline.add_stage(preprocess)
    pipeline.add_stage(inference)
    pipeline.add_stage(postprocess)

    chunk = max(1, len(failed_df) // max(1, num_workers))
    tasks = [
        DocumentBatch(dataset_name="stage3b", data=failed_df.iloc[i : i + chunk].reset_index(drop=True))
        for i in range(0, len(failed_df), chunk)
    ]
    result_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=tasks) or []

    frames = [t.to_pandas() for t in result_tasks]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def process_shard(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{args.shard_index:04d}.parquet"

    if out_path.exists():
        meta = pq.read_metadata(str(out_path))
        if meta.num_rows > 0:
            logger.info("SKIP shard {} — already done ({:,} rows)", args.shard_index, meta.num_rows)
            return {"status": "skipped", "shard": args.shard_index}

    failed_df = _load_failed_siblings(
        Path(args.propagation_results),
        Path(args.cluster_manifest),
        args.shard_index,
        args.num_shards,
    )
    if failed_df.empty:
        pq.write_table(
            pq.read_schema(str(next(Path(args.propagation_results).glob("*.parquet")))).empty_table(), str(out_path)
        )
        return {"status": "empty", "shard": args.shard_index, "fallback_rows": 0}

    result_df = run_llm_fallback(
        failed_df, args.model_name, args.server_url, args.max_concurrent_requests, args.workers
    )

    tmp = out_path.with_suffix(f".tmp_{os.getpid()}.parquet")
    result_df.to_parquet(str(tmp), index=False, compression="snappy")
    tmp.rename(out_path)

    elapsed = time.perf_counter() - t0
    ok = (
        int(result_df["dripper_content"].astype(str).str.len().gt(5).sum())
        if "dripper_content" in result_df.columns
        else 0
    )
    logger.info(
        "shard {} done  fallback_rows={:,} ok={} elapsed={:.1f}s output={}",
        args.shard_index,
        len(result_df),
        ok,
        elapsed,
        out_path,
    )
    return {"status": "done", "shard": args.shard_index, "fallback_rows": len(result_df), "ok": ok}


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 3b: GPU LLM fallback for failed propagation siblings")
    p.add_argument("--propagation-results", required=True, help="Stage 3 output dir")
    p.add_argument("--cluster-manifest", required=True, help="Stage 1b cluster assignment dir (needs html column)")
    p.add_argument("--output-dir", required=True, help="Output dir for stage3b results")
    p.add_argument("--model-name", default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact")
    p.add_argument("--server-url", default="http://localhost:8000/v1")
    p.add_argument("--shard-index", type=int, default=_DEFAULT_SHARD_INDEX)
    p.add_argument("--num-shards", type=int, default=_DEFAULT_NUM_SHARDS)
    p.add_argument("--max-concurrent-requests", type=int, default=64)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    import sys

    from loguru import logger as _log

    _log.remove()
    _log.add(sys.stdout, level=args.log_level.upper())

    process_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
