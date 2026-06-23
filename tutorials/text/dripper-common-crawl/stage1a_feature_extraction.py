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

"""Stage 1a: CPU-only DOM feature extraction via llm_web_kit get_feature()."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

OUTPUT_COLS = [
    "url",
    "url_host_name",
    "html",
    "dom_feature",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
]


class DOMFeatureExtractionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    name: str = "DOMFeatureExtractionStage"

    def __init__(self, cpus_per_actor: int = 4) -> None:
        super().__init__()
        self._resources = Resources(cpus=float(cpus_per_actor))
        self._web = None

    def setup(self, _worker_metadata: object = None) -> None:
        from nemo_curator.stages.text.experimental.dripper.stage import _load_llm_web_kit_bindings

        self._web = _load_llm_web_kit_bindings()

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()

        def _extract(html: object) -> str:
            if isinstance(html, bytes):
                html = html.decode("utf-8", errors="replace")
            if not isinstance(html, str) or not html.strip():
                return ""
            try:
                return json.dumps(self._web.get_feature(html))
            except Exception:
                return ""

        df["dom_feature"] = [_extract(h) for h in df["html"]]
        return DocumentBatch(dataset_name=batch.dataset_name, data=df)


def run(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    if inp.is_dir():
        exact = inp / f"shard_{args.shard_index:04d}.parquet"
        if exact.exists():
            inp = exact
        else:
            candidates = sorted(inp.glob("*.parquet"))
            if not candidates:
                raise FileNotFoundError(f"No parquet files in {args.input}")
            inp = candidates[0]

    pf = pq.ParquetFile(str(inp))
    total = pf.metadata.num_rows
    start = total * args.shard_index // args.num_shards
    end = total * (args.shard_index + 1) // args.num_shards
    need = ["url", "url_host_name", "html", "warc_filename", "warc_record_offset", "warc_record_length"]
    cols = [c for c in need if c in pf.schema_arrow.names]
    rows_seen, parts = 0, []
    for batch in pf.iter_batches(batch_size=65_536, columns=cols):
        df_b = batch.to_pandas()
        lo, hi = max(0, start - rows_seen), min(len(df_b), end - rows_seen)
        rows_seen += len(df_b)
        if lo < hi:
            parts.append(df_b.iloc[lo:hi])
        if rows_seen >= end:
            break
    shard_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    logger.info("shard {}/{}: {:,} pages", args.shard_index, args.num_shards, len(shard_df))
    if len(shard_df) == 0:
        return

    n_actors = max(1, (os.cpu_count() or 4) // max(1, args.cpus_per_actor))
    chunk = max(1, len(shard_df) // n_actors)
    tasks = [
        DocumentBatch(dataset_name="stage1a", data=shard_df.iloc[i : i + chunk].reset_index(drop=True))
        for i in range(0, len(shard_df), chunk)
    ]
    stage = DOMFeatureExtractionStage(cpus_per_actor=args.cpus_per_actor)
    pipeline = Pipeline(name="stage1a")
    pipeline.add_stage(stage)
    result_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=tasks) or []

    out_df = (
        pd.concat([t.to_pandas() for t in result_tasks if hasattr(t, "to_pandas")], ignore_index=True)
        if result_tasks
        else pd.DataFrame(columns=OUTPUT_COLS)
    )
    for col in OUTPUT_COLS:
        if col not in out_df.columns:
            out_df[col] = None

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / (f"shard_{args.shard_index:04d}.parquet" if args.num_shards > 1 else "shard_0000.parquet")
    tmp = out_path.with_suffix(".parquet.tmp")
    out_df.to_parquet(str(tmp), index=False, compression="snappy")
    tmp.rename(out_path)

    feat_ok = int((out_df["dom_feature"].astype(str) != "").sum())
    logger.info("feature_ok={}/{}  output -> {}", feat_ok, len(out_df), out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--cpus-per-actor", type=int, default=4)
    run(p.parse_args())


if __name__ == "__main__":
    main()
