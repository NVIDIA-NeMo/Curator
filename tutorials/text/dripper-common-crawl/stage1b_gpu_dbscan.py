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

"""Stage 1b: GPU DBSCAN clustering of DOM layout features → cluster assignments."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
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
    "cluster_id",
    "cluster_role",
    "layout_cluster_id",
    "is_representative",
    "cluster_size",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
]


def _singleton_row(url: str, host: str, html: object, src: dict, include_html: bool = True) -> dict:
    row: dict[str, Any] = {
        "url": url,
        "url_host_name": host,
        "cluster_id": "",
        "cluster_role": "singleton",
        "layout_cluster_id": "",
        "is_representative": False,
        "cluster_size": 1,
        "warc_filename": src.get("warc_filename"),
        "warc_record_offset": src.get("warc_record_offset"),
        "warc_record_length": src.get("warc_record_length"),
    }
    if include_html:
        row["html"] = html
    return row


@dataclass(kw_only=True)
class HostDBSCANStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """GPU DBSCAN clustering — one DocumentBatch per host, one GPU per Ray actor."""

    name: str = "host_dbscan"
    resources: Resources = field(default_factory=lambda: Resources(cpus=4.0, gpus=1.0))
    threshold: float = 0.95
    min_cluster_size: int = 2
    gpu_min_size: int = 5
    max_host_size: int = 3000
    _cluster_gpu: Any = field(init=False, repr=False, default=None)
    _has_gpu: bool = field(init=False, repr=False, default=False)
    _web: Any = field(init=False, repr=False, default=None)

    def setup(self, _worker_metadata: object = None) -> None:
        from nemo_curator.stages.text.experimental.dripper.gpu_layout_clustering import (
            _gpu_available,
            cluster_html_struct_gpu,
        )
        from nemo_curator.stages.text.experimental.dripper.stage import _load_llm_web_kit_bindings

        self._cluster_gpu = cluster_html_struct_gpu
        self._has_gpu = _gpu_available()
        self._web = _load_llm_web_kit_bindings()
        logger.info(
            "actor setup: has_gpu={} CUDA_VISIBLE_DEVICES={}",
            self._has_gpu,
            os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        )

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        samples = batch.to_pandas().to_dict("records")
        return DocumentBatch(
            dataset_name=batch.dataset_name, data=pd.DataFrame(self._cluster_host(batch.dataset_name, samples))
        )

    def _run_clustering(self, chunk: list[dict], chunk_idx: int | None = None) -> list[dict]:
        try:
            if self._cluster_gpu and self._has_gpu and len(chunk) >= self.gpu_min_size:
                cc, _ = self._cluster_gpu(chunk, threshold=self.threshold, gpu_min_size=self.gpu_min_size)
            elif self._web:
                cc, _ = self._web.cluster_html_struct(chunk, threshold=self.threshold)
            else:
                cc = chunk
                for i, s in enumerate(cc):
                    s["layout_id"] = 0 if i == 0 else -1
            if chunk_idx is not None:
                for s in cc:
                    lid = s.get("layout_id", -1)
                    if lid >= 0:
                        s["layout_id"] = chunk_idx * 100_000 + lid
        except Exception as exc:
            logger.warning("{} failed: {}", f"chunk {chunk_idx}" if chunk_idx is not None else "DBSCAN", exc)
            cc = chunk
        return cc

    def _cluster_host(self, host: str, samples: list[dict]) -> list[dict]:
        if len(samples) > self.max_host_size:
            clustered: list[dict] = []
            for ci, s in enumerate(range(0, len(samples), self.max_host_size)):
                clustered.extend(self._run_clustering(samples[s : s + self.max_host_size], chunk_idx=ci))
        else:
            clustered = self._run_clustering(samples)
        by_lid: dict[int, list] = defaultdict(list)
        for s in clustered:
            by_lid[int(s.get("layout_id", -1))].append(s)
        rows = []
        for lid, members in by_lid.items():
            if lid < 0 or len(members) < self.min_cluster_size:
                for m in members:
                    rows.append(_singleton_row(m["url"], host, None, m, include_html=False))
                continue
            cid = f"{host}:cluster_{lid}"
            try:
                rep_url = (
                    self._web.select_representative_html(
                        [{"track_id": m["url"], "html": m.get("html", "")} for m in members]
                    )["track_id"]
                    if self._web
                    else members[0]["url"]
                )
            except Exception:
                rep_url = members[0]["url"]
            for m in members:
                is_rep = m["url"] == rep_url
                rows.append(
                    {
                        "url": m["url"],
                        "url_host_name": host,
                        "cluster_id": cid,
                        "cluster_role": "representative" if is_rep else "sibling",
                        "layout_cluster_id": cid,
                        "is_representative": is_rep,
                        "cluster_size": len(members),
                        "warc_filename": m.get("warc_filename"),
                        "warc_record_offset": m.get("warc_record_offset"),
                        "warc_record_length": m.get("warc_record_length"),
                    }
                )
        return rows


def run(args: argparse.Namespace) -> None:
    inp = Path(args.input)
    if inp.is_dir():
        exact = inp / f"shard_{args.shard_index:04d}.parquet"
        inp = exact if exact.exists() else sorted(inp.glob("shard_*.parquet"))[0]
    pf = pq.ParquetFile(str(inp))
    total = pf.metadata.num_rows
    start = total * args.shard_index // args.num_shards
    end = total * (args.shard_index + 1) // args.num_shards
    need = ["url", "url_host_name", "dom_feature", "html", "warc_filename", "warc_record_offset", "warc_record_length"]
    cols = [c for c in need if c in pf.schema_arrow.names]
    rows_seen, parts = 0, []
    for batch in pf.iter_batches(batch_size=65_536, columns=cols):
        df = batch.to_pandas()
        lo, hi = max(0, start - rows_seen), min(len(df), end - rows_seen)
        rows_seen += len(df)
        if lo < hi:
            parts.append(df.iloc[lo:hi])
        if rows_seen >= end:
            break
    shard_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    logger.info("shard {}/{}: {:,} pages", args.shard_index, args.num_shards, len(shard_df))
    if len(shard_df) == 0:
        return

    html_lookup = {rec["url"]: rec.get("html") for rec in shard_df.to_dict("records")}
    by_host: dict[str, list] = defaultdict(list)
    singleton_rows: list[dict] = []
    for rec in shard_df.to_dict("records"):
        feat_json = rec.get("dom_feature", "")
        if not feat_json:
            singleton_rows.append(_singleton_row(rec["url"], rec.get("url_host_name", ""), rec.get("html"), rec))
            continue
        try:
            feat = json.loads(feat_json)
        except Exception:
            feat = None
        if feat is None:
            continue
        host = str(rec.get("url_host_name") or "")
        by_host[host].append(
            {
                "track_id": rec["url"],
                "url": rec["url"],
                "html": rec.get("html", ""),
                "feature": feat,
                "warc_filename": rec.get("warc_filename"),
                "warc_record_offset": rec.get("warc_record_offset"),
                "warc_record_length": rec.get("warc_record_length"),
            }
        )

    host_tasks = [DocumentBatch(dataset_name=h, data=pd.DataFrame(s)) for h, s in by_host.items()]
    t0 = time.perf_counter()
    stage = HostDBSCANStage(
        threshold=args.threshold,
        min_cluster_size=args.min_cluster_size,
        gpu_min_size=args.gpu_min_size,
        max_host_size=int(os.environ.get("STAGE1B_MAX_HOST_SIZE", "3000")),
    )
    pipeline = Pipeline(name="stage1b_dbscan")
    pipeline.add_stage(stage)
    output_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=host_tasks) if host_tasks else []
    elapsed = time.perf_counter() - t0

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"shard_{args.shard_index:04d}.parquet" if args.num_shards > 1 else "shard_0000.parquet")
    frames = []
    for task in output_tasks:
        df = task.to_pandas()
        if not df.empty:
            if "html" not in df.columns:
                df["html"] = df["url"].map(html_lookup)
            frames.append(df[[c for c in OUTPUT_COLS if c in df.columns]])
    if singleton_rows:
        sing_df = pd.DataFrame(singleton_rows)
        if "html" not in sing_df.columns or sing_df["html"].isna().all():
            sing_df["html"] = sing_df["url"].map(html_lookup)
        frames.append(sing_df[[c for c in OUTPUT_COLS if c in sing_df.columns]])
    out_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLS)
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(out_df, preserve_index=False), str(tmp), compression="snappy")
    tmp.rename(out_path)
    n_reps = int((out_df["cluster_role"] == "representative").sum())
    n_sing = int((out_df["cluster_role"] == "singleton").sum())
    logger.info(
        "GPU DBSCAN done in {:.1f}s  reps={} singletons={} call_reduction={:.1%}",
        elapsed,
        n_reps,
        n_sing,
        1.0 - (n_reps + n_sing) / max(len(out_df), 1),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--gpu-min-size", type=int, default=200)
    run(p.parse_args())


if __name__ == "__main__":
    main()
