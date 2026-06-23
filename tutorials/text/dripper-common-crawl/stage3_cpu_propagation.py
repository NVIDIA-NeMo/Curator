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

"""Stage 3: CPU propagation sharding wrapper (logic in DripperHTMLLayoutPropagationStage)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.experimental.dripper.propagation_stage import DripperHTMLLayoutPropagationStage
from nemo_curator.tasks import DocumentBatch

OUTPUT_COLUMNS = [
    "url",
    "url_host_name",
    "cluster_id",
    "cluster_role",
    "dripper_content",
    "dripper_html",
    "dripper_error",
    "dripper_time_s",
    "propagation_success",
    "propagation_method",
]
_MANIFEST_META_COLS = [
    "url",
    "url_host_name",
    "cluster_id",
    "cluster_role",
    "warc_filename",
    "warc_record_offset",
    "warc_record_length",
]
_INFERENCE_COLS = [
    "cluster_id",
    "layout_cluster_id",
    "url",
    "llm_output_raw",
    "inference_time_s",
    "error",
    "dripper_error",
    "dripper_content",
    "dripper_html",
    "mapping_json",
]
_NULL_VALS = frozenset(("none", "null", "nan", ""))
_DEFAULT_NUM_SHARDS = 80
_DEFAULT_NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "64"))


def _load_cluster_manifest_shard(path: str) -> pd.DataFrame:
    sn = pq.read_schema(path).names
    df = pq.read_table(path, columns=[c for c in _MANIFEST_META_COLS if c in sn]).to_pandas()
    df.setdefault("cluster_id", None)
    if "cluster_id" not in df.columns:
        df["cluster_id"] = None
    if "cluster_role" not in df.columns:
        df["cluster_role"] = "singleton"
    df["html"] = None
    if "html" in sn:
        smask = df["cluster_role"] == "sibling"
        if smask.any():
            hdf = pq.read_table(path, columns=["url", "html"]).to_pandas().drop_duplicates("url", keep="first")
            df.loc[smask, "html"] = df.loc[smask, "url"].map(hdf.set_index("url")["html"])
    return df


def _load_inference_results(path: str) -> pd.DataFrame:
    sn = pq.read_schema(path).names
    df = pq.read_table(path, columns=[c for c in _INFERENCE_COLS if c in sn]).to_pandas()
    if "cluster_id" not in df.columns and "layout_cluster_id" in df.columns:
        df = df.rename(columns={"layout_cluster_id": "cluster_id"})
    if "error" not in df.columns and "dripper_error" in df.columns:
        df = df.rename(columns={"dripper_error": "error"})
    return df


def _load_gpu_df(gpu_dir: Path, shard_index: int, cluster_ids: set, urls: set) -> pd.DataFrame:
    exact = gpu_dir / f"shard_{shard_index:04d}.parquet"
    files = (
        [exact] if exact.exists() else (sorted(gpu_dir.glob("shard_*.parquet")) or sorted(gpu_dir.glob("*.parquet")))
    )
    if not files:
        raise FileNotFoundError(f"No GPU inference result files found in {gpu_dir}")
    frames = []
    for f in files:
        try:
            sdf = _load_inference_results(str(f))
            if sdf.empty:
                continue
            mask = pd.Series(False, index=sdf.index)
            if "cluster_id" in sdf.columns and cluster_ids:
                mask |= sdf["cluster_id"].astype(str).isin(cluster_ids)
            if "url" in sdf.columns and urls:
                null_cid = sdf["cluster_id"].isna() | sdf["cluster_id"].astype(str).isin(_NULL_VALS)
                mask |= null_cid & sdf["url"].astype(str).isin(urls)
            if not (filt := sdf[mask]).empty:
                frames.append(filt)
        except OSError as exc:
            logger.warning("could not read GPU shard {}: {}", f, exc)
    gpu_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.info("{:,} GPU result rows loaded ({} files)", len(gpu_df), len(files))
    return gpu_df


def process_shard(
    cluster_manifest_dir: str,
    inference_results_dir: str,
    output_dir: str,
    shard_index: int,
    num_shards: int,
    num_workers: int,
) -> dict:
    t_start = time.perf_counter()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{shard_index:04d}.parquet"
    if out_path.exists():
        try:
            meta = pq.read_metadata(str(out_path))
            if meta.num_rows > 0:
                logger.info("SKIP shard {} — already exists ({:,} rows)", shard_index, meta.num_rows)
                return {"status": "skipped", "shard": shard_index, "rows": meta.num_rows}
            out_path.unlink(missing_ok=True)
        except OSError:
            out_path.unlink(missing_ok=True)

    manifest_dir = Path(cluster_manifest_dir)
    all_files = sorted(manifest_dir.glob("shard_*.parquet")) or sorted(manifest_dir.glob("*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No manifest shards found in {manifest_dir}")
    n = len(all_files)
    my_files = all_files[n * shard_index // num_shards : n * (shard_index + 1) // num_shards]
    if not my_files:
        logger.info("shard {}: no manifest files — writing empty shard", shard_index)
        pq.write_table(pa.table({c: [] for c in OUTPUT_COLUMNS}), str(out_path))
        return {"status": "empty", "shard": shard_index, "rows": 0}

    manifest_df = pd.concat([_load_cluster_manifest_shard(str(f)) for f in my_files], ignore_index=True)
    logger.info("shard {}/{}: {:,} rows from {} file(s)", shard_index, num_shards, len(manifest_df), len(my_files))

    cluster_ids = {str(r) for r in manifest_df["cluster_id"].dropna() if str(r).lower() not in _NULL_VALS}
    urls = set(manifest_df["url"].astype(str))
    gpu_df = _load_gpu_df(Path(inference_results_dir), shard_index, cluster_ids, urls)

    mapping_by_cluster: dict = {}
    for rec in gpu_df.to_dict("records"):
        cid = str(rec.get("cluster_id") or "")
        if cid and cid.lower() not in _NULL_VALS:
            mapping_by_cluster.setdefault(cid, rec.get("mapping_json") or rec.get("llm_output_raw", ""))

    manifest_df["dripper_layout_cluster"] = manifest_df["cluster_id"].astype(str)
    manifest_df["dripper_layout_representative"] = manifest_df["cluster_role"].isin(["representative", "singleton"])
    manifest_df["dripper_layout_mapping_json"] = (
        manifest_df["cluster_id"]
        .astype(str)
        .map(lambda cid: mapping_by_cluster.get(cid, "") if cid and cid.lower() not in _NULL_VALS else "")
    )
    manifest_df["dripper_layout_pending_propagation"] = manifest_df["cluster_role"] == "sibling"

    stage = DripperHTMLLayoutPropagationStage(use_static_lbp=True)
    pipeline = Pipeline(name="stage3_cpu_propagation")
    pipeline.add_stage(stage)
    chunk = max(1, len(manifest_df) // max(1, num_workers))
    doc_tasks = [
        DocumentBatch(dataset_name="stage3", data=manifest_df.iloc[i : i + chunk].reset_index(drop=True))
        for i in range(0, len(manifest_df), chunk)
    ]
    logger.info("submitting {:,} tasks ({} actors)...", len(doc_tasks), num_workers)
    output_doc_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=doc_tasks) or []

    frames = [t.to_pandas() for t in output_doc_tasks]
    result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    result_df = result_df.rename(
        columns={
            "dripper_layout_html": "dripper_html",
            "dripper_layout_content": "dripper_content",
            "dripper_layout_error": "dripper_error",
            "dripper_layout_postprocess_time_s": "dripper_time_s",
            "dripper_layout_propagation_success": "propagation_success",
            "dripper_layout_propagation_method": "propagation_method",
        }
    )
    for col in OUTPUT_COLUMNS:
        if col not in result_df.columns:
            result_df[col] = None

    tmp = out_path.with_suffix(f".tmp_{os.getpid()}.parquet")
    pq.write_table(
        pa.Table.from_pandas(result_df[OUTPUT_COLUMNS], preserve_index=False), str(tmp), compression="snappy"
    )
    tmp.rename(out_path)

    elapsed = time.perf_counter() - t_start
    ns = int(result_df.get("propagation_success", pd.Series()).fillna(False).sum())
    logger.info(
        "shard {} done  pages={:,} success={} elapsed={:.1f}s  output={}",
        shard_index,
        len(result_df),
        ns,
        elapsed,
        out_path,
    )
    metrics = {
        "shard_index": shard_index,
        "num_shards": num_shards,
        "total_pages": len(result_df),
        "success_pages": ns,
        "elapsed_s": elapsed,
        "output_path": str(out_path),
    }
    (out_dir / f"metrics_shard_{shard_index:04d}.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args
    _configs_dir = Path(__file__).parent / "configs"
    if str(_configs_dir) not in sys.path:
        sys.path.insert(0, str(_configs_dir))
    from dripper_config import DripperConfig

    cfg = DripperConfig.from_yaml(args.config)
    if args.num_shards == _DEFAULT_NUM_SHARDS:
        args.num_shards = cfg.num_shards
    if args.num_workers == _DEFAULT_NUM_WORKERS:
        stage_res = cfg.resources.get("stage3", {})
        args.num_workers = int(stage_res.get("num_workers", stage_res.get("cpus", args.num_workers)))
    return args


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stage 3: CPU template propagation", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--config", default=None)
    p.add_argument("--cluster-manifest", required=True)
    p.add_argument("--inference-results", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    p.add_argument("--num-shards", type=int, default=_DEFAULT_NUM_SHARDS)
    p.add_argument("--num-workers", type=int, default=_DEFAULT_NUM_WORKERS)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = _apply_config_defaults(p.parse_args())
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper())
    logger.info(
        "manifest={}  gpu={}  out={}  shard={}/{}  workers={}",
        args.cluster_manifest,
        args.inference_results,
        args.output_dir,
        args.shard_index,
        args.num_shards,
        args.num_workers,
    )
    metrics = process_shard(
        args.cluster_manifest,
        args.inference_results,
        args.output_dir,
        args.shard_index,
        args.num_shards,
        args.num_workers,
    )
    status = metrics.get("status", "done")
    logger.info(
        "Shard {} {}",
        args.shard_index,
        {"skipped": "already complete — skipped.", "empty": "had no input — wrote empty shard."}.get(
            status, "complete."
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
