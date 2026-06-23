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

"""Stage 1c + Stage 2 (vLLM) + Stage 2b GPU pipeline. Input: Stage 1b parquet."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

_REPO_ROOT = str(Path(__file__).parent.parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline_metrics import StageMetrics

OUTPUT_COLS = [
    "url",
    "url_host_name",
    "cluster_id",
    "cluster_role",
    "mapping_json",
    "dripper_content",
    "dripper_html",
    "dripper_error",
    "inference_time_s",
]
_GPU_SLICE_COLS = ["url", "prompt", "item_count", "cluster_id", "cluster_role", "url_host_name"]
_MIN_CONTENT_LEN, _MIN_ERROR_LEN, _MIN_PROMPT_LEN = 5, 2, 10


def run_stage1c(df: pd.DataFrame) -> pd.DataFrame:
    from nemo_curator.stages.text.experimental.dripper.preprocessing import DripperHTMLPreprocessStage

    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.tasks import DocumentBatch

    t0 = time.perf_counter()
    n_workers = max(1, (os.cpu_count() or 4) - 2)
    chunk = max(1, len(df) // n_workers)
    tasks = [
        DocumentBatch(dataset_name="stage1c", data=df.iloc[i : i + chunk].reset_index(drop=True))
        for i in range(0, len(df), chunk)
    ]
    stage = DripperHTMLPreprocessStage(html_col="html", url_col="url", worker_count=n_workers)
    pipeline = Pipeline(name="stage1c")
    pipeline.add_stage(stage)
    result_tasks = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=tasks) or []
    out = pd.concat([t.to_pandas() for t in result_tasks], ignore_index=True)
    ok = (out.get("prompt", out.get("_dripper_prompt", pd.Series())).astype(str).str.len() > _MIN_PROMPT_LEN).sum()
    logger.info("Stage 1c: {:,}/{:,} prompts in {:.1f}s", ok, len(df), time.perf_counter() - t0)
    return out


@dataclass
class _Cfg:
    model: str
    gpu_mem_util: float
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    max_tokens: int
    kv_cache_dtype: str


def _build_worker_prompts(rows, tok, max_model_len, max_tokens):
    from vllm import SamplingParams

    supports_think: list[bool] = [True]
    prompts, samplings, ridx, results, n_trunc = [], [], [], [None] * len(rows), 0
    for i, r in enumerate(rows):
        p = str(r.get("prompt", "") or "")
        if not p or p.startswith("ERROR:"):
            results[i] = {
                **r,
                "llm_response": "",
                "dripper_error": p if p.startswith("ERROR:") else "empty_prompt",
                "inference_time_s": 0.0,
            }
            continue
        ic = max(0, int(r.get("item_count", 0) or 0))
        max_tok = min(max_tokens, max(32, ic * 6 + 16) if ic > 0 else max_tokens)
        msgs = [{"role": "user", "content": p}]
        if supports_think[0]:
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                supports_think[0] = False
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        cap = max_model_len - max_tok - 8
        if len(ids) > cap:
            ids = ids[:cap]
            n_trunc += 1
        prompts.append({"prompt_token_ids": ids})
        samplings.append(SamplingParams(temperature=0.0, max_tokens=max_tok))
        ridx.append(i)
    return prompts, samplings, ridx, results, n_trunc


def run_stage2_worker(gpu_id: int, slice_path: str, out_path: str, cfg: _Cfg) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from transformers import AutoTokenizer
    from vllm import LLM

    from nemo_curator.utils.vllm_utils import pick_free_port, resolve_local_model_path

    local_model = resolve_local_model_path(cfg.model)
    tok = AutoTokenizer.from_pretrained(local_model, trust_remote_code=True)
    llm_kw: dict = {
        "model": local_model,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": cfg.gpu_mem_util,
        "max_model_len": cfg.max_model_len,
        "max_num_seqs": cfg.max_num_seqs,
        "max_num_batched_tokens": cfg.max_num_batched_tokens,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "enforce_eager": False,
        "trust_remote_code": True,
        "disable_log_stats": True,
    }
    if cfg.kv_cache_dtype and cfg.kv_cache_dtype != "auto":
        llm_kw["kv_cache_dtype"] = cfg.kv_cache_dtype
    os.environ["MASTER_PORT"] = str(pick_free_port())
    t_setup = time.perf_counter()
    llm = LLM(**llm_kw)
    setup_s = time.perf_counter() - t_setup
    rows = pq.ParquetFile(slice_path).read().to_pandas().to_dict("records")
    prompts, samplings, ridx, results, n_trunc = _build_worker_prompts(rows, tok, cfg.max_model_len, cfg.max_tokens)
    t1 = time.perf_counter()
    outs = llm.generate(prompts, samplings) if prompts else []
    infer_s = time.perf_counter() - t1
    for j, o in enumerate(outs):
        i = ridx[j]
        resp = o.outputs[0].text if o.outputs else ""
        results[i] = {
            **rows[i],
            "llm_response": resp,
            "dripper_error": "" if resp else "empty_response",
            "inference_time_s": infer_s / max(len(outs), 1),
        }
    pd.DataFrame([x for x in results if x is not None]).to_parquet(out_path, index=False, compression="snappy")
    logger.info(
        "gpu{} DONE {} prompts ({} trunc) setup={:.1f}s infer={:.1f}s {:.1f} pages/s",
        gpu_id,
        len(prompts),
        n_trunc,
        setup_s,
        infer_s,
        len(prompts) / max(infer_s, 1e-6),
    )


def _detect_gpus() -> int:
    n = os.environ.get("SLURM_GPUS_ON_NODE") or os.environ.get("SLURM_GPUS_PER_NODE", "")
    if n:
        try:
            return int(n.split(":")[-1])
        except ValueError:
            pass
    try:
        r = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True, timeout=5)
        return max(1, sum(1 for ln in r.stdout.splitlines() if ln.startswith("GPU")))
    except OSError:
        return 1


def run_stage2(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    n_gpus = args.replicas if args.replicas > 0 else _detect_gpus()
    logger.info("Stage 2: {:,} pages over {} GPUs", len(df), n_gpus)
    tmp = Path(args.output) / "_gpu_slices"
    tmp.mkdir(parents=True, exist_ok=True)
    cost = df["prompt"].astype(str).str.len().to_numpy()
    order = sorted(range(len(df)), key=lambda i: -cost[i])
    bins: list[list[int]] = [[] for _ in range(n_gpus)]
    load = [0] * n_gpus
    for i in order:
        g = min(range(n_gpus), key=lambda k: load[k])
        bins[g].append(i)
        load[g] += int(cost[i])
    sl = [str(tmp / f"slice_{g}.parquet") for g in range(n_gpus)]
    ol = [str(tmp / f"out_{g}.parquet") for g in range(n_gpus)]
    cols = [c for c in _GPU_SLICE_COLS if c in df.columns]
    for g in range(n_gpus):
        df[cols].iloc[bins[g]].to_parquet(sl[g], index=False)
    w_base = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--model",
        args.model,
        "--max-tokens",
        str(args.max_tokens),
        "--gpu-mem-util",
        str(args.gpu_mem_util),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--kv-cache-dtype",
        args.kv_cache_dtype,
    ]
    t0 = time.perf_counter()
    procs = [
        subprocess.Popen([*w_base, "--gpu", str(g), "--slice", sl[g], "--slice-out", ol[g]]) for g in range(n_gpus)
    ]
    rcs = [p.wait() for p in procs]
    logger.info("Stage 2 workers done in {:.1f}s codes={}", time.perf_counter() - t0, rcs)
    frames = [pq.ParquetFile(o).read().to_pandas() for o in ol if Path(o).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_stage2b(df: pd.DataFrame) -> pd.DataFrame:
    from nemo_curator.stages.text.experimental.dripper.preprocessing import DripperHTMLPostprocessStage

    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.tasks import DocumentBatch

    t0 = time.perf_counter()
    n_workers = max(1, (os.cpu_count() or 4) - 2)
    stage_df = df.copy()
    if "dripper_response" not in stage_df.columns and "llm_response" in stage_df.columns:
        stage_df["dripper_response"] = stage_df["llm_response"]
    stage = DripperHTMLPostprocessStage(html_col="html", url_col="url", worker_count=n_workers)
    pipeline = Pipeline(name="stage2b")
    pipeline.add_stage(stage)
    chunks = [
        DocumentBatch(dataset_name="stage2b", data=stage_df.iloc[i : i + 1000].reset_index(drop=True))
        for i in range(0, len(stage_df), 1000)
    ]
    output = pipeline.run(executor=RayActorPoolExecutor(), initial_tasks=chunks) or []
    out = pd.concat([t.to_pandas() for t in output], ignore_index=True) if output else stage_df
    if "mapping_json" not in out.columns:
        out["mapping_json"] = ""
    logger.info(
        "Stage 2b: content_ok={:,} mapping_ok={:,} in {:.1f}s",
        (out["dripper_content"].astype(str).str.len() > _MIN_CONTENT_LEN).sum(),
        (out["mapping_json"].astype(str).str.len() > _MIN_CONTENT_LEN).sum(),
        time.perf_counter() - t0,
    )
    return out


def run(args: argparse.Namespace) -> None:
    tracker = StageMetrics(
        "stage_gpu_pipeline",
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        n_gpus=args.replicas or _detect_gpus(),
    )
    tracker.start()
    t_total = time.perf_counter()
    inp = Path(args.input)
    if inp.is_dir():
        exact = inp / f"shard_{args.shard_index:04d}.parquet"
        inp = exact if exact.exists() else sorted(inp.glob("shard_*.parquet"))[0]
    all_df = pq.ParquetFile(str(inp)).read().to_pandas()
    rep_df = (
        all_df[all_df["cluster_role"].isin(["representative", "singleton"])]
        if "cluster_role" in all_df.columns
        else all_df
    ).reset_index(drop=True)
    logger.info(
        "{:,}/{:,} pages sent to LLM ({:.1f}%)", len(rep_df), len(all_df), len(rep_df) / max(len(all_df), 1) * 100
    )
    _t = time.perf_counter()
    rep_df = run_stage1c(rep_df)
    t1c_s = time.perf_counter() - _t
    _t = time.perf_counter()
    infer_df = run_stage2(rep_df, args)
    t2_s = time.perf_counter() - _t
    _t = time.perf_counter()
    passthrough = rep_df[["url"] + [c for c in ["simp_html", "map_html", "html"] if c in rep_df.columns]]
    infer_df = infer_df.merge(passthrough, on="url", how="left", suffixes=("", "_1c"))
    for c in ["simp_html", "map_html", "html"]:
        if f"{c}_1c" in infer_df.columns:
            infer_df[c] = infer_df[c].fillna(infer_df[f"{c}_1c"])
            infer_df = infer_df.drop(columns=[f"{c}_1c"])
    result_df = run_stage2b(infer_df)
    t2b_s = time.perf_counter() - _t
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"shard_{args.shard_index:04d}.parquet" if args.num_shards > 1 else "pipeline_results.parquet"
    out_path = out_dir / fname
    for col in OUTPUT_COLS:
        if col not in result_df.columns:
            result_df[col] = None
    tmp = out_path.with_suffix(".parquet.tmp")
    result_df.to_parquet(str(tmp), index=False, compression="snappy")
    tmp.rename(out_path)
    total_s = time.perf_counter() - t_total
    ok = int((result_df["dripper_content"].astype(str).str.len() > _MIN_CONTENT_LEN).sum())
    errs = int((result_df["dripper_error"].astype(str).str.len() > _MIN_ERROR_LEN).sum())
    logger.info(
        "ALL DONE: {:,} pages ok={} total={:.1f}s (1c={:.1f}s 2={:.1f}s 2b={:.1f}s) -> {}",
        len(result_df),
        ok,
        total_s,
        t1c_s,
        t2_s,
        t2b_s,
        out_path,
    )
    tracker.finish(total_pages=len(result_df), errors=errs)
    tracker.extra = {
        "stage1c_s": round(t1c_s, 1),
        "stage2_s": round(t2_s, 1),
        "stage2b_s": round(t2b_s, 1),
        "content_ok": ok,
    }
    tracker.save(args.output)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--slice")
    p.add_argument("--slice-out")
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--replicas", type=int, default=int(os.environ.get("N_GPU_REPLICAS", "0")))
    p.add_argument("--model", default="opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact")
    p.add_argument("--hf-cache", default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--kv-cache-dtype", default="fp8")
    args = p.parse_args()
    os.environ.setdefault("HF_HOME", args.hf_cache)
    if args.worker:
        run_stage2_worker(
            args.gpu,
            args.slice,
            args.slice_out,
            _Cfg(
                args.model,
                args.gpu_mem_util,
                args.max_model_len,
                args.max_num_seqs,
                args.max_num_batched_tokens,
                args.max_tokens,
                args.kv_cache_dtype,
            ),
        )
    else:
        if not args.input or not args.output:
            p.error("--input and --output required in main mode")
        run(args)


if __name__ == "__main__":
    main()
