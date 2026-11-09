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

"""Evaluate video caption quality using Summarize-then-Align with CosmosEmbed1.

Given one or more pipeline output directories, this script:

  1. Loads pre-computed CosmosEmbed1 video embeddings.
  2. Reads per-window captions from the JSON metadata files.
  3. Summarizes each caption to <=80 words using an LLM (via vLLM), extracting
     only visual elements and removing narrative commentary.
  4. Encodes each summary with the CosmosEmbed1 text encoder (single chunk).
  5. Computes per-clip cosine similarity between video and text embeddings.
  6. Prints per-model mean scores and writes per-clip scores to a CSV.

The summarizer LLM should be from a different model family than the captioning
models being evaluated to avoid vocabulary/phrasing bias.

Use ``--save-summaries`` to cache LLM summaries to a JSON file, and
``--load-summaries`` to reuse them for deterministic scoring.

Example:

    python caption_clipscore.py \\
        --embedding-dir /path/to/captions_qwen25/ce1_embd \\
        --cosmos-model-dir /path/to/models \\
        --summarizer-model /path/to/Llama-3.1-8B-Instruct \\
        --caption-dirs \\
            qwen25=/path/to/captions_qwen25 \\
            qwen3=/path/to/captions_qwen3 \\
            nemotron=/path/to/captions_nemotron \\
        --uid-list /path/to/benchmark_200/selected_uids.txt \\
        --save-summaries summaries.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from nemo_curator.models.cosmos_embed1 import CosmosEmbed1

_SUMMARIZE_SYSTEM = (
    "You are a visual description extractor. You output ONLY the visual elements "
    "from a video caption. Keep colors, objects, actions, positions, clothing, text "
    "visible on screen. Remove all narrative commentary, emotional interpretation, "
    "aesthetic judgments, and editorial language. Output a single paragraph under 80 "
    "words. Do not include word counts, revisions, or meta-commentary."
)

_UID_LINE_FIELDS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_window_captions(meta_path: str) -> list[str]:
    """Return one caption string per window from a metadata JSON."""
    with open(meta_path) as f:
        data = json.load(f)
    captions = []
    for window in data.get("windows", []):
        for key, value in window.items():
            if "caption" in key and isinstance(value, str) and value.strip():
                captions.append(value.strip())
                break
    return captions


def _get_source_video(meta_path: str) -> str:
    """Return the source video path from a clip metadata JSON."""
    with open(meta_path) as f:
        data = json.load(f)
    return data.get("source_video", data.get("video_path", "unknown"))


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / a.norm()
    b = b / b.norm()
    return float((a * b).sum())


def _load_uid_list(
    uid_list_path: str,
    embedding_dir: str,
    caption_dirs: dict[str, str],
) -> set[str]:
    """Load UIDs from a file, resolving by (source_video, span) if needed.

    The UID list may use tab-separated format: uid<TAB>source_video<TAB>start<TAB>end.
    If UIDs don't match the embedding directory directly, resolution falls back to
    matching by (source_video_basename, duration_span) as a stable identifier.
    """
    requested_uids = set()
    requested_keys: dict[tuple, str] = {}
    with open(uid_list_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            uid = parts[0]
            if not uid:
                continue
            requested_uids.add(uid)
            if len(parts) >= _UID_LINE_FIELDS:
                key = (parts[1], round(float(parts[2]), 4), round(float(parts[3]), 4))
                requested_keys[key] = uid

    emb_uuids = {Path(f).stem for f in glob.glob(f"{embedding_dir}/*.pickle")}
    direct_match = requested_uids & emb_uuids
    if len(direct_match) == len(requested_uids):
        return requested_uids

    if not requested_keys:
        logger.warning(
            f"UIDs don't match and no (source_video, span) info in uid list. Using {len(direct_match)} direct matches."
        )
        return direct_match

    first_cap_dir = next(iter(caption_dirs.values()))
    resolved_uids = set()
    for fp in glob.glob(f"{first_cap_dir}/metas/v0/*.json"):
        with open(fp) as fh:
            d = json.load(fh)
        uid = Path(fp).stem
        src = Path(d.get("source_video", "")).name
        span = d.get("duration_span", [0, 0])
        key = (src, round(span[0], 4), round(span[1], 4))
        if key in requested_keys and uid in emb_uuids:
            resolved_uids.add(uid)

    logger.info(f"Resolved {len(resolved_uids)}/{len(requested_uids)} UIDs via (source_video, span)")
    return resolved_uids


def _summarize_captions(
    tasks: list[tuple[str, str, str]],
    summarizer_model: str,
) -> list[str]:
    """Batch-summarize captions using vLLM."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    logger.info(f"Loading summarizer: {summarizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(summarizer_model)
    llm = LLM(model=summarizer_model, dtype="bfloat16", gpu_memory_utilization=0.5)

    prompts = []
    for _, _, caption in tasks:
        messages = [
            {"role": "system", "content": _SUMMARIZE_SYSTEM},
            {"role": "user", "content": caption},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    stop_tokens = ["\n\n"]
    if tokenizer.eos_token:
        stop_tokens.append(tokenizer.eos_token)
    sampling = SamplingParams(temperature=0.0, max_tokens=120, stop=stop_tokens)
    logger.info(f"Summarizing {len(prompts)} captions ...")
    outputs = llm.generate(prompts, sampling)
    summaries = [o.outputs[0].text.strip() for o in outputs]
    logger.info("Summarization done")

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return summaries


def _collect_tasks(
    common_uuids: list[str],
    caption_dirs: dict[str, str],
) -> list[tuple[str, str, str]]:
    """Collect (uid, label, full_caption) tuples for all clips and models."""
    tasks = []
    for uid in common_uuids:
        for label, cap_dir in caption_dirs.items():
            captions = _get_window_captions(f"{cap_dir}/metas/v0/{uid}.json")
            tasks.append((uid, label, " ".join(captions)))
    return tasks


def _load_video_embeddings(
    common_uuids: list[str],
    embedding_dir: str,
) -> dict[str, torch.Tensor]:
    """Load and cache video embeddings for all UIDs."""
    cache = {}
    for uid in common_uuids:
        with open(f"{embedding_dir}/{uid}.pickle", "rb") as f:
            arr = pickle.load(f)  # noqa: S301
        cache[uid] = torch.from_numpy(arr).squeeze(0)
    return cache


def _score_summaries(
    tasks: list[tuple[str, str, str]],
    summaries: list[str],
    vid_emb_cache: dict[str, torch.Tensor],
    cosmos_model_dir: str,
    variant: str,
) -> dict[str, dict[str, float]]:
    """Encode summaries with CosmosEmbed1 and compute per-clip scores."""
    logger.info(f"Loading CosmosEmbed1-{variant} from {cosmos_model_dir} ...")
    model = CosmosEmbed1(variant=variant, utils_only=False, model_dir=cosmos_model_dir)
    model.setup()

    clip_scores: dict[str, dict[str, float]] = {}
    logger.info("Encoding summaries and scoring ...")
    for i, (uid, label, _caption) in enumerate(tqdm(tasks, unit="cap")):
        vid_emb = vid_emb_cache[uid]
        text_emb = model.get_text_embedding(summaries[i]).squeeze(0)
        clip_scores.setdefault(uid, {})[label] = _cosine_sim(vid_emb, text_emb)

    return clip_scores


def evaluate(  # noqa: PLR0913, C901, PLR0915
    embedding_dir: str,
    caption_dirs: dict[str, str],
    cosmos_model_dir: str,
    summarizer_model: str | None,
    variant: str,
    output_csv: str,
    uid_list: str | None = None,
    save_summaries: str | None = None,
    load_summaries: str | None = None,
) -> None:
    """Run the full Summarize-then-Align evaluation pipeline."""
    # Collect UUIDs present in all caption directories
    uuid_sets = []
    for label, cap_dir in caption_dirs.items():
        metas = glob.glob(f"{cap_dir}/metas/v0/*.json")
        uuids = {Path(f).stem for f in metas}
        logger.info(f"{label}: {len(uuids)} clips in {cap_dir}/metas/v0/")
        uuid_sets.append(uuids)

    emb_uuids = {Path(f).stem for f in glob.glob(f"{embedding_dir}/*.pickle")}
    common_uuids = emb_uuids.intersection(*uuid_sets)

    if uid_list is not None:
        logger.info(f"Filtering to UIDs from: {uid_list}")
        allowed = _load_uid_list(uid_list, embedding_dir, caption_dirs)
        common_uuids = common_uuids & allowed

    common_uuids = sorted(common_uuids)
    logger.info(f"Evaluating {len(common_uuids)} clips")

    labels = list(caption_dirs.keys())
    tasks = _collect_tasks(common_uuids, caption_dirs)
    logger.info(f"Collected {len(tasks)} captions")

    # Get summaries (from cache or LLM)
    if load_summaries is not None:
        logger.info(f"Loading cached summaries from: {load_summaries}")
        with open(load_summaries) as f:
            summary_cache = json.load(f)
        summaries = []
        missing = 0
        for uid, label, _ in tasks:
            s = summary_cache.get(uid, {}).get(label, "")
            if not s:
                missing += 1
            summaries.append(s)
        if missing:
            logger.warning(f"{missing}/{len(tasks)} summaries not found in cache — scores may be invalid")
    else:
        if summarizer_model is None:
            msg = "--summarizer-model is required when not using --load-summaries"
            raise ValueError(msg)
        summaries = _summarize_captions(tasks, summarizer_model)

    if save_summaries is not None:
        summary_cache = {}
        for i, (uid, label, _) in enumerate(tasks):
            summary_cache.setdefault(uid, {})[label] = summaries[i]
        with open(save_summaries, "w") as f:
            json.dump(summary_cache, f, indent=2)
        logger.info(f"Summaries saved to: {save_summaries}")

    # Load video embeddings once (shared across all models)
    vid_emb_cache = _load_video_embeddings(common_uuids, embedding_dir)

    # Score
    clip_scores = _score_summaries(tasks, summaries, vid_emb_cache, cosmos_model_dir, variant)

    # Print statistics
    logger.info("Results (per-clip scores):")
    for label in labels:
        scores = [clip_scores[uid][label] for uid in common_uuids if label in clip_scores.get(uid, {})]
        logger.info(f"  {label}: mean={np.mean(scores):.4f} (n={len(scores)})")

    # Export CSV
    csv_path = Path(output_csv)
    fieldnames = ["uuid", "source_video", *labels]
    rows = []
    for uid in common_uuids:
        first_cap_dir = next(iter(caption_dirs.values()))
        source_video = _get_source_video(f"{first_cap_dir}/metas/v0/{uid}.json")
        row = {"uuid": uid, "source_video": source_video}
        row.update(clip_scores.get(uid, {}))
        rows.append(row)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Per-clip scores written to: {csv_path.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate caption quality with Summarize-then-Align (CosmosEmbed1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--embedding-dir",
        required=True,
        help="Directory containing pre-computed CosmosEmbed1 pickle files named <uuid>.pickle.",
    )
    parser.add_argument(
        "--cosmos-model-dir",
        required=True,
        help="Root model directory containing nvidia/Cosmos-Embed1-<variant>/.",
    )
    parser.add_argument(
        "--summarizer-model",
        default=None,
        help="Path to the summarizer LLM (e.g. Llama-3.1-8B-Instruct). "
        "Required unless --load-summaries is provided. "
        "Should be a different model family than the captioners to avoid bias.",
    )
    parser.add_argument(
        "--caption-dirs",
        required=True,
        nargs="+",
        metavar="LABEL=PATH",
        help=(
            "One or more label=path pairs pointing to pipeline output directories. "
            "Each must contain a metas/v0/ subdirectory with per-clip JSON files."
        ),
    )
    parser.add_argument(
        "--uid-list",
        default=None,
        help="Optional file listing clip UUIDs to score (one per line). "
        "If provided, only these clips are evaluated. UIDs are resolved "
        "by (source_video, duration_span) if they don't match directly.",
    )
    parser.add_argument(
        "--save-summaries",
        default=None,
        help="Save LLM summaries to a JSON file for deterministic re-scoring.",
    )
    parser.add_argument(
        "--load-summaries",
        default=None,
        help="Load cached summaries from a JSON file instead of running the LLM. Skips --summarizer-model entirely.",
    )
    parser.add_argument(
        "--variant",
        default="224p",
        choices=["224p", "336p", "448p"],
        help="CosmosEmbed1 variant (default: 224p).",
    )
    parser.add_argument(
        "--output-csv",
        default="clipscore_results.csv",
        help="Path to write per-clip CSV results (default: clipscore_results.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    caption_dirs: dict[str, str] = {}
    for item in args.caption_dirs:
        if "=" not in item:
            msg = f"--caption-dirs entries must be LABEL=PATH, got: {item!r}"
            raise ValueError(msg)
        label, path = item.split("=", 1)
        caption_dirs[label] = path

    evaluate(
        embedding_dir=args.embedding_dir,
        caption_dirs=caption_dirs,
        cosmos_model_dir=args.cosmos_model_dir,
        summarizer_model=args.summarizer_model,
        variant=args.variant,
        output_csv=args.output_csv,
        uid_list=args.uid_list,
        save_summaries=args.save_summaries,
        load_summaries=args.load_summaries,
    )


if __name__ == "__main__":
    main()
