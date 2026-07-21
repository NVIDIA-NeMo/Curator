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

"""MSR-VTT text-to-video retrieval benchmark using CosmosEmbed1.

Evaluates ``nemo_curator.models.cosmos_embed1.CosmosEmbed1`` on MSR-VTT
text-to-video retrieval:

  - Corpus : one VIDEO embedding per unique video via CosmosEmbed1.
  - Queries: every caption embedded with CosmosEmbed1's text tower.
  - Retrieval: cosine-similarity ranking of each query over the video corpus.
  - Metrics : R@1, R@5, R@10, MedR, MeanR, MRR, NDCG@10.

Usage
-----
  python eval/video/bench_msrvtt_retrieval.py \\
      --model-dir /path/to/models \\
      --variant 336p \\
      --split test \\
      --top-k 10 \\
      --output results.json

  # quick smoke test on 50 captions
  python eval/video/bench_msrvtt_retrieval.py \\
      --model-dir /path/to/models --variant 224p --limit 50
"""

import argparse
import json
import logging
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from nemo_curator.models.cosmos_embed1 import CosmosEmbed1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOG = logging.getLogger("bench_msrvtt_retrieval")

HF_DATASET = "friedrichor/MSR-VTT"
HF_VIDEO_ZIP = "MSRVTT_Videos.zip"


# ---------------------------------------------------------------------------
# Dataset loading (captions + video paths)
# ---------------------------------------------------------------------------

def ensure_msrvtt_videos(target_dir: Path) -> Path:
    """Download + extract MSRVTT_Videos.zip from the HF Hub (once)."""
    target_dir = target_dir.expanduser().resolve()
    done_flag = target_dir / ".videos_extracted"
    if done_flag.exists():
        LOG.info("MSR-VTT videos already extracted in %s", target_dir)
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading %s from HF Hub (%s) ...", HF_VIDEO_ZIP, HF_DATASET)
    zip_path = Path(
        hf_hub_download(
            repo_id=HF_DATASET,
            filename=HF_VIDEO_ZIP,
            repo_type="dataset",
            resume_download=True,
        )
    )
    LOG.info("Extracting %s -> %s (may take a while)", zip_path, target_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    done_flag.touch()
    LOG.info("MSR-VTT videos ready in %s", target_dir)
    return target_dir


def _build_video_index(video_dir: Path) -> Dict[str, str]:
    """Map {video_id_stem: absolute mp4 path} for everything under video_dir."""
    index: Dict[str, str] = {}
    for p in video_dir.rglob("*.mp4"):
        index.setdefault(p.stem, str(p))
    return index


def load_msrvtt(
    split: str,
    video_dir: Path,
    config: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    """Return records: [{video_id, video_path, caption}].

    Each caption row becomes one query; rows are de-duplicated to unique videos
    when building the retrieval corpus downstream.
    """
    LOG.info("Loading %s  split=%s  config=%s", HF_DATASET, split, config)
    ds = load_dataset(HF_DATASET, config, split=split)

    video_index = _build_video_index(video_dir)
    LOG.info("Indexed %d mp4 files under %s", len(video_index), video_dir)

    records: List[Dict] = []
    missing = 0
    # Column name differs by config: test_1k uses "caption", train configs use "sentence"
    cap_col = "caption" if "caption" in ds.column_names else "sentence"
    for row in ds:
        vid = str(row["video_id"])
        cap = str(row[cap_col])
        vpath = video_index.get(vid)
        if vpath is None:
            missing += 1
            continue
        records.append({"video_id": vid, "video_path": vpath, "caption": cap})

    if missing:
        LOG.warning("%d caption rows had no matching .mp4 and were dropped", missing)

    if limit and limit > 0:
        records = records[:limit]

    LOG.info("Loaded %d caption rows over %d unique videos",
             len(records), len({r["video_id"] for r in records}))
    return records


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _load_frames(path: str, sample_fps: float = 2.0) -> List[np.ndarray]:
    """Extract frames from a video at sample_fps using cv2 (software decoding).

    Uses cv2.VideoCapture which defaults to CPU/software decoding, avoiding
    libnvcuvid dependency that may be absent in some container environments.
    Returns list of (H, W, C) uint8 RGB arrays.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(video_fps / sample_fps))
    frames = []
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError("empty video")
    return frames


def embed_texts(model: CosmosEmbed1, texts: List[str], batch_size: int) -> np.ndarray:
    """Embed a list of strings; returns L2-normalised (N, D) float32 array."""
    embs: List[np.ndarray] = []
    n = len(texts)
    for i, text in enumerate(texts):
        if i % batch_size == 0:
            LOG.info("Text embed %d / %d", i + 1, n)
        t = model.get_text_embedding(text)   # (1, D) float16 cpu tensor
        embs.append(t.float().numpy()[0])
    return _normalise(np.stack(embs, axis=0))


def embed_videos(
    model: CosmosEmbed1, video_paths: List[str], batch_size: int, sample_fps: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Embed videos. Returns (embeddings (N, D), valid_mask (N,) bool).

    Videos are embedded one at a time because clips have varying native
    resolutions. Failed/undecodable videos get a zero embedding and valid=False.
    """
    dim: Optional[int] = None
    all_embs: List[Optional[np.ndarray]] = [None] * len(video_paths)
    valid = np.zeros(len(video_paths), dtype=bool)

    n = len(video_paths)
    for j in range(n):
        if j % max(batch_size, 1) == 0:
            LOG.info("Video embed %d / %d", j + 1, n)
        try:
            frames = _load_frames(video_paths[j], sample_fps)
            processed = model.formulate_input_frames(frames)
            if processed is None:
                LOG.warning("Not enough frames in %s", video_paths[j])
                continue
            emb = model.encode_video_frames(processed)   # (1, D) float16 cpu
            all_embs[j] = emb.float().numpy()[0]         # (D,)
            dim = all_embs[j].shape[0]
            valid[j] = True
        except Exception as exc:
            LOG.warning("Failed to embed %s: %s", video_paths[j], exc)

    if dim is None:
        raise RuntimeError("No videos could be decoded/embedded.")
    filled = np.stack(
        [e if e is not None else np.zeros(dim, dtype=np.float32) for e in all_embs],
        axis=0,
    )
    return _normalise(filled), valid


# ---------------------------------------------------------------------------
# Corpus / ranking / metrics
# ---------------------------------------------------------------------------

def build_corpus(records: List[Dict]) -> Tuple[List[str], List[str]]:
    """One entry per unique video. Returns (video_ids, video_paths)."""
    seen: Dict[str, str] = {}
    for r in records:
        seen.setdefault(r["video_id"], r["video_path"])
    video_ids = list(seen.keys())
    video_paths = [seen[v] for v in video_ids]
    return video_ids, video_paths


def rank_queries(
    query_embs: np.ndarray, corpus_embs: np.ndarray, batch_size: int = 512
) -> np.ndarray:
    """Return (Q, C) 1-based rank matrix via batched cosine similarity."""
    Q, C = query_embs.shape[0], corpus_embs.shape[0]
    ranks = np.empty((Q, C), dtype=np.int32)
    for start in range(0, Q, batch_size):
        end = min(start + batch_size, Q)
        sims = query_embs[start:end] @ corpus_embs.T
        order = np.argsort(-sims, axis=1)
        bs = end - start
        rank_mat = np.empty((bs, C), dtype=np.int32)
        rows = np.arange(bs)[:, None]
        rank_mat[rows, order] = np.arange(1, C + 1, dtype=np.int32)
        ranks[start:end] = rank_mat
    return ranks


def compute_metrics(ranks_of_gt: np.ndarray, top_k: int = 10) -> Dict:
    """Standard text-to-video retrieval metrics from per-query GT ranks."""
    r1 = float(np.mean(ranks_of_gt == 1))
    r5 = float(np.mean(ranks_of_gt <= 5))
    r10 = float(np.mean(ranks_of_gt <= top_k))
    med_r = float(np.median(ranks_of_gt))
    mean_r = float(np.mean(ranks_of_gt))
    mrr = float(np.mean(1.0 / ranks_of_gt))
    dcg = np.where(ranks_of_gt <= top_k, 1.0 / np.log2(ranks_of_gt + 1.0), 0.0)
    ndcg = float(np.mean(dcg))  # single relevant item -> IDCG = 1.0
    result = {
        "recall_at_1": r1,
        f"recall_at_{top_k}": r10,
        "median_rank": med_r,
        "mean_rank": mean_r,
        "mrr": mrr,
        f"ndcg_at_{top_k}": ndcg,
        "total_queries": int(ranks_of_gt.shape[0]),
        "corpus_size": None,
    }
    if top_k != 5:
        result["recall_at_5"] = r5
    return result


def t2v_gt_ranks(
    text_embs: np.ndarray,
    video_embs: np.ndarray,
    query_gt_video_ids: List[str],
    vid_to_corpus_idx: Dict[str, int],
) -> np.ndarray:
    """Text-to-video: each caption query ranks videos; GT = its own video.

    Returns (Q,) 1-based rank of the ground-truth video per text query.
    """
    rank_matrix = rank_queries(text_embs, video_embs)  # (Q, V)
    n_v = video_embs.shape[0]
    return np.array(
        [
            rank_matrix[q, vid_to_corpus_idx[vid]] if vid in vid_to_corpus_idx else n_v + 1
            for q, vid in enumerate(query_gt_video_ids)
        ],
        dtype=np.float32,
    )


def v2t_gt_ranks(
    video_embs: np.ndarray,
    text_embs: np.ndarray,
    corpus_video_ids: List[str],
    query_gt_video_ids: List[str],
) -> np.ndarray:
    """Video-to-text: each video query ranks captions; relevant = its captions.

    With multiple captions per video, uses the rank of the FIRST relevant
    caption (standard retrieval convention). Returns (V,) 1-based ranks.
    """
    rank_matrix = rank_queries(video_embs, text_embs)  # (V, Q)
    n_q = text_embs.shape[0]
    vid_to_text_idxs: Dict[str, List[int]] = defaultdict(list)
    for ti, vid in enumerate(query_gt_video_ids):
        vid_to_text_idxs[vid].append(ti)
    ranks = np.empty(len(corpus_video_ids), dtype=np.float32)
    for v, vid in enumerate(corpus_video_ids):
        rel = vid_to_text_idxs.get(vid, [])
        ranks[v] = min((rank_matrix[v, ti] for ti in rel), default=n_q + 1)
    return ranks


def print_summary(metrics: Dict, top_k: int, direction: str) -> None:
    label = "Text-to-Video" if direction == "t2v" else "Video-to-Text"
    qlabel = "captions" if direction == "t2v" else "videos"
    clabel = "videos" if direction == "t2v" else "captions"
    print(f"\n=== MSR-VTT {label} Retrieval ===")
    print(f"Queries : {metrics['total_queries']} {qlabel}   "
          f"Corpus : {metrics['corpus_size']} {clabel}   Top-K : {top_k}")
    print(f"R@1     : {metrics['recall_at_1']:.4f}")
    print(f"R@5     : {metrics['recall_at_5']:.4f}")
    print(f"R@{top_k:<4d}  : {metrics[f'recall_at_{top_k}']:.4f}")
    print(f"MedR    : {metrics['median_rank']:.1f}")
    print(f"MeanR   : {metrics['mean_rank']:.1f}")
    print(f"MRR     : {metrics['mrr']:.4f}")
    print(f"NDCG@{top_k} : {metrics[f'ndcg_at_{top_k}']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MSR-VTT video-text retrieval benchmark (CosmosEmbed1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", required=True,
                   help="Root model directory passed to CosmosEmbed1 (contains nvidia/Cosmos-Embed1-<variant>/).")
    p.add_argument("--variant", default="336p", choices=["224p", "336p", "448p"],
                   help="CosmosEmbed1 variant.")
    p.add_argument("--direction", choices=["t2v", "v2t", "both"], default="both",
                   help="Retrieval direction: text->video, video->text, or both.")
    p.add_argument("--split", default="test", help="MSR-VTT split.")
    p.add_argument("--config", default=None,
                   help="HF dataset config name (e.g. test_1k, train_9k). "
                        "Defaults to 'test_1k' when split='test', else required.")
    p.add_argument("--video-dir", default=None,
                   help="Local dir of MSR-VTT mp4s; if omitted, download from HF Hub.")
    p.add_argument("--top-k", type=int, default=10, help="K for Recall@K and NDCG@K.")
    p.add_argument("--limit", type=int, default=0,
                   help="Limit caption rows / queries (0 = all). For quick checks.")
    p.add_argument("--sample-fps", type=float, default=2.0,
                   help="Frame rate used to sample each video before embedding.")
    p.add_argument("--text-batch-size", type=int, default=64, help="Text embedding batch size.")
    p.add_argument("--video-batch-size", type=int, default=8, help="Video embedding batch size.")
    p.add_argument("--output", default=None, help="Path to write JSON results (optional).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # 1. Resolve videos
    if args.video_dir:
        video_dir = Path(args.video_dir).expanduser().resolve()
        if not video_dir.exists():
            LOG.error("--video-dir does not exist: %s", video_dir)
            return 2
    else:
        video_dir = ensure_msrvtt_videos(Path(tempfile.gettempdir()) / "msrvtt_videos")

    # 2. Load dataset (captions + resolved video paths)
    hf_config = args.config or ("test_1k" if args.split == "test" else None)
    if hf_config is None:
        LOG.error("--config is required for --split %s (e.g. train_9k, train_7k).", args.split)
        return 2
    records = load_msrvtt(
        split=args.split, video_dir=video_dir,
        config=hf_config,
        limit=args.limit if args.limit > 0 else None,
    )
    if not records:
        LOG.error("No MSR-VTT records resolved; nothing to evaluate.")
        return 1

    # 3. Build corpus of unique videos; queries are all caption rows
    corpus_video_ids, corpus_video_paths = build_corpus(records)
    vid_to_corpus_idx = {vid: i for i, vid in enumerate(corpus_video_ids)}
    query_captions = [r["caption"] for r in records]
    query_gt_video_ids = [r["video_id"] for r in records]
    LOG.info("Corpus: %d videos | Queries: %d captions",
             len(corpus_video_ids), len(query_captions))

    # 4. Embed
    LOG.info("Loading CosmosEmbed1-%s from %s", args.variant, args.model_dir)
    model = CosmosEmbed1(variant=args.variant, utils_only=False, model_dir=args.model_dir)
    model.setup()
    LOG.info("Embedding video corpus at %.1f fps...", args.sample_fps)
    corpus_embs, valid = embed_videos(
        model, corpus_video_paths, batch_size=args.video_batch_size, sample_fps=args.sample_fps
    )
    n_bad = int((~valid).sum())
    if n_bad:
        LOG.warning("%d/%d corpus videos failed to embed; dropping from corpus.", n_bad, len(valid))
        corpus_embs = corpus_embs[valid]
        corpus_video_ids = [vid for vid, v in zip(corpus_video_ids, valid) if v]
        vid_to_corpus_idx = {vid: i for i, vid in enumerate(corpus_video_ids)}
    LOG.info("Embedding text queries...")
    query_embs = embed_texts(model, query_captions, batch_size=args.text_batch_size)

    # 5. Rank + score in the requested direction(s)
    n_videos = len(corpus_video_ids)
    n_texts = len(query_captions)
    common = {
        "model": f"CosmosEmbed1-{args.variant}",
        "split": args.split,
        "top_k": args.top_k,
        "sample_fps": args.sample_fps,
        "failed_videos": n_bad,
    }
    directions = ["t2v", "v2t"] if args.direction == "both" else [args.direction]
    all_metrics: Dict[str, Dict] = {}

    for direction in directions:
        if direction == "t2v":
            LOG.info("Ranking %d caption queries against %d videos (t2v)...", n_texts, n_videos)
            ranks = t2v_gt_ranks(query_embs, corpus_embs, query_gt_video_ids, vid_to_corpus_idx)
            corpus_size = n_videos
        else:
            LOG.info("Ranking %d video queries against %d captions (v2t)...", n_videos, n_texts)
            ranks = v2t_gt_ranks(corpus_embs, query_embs, corpus_video_ids, query_gt_video_ids)
            corpus_size = n_texts

        metrics = compute_metrics(ranks, top_k=args.top_k)
        metrics["corpus_size"] = corpus_size
        metrics["direction"] = direction
        metrics.update(common)
        print_summary(metrics, top_k=args.top_k, direction=direction)
        all_metrics[direction] = metrics

    # 6. Persist
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = all_metrics[directions[0]] if len(directions) == 1 else all_metrics
        with out_path.open("w") as fh:
            json.dump(payload, fh, indent=2)
        LOG.info("Results written to %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
