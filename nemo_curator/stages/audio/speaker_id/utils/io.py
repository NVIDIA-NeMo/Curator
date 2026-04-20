"""I/O utilities: save/load embeddings, wav.scp, utt lists."""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np


def save_embeddings(
    embeddings: np.ndarray,
    utt_ids: List[str],
    output_dir: str,
    suffix: str = "",
) -> Tuple[str, str]:
    """Save embeddings and utterance IDs to disk.

    Returns (emb_path, utt_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    emb_path = os.path.join(output_dir, f"embeddings{suffix}.npy")
    utt_path = os.path.join(output_dir, f"utt_names{suffix}.txt")

    np.save(emb_path, embeddings)
    with open(utt_path, "w") as f:
        f.write("\n".join(utt_ids) + "\n")

    return emb_path, utt_path


def load_embeddings(output_dir: str, suffix: str = "") -> Tuple[np.ndarray, List[str]]:
    """Load embeddings and utterance IDs from disk."""
    emb_path = os.path.join(output_dir, f"embeddings{suffix}.npy")
    utt_path = os.path.join(output_dir, f"utt_names{suffix}.txt")

    embeddings = np.load(emb_path)
    with open(utt_path) as f:
        utt_ids = [line.strip() for line in f if line.strip()]

    return embeddings, utt_ids


def merge_embedding_shards(output_dir: str, num_shards: int) -> Tuple[str, str]:
    """Merge per-GPU embedding shards into a single file.

    Expects files named ``embeddings_gpu{i}.npy`` and ``utt_names_gpu{i}.txt``.
    Returns (merged_emb_path, merged_utt_path).
    """
    all_embs = []
    all_utt_ids = []

    for i in range(num_shards):
        embs, utts = load_embeddings(output_dir, suffix=f"_gpu{i}")
        all_embs.append(embs)
        all_utt_ids.extend(utts)

    merged = np.concatenate(all_embs, axis=0)
    return save_embeddings(merged, all_utt_ids, output_dir)
