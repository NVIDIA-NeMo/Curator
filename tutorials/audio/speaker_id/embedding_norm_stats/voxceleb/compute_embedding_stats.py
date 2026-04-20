#!/usr/bin/env python3
"""Compute and save global mean / variance / std for a cohort of speaker embeddings.

Use this on **VoxCeleb2 dev** (or any) embeddings so you can later center or
whiten YouTube-scraped embeddings in the same model space.

Outputs (under --out_dir):
  cohort_mean.npy    shape (D,)  — mean of each dimension
  cohort_var.npy     shape (D,)  — population variance (ddof=0) per dimension
  cohort_std.npy     shape (D,)  — sqrt(var), for scaling (add eps in your code)
  meta.json          counts, paths, optional note

Inputs (one of):
  --xvector_scp      Kaldi xvector.scp (requires kaldiio)
  --embeddings_npy   float32 array shape (N, D) or .npz with key 'embeddings'

Example (WeSpeaker ResNet after vox2_dev extract):
  python compute_embedding_stats.py \\
    --xvector_scp /path/to/exp/embeddings/vox2_dev/xvector.scp \\
    --out_dir Curator/tutorials/audio/speaker_id/embedding_norm_stats/voxceleb/wespeaker_resnet293_lm \\
    --label wespeaker_resnet293_lm_vox2dev

Example (TitaNet / NeMo, same layout):
  python compute_embedding_stats.py \\
    --xvector_scp /path/to/titanet/exp/embeddings/vox2_dev/xvector.scp \\
    --out_dir Curator/tutorials/audio/speaker_id/embedding_norm_stats/voxceleb/nemo_titanet_large \\
    --label nemo_titanet_large_vox2dev
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone


def load_embeddings_from_scp(scp_path: str) -> tuple[list[str], "np.ndarray"]:
    import numpy as np

    try:
        import kaldiio
    except ImportError as e:
        raise SystemExit(
            "kaldiio is required for --xvector_scp.  pip install kaldiio"
        ) from e

    utts: list[str] = []
    rows: list[np.ndarray] = []
    for utt, vec in kaldiio.load_scp_sequential(scp_path):
        utts.append(utt)
        rows.append(np.asarray(vec, dtype=np.float64).ravel())
    if not rows:
        raise ValueError(f"No vectors read from {scp_path}")
    emb = np.stack(rows, axis=0)
    return utts, emb


def load_embeddings_npy(path: str) -> "np.ndarray":
    import numpy as np

    if path.endswith(".npz"):
        z = np.load(path)
        if "embeddings" in z.files:
            x = z["embeddings"]
        else:
            x = z[z.files[0]]
    else:
        x = np.load(path)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected (N, D) array, got shape {x.shape}")
    return x


def compute_stats(emb: "np.ndarray", ddof: int = 0) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    import numpy as np

    mean = emb.mean(axis=0)
    var = emb.var(axis=0, ddof=ddof)
    std = np.sqrt(np.maximum(var, 0.0))
    return mean.astype(np.float32), var.astype(np.float32), std.astype(np.float32)


def apply_cohort_affine(
    embeddings: "np.ndarray",
    mean: "np.ndarray",
    std: "np.ndarray | None" = None,
    eps: float = 1e-8,
    l2_normalize_rows: bool = False,
) -> "np.ndarray":
    """Subtract cohort mean; optionally divide by cohort std (per dimension)."""
    import numpy as np

    x = np.asarray(embeddings, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    if std is not None:
        s = np.asarray(std, dtype=np.float64) + eps
        x = x / s
    x = x.astype(np.float32)
    if l2_normalize_rows:
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n = np.maximum(n, eps)
        x = (x / n).astype(np.float32)
    return x


def main():
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--xvector_scp", help="Kaldi xvector.scp from WeSpeaker/NeMo extract")
    src.add_argument("--embeddings_npy", help=".npy (N,D) or .npz with embeddings")
    ap.add_argument("--out_dir", required=True, help="Directory to write .npy + meta.json")
    ap.add_argument(
        "--label",
        default="cohort",
        help="Short name stored in meta.json (e.g. model + dataset)",
    )
    ap.add_argument(
        "--ddof",
        type=int,
        default=0,
        help="Variance ddof (0=population, 1=sample)",
    )
    args = ap.parse_args()

    if args.xvector_scp:
        _, emb = load_embeddings_from_scp(args.xvector_scp)
        source = {"type": "xvector_scp", "path": os.path.abspath(args.xvector_scp)}
    else:
        emb = load_embeddings_npy(args.embeddings_npy)
        source = {"type": "npy", "path": os.path.abspath(args.embeddings_npy)}

    n, d = emb.shape
    mean, var, std = compute_stats(emb, ddof=args.ddof)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "cohort_mean.npy"), mean)
    np.save(os.path.join(args.out_dir, "cohort_var.npy"), var)
    np.save(os.path.join(args.out_dir, "cohort_std.npy"), std)

    meta = {
        "label": args.label,
        "n_vectors": int(n),
        "embedding_dim": int(d),
        "variance_ddof": int(args.ddof),
        "source": source,
        "files": ["cohort_mean.npy", "cohort_var.npy", "cohort_std.npy"],
        "description": (
            "cohort_mean: subtract from embeddings (same as WeSpeaker cal_mean). "
            "cohort_std: optional per-dim scaling (whitening-lite); use small eps when dividing."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote stats for N={n}, D={d} to {args.out_dir}")
    print(f"  mean L2 norm: {np.linalg.norm(mean):.4f}")
    print(f"  mean std across dims: {std.mean():.6f}")


if __name__ == "__main__":
    main()
