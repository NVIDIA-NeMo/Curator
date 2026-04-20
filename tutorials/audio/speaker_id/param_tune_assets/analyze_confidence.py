"""Per-utterance speaker-ID confidence analysis on the LibriSpeech sweep.

Reproduces the *winning* clustering cell (BIRCH cosine floor 0.95, AHC 0.50)
and produces the artefacts referenced from PARAM_TUNE.md:

  * ``confidence_examples.json`` -
        ten typical ``correct`` and ten typical ``wrong`` per-utterance
        examples, each with the original manifest record and the silhouette
        confidence score.
  * ``confidence_threshold_sweep.csv`` -
        retention vs. cluster-purity vs. error-rate vs. coverage as we sweep
        the confidence threshold from 0.00 to 1.00.
  * ``confidence_threshold_sweep.png`` -
        plot of the sweep used in PARAM_TUNE.md.
  * ``conservative_threshold.json`` -
        the recommended "very conservative" threshold.

Definitions (silhouette score, matches Curator production)
----------------------------------------------------------

We use the **same** confidence formula that Curator's
``cluster_embeddings_large_scale`` writes into output manifests as
``confidence_score``, so this analysis and the production output are
directly comparable.  For every utterance ``i`` with predicted cluster
``k = pred(i)`` and surviving cluster-centroids ``{c_1, ..., c_K}``:

    a = cos(emb_i, c_k)                    # self-similarity to own centroid
    b = max_{k' != k} cos(emb_i, c_{k'})   # best similarity to any other
    confidence_i = clamp((a - b) / max(a, b),  0,  1)

This is the cluster-silhouette score adapted to cosine distance and
centroids.  Range is ``[0, 1]``: 1.0 = utterance is right at its own
centroid and far from all others; 0.0 = utterance is no closer to its own
centroid than to the next-best one (i.e. an ambiguous frontier point).

Centroids are L2-normalised means of the L2-normalised utterance
embeddings inside each cluster.  Singleton clusters and dropped utterances
get ``confidence = 0.0`` (no separation signal).

Majority-voted oracle
---------------------

Why majority-vote is the right oracle here: the clustering doesn't know the
LibriSpeech speaker IDs, only that each cluster *should* be one speaker.
Comparing each utterance to its cluster's plurality label is the standard
``cluster purity`` decomposition (and matches the purity numbers in
PARAM_TUNE.md  4.4).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Helpers (intentionally vendored from the tuning script for self-containment)
# --------------------------------------------------------------------------
def center_global(embeddings: np.ndarray) -> np.ndarray:
    return embeddings - embeddings.mean(axis=0, keepdims=True)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32, copy=False)


# --------------------------------------------------------------------------
def reproduce_best_clustering(
    labels_path: str, repo_path: str, birch_floor: float, ahc_thr: float,
    branching_factor: int = 50, partial_fit_batch: int = 50_000,
    assign_tile: int = 16_384,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (cut_ids, normed_embs, pred, true)."""
    print(f"  Loading labels from {labels_path}", flush=True)
    data = np.load(labels_path, allow_pickle=True)
    cut_ids = np.asarray(data["cut_ids"], dtype=object)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    true = np.asarray(data["true_speakers"], dtype=np.int64)
    print(f"    N={embeddings.shape[0]:,}, D={embeddings.shape[1]}", flush=True)

    print("  Applying center_global (matches Curator default)", flush=True)
    embeddings = center_global(embeddings)

    if repo_path and os.path.isdir(repo_path) and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from nemo_curator.stages.audio.speaker_id.clustering import large_scale_clustering_and_scoring as ls  # noqa: E402

    print("  L2-normalising", flush=True)
    normed = l2_normalize(embeddings)

    radius = float(np.sqrt(2.0 * (1.0 - birch_floor)))
    print(f"  Building BIRCH (floor={birch_floor}, radius={radius:.4f})", flush=True)
    birch = ls._build_birch_tree(
        normed,
        birch_threshold=radius,
        branching_factor=branching_factor,
        partial_fit_batch=partial_fit_batch,
    )
    leaves = l2_normalize(np.asarray(birch.subcluster_centers_, dtype=np.float32))
    print(f"  BIRCH produced {leaves.shape[0]:,} leaves", flush=True)

    print("  Assigning each utt to its nearest leaf", flush=True)
    leaf_idx = ls._assign_to_nearest_leaf(normed, leaves, tile=assign_tile)

    print("  Running AHC on leaf centroids", flush=True)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    sim = leaves @ leaves.T
    np.clip(sim, -1.0, 1.0, out=sim)
    cond = squareform(1.0 - sim, checks=False)
    Z = linkage(cond, method="average")
    centroid_labels = fcluster(Z, t=1.0 - ahc_thr, criterion="distance")
    pred = centroid_labels[leaf_idx].astype(np.int64, copy=False)
    print(f"  Got {len(set(pred.tolist())):,} predicted clusters", flush=True)
    return cut_ids, normed, pred, true


# --------------------------------------------------------------------------
def per_utt_confidence(
    normed: np.ndarray, pred: np.ndarray, tile: int = 16_384,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-utterance silhouette-style confidence.

    For each utterance i with predicted cluster ``k = pred(i)``:

        a = cos(normed_i, centroid_k)
        b = max over k' != k of cos(normed_i, centroid_k')
        confidence = clamp((a - b) / max(a, b),  0,  1)

    Singletons and any cluster of size < 2 get confidence ``0.0`` (no
    separation signal).  Returns ``(confidence per utt, cluster index per
    utt)`` where the second array is a dense 0..K-1 mapping suitable for
    downstream centroid-indexed lookups.
    """
    print("  Computing per-cluster centroids and silhouette confidences", flush=True)
    uniq, inv = np.unique(pred, return_inverse=True)
    n_clusters = uniq.shape[0]
    sums = np.zeros((n_clusters, normed.shape[1]), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)
    np.add.at(sums, inv, normed)
    np.add.at(counts, inv, 1)
    centroids = (sums / np.maximum(counts[:, None], 1)).astype(np.float32)
    cnorm = np.linalg.norm(centroids, axis=1, keepdims=True)
    cnorm[cnorm == 0] = 1.0
    centroids /= cnorm

    n = normed.shape[0]
    confidences = np.zeros(n, dtype=np.float32)
    K = centroids.shape[0]
    print(f"    K={K} centroids, tiling N={n:,} in chunks of {tile:,}",
          flush=True)
    for start in range(0, n, tile):
        end = min(start + tile, n)
        sims = normed[start:end] @ centroids.T  # (B, K)
        np.clip(sims, -1.0, 1.0, out=sims)

        own_k = inv[start:end]                  # (B,)
        rows = np.arange(end - start)
        a = sims[rows, own_k]                   # self-cosine

        # Mask own centroid before taking the max over the other ones.
        sims[rows, own_k] = -2.0
        b = sims.max(axis=1)                    # next-best cosine

        denom = np.maximum(a, b)
        nonzero = denom > 0.0
        raw = np.zeros(end - start, dtype=np.float32)
        raw[nonzero] = (a[nonzero] - b[nonzero]) / denom[nonzero]
        np.clip(raw, 0.0, 1.0, out=raw)

        # Singletons (cluster size < 2): confidence is meaningless, force 0.
        is_singleton = counts[own_k] < 2
        raw[is_singleton] = 0.0

        confidences[start:end] = raw

    return confidences, inv


def majority_label_per_cluster(
    pred_dense: np.ndarray, true: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each cluster, return (majority-true-label, cluster size).

    pred_dense is an integer in ``[0, n_clusters)``.
    """
    n_clusters = int(pred_dense.max()) + 1
    majority = np.full(n_clusters, -1, dtype=np.int64)
    sizes = np.zeros(n_clusters, dtype=np.int64)
    order = np.argsort(pred_dense, kind="stable")
    p_sorted = pred_dense[order]
    t_sorted = true[order]
    boundaries = np.flatnonzero(np.diff(p_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(p_sorted)]])
    for s, e in zip(starts, ends):
        c = int(p_sorted[s])
        sizes[c] = e - s
        # bincount over the small slice -- much faster than Counter at scale.
        bc = np.bincount(t_sorted[s:e].astype(np.int64))
        majority[c] = int(bc.argmax())
    return majority, sizes


# --------------------------------------------------------------------------
def find_manifest_record(manifest_glob: str, audio_filepath: str) -> Dict:
    """Return the manifest record whose ``audio_filepath`` matches.

    Linear scan; called once per example only.  ``manifest_glob`` follows
    the NeMo brace syntax (``manifest_{0..511}.json``).
    """
    import glob
    import re
    paths: List[str] = []
    m = re.search(r"\{(\d+)\.\.(\d+)\}", manifest_glob)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        prefix = manifest_glob[: m.start()]
        suffix = manifest_glob[m.end():]
        paths = [f"{prefix}{i}{suffix}" for i in range(a, b + 1)]
    else:
        paths = sorted(glob.glob(manifest_glob))
    for mp in paths:
        if not os.path.isfile(mp):
            continue
        with open(mp, "r", encoding="utf-8") as f:
            for raw in f:
                if audio_filepath not in raw:
                    continue
                rec = json.loads(raw)
                if rec.get("audio_filepath") == audio_filepath:
                    return rec
    return {"audio_filepath": audio_filepath, "_note": "manifest record not found"}


def index_manifest_records(manifest_glob: str, wanted: set[str]) -> Dict[str, Dict]:
    """Single pass over the 512 shards, return only the records we want."""
    import glob
    import re
    paths: List[str] = []
    m = re.search(r"\{(\d+)\.\.(\d+)\}", manifest_glob)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        prefix = manifest_glob[: m.start()]
        suffix = manifest_glob[m.end():]
        paths = [f"{prefix}{i}{suffix}" for i in range(a, b + 1)]
    else:
        paths = sorted(glob.glob(manifest_glob))
    out: Dict[str, Dict] = {}
    remaining = set(wanted)
    for mp in paths:
        if not remaining:
            break
        if not os.path.isfile(mp):
            continue
        with open(mp, "r", encoding="utf-8") as f:
            for raw in f:
                if not remaining:
                    break
                rec = json.loads(raw)
                afp = rec.get("audio_filepath")
                if afp in remaining:
                    out[afp] = rec
                    remaining.discard(afp)
    return out


# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True,
                   help="Path to labels.npz from the tune phase.")
    p.add_argument("--repo", required=False, default="",
                   help=(
                       "DEPRECATED: legacy path to speaker_id_for_asr_data. "
                       "The clustering module now lives in Curator at "
                       "nemo_curator.stages.audio.speaker_id.clustering."
                   ))
    p.add_argument("--manifest_glob", required=True,
                   help=(
                       "NeMo-style glob over per-shard manifests, e.g. "
                       "'${DATA_ROOT}/raw_sharded_manifests/manifest_{0..511}.json'. "
                       "On NVIDIA clusters ${DATA_ROOT} is "
                       "/lustre/fs11/.../tarred_train (cs-oci-ord) or "
                       "/lustre/fs12/.../tarred_train (draco-oci-iad)."
                   ))
    p.add_argument("--out", required=True,
                   help="Output directory for artefacts.")
    p.add_argument("--birch_floor", type=float, default=0.95)
    p.add_argument("--ahc_thr", type=float, default=0.50)
    p.add_argument("--n_examples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cut_ids, normed, pred_raw, true = reproduce_best_clustering(
        args.labels, args.repo, args.birch_floor, args.ahc_thr,
    )

    # Densify pred labels to [0, K).
    uniq, pred = np.unique(pred_raw, return_inverse=True)
    n_clusters = uniq.shape[0]
    print(f"\nDense cluster id range: [0, {n_clusters - 1}]")

    confidences, _cluster_idx = per_utt_confidence(normed, pred)
    majority, sizes = majority_label_per_cluster(pred, true)
    correct_mask = (majority[pred] == true)

    err_count = int((~correct_mask).sum())
    n = correct_mask.shape[0]
    print(f"\nOverall: {err_count:,}/{n:,} mismatch with cluster majority "
          f"({100 * err_count / n:.3f} %)")

    # ---------- 1. Confidence-threshold sweep --------------------------------
    print("\n[1/3] Confidence-threshold sweep")
    # Silhouette score lives in [0, 1].  We sample fine-grained.
    grid = np.round(np.arange(0.00, 1.001, 0.01), 4)
    rows = []
    for thr in grid:
        keep = confidences >= thr
        kept = int(keep.sum())
        kept_corr = int(correct_mask[keep].sum())
        # Per-true-speaker coverage: how many of the 2,337 speakers still
        # have at least one (correct) utterance kept after filtering.
        kept_corr_mask = keep & correct_mask
        spk_kept = int(np.unique(true[kept_corr_mask]).shape[0]) if kept_corr_mask.any() else 0
        rows.append({
            "confidence_threshold": float(thr),
            "kept_utts": kept,
            "kept_correct": kept_corr,
            "kept_wrong": kept - kept_corr,
            "retention_pct": 100.0 * kept / n,
            "purity_pct": 100.0 * kept_corr / max(kept, 1),
            "speaker_coverage_pct": 100.0 * spk_kept / 2337,
        })

    csv_path = os.path.join(args.out, "confidence_threshold_sweep.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])
                             for k in keys) + "\n")
    print(f"  wrote {csv_path}")

    # Print a small table for the doc.
    print("\n  Confidence sweep (selected rows):")
    print(f"  {'thr':>5}  {'kept':>9}  {'ret%':>6}  {'pur%':>7}  {'wrong':>7}  {'spk_cov%':>9}")
    table_thresholds = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
                        0.80, 0.85, 0.90, 0.95, 0.99)
    for r in rows:
        if r["confidence_threshold"] in table_thresholds:
            print(f"  {r['confidence_threshold']:5.2f}  {r['kept_utts']:9,d}  "
                  f"{r['retention_pct']:6.2f}  {r['purity_pct']:7.4f}  "
                  f"{r['kept_wrong']:7,d}  {r['speaker_coverage_pct']:9.2f}")

    # ---------- 2. Pick the recommended conservative threshold ---------------
    # Rule: smallest silhouette cutoff that yields >= 99.95% purity AND keeps
    # at least 80% of true speakers.  This is intentionally aggressive to
    # discard borderline samples.
    target_purity = 99.95
    min_spk_cov = 80.0
    chosen = None
    for r in rows:
        if r["purity_pct"] >= target_purity and r["speaker_coverage_pct"] >= min_spk_cov:
            chosen = r
            break
    if chosen is None:
        chosen = max(rows, key=lambda r: (r["purity_pct"], r["speaker_coverage_pct"]))
    print(f"\n  Recommended conservative confidence threshold: "
          f"{chosen['confidence_threshold']:.2f}  "
          f"(retention={chosen['retention_pct']:.2f}%, "
          f"purity={chosen['purity_pct']:.4f}%, "
          f"speaker coverage={chosen['speaker_coverage_pct']:.2f}%)")
    with open(os.path.join(args.out, "conservative_threshold.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "rule": ("smallest threshold s.t. cluster-majority purity >= "
                     f"{target_purity}% AND speaker_coverage >= "
                     f"{min_spk_cov}%"),
            "chosen": chosen,
            "best_clustering_cell": {
                "ahc_threshold": args.ahc_thr,
                "birch_cosine_floor": args.birch_floor,
            },
        }, f, indent=2)

    # ---------- 3. Plot -----------------------------------------------------
    print("\n[2/3] Confidence sweep plot")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(10, 5))
    thrs = np.array([r["confidence_threshold"] for r in rows])
    ret = np.array([r["retention_pct"] for r in rows])
    pur = np.array([r["purity_pct"] for r in rows])
    cov = np.array([r["speaker_coverage_pct"] for r in rows])

    color1, color2, color3 = "#1f3b73", "#c0392b", "#27ae60"
    ax1.plot(thrs, ret, "-o", color=color1, ms=3, label="Retention (kept / N)")
    ax1.plot(thrs, cov, "-s", color=color3, ms=3,
             label="Speaker coverage (true speakers w/ >= 1 correct kept utt)")
    ax1.set_xlabel("Per-utterance silhouette confidence threshold  "
                   "((a-b)/max(a,b),  a=self-centroid cos,  b=best-other cos)")
    ax1.set_ylabel("Retention / Speaker coverage  (%)")
    ax1.set_ylim(0, 105)
    ax1.set_xlim(0.0, 1.0)
    ax1.grid(alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(thrs, pur, "-^", color=color2, ms=3,
             label="Purity (cluster-majority agreement)")
    ax2.set_ylabel("Purity  (%)", color=color2)
    ax2.tick_params(axis="y", colors=color2)
    ax2.set_ylim(95, 100.05)

    # Annotations for chosen and best-F1 baseline.
    ax1.axvline(chosen["confidence_threshold"], color="black", lw=1.0, ls="--")
    ax1.text(
        chosen["confidence_threshold"] + 0.005, 50,
        f"  recommended\n  conservative\n  thr = {chosen['confidence_threshold']:.2f}\n"
        f"  ret = {chosen['retention_pct']:.1f}%\n"
        f"  pur = {chosen['purity_pct']:.3f}%\n"
        f"  spk cov = {chosen['speaker_coverage_pct']:.1f}%",
        fontsize=9, va="center")

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="lower left", fontsize=9)
    ax1.set_title(
        "LibriSpeech 280k - filtering by per-utterance silhouette confidence\n"
        "(BIRCH floor 0.95, AHC 0.50; baseline purity at thr=0 is "
        f"{100 * (correct_mask.sum() / n):.3f}%)"
    )
    plot_path = os.path.join(args.out, "confidence_threshold_sweep.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {plot_path}")

    # ---------- 4. Pick examples --------------------------------------------
    print("\n[3/3] Picking representative correct/wrong examples")
    # "Wrong" examples that span the confidence range (so the doc shows the
    # whole story, not just the obvious low-confidence ones).
    wrong_idx = np.flatnonzero(~correct_mask)
    correct_idx = np.flatnonzero(correct_mask)

    # Buckets of the silhouette confidence axis for wrong samples (0..1), so
    # we get one per band.  Silhouette values for LibriSpeech land mostly in
    # [0.30, 0.75] -- bands chosen to cover that range and the long tails.
    bands = [(0.0, 0.20), (0.20, 0.40), (0.40, 0.55), (0.55, 0.70), (0.70, 1.01)]
    chosen_wrong: List[int] = []
    for lo, hi in bands:
        m = (confidences[wrong_idx] >= lo) & (confidences[wrong_idx] < hi)
        pool = wrong_idx[m]
        if pool.shape[0] == 0:
            continue
        pick = rng.choice(pool, size=min(2, pool.shape[0]), replace=False)
        chosen_wrong.extend(int(x) for x in pick)
    chosen_wrong = chosen_wrong[: args.n_examples]

    # Correct examples likewise spread across the confidence range.
    chosen_correct: List[int] = []
    for lo, hi in bands:
        m = (confidences[correct_idx] >= lo) & (confidences[correct_idx] < hi)
        pool = correct_idx[m]
        if pool.shape[0] == 0:
            continue
        pick = rng.choice(pool, size=min(2, pool.shape[0]), replace=False)
        chosen_correct.extend(int(x) for x in pick)
    chosen_correct = chosen_correct[: args.n_examples]

    wanted_afps = set(str(cut_ids[i]) for i in chosen_wrong + chosen_correct)
    print(f"  Looking up {len(wanted_afps)} manifest records...", flush=True)
    records = index_manifest_records(args.manifest_glob, wanted_afps)
    print(f"  Found {len(records)} / {len(wanted_afps)} records.", flush=True)

    # Build the JSON payload.  We carry the full original manifest record
    # plus the analysis annotations.
    def example_dict(i: int) -> Dict:
        afp = str(cut_ids[i])
        rec = records.get(afp, {"audio_filepath": afp, "_note": "manifest record not found"})
        c = int(pred[i])
        true_int = int(true[i])
        maj_int = int(majority[c])
        # Map int -> string speaker id via the labels.npz table.
        return {
            "audio_filepath": afp,
            "manifest_record": rec,
            "true_speaker_id": _spk_str(args.labels, true_int),
            "cluster_majority_speaker_id": _spk_str(args.labels, maj_int),
            "cluster_id": c,
            "cluster_size": int(sizes[c]),
            "speaker_id_confidence": float(round(float(confidences[i]), 4)),
            "is_correct": bool(correct_mask[i]),
        }

    payload = {
        "best_clustering_cell": {
            "ahc_threshold": args.ahc_thr,
            "birch_cosine_floor": args.birch_floor,
            "n_predicted_clusters": int(n_clusters),
            "n_true_speakers": int(np.unique(true).shape[0]),
            "overall_purity_pct": float(round(100.0 * correct_mask.mean(), 4)),
        },
        "scoring_definition": {
            "confidence": ("silhouette-style score in [0, 1]:  "
                            "(a - b) / max(a, b)  where  "
                            "a = cos(emb_i, centroid_pred(i))  and  "
                            "b = max over k != pred(i) of "
                            "cos(emb_i, centroid_k).  Centroids are "
                            "L2-normalised means of the L2-normalised "
                            "(centred) utterance embeddings inside each "
                            "predicted cluster.  Same field that Curator's "
                            "cluster_embeddings_large_scale writes into "
                            "output manifests as confidence_score."),
            "is_correct": ("the utterance's true LibriSpeech speaker id "
                            "equals the plurality (most-frequent) true "
                            "speaker id within its predicted cluster"),
        },
        "wrong_examples": [example_dict(i) for i in chosen_wrong],
        "correct_examples": [example_dict(i) for i in chosen_correct],
        "recommended_conservative_threshold": chosen,
    }
    out_json = os.path.join(args.out, "confidence_examples.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {out_json}")
    return 0


_spk_cache: Dict[str, np.ndarray] = {}


def _spk_str(labels_path: str, int_id: int) -> str:
    """Decode integer speaker id back to LibriSpeech string id."""
    if labels_path not in _spk_cache:
        d = np.load(labels_path, allow_pickle=True)
        _spk_cache[labels_path] = np.asarray(d["speaker_id_str"], dtype=object)
    if int_id < 0:
        return ""
    arr = _spk_cache[labels_path]
    return str(arr[int_id]) if int_id < arr.shape[0] else f"<unknown:{int_id}>"


if __name__ == "__main__":
    raise SystemExit(main())
