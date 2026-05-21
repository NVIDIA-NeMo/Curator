"""Codename + on-disk config sidecar for the speaker-clustering pipeline.

Every clustering run leaves a single ``cluster_config.json`` file next to its
``clusters.jsonl`` so that downstream consumers (and humans, six months
later) can answer one question without having to re-derive it from logs:

    "What configuration produced these cluster ids?"

Codename
--------

We call the configuration family **SCOTCH** -- *Scalable Centroid-based
Two-stage Clustering with Hierarchical refinement*.  Concretely:

    BIRCH (streaming, cosine-floor) -> per-utt assignment to nearest leaf
        -> AHC on the leaf centroids (cosine, average linkage)
        -> min-cluster-size purity filter
        -> silhouette-based per-utt confidence in [0, 1]

The same backend can also degenerate to a single-stage AHC when the
dataset is small enough (the ``standard`` backend path); we still call
that **SCOTCH-mini** so that every artefact in the result tree carries a
recognisable family name.

Versioning
----------

* ``CONFIG_FAMILY``      -- ``"SCOTCH"``.  Never changes.
* ``CONFIG_VERSION``     -- bumped whenever the *algorithm* (not the
  parameter values) changes in a way that can move cluster ids.
* ``CONFIG_PRESET``      -- a short tag for a *named parameter set*.  The
  default preset, ``"librispeech-2026-04"``, is the one tuned in
  ``PARAM_TUNE.md``.  Custom runs use ``"custom"``.
* ``config_id``          -- the human-readable, machine-greppable
  identifier written into every JSON sidecar:

      ``SCOTCH-v1.standard.librispeech-2026-04``
      ``SCOTCH-v1.large_scale.custom``

Use :func:`build_cluster_config` to assemble the dict and
:func:`write_cluster_config` to persist it.
"""
from __future__ import annotations

import json
import logging
import math
import os
import platform
import socket
import sys
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Codename / version constants -- bump CONFIG_VERSION when the algorithm
# changes in a way that can move cluster ids.  Bump CONFIG_PRESET (or
# add a new one) when tuned threshold values change.
# --------------------------------------------------------------------------
CONFIG_FAMILY = "SCOTCH"
CONFIG_VERSION = "v1"
DEFAULT_PRESET = "librispeech-2026-04"
SIDECAR_FILENAME = "cluster_config.json"

# Reference values that match the preset.  These are documented in
# PARAM_TUNE.md  3 and are the ones the on-disk sidecar will record as
# the "expected" values for the preset.  A run is still tagged with the
# preset name even when the user overrides individual values, but the
# sidecar carries an ``overrides`` block so the difference is visible.
PRESETS: Dict[str, Dict[str, Any]] = {
    "librispeech-2026-04": {
        "cluster_threshold": 0.50,            # AHC cosine cutoff
        "cluster_linkage": "average",
        "min_cluster_size": 30,               # production default
        "birch_cosine_floor": 0.95,           # -> radius = sqrt(2*(1-0.95)) = 0.3162
        "birch_branching_factor": 50,
        "birch_partial_fit_batch": 50_000,
        "assign_tile": 16_384,
        "embedding_normalization": "center_global",
        "tuning_corpus": "LibriSpeech-train-960h",
        "tuning_metric": "B-cubed F1",
        "tuning_score": 0.9862,
        "tuning_doc": "tutorials/audio/speaker_id/PARAM_TUNE.md",
    },
    "custom": {
        # Empty -- "custom" means "the user knows what they're doing".
    },
}


def cosine_floor_to_birch_radius(cos_floor: float) -> float:
    """Convert a cosine floor in ``[-1, 1]`` to the equivalent Euclidean

    radius on the unit sphere (``r = sqrt(2 * (1 - c))``)."""
    return math.sqrt(2.0 * (1.0 - cos_floor))


def make_config_id(backend: str, preset: str = DEFAULT_PRESET) -> str:
    """Return e.g. ``"SCOTCH-v1.large_scale.librispeech-2026-04"``."""
    if backend not in ("standard", "large_scale"):
        raise ValueError(f"Unknown backend: {backend!r}")
    return f"{CONFIG_FAMILY}-{CONFIG_VERSION}.{backend}.{preset}"


# --------------------------------------------------------------------------
# Config builder
# --------------------------------------------------------------------------
def build_cluster_config(
    backend: str,
    preset: str = DEFAULT_PRESET,
    *,
    cluster_threshold: float,
    cluster_linkage: str,
    min_cluster_size: int,
    n_input: int,
    embedding_dim: int,
    embedding_normalization: str = "none",
    confidence_enabled: bool = True,
    # --- BIRCH-only knobs (ignored by the standard backend, but always
    # recorded so the sidecar shape is identical across backends).
    birch_cosine_floor: Optional[float] = None,
    birch_radius: Optional[float] = None,
    birch_branching_factor: Optional[int] = None,
    birch_partial_fit_batch: Optional[int] = None,
    assign_tile: Optional[int] = None,
    n_leaf_subclusters: Optional[int] = None,
    # --- Outcome metrics from the run itself (filled in after clustering).
    n_clusters_raw: Optional[int] = None,
    n_clusters_kept: Optional[int] = None,
    n_utts_kept: Optional[int] = None,
    n_utts_dropped: Optional[int] = None,
    runtime_seconds: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the JSON-able config dict written next to ``clusters.jsonl``."""
    config_id = make_config_id(backend, preset)
    expected = PRESETS.get(preset, {})

    # Compute a small overrides block so the sidecar makes "what's
    # different from the named preset" trivially auditable.
    overrides: Dict[str, Any] = {}
    actual = {
        "cluster_threshold": cluster_threshold,
        "cluster_linkage": cluster_linkage,
        "min_cluster_size": min_cluster_size,
        "embedding_normalization": embedding_normalization,
        "birch_cosine_floor": birch_cosine_floor,
        "birch_branching_factor": birch_branching_factor,
        "birch_partial_fit_batch": birch_partial_fit_batch,
        "assign_tile": assign_tile,
    }
    for k, expected_v in expected.items():
        if k in actual and actual[k] is not None and actual[k] != expected_v:
            overrides[k] = {"preset": expected_v, "actual": actual[k]}

    cfg: Dict[str, Any] = {
        # ---- identity
        "config_id": config_id,
        "config_family": CONFIG_FAMILY,
        "config_version": CONFIG_VERSION,
        "preset": preset,
        "backend": backend,
        # ---- algorithm
        "algorithm": {
            "stage1": (
                "BIRCH (streaming partial_fit on L2-normalised embeddings)"
                if backend == "large_scale" else "(none)"
            ),
            "stage2": "AHC on centroids (cosine, average linkage)",
            "post1": (
                f"min_cluster_size={min_cluster_size} purity filter"
                if min_cluster_size and min_cluster_size > 1 else "no purity filter"
            ),
            "post2": (
                "silhouette per-utterance confidence in [0, 1] "
                "((a-b)/max(a,b), a=self-centroid cos, b=best-other cos)"
                if confidence_enabled else "(no confidence)"
            ),
        },
        # ---- the actual numerical knobs in effect for this run
        "parameters": {
            "cluster_threshold": cluster_threshold,
            "cluster_linkage": cluster_linkage,
            "min_cluster_size": min_cluster_size,
            "embedding_normalization": embedding_normalization,
            "confidence_enabled": confidence_enabled,
            "birch": {
                "cosine_floor": birch_cosine_floor,
                "radius": birch_radius,
                "branching_factor": birch_branching_factor,
                "partial_fit_batch": birch_partial_fit_batch,
                "assign_tile": assign_tile,
            } if backend == "large_scale" else None,
        },
        "preset_expected": expected,
        "overrides": overrides,
        # ---- outcome
        "outcome": {
            "n_input": n_input,
            "embedding_dim": embedding_dim,
            "n_leaf_subclusters": n_leaf_subclusters,
            "n_clusters_raw": n_clusters_raw,
            "n_clusters_kept": n_clusters_kept,
            "n_utts_kept": n_utts_kept,
            "n_utts_dropped": n_utts_dropped,
            "runtime_seconds": runtime_seconds,
        },
        # ---- provenance (so we can later reproduce a result)
        "provenance": {
            "host": socket.gethostname(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": list(sys.argv),
        },
    }
    if extra:
        cfg["extra"] = extra
    return cfg


def write_cluster_config(output_dir: str, config: Dict[str, Any]) -> str:
    """Write the config dict to ``output_dir/cluster_config.json``."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, SIDECAR_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Wrote cluster config sidecar: %s  [%s]",
                path, config.get("config_id", "<no id>"))
    return path
