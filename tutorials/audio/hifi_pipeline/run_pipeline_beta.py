# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Beta variant of the HIFI pipeline.

Differences vs ``run_pipeline.py``:

* **ASR replaces the 3-pass transcription cascade.**  The first stage is
  now :class:`InferenceAsrNemoStage` (default ``nvidia/canary-1b-flash``)
  instead of the QwenOmni/QwenLLM cascade.  Order is::

      ASR -> SED -> SED postprocess -> segment -> diarize ->
             embed -> utmos2 -> cluster_scotch

* **Corpus-wide SCOTCH clustering** replaces per-video AHC clustering.
  After the streaming stages finish, we gather the per-shard embedding
  NPZ files plus the diarized manifest shards and call SCOTCH (BIRCH +
  AHC) once across the corpus.

* **One Slurm allocation, one Ray cluster.**  Use ``--slurm`` to spin up
  a multi-node Ray cluster via :class:`SlurmRayClient`.  Each Curator
  ``Pipeline.run()`` then dispatches its stage-specific actor pool onto
  that cluster.  Per-stage ``Resources`` (CPU/GPU/memory) drives Xenna's
  scheduler — no manual sbatch chunking.

Usage (local)::

    python run_pipeline_beta.py \\
        --input_manifest /data/segment_manifest \\
        --output_dir output/hifi_beta \\
        --num_shards 64

Usage (multi-node Slurm)::

    sbatch tutorials/audio/hifi_pipeline/submit_beta.sh \\
        --corpus ytc_ru
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from dataclasses import dataclass

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


# When the source is NemoTarredAudioReader, AudioTasks carry ``waveform``
# (numpy array) and a few helper fields that aren't JSON-serializable.
# ManifestWriterStage calls json.dumps directly, so we strip them in a
# tiny pass-through stage right before writing.  Stages that need the
# waveform consume it inline (e.g. UTMOSv2 reads ``waveform`` directly);
# anything reaching the writer no longer needs it.
_NON_SERIALIZABLE_KEYS = ("waveform",)


@dataclass
class _StripAudioBytesStage(ProcessingStage[AudioTask, AudioTask]):
    name: str = "strip_audio_bytes"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: AudioTask) -> AudioTask:
        for key in _NON_SERIALIZABLE_KEYS:
            task.data.pop(key, None)
        return task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Beta HIFI pipeline (ASR-first, SCOTCH cluster)")

    p.add_argument("--input_manifest", type=str, default="",
                   help="Directory or file containing input JSONL manifest(s).  "
                        "Use this OR --data_config (not both).")
    p.add_argument("--data_config", type=str, default="",
                   help="Granary YAML data config for AIS-streamed input.  "
                        "Pairs each shard's manifest with its tar from AIS/S3 "
                        "and emits AudioTask objects carrying waveform + sample_rate.  "
                        "Use this OR --input_manifest.")
    p.add_argument("--corpus_filter", type=str, nargs="*", default=None,
                   help="Only process these corpora from --data_config "
                        "(matches the YAML 'corpus' field).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_shards", type=int, default=64,
                   help="Used by cluster_scotch when scanning per-shard outputs.")

    # Stage selection
    p.add_argument(
        "--stages",
        type=str,
        default="asr,sed,sed_post,segment,diarize,embed,utmos2,cluster_scotch",
        help="Comma-separated. Available: asr, sed, sed_post, segment, "
             "diarize, embed, utmos2, cluster_scotch",
    )
    p.add_argument("--start_from", type=str, default=None,
                   help="Skip stages before this one. Assumes prior outputs "
                        "already exist under --output_dir/<stage>/.")

    # Slurm / Ray
    p.add_argument("--slurm", action="store_true",
                   help="Use SlurmRayClient (multi-node). Default is local "
                        "RayClient. Workers block in start(); only the head "
                        "continues into the pipeline.")

    # ASR
    p.add_argument("--asr_model", type=str, default="nvidia/canary-1b-flash",
                   help="NeMo ASR model name (multilingual; native Russian).")
    p.add_argument("--asr_cache_dir", type=str, default=None)
    p.add_argument("--asr_batch_size", type=int, default=16)

    # SED
    p.add_argument("--sed_checkpoint", type=str, default="")
    p.add_argument("--sed_model_type", type=str, default="Cnn14_DecisionLevelMax")
    p.add_argument("--sed_threshold", type=float, default=0.5)

    # Segment (fan-out from speech events).  Default is what SEDPostprocessing
    # writes; for inputs that already carry pre-computed events under a
    # different key (Granary ``speech_events``), override here.
    p.add_argument("--segment_events_key", type=str, default="predicted_events")

    # Speaker model (embed)
    p.add_argument("--speaker_model", type=str,
                   default="nvidia/speakerverification_en_titanet_large")
    p.add_argument("--embed_batch_size", type=int, default=64)

    # UTMOS
    p.add_argument("--utmos_batch_size", type=int, default=16)

    # SCOTCH (corpus-wide clustering)
    p.add_argument("--scotch_preset", type=str, default="librispeech-2026-04")
    p.add_argument("--scotch_max_leaves", type=int, default=150_000)
    p.add_argument("--scotch_birch_floor", type=float, default=None,
                   help="Override BIRCH starting cosine floor (default: preset).")
    p.add_argument("--audio_filepath_key", type=str, default="audio_filepath")

    # AsrBridgeStage temp dir (active when --data_config is set).  Default
    # /tmp is node-local and slurm-cleaned; override for a shared FS only
    # if you have multi-node actors sharing audio files (uncommon).
    p.add_argument("--temp_dir", type=str, default="/tmp",
                   help="Where AsrBridgeStage writes per-task temp WAVs.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage helpers — each builds a small Pipeline(name=...) + run()
# ---------------------------------------------------------------------------

def _stage_dir(args: argparse.Namespace, name: str) -> str:
    return os.path.join(args.output_dir, name)


def _audio_pipeline(
    name: str,
    args: argparse.Namespace,
    current_manifest: str,
    needs_audio: bool = True,
) -> Pipeline:
    """Pipeline source: AIS-streamed (with AsrBridge) or file-based JSONL.

    ``needs_audio=True`` (default): stages that load audio by filepath
      (SED, diarize, embed, utmos2).  With --data_config, uses
      NemoTarredAudioReader → AsrBridgeStage so actors see real WAV paths.

    ``needs_audio=False``: data-flow stages (sed_post, segment) that only
      consume prior stage's JSONL fields (npz_filepath, predicted_events).
      Always uses ManifestReader on current_manifest regardless of
      --data_config, so they see the correct upstream outputs.
    """
    from nemo_curator.stages.audio.common import ManifestReader

    pipeline = Pipeline(name=name)
    if args.data_config and needs_audio:
        from nemo_curator.stages.audio.io.nemo_tarred_reader import NemoTarredAudioReader
        from nemo_curator.stages.audio.preprocessing import AsrBridgeStage
        pipeline.add_stage(NemoTarredAudioReader(
            yaml_path=args.data_config,
            corpus_filter=args.corpus_filter,
        ))
        pipeline.add_stage(AsrBridgeStage(temp_dir=args.temp_dir))
    else:
        pipeline.add_stage(ManifestReader(manifest_path=current_manifest))
    return pipeline


def _doc_pipeline(name: str, current_manifest: str) -> Pipeline:
    """Pipeline starting with JsonlReader (DocumentBatch-shaped)."""
    from nemo_curator.stages.text.io.reader.jsonl import JsonlReader

    pipeline = Pipeline(name=name)
    pipeline.add_stage(JsonlReader(file_paths=current_manifest))
    return pipeline


def _xenna_executor():
    from nemo_curator.backends.xenna import XennaExecutor

    return XennaExecutor(config={"execution_mode": "streaming"})


def _ray_data_executor():
    try:
        from nemo_curator.backends.ray_data import RayDataExecutor
    except (ImportError, ModuleNotFoundError):
        from nemo_curator.backends.experimental.ray_data import RayDataExecutor
    return RayDataExecutor()


def run_asr(args: argparse.Namespace, current: str) -> str:
    from nemo_curator.stages.audio.common import ManifestWriterStage
    from nemo_curator.stages.audio.inference.asr.asr_nemo import InferenceAsrNemoStage

    out_dir = _stage_dir(args, "asr")
    out_path = os.path.join(out_dir, "manifest.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[beta] asr ({args.asr_model})")
    pipeline = _audio_pipeline("asr", args, current)
    pipeline.add_stage(InferenceAsrNemoStage(
        model_name=args.asr_model,
        cache_dir=args.asr_cache_dir,
        batch_size=args.asr_batch_size,
    ))
    pipeline.add_stage(_StripAudioBytesStage())
    pipeline.add_stage(ManifestWriterStage(output_path=out_path))
    pipeline.run(executor=_xenna_executor())
    return out_path


def run_sed(args: argparse.Namespace, current: str) -> str:
    from nemo_curator.stages.audio.common import ManifestWriterStage
    from nemo_curator.stages.audio.inference.sed import SEDInferenceStage
    from nemo_curator.stages.resources import Resources

    if not args.sed_checkpoint:
        msg = "--sed_checkpoint is required for the SED stage."
        raise ValueError(msg)

    out_dir = _stage_dir(args, "sed")
    out_path = os.path.join(out_dir, "manifest.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[beta] sed ({args.sed_model_type})")
    pipeline = _audio_pipeline("sed", args, current)
    pipeline.add_stage(
        SEDInferenceStage(
            checkpoint_path=args.sed_checkpoint,
            model_type=args.sed_model_type,
            output_dir=out_dir,
            resources=Resources(cpus=1.0, gpu_memory_gb=4.0),
        )
    )
    pipeline.add_stage(_StripAudioBytesStage())
    pipeline.add_stage(ManifestWriterStage(output_path=out_path))
    pipeline.run(executor=_xenna_executor())
    return out_path


def run_sed_post(args: argparse.Namespace, current: str) -> str:
    from nemo_curator.stages.audio.common import ManifestWriterStage
    from nemo_curator.stages.audio.postprocessing.sed_postprocessing import SEDPostprocessingStage

    out_dir = _stage_dir(args, "sed_post")
    out_path = os.path.join(out_dir, "manifest.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    print("[beta] sed_post")
    pipeline = _audio_pipeline("sed_post", args, current, needs_audio=False)
    pipeline.add_stage(SEDPostprocessingStage(
        speech_threshold=args.sed_threshold,
        min_duration_sec=0.3,
    ))
    pipeline.add_stage(_StripAudioBytesStage())
    pipeline.add_stage(ManifestWriterStage(output_path=out_path))
    pipeline.run(executor=_xenna_executor())
    return out_path


def run_segment(args: argparse.Namespace, current: str) -> str:
    from nemo_curator.stages.audio.common import ManifestWriterStage
    from nemo_curator.stages.audio.segmentation.segment_extractor import SegmentExtractorStage

    out_dir = _stage_dir(args, "segment")
    out_path = os.path.join(out_dir, "manifest.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    print("[beta] segment (fan-out)")
    pipeline = _audio_pipeline("segment", args, current, needs_audio=False)
    pipeline.add_stage(SegmentExtractorStage(events_key=args.segment_events_key))
    pipeline.add_stage(_StripAudioBytesStage())
    pipeline.add_stage(ManifestWriterStage(output_path=out_path))
    pipeline.run(executor=_xenna_executor())
    return out_path


def run_diarize(args: argparse.Namespace, current: str) -> str:
    from nemo_curator.stages.audio.common import ManifestWriterStage
    from nemo_curator.stages.audio.inference.sortformer import InferenceSortformerStage

    out_dir = _stage_dir(args, "diarize")
    out_path = os.path.join(out_dir, "manifest.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    print("[beta] diarize (Sortformer)")
    pipeline = _audio_pipeline("diarize", args, current)
    pipeline.add_stage(InferenceSortformerStage())
    pipeline.add_stage(_StripAudioBytesStage())
    pipeline.add_stage(ManifestWriterStage(output_path=out_path))
    pipeline.run(executor=_xenna_executor())
    return out_path


def run_embed(args: argparse.Namespace, current: str) -> tuple[str, str]:
    """Run TitaNet embeddings via SpeakerEmbeddingRequestStage.

    Returns ``(updated_manifest, embedding_dir)``.  ``embedding_dir`` holds
    one ``embeddings.npz`` file consumed by ``cluster_scotch``.
    """
    from nemo_curator.stages.audio.speaker_id.speaker_embedding_request import SpeakerEmbeddingRequestStage
    from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter

    embedding_dir = _stage_dir(args, "embed")
    out_manifest = os.path.join(embedding_dir, "manifest")
    os.makedirs(embedding_dir, exist_ok=True)

    print(f"[beta] embed (TitaNet, {args.speaker_model})")
    pipeline = _doc_pipeline("embed", current)
    pipeline.add_stage(SpeakerEmbeddingRequestStage(
        model_name=args.speaker_model,
        audio_filepath_key=args.audio_filepath_key,
        output_path=os.path.join(embedding_dir, "embeddings"),
        output_format="npz",
        batch_size=args.embed_batch_size,
    ))
    pipeline.add_stage(JsonlWriter(path=out_manifest, write_kwargs={"force_ascii": False}).with_(batch_size=1))
    pipeline.run(executor=_ray_data_executor())
    return out_manifest, embedding_dir


def run_utmos2(args: argparse.Namespace, current: str) -> str:
    from nemo_curator.stages.audio.common import ManifestWriterStage
    from nemo_curator.stages.audio.metrics.utmosv2_score import GetUtmosv2ScoreStage

    out_dir = _stage_dir(args, "utmos2")
    out_path = os.path.join(out_dir, "manifest.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    print("[beta] utmos2 (UTMOSv2)")
    pipeline = _audio_pipeline("utmos2", args, current)
    pipeline.add_stage(GetUtmosv2ScoreStage(inference_batch_size=args.utmos_batch_size))
    pipeline.add_stage(_StripAudioBytesStage())
    pipeline.add_stage(ManifestWriterStage(output_path=out_path))
    pipeline.run(executor=_xenna_executor())
    return out_path


# ---------------------------------------------------------------------------
# Post-pipeline: corpus-wide SCOTCH clustering
# ---------------------------------------------------------------------------

def run_cluster_scotch(
    args: argparse.Namespace,
    manifest_dir: str,
    embedding_dir: str,
) -> str:
    """Gather embeddings + manifests, run BIRCH+AHC once, scatter labels.

    Reuses helpers from :mod:`tutorials.audio.hifi_pipeline.slurm_e2e.run_cluster_scotch`.
    """
    import time

    import numpy as np

    from nemo_curator.stages.audio.speaker_id.clustering.cluster_config import (
        PRESETS,
        build_cluster_config,
        cosine_floor_to_birch_radius,
        write_cluster_config,
    )
    from nemo_curator.stages.audio.speaker_id.clustering.large_scale_clustering_and_scoring import (
        DROPPED_LABEL,
        cluster_embeddings_large_scale,
        print_large_scale_summary,
    )
    from tutorials.audio.hifi_pipeline.slurm_e2e.run_cluster_scotch import (
        gather_corpus,
        scatter_labels,
        write_shards,
    )

    if args.scotch_preset not in PRESETS:
        msg = f"Unknown scotch preset {args.scotch_preset!r}. Known: {sorted(PRESETS)}"
        raise ValueError(msg)
    preset = PRESETS[args.scotch_preset]

    cluster_threshold = float(preset["cluster_threshold"])
    linkage_method = str(preset["cluster_linkage"])
    min_cluster_size = int(preset["min_cluster_size"])
    birch_cosine_floor = float(args.scotch_birch_floor or preset["birch_cosine_floor"])
    branching_factor = int(preset["birch_branching_factor"])
    partial_fit_batch = int(preset["birch_partial_fit_batch"])
    assign_tile = int(preset["assign_tile"])
    embedding_normalization = str(preset["embedding_normalization"])
    birch_radius = cosine_floor_to_birch_radius(birch_cosine_floor)

    print(f"[beta] cluster_scotch (preset={args.scotch_preset})")
    embeddings, origin, shard_rows = gather_corpus(
        manifest_dir, embedding_dir, args.num_shards, args.audio_filepath_key,
    )
    if embedding_normalization == "center_global":
        embeddings -= embeddings.mean(axis=0, keepdims=True)

    t0 = time.time()
    labels, confidence, stats = cluster_embeddings_large_scale(
        embeddings,
        threshold=cluster_threshold,
        linkage_method=linkage_method,
        min_cluster_size=min_cluster_size,
        birch_threshold=birch_radius,
        branching_factor=branching_factor,
        partial_fit_batch=partial_fit_batch,
        assign_tile=assign_tile,
        compute_confidence=True,
        dropped_label=DROPPED_LABEL,
        max_leaf_subclusters=args.scotch_max_leaves,
    )
    runtime = time.time() - t0
    print(f"[beta] cluster_scotch runtime: {runtime:.1f}s")
    print_large_scale_summary(labels, stats)

    out_dir = _stage_dir(args, "clustered_scotch")
    scatter_labels(shard_rows, origin, labels, confidence, DROPPED_LABEL)
    write_shards(shard_rows, out_dir)

    n_kept = int((labels != DROPPED_LABEL).sum())
    n_dropped = int((labels == DROPPED_LABEL).sum())
    effective_radius = float(stats.get("effective_birch_threshold", birch_radius))
    effective_cos_floor = max(-1.0, min(1.0, 1.0 - (effective_radius ** 2) / 2.0))
    cfg = build_cluster_config(
        backend="large_scale",
        preset=args.scotch_preset,
        cluster_threshold=cluster_threshold,
        cluster_linkage=linkage_method,
        min_cluster_size=min_cluster_size,
        n_input=int(embeddings.shape[0]),
        embedding_dim=int(embeddings.shape[1]),
        embedding_normalization=embedding_normalization,
        confidence_enabled=True,
        birch_cosine_floor=effective_cos_floor,
        birch_radius=effective_radius,
        birch_branching_factor=branching_factor,
        birch_partial_fit_batch=partial_fit_batch,
        assign_tile=assign_tile,
        n_leaf_subclusters=stats.get("n_leaf_subclusters"),
        n_clusters_raw=stats.get("n_clusters_raw"),
        n_clusters_kept=stats.get("filter", {}).get("n_clusters_after"),
        n_utts_kept=n_kept,
        n_utts_dropped=n_dropped,
        runtime_seconds=runtime,
        extra={
            "birch_retries": int(stats.get("birch_retries", 0)),
            "max_leaf_subclusters": int(stats.get("max_leaf_subclusters", args.scotch_max_leaves)),
            "birch_cosine_floor_requested": birch_cosine_floor,
            "birch_radius_requested": birch_radius,
        },
    )
    write_cluster_config(out_dir, cfg)
    return out_dir


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _start_ray(args: argparse.Namespace):
    """Start the Ray cluster.  Worker nodes block in start() and never return."""
    from nemo_curator.core.client import RayClient, SlurmRayClient

    client = SlurmRayClient() if args.slurm else RayClient()
    client.start()
    return client


def main() -> None:
    args = parse_args()
    if not args.input_manifest and not args.data_config:
        print("ERROR: provide --input_manifest OR --data_config")
        sys.exit(1)
    if args.input_manifest and args.data_config:
        print("ERROR: --input_manifest and --data_config are mutually exclusive")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.start_from:
        if args.start_from not in requested:
            print(f"ERROR: --start_from={args.start_from} not in --stages")
            sys.exit(1)
        requested = requested[requested.index(args.start_from):]

    ray_client = _start_ray(args)
    try:
        current = args.input_manifest
        embedding_dir: str | None = None
        manifest_pre_cluster: str | None = None

        if "asr" in requested:
            current = run_asr(args, current)
        if "sed" in requested:
            current = run_sed(args, current)
        if "sed_post" in requested:
            current = run_sed_post(args, current)
        if "segment" in requested:
            current = run_segment(args, current)
        if "diarize" in requested:
            current = run_diarize(args, current)
            manifest_pre_cluster = current
        if "embed" in requested:
            current, embedding_dir = run_embed(args, current)
        if "utmos2" in requested:
            current = run_utmos2(args, current)
        if "cluster_scotch" in requested:
            if not embedding_dir:
                embedding_dir = _stage_dir(args, "embed")
            if not manifest_pre_cluster:
                manifest_pre_cluster = _stage_dir(args, "diarize")
            current = run_cluster_scotch(args, manifest_pre_cluster, embedding_dir)

        print(f"\n[beta] All stages complete. Final output: {current}")
        with open(os.path.join(args.output_dir, "_done.json"), "w") as f:
            json.dump({"final_output": current, "stages": requested}, f, indent=2)
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
