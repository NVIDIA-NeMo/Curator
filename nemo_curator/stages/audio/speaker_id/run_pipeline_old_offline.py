#!/usr/bin/env python3
"""Integrated speaker-ID pipeline for NeMo-tarred ASR datasets.

Downloads data from S3, extracts wav files, computes speaker embeddings
(multi-GPU), and clusters speakers — all in one script.

Usage:
    # Process multiple languages (space-separated)
    python run_pipeline.py --dataset yodas --languages da hr de cs --base-dir /data/Yodas/

    # Override GPU count and model
    python run_pipeline.py --dataset yodas --languages da hr --num-gpus 4

    # Skip download (data already on disk)
    python run_pipeline.py --dataset yodas --languages da hr --skip-download

    # Different dataset
    python run_pipeline.py --dataset ytc --languages en es fr --base-dir /data/YTC/
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("speaker_id_pipeline")

# ═══════════════════════════════════════════════════════════════════════════════
# Clustering constants (ResNet293_LM on VoxCeleb1-O cleaned, 37,611 pairs)
# ═══════════════════════════════════════════════════════════════════════════════
EER_THRESHOLD = 0.362870
DEFAULT_THRESHOLD = 0.363

RANGE_PATTERN = re.compile(r"_OP_(\d+)\.\.(\d+)_CL_")
_DONE_RE = re.compile(r"done=(\d+)")

KNOWN_DATASETS = [
    "ami", "fleurs", "google_speech_commands", "librilight", "mcv",
    "mmlpc", "mosel", "non_speech", "riva", "suno", "yodas", "ytc",
]

HOURS_TOLERANCE = 0.001  # 0.1 % tolerance when comparing hours


# ═══════════════════════════════════════════════════════════════════════════════
# Corpus metadata from corpusview YAML
# ═══════════════════════════════════════════════════════════════════════════════

class CorpusMetadata:
    """Expected counts per (language, subset) from a corpusview YAML."""

    def __init__(self, yaml_path: str, dataset: str):
        self._entries: Dict[Tuple[str, str], dict] = {}
        with open(yaml_path, encoding="utf-8") as f:
            docs = yaml.safe_load(f)

        for entry in docs:
            if entry.get("corpus") != dataset:
                continue
            lang = entry["language"]
            audio_path = entry.get("paths", {}).get("pdx", {}).get("audio_filepaths", "")
            subset = self._extract_subset(audio_path, dataset)
            if not subset:
                continue
            hours = entry.get("hours", 0.0)
            expected_tars = _count_expanded(os.path.basename(audio_path))
            key = (lang, subset)
            if key in self._entries:
                logger.warning(
                    "Duplicate YAML entry for (%s, %s) — accumulating hours "
                    "and tar counts. Verify subset extraction logic if this "
                    "is unexpected.",
                    lang, subset,
                )
                self._entries[key]["hours"] += hours
                self._entries[key]["expected_tars"] += expected_tars
            else:
                self._entries[key] = {
                    "hours": hours,
                    "expected_tars": expected_tars,
                    "audio_pattern": audio_path,
                }

    @staticmethod
    def _extract_subset(audio_path: str, dataset: str) -> Optional[str]:
        """Extract subset name from an S3 audio path.

        For yodas: ``s3://yodas2/da/0_by_whisper/audio__OP_0..63_CL_.tar``
                   → ``0_by_whisper``
        For ytc:   ``s3://YTC/da/tarred_train/audio__OP_0..63_CL_.tar``
                   → ``tarred_train``
        """
        if not audio_path:
            return None
        parts = audio_path.rstrip("/").split("/")
        if len(parts) < 3:
            return None
        return parts[-2]

    def get(self, language: str, subset: str) -> Optional[dict]:
        return self._entries.get((language, subset))

    def expected_hours(self, language: str, subsets: List[str]) -> float:
        return sum(
            (self._entries.get((language, s), {}).get("hours", 0.0))
            for s in subsets
        )

    def expected_tars(self, language: str, subset: str) -> Optional[int]:
        info = self._entries.get((language, subset))
        return info["expected_tars"] if info else None

    def audio_s3_pattern(self, language: str, subset: str) -> Optional[str]:
        """Return the raw S3 audio pattern (with ``_OP_..._CL_`` ranges)."""
        info = self._entries.get((language, subset))
        return info["audio_pattern"] if info else None

    def has_language(self, language: str, subsets: List[str]) -> bool:
        return any((language, s) in self._entries for s in subsets)


def _resolve_model_source(
    model_name_or_path: str,
    model_checkpoint: Optional[str],
) -> str:
    """Resolve model input.

    Priority:
    1) If --model-checkpoint is provided, always use local checkpoint/folder.
    2) Otherwise use --model (hub name or local path), which may download.
    """
    if not model_checkpoint:
        return model_name_or_path

    ckpt_path = os.path.abspath(os.path.expanduser(model_checkpoint))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"--model-checkpoint path does not exist: {ckpt_path}"
        )

    if os.path.isdir(ckpt_path):
        config_path = os.path.join(ckpt_path, "config.yaml")
        avg_model_path = os.path.join(ckpt_path, "avg_model.pt")
        if not os.path.isfile(config_path) or not os.path.isfile(avg_model_path):
            raise FileNotFoundError(
                "Checkpoint folder must contain config.yaml and avg_model.pt: "
                f"{ckpt_path}"
            )
        return ckpt_path

    # File path mode: expect avg_model.pt, then use its parent folder.
    if os.path.basename(ckpt_path) != "avg_model.pt":
        raise ValueError(
            "--model-checkpoint file must be avg_model.pt, "
            f"got: {ckpt_path}"
        )

    ckpt_dir = os.path.dirname(ckpt_path)
    config_path = os.path.join(ckpt_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"config.yaml not found next to checkpoint: {config_path}"
        )
    return ckpt_dir


# ═══════════════════════════════════════════════════════════════════════════════
# S3 Client
# ═══════════════════════════════════════════════════════════════════════════════

class S3Client:
    """Lightweight boto3 client that reads credentials from .s3cfg files."""

    def __init__(self, s3cfg: str = "~/.s3cfg[default]"):
        import configparser

        import boto3
        from botocore.config import Config

        path, section = s3cfg.rsplit("[", 1)
        section = section.rstrip("]")
        config_path = Path(path).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = configparser.ConfigParser()
        config.read(config_path)
        if section not in config:
            raise ValueError(f"No [{section}] section found in {config_path}")

        cfg = config[section]
        protocol = "https" if cfg.getboolean("use_https", fallback=True) else "http"
        self.client = boto3.client(
            "s3",
            endpoint_url=f"{protocol}://{cfg.get('host_base', '')}",
            aws_access_key_id=cfg.get("access_key"),
            aws_secret_access_key=cfg.get("secret_key"),
            region_name=cfg.get("bucket_location"),
            config=Config(connect_timeout=60, read_timeout=600, retries={"max_attempts": 3}),
        )

    def download_file(self, bucket: str, key: str, local_path: str) -> None:
        self.client.download_file(bucket, key, local_path)

    def list_prefixes(self, bucket: str, prefix: str, delimiter: str = "/") -> List[str]:
        """List immediate sub-prefixes under *prefix* (like listing directories)."""
        paginator = self.client.get_paginator("list_objects_v2")
        prefixes = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter=delimiter):
            for cp in page.get("CommonPrefixes", []):
                prefixes.append(cp["Prefix"])
        return prefixes


# ═══════════════════════════════════════════════════════════════════════════════
# S3 helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_s3_path(s3_path: str) -> Tuple[str, str]:
    parsed = urlparse(str(s3_path))
    return parsed.netloc, parsed.path.lstrip("/")


def _expand_ranges(pattern: str) -> Generator[str, None, None]:
    match = RANGE_PATTERN.search(pattern)
    if not match:
        yield pattern
        return
    start, end = int(match.group(1)), int(match.group(2))
    width = len(match.group(1))
    fmt = f"{{:0{width}d}}"
    for i in range(start, end + 1):
        expanded = RANGE_PATTERN.sub(fmt.format(i), pattern, count=1)
        yield from _expand_ranges(expanded)


def _count_expanded(pattern: str) -> int:
    matches = RANGE_PATTERN.findall(pattern)
    total = 1
    for start, end in matches:
        total *= int(end) - int(start) + 1
    return total


def _download_one(s3_path, output_dir, s3_client, cut_prefix, force):
    try:
        bucket, key = _parse_s3_path(s3_path)
        local_key = f"{bucket}/{key}"
        if cut_prefix:
            normalized = cut_prefix.strip("/")
            if local_key.startswith(normalized + "/"):
                local_key = local_key[len(normalized) + 1:]
            elif local_key.startswith(normalized):
                local_key = local_key[len(normalized):]
        local_path = Path(output_dir) / local_key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not force and local_path.exists():
            return (s3_path, True, f"Already exists: {local_path}")
        s3_client.download_file(bucket, key, str(local_path))
        return (s3_path, True, f"Downloaded: {local_path}")
    except Exception as e:
        return (s3_path, False, f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _save_embeddings(embeddings, utt_ids, output_dir, suffix=""):
    os.makedirs(output_dir, exist_ok=True)
    emb_path = os.path.join(output_dir, f"embeddings{suffix}.npy")
    utt_path = os.path.join(output_dir, f"utt_names{suffix}.txt")
    np.save(emb_path, embeddings)
    with open(utt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(utt_ids) + "\n")
    return emb_path, utt_path


def _load_embeddings(output_dir, suffix=""):
    emb_path = os.path.join(output_dir, f"embeddings{suffix}.npy")
    utt_path = os.path.join(output_dir, f"utt_names{suffix}.txt")
    embeddings = np.load(emb_path)
    with open(utt_path, encoding="utf-8") as f:
        utt_ids = [line.strip() for line in f if line.strip()]
    return embeddings, utt_ids


def _merge_embedding_shards(output_dir, num_shards):
    all_embs, all_utt_ids = [], []
    for i in range(num_shards):
        embs, utts = _load_embeddings(output_dir, suffix=f"_gpu{i}")
        all_embs.append(embs)
        all_utt_ids.extend(utts)
    merged = np.concatenate(all_embs, axis=0)
    return _save_embeddings(merged, all_utt_ids, output_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# Clustering helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _l2_normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-8)


def _cosine_similarity_matrix(embeddings):
    normed = _l2_normalize(embeddings)
    sim = normed @ normed.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return sim


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline class
# ═══════════════════════════════════════════════════════════════════════════════

class SpeakerIDPipeline:
    """End-to-end speaker-ID pipeline: download -> extract -> embed -> cluster.

    Instantiate once with global settings, then call ``run(languages)`` to
    process a list of language codes sequentially.
    """

    def __init__(
        self,
        dataset: str,
        base_dir: str,
        result_dir: Optional[str] = None,
        s3cfg: str = "~/.s3cfg[default]",
        subsets: Optional[List[str]] = None,
        model: str = "voxceleb_resnet293_LM",
        model_checkpoint: Optional[str] = None,
        corpusview_path: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        num_gpus: int = 1,
        batch_dur: float = 600.0,
        sample_rate: int = 16000,
        num_mel_bins: int = 80,
        download_workers: int = 16,
        audio_pattern: str = "s3://yodas2/{language}/{subset}/audio__OP_0..63_CL_.tar",
        manifest_pattern: str = (
            "s3://granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/"
            "YODAS2/{language}/{subset}/sharded_manifests_updated/"
            "manifest__OP_0..63_CL_.json"
        ),
        cluster_threshold: float = DEFAULT_THRESHOLD,
        cluster_linkage: str = "average",
        skip_download: bool = False,
        skip_extract: bool = False,
        skip_embed: bool = False,
        no_cluster: bool = False,
        no_confidence: bool = False,
        force_download: bool = False,
    ):
        self.dataset = dataset
        self.base_dir = base_dir
        self.result_dir = (
            os.path.abspath(os.path.expanduser(result_dir))
            if result_dir else None
        )
        self.s3cfg = s3cfg
        self.subsets = subsets or ["0_by_whisper", "0_from_captions"]
        self.model = _resolve_model_source(model, model_checkpoint)
        self.model_checkpoint = model_checkpoint
        self.corpusview_path = (
            os.path.abspath(os.path.expanduser(corpusview_path))
            if corpusview_path else None
        )
        self.model_cache_dir = model_cache_dir
        self.num_gpus = num_gpus
        self.batch_dur = batch_dur
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.download_workers = download_workers
        self.audio_pattern = audio_pattern
        self.manifest_pattern = manifest_pattern
        self.cluster_threshold = cluster_threshold
        self.cluster_linkage = cluster_linkage
        self.skip_download = skip_download
        self.skip_extract = skip_extract
        self.skip_embed = skip_embed
        self.no_cluster = no_cluster
        self.no_confidence = no_confidence
        self.force_download = force_download

        self._s3_client = None
        self._tars_verified: set = set()
        self._corpus_meta: Optional[CorpusMetadata] = None
        if self.corpusview_path:
            yaml_path = os.path.join(
                self.corpusview_path, "corpus", "asr", f"{self.dataset}.yaml",
            )
            if os.path.isfile(yaml_path):
                self._corpus_meta = CorpusMetadata(yaml_path, self.dataset)
                logger.info("Loaded corpus metadata from %s", yaml_path)
            else:
                logger.warning(
                    "Corpus YAML not found at %s — count validation disabled.",
                    yaml_path,
                )

    # ── public API ────────────────────────────────────────────────────────

    def run(self, languages: List[str]) -> Dict[str, str]:
        """Process a list of languages end-to-end.

        Returns a dict mapping language -> output directory for successful runs.
        """
        results: Dict[str, str] = {}

        print(f"\n{'='*60}")
        print("  Speaker-ID Pipeline")
        print(f"  Dataset:   {self.dataset}")
        print(f"  Languages: {' '.join(languages)}")
        print(f"  Model:     {self.model}")
        if self.result_dir:
            print(f"  Result dir:{self.result_dir}")
        if self.corpusview_path:
            print(f"  CorpusView:{self.corpusview_path}")
        print(f"  GPUs:      {self.num_gpus}")
        print(f"{'='*60}\n")

        for lang in languages:
            print(f"\n{'─'*60}")
            print(f"  Processing language: {lang}")
            print(f"{'─'*60}")

            output_dir = self._language_output_dir(lang)
            if self._is_language_complete(output_dir):
                logger.info(
                    "[%s] Already complete at %s, skipping language.",
                    lang, output_dir,
                )
                results[lang] = output_dir
                continue

            if not self._language_exists(lang):
                logger.warning(
                    "No such dataset for language '%s', skipping.", lang,
                )
                continue

            try:
                output_dir = self._process_language(lang)
                results[lang] = output_dir
                logger.info("Language %s complete -> %s", lang, output_dir)
            except Exception:
                logger.exception("Language %s FAILED", lang)

        print(f"\n{'='*60}")
        print(f"  Pipeline finished: {len(results)}/{len(languages)} languages OK")
        for lang, out in results.items():
            print(f"    {lang} -> {out}")
        print(f"{'='*60}\n")

        return results

    # ── language validation ───────────────────────────────────────────────

    def _language_exists(self, language: str) -> bool:
        """Check whether S3 has data for this language by probing the audio bucket."""
        if self.skip_download:
            # With --skip-download, do not silently skip based on base_dir.
            # _process_language() will raise a clear error if input data is missing.
            return True

        try:
            client = self._get_s3_client()
            probe = self.audio_pattern.format(
                language=language, subset=self.subsets[0],
            )
            bucket, key = _parse_s3_path(probe)
            prefix = key.rsplit("/", 1)[0] + "/"
            prefixes = client.list_prefixes(bucket, prefix)
            return len(prefixes) > 0
        except Exception:
            lang_dir = os.path.join(self.base_dir, language)
            return os.path.isdir(lang_dir)

    # ── per-language pipeline ─────────────────────────────────────────────

    def _process_language(self, language: str) -> str:
        lang_dir = os.path.join(self.base_dir, language)
        output_dir = self._language_output_dir(language)

        should_download = not self.skip_download
        if self.skip_download and not os.path.isdir(lang_dir):
            logger.warning(
                "[%s] Input directory missing under --skip-download (%s). "
                "Falling back to S3 download for this language.",
                language, lang_dir,
            )
            should_download = True

        if should_download:
            self._step_download(language)

        if not self.skip_extract:
            self._step_extract_tars(lang_dir)

        manifest_count: Optional[int] = None
        if not self.skip_embed:
            if self._embeddings_exist(output_dir):
                logger.info(
                    "[%s] Embeddings already exist at %s, skipping extraction.",
                    language, output_dir,
                )
            else:
                all_entries = self._step_build_manifest(lang_dir, output_dir)
                manifest_count = len(all_entries)

                if self.num_gpus > 1:
                    self._step_embed_multigpu(language, all_entries, output_dir)
                else:
                    self._step_embed_single(all_entries, output_dir)

                self._validate_embeddings_vs_manifest(output_dir, manifest_count, language)

        if not self.no_cluster:
            self._step_cluster(output_dir)
            self._validate_cluster_output(output_dir, language)

        self._cleanup_wavs(lang_dir, output_dir)

        return output_dir

    # ── result validation + cleanup ───────────────────────────────────────

    def _language_output_dir(self, language: str) -> str:
        if self.result_dir:
            return os.path.join(self.result_dir, language, "wespeaker_embeddings")
        return os.path.join(self.base_dir, language, "wespeaker_embeddings")

    def _is_language_complete(self, output_dir: str) -> bool:
        if not self._embeddings_exist(output_dir):
            return False
        if self.no_cluster:
            return True
        clusters_path = os.path.join(output_dir, "clusters.jsonl")
        if not os.path.isfile(clusters_path) or os.path.getsize(clusters_path) == 0:
            return False

        emb_path = os.path.join(output_dir, "embeddings.npy")
        try:
            emb_count = np.load(emb_path).shape[0]
            with open(clusters_path, encoding="utf-8") as f:
                cluster_count = sum(1 for line in f if line.strip())
            if cluster_count != emb_count:
                logger.warning(
                    "clusters.jsonl lines (%d) != embeddings (%d). Will re-run.",
                    cluster_count, emb_count,
                )
                return False
        except Exception:
            return False
        return True

    def _embeddings_exist(self, output_dir: str) -> bool:
        """Check if embeddings.npy and utt_names.txt already exist and are non-empty."""
        emb_path = os.path.join(output_dir, "embeddings.npy")
        utt_path = os.path.join(output_dir, "utt_names.txt")
        if not os.path.isfile(emb_path) or not os.path.isfile(utt_path):
            return False
        if os.path.getsize(emb_path) == 0 or os.path.getsize(utt_path) == 0:
            return False
        try:
            emb = np.load(emb_path)
            with open(utt_path, encoding="utf-8") as f:
                utt_count = sum(1 for line in f if line.strip())
            if emb.shape[0] == 0 or utt_count == 0:
                return False
            if emb.shape[0] != utt_count:
                logger.warning(
                    "Embedding count mismatch: %d embeddings vs %d utt names. "
                    "Will re-extract.", emb.shape[0], utt_count,
                )
                return False
            logger.info(
                "Found existing embeddings: %d utterances, dim %d",
                emb.shape[0], emb.shape[1],
            )
            return True
        except Exception:
            return False

    def _cleanup_wavs(self, lang_dir: str, output_dir: str) -> None:
        """Remove extracted wav files after confirming all outputs exist."""
        wav_dir = os.path.join(lang_dir, "wavs")
        if not os.path.isdir(wav_dir):
            return

        emb_path = os.path.join(output_dir, "embeddings.npy")
        utt_path = os.path.join(output_dir, "utt_names.txt")

        if not os.path.isfile(emb_path) or not os.path.isfile(utt_path):
            logger.info("Keeping wavs — embeddings not yet generated.")
            return

        try:
            emb = np.load(emb_path)
            with open(utt_path, encoding="utf-8") as f:
                utt_count = sum(1 for line in f if line.strip())
        except Exception:
            logger.warning("Could not verify embeddings, keeping wavs.")
            return

        if emb.shape[0] == 0 or utt_count == 0 or emb.shape[0] != utt_count:
            logger.warning(
                "Embedding validation failed (%d embeddings, %d utt names), keeping wavs.",
                emb.shape[0], utt_count,
            )
            return

        language = os.path.basename(os.path.normpath(lang_dir))
        wav_count = sum(1 for _ in Path(wav_dir).rglob("*.wav"))
        logger.info(
            "[%s] All outputs verified (%d embeddings). "
            "Removing %d extracted wav files from %s",
            language, emb.shape[0], wav_count, wav_dir,
        )
        shutil.rmtree(wav_dir)
        logger.info("[%s] Cleanup complete.", language)

    # ── count validation helpers ──────────────────────────────────────────

    def _validate_tar_counts(self, lang_dir: str, language: str) -> None:
        """Verify that the number of downloaded tar files matches the YAML."""
        if not self._corpus_meta:
            return
        cache_key = ("tar_counts", language)
        if cache_key in self._tars_verified:
            return
        for subset in self.subsets:
            expected = self._corpus_meta.expected_tars(language, subset)
            if expected is None:
                continue
            tar_dir = Path(lang_dir) / subset
            actual = len(list(tar_dir.glob("audio_*.tar"))) if tar_dir.is_dir() else 0
            if actual != expected:
                raise RuntimeError(
                    f"[{language}/{subset}] Tar file count mismatch: "
                    f"expected {expected}, found {actual} in {tar_dir}"
                )
            logger.info(
                "[%s/%s] Tar count OK: %d files", language, subset, actual,
            )
        self._tars_verified.add(cache_key)

    def _verify_and_repair_tars(self, lang_dir: str, language: str) -> None:
        """Check every tar file for integrity; re-download corrupt ones from S3."""
        cache_key = ("tar_integrity", language)
        if cache_key in self._tars_verified:
            return
        for subset in self.subsets:
            tar_dir = Path(lang_dir) / subset
            if not tar_dir.is_dir():
                continue

            s3_pattern = None
            if self._corpus_meta:
                s3_pattern = self._corpus_meta.audio_s3_pattern(language, subset)

            tar_files = sorted(tar_dir.glob("audio_*.tar"))
            if not tar_files:
                continue

            s3_url_map: Dict[str, str] = {}
            if s3_pattern:
                for url in _expand_ranges(s3_pattern):
                    fname = url.rstrip("/").rsplit("/", 1)[-1]
                    s3_url_map[fname] = url
            else:
                audio_pat = self.audio_pattern.format(
                    language=language, subset=subset,
                )
                for url in _expand_ranges(audio_pat):
                    fname = url.rstrip("/").rsplit("/", 1)[-1]
                    s3_url_map[fname] = url

            corrupt = []
            for tf_path in tqdm(tar_files, desc=f"Verifying {subset} tars"):
                try:
                    with tarfile.open(tf_path, "r") as tf:
                        tf.getmembers()
                except (tarfile.TarError, EOFError, OSError):
                    corrupt.append(tf_path)

            if not corrupt:
                logger.info(
                    "[%s/%s] All %d tar files are intact.",
                    language, subset, len(tar_files),
                )
                continue

            logger.warning(
                "[%s/%s] %d/%d tar files are corrupt — re-downloading.",
                language, subset, len(corrupt), len(tar_files),
            )
            client = self._get_s3_client()
            for tf_path in corrupt:
                fname = tf_path.name
                s3_url = s3_url_map.get(fname)
                if not s3_url:
                    raise RuntimeError(
                        f"[{language}/{subset}] Cannot determine S3 URL for "
                        f"corrupt tar {fname}. Check audio pattern."
                    )
                logger.info("Re-downloading %s -> %s", s3_url, tf_path)
                tf_path.unlink(missing_ok=True)
                bucket, key = _parse_s3_path(s3_url)
                client.download_file(bucket, key, str(tf_path))

                try:
                    with tarfile.open(tf_path, "r") as tf:
                        tf.getmembers()
                except (tarfile.TarError, EOFError, OSError) as exc:
                    raise RuntimeError(
                        f"[{language}/{subset}] Re-downloaded {fname} is still "
                        f"corrupt: {exc}"
                    ) from exc

            logger.info(
                "[%s/%s] Successfully re-downloaded %d corrupt tar files.",
                language, subset, len(corrupt),
            )
        self._tars_verified.add(cache_key)

    def _validate_hours(
        self, language: str, actual_hours: float,
    ) -> None:
        """Warn if total manifest hours deviate from the YAML expectation."""
        if not self._corpus_meta:
            return
        expected = self._corpus_meta.expected_hours(language, self.subsets)
        if expected <= 0:
            return
        rel_diff = abs(actual_hours - expected) / expected
        if rel_diff > HOURS_TOLERANCE:
            logger.warning(
                "[%s] Hours mismatch: YAML expects %.2f h, manifest has %.2f h "
                "(%.1f%% off). Some audio may be missing.",
                language, expected, actual_hours, rel_diff * 100,
            )
        else:
            logger.info(
                "[%s] Hours OK: %.2f h (expected %.2f h, %.1f%% diff)",
                language, actual_hours, expected, rel_diff * 100,
            )

    def _validate_embeddings_vs_manifest(
        self, output_dir: str, manifest_count: int, language: str,
    ) -> None:
        """Verify that the number of extracted embeddings == manifest entries."""
        emb_path = os.path.join(output_dir, "embeddings.npy")
        utt_path = os.path.join(output_dir, "utt_names.txt")
        if not os.path.isfile(emb_path) or not os.path.isfile(utt_path):
            raise RuntimeError(
                f"[{language}] Embedding files missing after extraction: "
                f"{emb_path}, {utt_path}"
            )
        emb = np.load(emb_path)
        with open(utt_path, encoding="utf-8") as f:
            utt_count = sum(1 for line in f if line.strip())

        if emb.shape[0] != utt_count:
            raise RuntimeError(
                f"[{language}] embeddings.npy rows ({emb.shape[0]}) != "
                f"utt_names.txt lines ({utt_count})"
            )
        if emb.shape[0] != manifest_count:
            raise RuntimeError(
                f"[{language}] Embedding count ({emb.shape[0]}) != "
                f"manifest entry count ({manifest_count}). "
                f"Some utterances were lost during extraction."
            )
        logger.info(
            "[%s] Embedding count OK: %d (matches manifest)",
            language, emb.shape[0],
        )

    def _validate_cluster_output(self, output_dir: str, language: str) -> None:
        """Verify clusters.jsonl line count == embedding count."""
        clusters_path = os.path.join(output_dir, "clusters.jsonl")
        emb_path = os.path.join(output_dir, "embeddings.npy")

        if not os.path.isfile(clusters_path):
            raise RuntimeError(
                f"[{language}] clusters.jsonl not found at {clusters_path}"
            )
        if not os.path.isfile(emb_path):
            raise RuntimeError(
                f"[{language}] embeddings.npy not found at {emb_path}"
            )

        emb = np.load(emb_path)
        emb_count = emb.shape[0]

        with open(clusters_path, encoding="utf-8") as f:
            cluster_count = sum(1 for line in f if line.strip())

        if cluster_count != emb_count:
            raise RuntimeError(
                f"[{language}] clusters.jsonl lines ({cluster_count}) != "
                f"embedding count ({emb_count})"
            )
        logger.info(
            "[%s] Cluster output OK: %d lines (matches embeddings)",
            language, cluster_count,
        )

    # ── Step 1: download ──────────────────────────────────────────────────

    def _step_download(self, language: str) -> None:
        logger.info("[%s] Step 1/4: Downloading data from S3...", language)
        client = self._get_s3_client()

        for subset in self.subsets:
            fmt = dict(language=language, subset=subset)
            audio_pat = self.audio_pattern.format(**fmt)
            manifest_pat = self.manifest_pattern.format(**fmt)

            manifest_output = os.path.join(
                self.base_dir, language, subset, "sharded_manifests_updated",
            )

            logger.info("[%s] Downloading audio tars for %s...", language, subset)
            self._download_pattern(
                audio_pat, self.base_dir, client.client, cut_prefix="yodas2/",
            )

            logger.info("[%s] Downloading manifests for %s...", language, subset)
            manifest_cut = (
                f"granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/"
                f"YODAS2/{language}/{subset}/sharded_manifests_updated/"
            )
            self._download_pattern(
                manifest_pat, manifest_output, client.client, cut_prefix=manifest_cut,
            )

        lang_dir = os.path.join(self.base_dir, language)
        self._validate_tar_counts(lang_dir, language)
        self._verify_and_repair_tars(lang_dir, language)
        logger.info("[%s] Download complete.", language)

    def _download_pattern(self, pattern, output_dir, s3_client, cut_prefix=None):
        total = _count_expanded(pattern)
        urls = list(_expand_ranges(pattern))
        failed = 0
        failed_urls: List[str] = []
        with ThreadPoolExecutor(max_workers=self.download_workers) as pool:
            futures = {
                pool.submit(
                    _download_one, url, output_dir, s3_client, cut_prefix,
                    self.force_download,
                ): url
                for url in urls
            }
            for future in tqdm(as_completed(futures), total=total, desc="Downloading"):
                url_key = futures[future]
                _, success, msg = future.result()
                if not success:
                    failed += 1
                    failed_urls.append(url_key)
                    logger.warning("Download failed: %s", msg)
        if failed:
            raise RuntimeError(
                f"{failed}/{total} downloads failed. "
                f"First few: {failed_urls[:5]}"
            )

    # ── Step 2: extract tars ──────────────────────────────────────────────

    def _step_extract_tars(self, lang_dir: str) -> None:
        language = os.path.basename(os.path.normpath(lang_dir))
        logger.info("[%s] Step 2/4: Extracting wav files from tar archives...", language)

        self._validate_tar_counts(lang_dir, language)
        self._verify_and_repair_tars(lang_dir, language)

        wav_dir = Path(lang_dir) / "wavs"

        for subset in self.subsets:
            tar_dir = Path(lang_dir) / subset
            extract_dir = wav_dir / subset
            extract_dir.mkdir(parents=True, exist_ok=True)

            tar_files = sorted(tar_dir.glob("audio_*.tar"))
            logger.info("Subset %s: %d tar files", subset, len(tar_files))

            for tf_path in tqdm(tar_files, desc=f"Extracting {subset}"):
                with tarfile.open(tf_path, "r") as tf:
                    tf.extractall(path=extract_dir)

            wav_count = len(list(extract_dir.glob("*.wav")))
            logger.info("Extracted %d wav files to %s", wav_count, extract_dir)

    # ── Step 3: build manifest + extract embeddings ───────────────────────

    def _step_build_manifest(self, lang_dir: str, output_dir: str) -> List[dict]:
        language = os.path.basename(os.path.normpath(lang_dir))
        logger.info("[%s] Step 3/4: Building manifest...", language)

        wav_dir = os.path.join(lang_dir, "wavs")
        manifest_dir = os.path.join(lang_dir, "manifests_merged")
        os.makedirs(manifest_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        all_entries: List[dict] = []
        missing = 0

        for subset in self.subsets:
            manifest_base = os.path.join(lang_dir, subset, "sharded_manifests_updated")
            extract_dir = os.path.join(wav_dir, subset)
            manifest_files = sorted(Path(manifest_base).glob("manifest_*.json"))
            logger.info("Subset %s: %d manifest shards", subset, len(manifest_files))

            for mf in tqdm(manifest_files, desc=f"Reading {subset} manifests"):
                with open(mf, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        audio_fn = entry["audio_filepath"]
                        wav_path = os.path.join(extract_dir, audio_fn)
                        if not os.path.isfile(wav_path):
                            missing += 1
                            continue
                        utt_id = f"{subset}__{Path(audio_fn).stem}"
                        entry["_abs_wav_path"] = wav_path
                        entry["_utt_id"] = utt_id
                        entry["_subset"] = subset
                        all_entries.append(entry)

        if missing > 0:
            raise RuntimeError(
                f"[{language}] {missing} wav files referenced in manifests "
                f"are missing on disk. Cannot proceed with incomplete data."
            )

        logger.info("Total utterances: %d (missing wavs: %d)", len(all_entries), missing)

        wav_scp_path = os.path.join(output_dir, "wav.scp")
        with open(wav_scp_path, "w", encoding="utf-8") as f:
            for entry in all_entries:
                f.write(f"{entry['_utt_id']} {entry['_abs_wav_path']}\n")
        logger.info("wav.scp: %s (%d entries)", wav_scp_path, len(all_entries))

        total_hours = sum(e.get("duration", 0) for e in all_entries) / 3600
        logger.info("Total duration: %.2f hours", total_hours)

        self._validate_hours(language, total_hours)

        return all_entries

    def _step_embed_single(self, all_entries: List[dict], output_dir: str) -> None:
        """Single-GPU embedding extraction (runs in-process)."""
        from nemo_curator.stages.audio.speaker_id.embedding.extractor import extract_embeddings
        from nemo_curator.stages.audio.speaker_id.embedding.model_loader import load_wespeaker_model

        logger.info("Loading model %s on cuda:0...", self.model)
        loaded = load_wespeaker_model(
            self.model, device="cuda:0", model_cache_dir=self.model_cache_dir,
        )

        logger.info("Extracting embeddings for %d utterances...", len(all_entries))
        embeddings, utt_ids = extract_embeddings(
            entries=all_entries,
            model=loaded.model,
            frontend_type=loaded.frontend_type,
            device=loaded.device,
            batch_dur=self.batch_dur,
            sample_rate=self.sample_rate,
            num_mel_bins=self.num_mel_bins,
        )

        emb_path, _ = _save_embeddings(embeddings, utt_ids, output_dir)
        logger.info("Saved %d embeddings -> %s", len(utt_ids), emb_path)

    def _step_embed_multigpu(
        self, language: str, all_entries: List[dict], output_dir: str,
    ) -> None:
        """Multi-GPU embedding extraction via subprocesses."""
        total_utts = len(all_entries)
        total_hours = sum(e.get("duration", 0) for e in all_entries) / 3600
        durs = sorted([e.get("duration", 0) for e in all_entries], reverse=True)

        print(f"\n  Multi-GPU extraction: {self.num_gpus} GPUs, {total_utts:,} utts, "
              f"{total_hours:.1f}h audio, longest={durs[0]:.1f}s")

        script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "scripts", "extract_embeddings.py",
        )

        procs, log_files = [], []
        for gpu_id in range(self.num_gpus):
            log_path = os.path.join(output_dir, f"log_gpu{gpu_id}.txt")
            log_files.append(log_path)

            cmd = [
                sys.executable, script,
                "--base-dir", self.base_dir,
                "--language", language,
                "--subsets", *self.subsets,
                "--model", self.model,
                "--device", f"cuda:{gpu_id}",
                "--gpu-id", str(gpu_id),
                "--num-gpus", str(self.num_gpus),
                "--batch-dur", str(self.batch_dur),
                "--sample-rate", str(self.sample_rate),
                "--num-mel-bins", str(self.num_mel_bins),
                "--output-dir", output_dir,
                "--skip-extract",
            ]
            if self.model_cache_dir:
                cmd.extend(["--model-cache-dir", self.model_cache_dir])

            logger.info("Launching GPU %d...", gpu_id)
            with open(log_path, "w", encoding="utf-8") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            procs.append((gpu_id, proc))

        pbar = tqdm(total=total_utts, desc="All GPUs", unit="utt", dynamic_ncols=True)
        prev_total = 0
        while True:
            all_done = all(proc.poll() is not None for _, proc in procs)
            current_total = sum(self._count_done_in_log(lp) for lp in log_files)
            delta = current_total - prev_total
            if delta > 0:
                pbar.update(delta)
                prev_total = current_total
            per_gpu_done = [self._count_done_in_log(lp) for lp in log_files]
            pbar.set_postfix_str(
                "  ".join(f"gpu{i}={d}" for i, d in enumerate(per_gpu_done))
            )
            if all_done:
                current_total = sum(self._count_done_in_log(lp) for lp in log_files)
                delta = current_total - prev_total
                if delta > 0:
                    pbar.update(delta)
                break
            time.sleep(2)
        pbar.close()

        failed = 0
        for gpu_id, proc in procs:
            if proc.returncode != 0:
                failed += 1
                logger.error("GPU %d FAILED (exit %d). See %s", gpu_id, proc.returncode, log_files[gpu_id])
        if failed:
            raise RuntimeError(f"{failed}/{self.num_gpus} GPU processes failed. Check logs in {output_dir}")

        logger.info("Merging %d GPU shards...", self.num_gpus)
        merged_emb_path, _ = _merge_embedding_shards(output_dir, self.num_gpus)
        logger.info("Merged embeddings: %s", merged_emb_path)

    @staticmethod
    def _count_done_in_log(log_path: str) -> int:
        if not os.path.isfile(log_path):
            return 0
        try:
            with open(log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 4096))
                tail = f.read().decode("utf-8", errors="replace")
            matches = _DONE_RE.findall(tail)
            return int(matches[-1]) if matches else 0
        except Exception:
            return 0

    # ── Step 4: cluster ───────────────────────────────────────────────────

    def _step_cluster(self, output_dir: str) -> None:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        logger.info("Step 4/4: Clustering speakers (threshold=%.4f)...", self.cluster_threshold)
        embeddings, utt_ids = _load_embeddings(output_dir)
        logger.info("Loaded %d embeddings of dim %d", embeddings.shape[0], embeddings.shape[1])

        n = embeddings.shape[0]
        if n <= 1:
            labels = np.ones(n, dtype=int)
        else:
            pbar = tqdm(total=4, desc="AHC clustering", unit="step")

            pbar.set_postfix_str("cosine distance matrix")
            dist_mat = 1.0 - _cosine_similarity_matrix(embeddings)
            pbar.update(1)

            pbar.set_postfix_str("condensing to 1-D")
            condensed = squareform(dist_mat, checks=False)
            pbar.update(1)

            pbar.set_postfix_str(f"linkage ({self.cluster_linkage})")
            Z = linkage(condensed, method=self.cluster_linkage)
            pbar.update(1)

            pbar.set_postfix_str(f"fcluster (threshold={self.cluster_threshold:.4f})")
            labels = fcluster(Z, t=1.0 - self.cluster_threshold, criterion="distance")
            pbar.update(1)

            pbar.close()

        self._print_cluster_summary(labels)

        conf_scores = None
        if not self.no_confidence:
            conf_scores = self._speaker_confidence_with_progress(embeddings, labels)

        output_path = os.path.join(output_dir, "clusters.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for i, (utt_id, label) in enumerate(zip(utt_ids, labels)):
                record = {"utt_id": utt_id, "cluster_id": int(label)}
                if conf_scores is not None:
                    record["speaker_confidence"] = round(float(conf_scores[i]), 4)
                f.write(json.dumps(record) + "\n")

        logger.info("Cluster assignments written to %s", output_path)

    def _print_cluster_summary(self, labels: np.ndarray) -> None:
        counts = Counter(labels.tolist())
        sizes = sorted(counts.values(), reverse=True)
        n = len(labels)
        singletons = sum(1 for s in sizes if s == 1)
        print(f"\n{'='*50}")
        print(f"  AHC Clustering (threshold={self.cluster_threshold:.4f})")
        print(f"{'='*50}")
        print(f"  Utterances      : {n:,}")
        print(f"  Speakers found  : {len(counts):,}")
        print(f"  Largest cluster : {sizes[0]:,} utts")
        print(f"  Singletons      : {singletons:,}")
        top = min(20, len(sizes))
        print(f"  Top-{top} sizes   : {sizes[:top]}")
        print(f"{'='*50}\n")

    @staticmethod
    def _speaker_confidence_with_progress(
        embeddings: np.ndarray, labels: np.ndarray,
    ) -> np.ndarray:
        n = len(labels)

        pbar = tqdm(total=3, desc="Confidence scoring", unit="step")

        pbar.set_postfix_str("similarity matrix")
        sim_mat = _cosine_similarity_matrix(embeddings)
        pbar.update(1)

        pbar.set_postfix_str("building membership")
        cluster_indices = defaultdict(list)
        for i, lab in enumerate(labels):
            cluster_indices[lab].append(i)

        unique_labels = sorted(cluster_indices.keys())
        label_to_k = {lab: k for k, lab in enumerate(unique_labels)}
        K = len(unique_labels)

        membership = np.zeros((n, K), dtype=np.float32)
        cluster_sizes = np.zeros(K, dtype=np.float32)
        for lab, idxs in cluster_indices.items():
            k = label_to_k[lab]
            membership[idxs, k] = 1.0
            cluster_sizes[k] = len(idxs)

        mean_sim = (sim_mat @ membership) / np.maximum(cluster_sizes, 1.0)
        pbar.update(1)

        pbar.set_postfix_str(f"scoring {n:,} utterances")
        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            my_k = label_to_k[labels[i]]
            my_size = cluster_sizes[my_k]
            if my_size < 2:
                continue
            cohesion = (mean_sim[i, my_k] * my_size - 1.0) / (my_size - 1.0)
            rival_sims = mean_sim[i].copy()
            rival_sims[my_k] = -2.0
            best_rival = rival_sims.max()
            if best_rival <= -2.0:
                scores[i] = 1.0
                continue
            denom = max(cohesion, best_rival)
            if denom <= 0:
                scores[i] = 0.0
            else:
                raw = (cohesion - best_rival) / denom
                scores[i] = max(0.0, min(1.0, raw))
        pbar.update(1)

        pbar.close()
        return scores

    # ── internal helpers ──────────────────────────────────────────────────

    def _get_s3_client(self) -> S3Client:
        if self._s3_client is None:
            self._s3_client = S3Client(self.s3cfg)
        return self._s3_client


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_detect_gpus() -> int:
    try:
        import torch
        count = torch.cuda.device_count()
        return count if count > 0 else 1
    except Exception:
        return 1


def main():
    p = argparse.ArgumentParser(
        description="Integrated speaker-ID pipeline: download -> extract -> embed -> cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--dataset", required=True, choices=KNOWN_DATASETS,
                    help="Dataset name (e.g. yodas, ytc, mcv, fleurs)")
    p.add_argument("--languages", required=True, nargs="+",
                    help="Space-separated language codes (e.g. da hr de cs)")
    p.add_argument("--base-dir", default="/disk_f_nvd/datasets/Yodas/",
                    help="Root data directory")
    p.add_argument("--result-dir", default=None,
                    help="Root directory for output artifacts; defaults to <base-dir>/<lang>/wespeaker_embeddings")
    p.add_argument("--subsets", nargs="+", default=["0_by_whisper", "0_from_captions"],
                    help="Dataset subsets to process")
    p.add_argument("--model", default="voxceleb_resnet293_LM",
                    help="WeSpeaker hub model name (used when --model-checkpoint is not provided)")
    p.add_argument("--model-checkpoint", default=None,
                    help="Local checkpoint path: folder containing config.yaml+avg_model.pt, "
                         "or direct path to avg_model.pt. When set, no hub download is attempted.")
    p.add_argument("--corpusview-path", default=None,
                    help="Path to local corpusview repository checkout")
    p.add_argument("--model-cache-dir", default=None,
                    help="Directory to cache downloaded models")
    p.add_argument("--num-gpus", type=int, default=None,
                    help="Number of GPUs (default: auto-detect)")
    p.add_argument("--batch-dur", type=float, default=600.0,
                    help="Max audio seconds per dynamic batch")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--num-mel-bins", type=int, default=80)
    p.add_argument("--s3cfg", default="~/.s3cfg[default]",
                    help="S3 credentials file + section")
    p.add_argument("--workers", type=int, default=16,
                    help="Parallel download threads")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="Cosine-similarity threshold for clustering")
    p.add_argument("--linkage", default="average",
                    choices=["average", "complete", "single"],
                    help="AHC linkage method")
    p.add_argument("--skip-download", action="store_true",
                    help="Skip S3 download (data already on disk)")
    p.add_argument("--skip-extract", action="store_true",
                    help="Skip tar extraction (wavs already extracted)")
    p.add_argument("--skip-embed", action="store_true",
                    help="Skip embedding extraction")
    p.add_argument("--no-cluster", action="store_true",
                    help="Skip speaker clustering")
    p.add_argument("--no-confidence", action="store_true",
                    help="Skip confidence scoring (faster clustering)")
    p.add_argument("--force-download", action="store_true",
                    help="Re-download even if files exist")

    args = p.parse_args()

    languages = args.languages
    num_gpus = args.num_gpus or _auto_detect_gpus()

    pipeline = SpeakerIDPipeline(
        dataset=args.dataset,
        base_dir=args.base_dir,
        result_dir=args.result_dir,
        s3cfg=args.s3cfg,
        subsets=args.subsets,
        model=args.model,
        model_checkpoint=args.model_checkpoint,
        corpusview_path=args.corpusview_path,
        model_cache_dir=args.model_cache_dir,
        num_gpus=num_gpus,
        batch_dur=args.batch_dur,
        sample_rate=args.sample_rate,
        num_mel_bins=args.num_mel_bins,
        download_workers=args.workers,
        cluster_threshold=args.threshold,
        cluster_linkage=args.linkage,
        skip_download=args.skip_download,
        skip_extract=args.skip_extract,
        skip_embed=args.skip_embed,
        no_cluster=args.no_cluster,
        no_confidence=args.no_confidence,
        force_download=args.force_download,
    )

    pipeline.run(languages)


if __name__ == "__main__":
    main()
