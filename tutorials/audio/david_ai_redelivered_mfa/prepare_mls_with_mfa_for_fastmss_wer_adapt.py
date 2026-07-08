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

"""MLS dataset preparation with ASR-WER filtering, MFA adaptation, and alignment.

Extended pipeline that adds ASR-based transcript verification and MFA model
adaptation on top of the base MLS+MFA preparation.

Stages:
   1. (reserved — download / extract)
   2. Prepare Lhotse recordings + supervisions via ``lhotse.recipes.mls``
   3. Download MFA acoustic, dictionary, G2P, and tokenizer models
   4. Create Lhotse cutsets from recordings + supervisions
   5. Resample audio to 16 kHz WAV; write ``stage5_tasks.jsonl``
   6. Transcript normalization + num2words; write ``stage6_tasks.jsonl``
   7. ASR transcription (NeMo / Whisper) + WER filtering; write ``stage7_tasks.jsonl``
   8. MFA model adaptation (optional 200 h subset, OOV dictionary from full set)
   9. MFA forced alignment (uses adapted model if available)
  10. Merge MFA TextGrids back into Lhotse aligned cutsets
  11. Rewrite recording paths to point at resampled WAV files

Usage::

    python prepare_mls_with_mfa_for_fastmss_wer_adapt.py \\
        --config-path . --config-name input_french_wer_adapt \\
        mls_src_dir=~/multilingual_librispeech \\
        data_dir=mls_workdir \\
        manifests_dir=mls_manifests
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import unicodedata
import urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import hydra
from omegaconf import DictConfig

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
logger = logging.getLogger("mls_mfa_pipeline")

DEFAULT_TARGET_SR = 16000
DEFAULT_SILENCE_MARKERS = ["", "sp", "sil", "spn", "<eps>"]
MLS_BASE_URL = "https://dl.fbaipublicfiles.com/mls"

_MLS_LANGUAGE_TO_NUM2WORDS = {
    "french": "fr",
    "english": "en",
    "german": "de",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
}


# ═══════════════════════════════════════════════════════════════════════════════
# MFA TextGrid helpers (Stage 10)
# ═══════════════════════════════════════════════════════════════════════════════


def _build_textgrid_index(textgrid_dir: str) -> dict[str, str]:
    """Map cut_id -> TextGrid path, searching recursively for speaker sub-dirs."""
    return {p.stem: str(p) for p in Path(textgrid_dir).rglob("*.TextGrid")}


def _align_single_cut(
    cut, *, textgrid_dir: str, tg_index: dict[str, str] | None,
    silence_markers: list[str] | None = None,
):
    """Attach MFA word alignments from a TextGrid to a single Lhotse cut."""
    import textgrid as tg_mod
    from lhotse.supervision import AlignmentItem

    tg_path = tg_index.get(cut.id) if tg_index else os.path.join(textgrid_dir, f"{cut.id}.TextGrid")
    if not tg_path or not os.path.exists(tg_path):
        return None

    try:
        tg = tg_mod.TextGrid.fromFile(tg_path)
    except Exception:
        logger.warning("Skipped '%s': failed to parse TextGrid", cut.id)
        return None

    skip_marks = set(silence_markers) if silence_markers is not None else set(DEFAULT_SILENCE_MARKERS)
    try:
        words_tier = tg.getFirst("words")
    except ValueError:
        logger.warning("Skipped '%s': 'words' tier not found", cut.id)
        return None

    words = [
        AlignmentItem(
            symbol=iv.mark,
            start=round(iv.minTime, 6) + cut.start,
            duration=round(iv.maxTime - iv.minTime, 6),
        )
        for iv in words_tier.intervals
        if iv.mark and iv.mark not in skip_marks
    ]
    if not words:
        return None

    for sup in cut.supervisions:
        if sup.alignment is None:
            sup.alignment = {}
        sup.alignment["word"] = words
    return cut


# ═══════════════════════════════════════════════════════════════════════════════
# WER text normalization
# ═══════════════════════════════════════════════════════════════════════════════


def _normalize_for_wer(text: str) -> str:
    """Normalize text for WER comparison: NFKC, lowercase, strip punctuation."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# AudioTask JSONL helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _write_audio_tasks_jsonl(tasks: list, path: str) -> None:
    """Serialize AudioTask list to JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            row = {
                "task_id": t.task_id,
                "dataset_name": t.dataset_name,
                "data": dict(t.data),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("  Wrote %d tasks to %s", len(tasks), path)


def _read_audio_tasks_jsonl(path: str) -> list:
    """Deserialize AudioTask list from JSONL."""
    from nemo_curator.tasks import AudioTask

    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tasks.append(
                AudioTask(
                    task_id=row["task_id"],
                    dataset_name=row.get("dataset_name", ""),
                    data=row["data"],
                )
            )
    logger.info("  Loaded %d tasks from %s", len(tasks), path)
    return tasks


# ═══════════════════════════════════════════════════════════════════════════════
# MFA corpus export helper (Stage 5)
# ═══════════════════════════════════════════════════════════════════════════════


def _export_cut_to_mfa(cut, output_dir: Path, target_sr: int = DEFAULT_TARGET_SR) -> bool:
    """Convert one Lhotse cut to resampled WAV + transcript TXT for MFA."""
    speakers = {s.speaker for s in cut.supervisions if s.speaker}
    dest = output_dir / sorted(speakers)[0] if speakers else output_dir
    dest.mkdir(parents=True, exist_ok=True)

    wav_path = dest / f"{cut.id}.wav"
    txt_path = dest / f"{cut.id}.txt"

    if wav_path.exists() and txt_path.exists():
        return True

    text = " ".join(s.text for s in cut.supervisions if s.text).strip()
    if not text:
        return False

    src_path = cut.recording.sources[0].source
    if cut.recording.sampling_rate == target_sr and os.path.isfile(src_path):
        os.symlink(os.path.abspath(src_path), str(wav_path))
    else:
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-ar", str(target_sr), "-ac", "1",
            "-acodec", "pcm_s16le",
            "-af", "aresample=dither_method=none",
            str(wav_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
        if result.returncode != 0:
            logger.warning("ffmpeg failed for '%s': %s", cut.id, result.stderr[-200:])
            return False

    txt_path.write_text(text, encoding="utf-8")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline class
# ═══════════════════════════════════════════════════════════════════════════════


class MLSMFAPipeline:
    """End-to-end MLS preparation with ASR WER filtering + MFA adaptation + alignment."""

    def __init__(
        self,
        *,
        language: str,
        mfa_acoustic: str,
        mfa_dict: str,
        mls_src_dir: str,
        data_dir: str,
        manifests_dir: str,
        mfa_g2p: str = "",
        mfa_tokenizer: str = "",
        mfa_root_dir: str = "",
        num_jobs: int = 16,
        splits: list[str] | None = None,
        start_stage: int = 1,
        target_sample_rate: int = DEFAULT_TARGET_SR,
        beam: int = 100,
        retry_beam: int = 400,
        output_format: str = "long_textgrid",
        opus: bool = False,
        silence_markers: list[str] | None = None,
        asr_model_name: str = "nvidia/stt_fr_conformer_transducer_large",
        asr_backend: str = "nemo",
        whisper_model_size: str = "large-v3",
        whisper_language: str = "french",
        adapt_subset_hours: float = 0,
    ):
        self.language = language
        self.mfa_acoustic = mfa_acoustic
        self.mfa_dict = mfa_dict
        self.mls_src_dir = os.path.abspath(os.path.expanduser(mls_src_dir))
        self.data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.manifests_dir = os.path.abspath(os.path.expanduser(manifests_dir))
        self.mfa_g2p = mfa_g2p
        self.mfa_tokenizer = mfa_tokenizer
        self.num_jobs = num_jobs
        self.splits = splits or ["train", "dev", "test"]
        self.start_stage = start_stage
        self.target_sample_rate = target_sample_rate
        self.beam = beam
        self.retry_beam = retry_beam
        self.output_format = output_format
        self.opus = opus
        self.silence_markers = silence_markers if silence_markers is not None else DEFAULT_SILENCE_MARKERS

        self.asr_model_name = asr_model_name
        self.asr_backend = asr_backend
        self.whisper_model_size = whisper_model_size
        self.whisper_language = whisper_language
        self.adapt_subset_hours = adapt_subset_hours

        if mfa_root_dir:
            os.environ["MFA_ROOT_DIR"] = os.path.abspath(os.path.expanduser(mfa_root_dir))

        self.mls_lang_dir = os.path.join(self.mls_src_dir, f"mls_{self.language}")
        self.mls_manifests_dir = os.path.join(self.manifests_dir, f"mls_{self.language}")
        self.manifest_prefix = f"mls-{self.language}"

        log_dir = os.path.join(self.data_dir, "mls", "logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "pipeline.log"))
        fh.setFormatter(logging.Formatter(_LOG_FMT))
        logger.addHandler(fh)

    # ── checkpoint paths ─────────────────────────────────────────────────

    def _pipeline_state_dir(self, split: str) -> str:
        d = os.path.join(self.data_dir, "mls", "pipeline_state", f"{self.language}_{split}")
        os.makedirs(d, exist_ok=True)
        return d

    # ── public API ────────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full pipeline from :attr:`start_stage` onwards."""
        print(f"\n{'=' * 60}")
        print("  MLS + MFA Preparation Pipeline (WER+Adapt)")
        print(f"  Language:        {self.language}")
        print(f"  Source:          {self.mls_lang_dir}")
        print(f"  Manifests:       {self.mls_manifests_dir}")
        print(f"  Data dir:        {self.data_dir}")
        print(f"  Splits:          {' '.join(self.splits)}")
        print(f"  Start stage:     {self.start_stage}")
        print(f"  MFA acoustic:    {self.mfa_acoustic}")
        print(f"  MFA dict:        {self.mfa_dict}")
        if self.mfa_g2p:
            print(f"  MFA G2P:         {self.mfa_g2p}")
        if self.mfa_tokenizer:
            print(f"  MFA tokenizer:   {self.mfa_tokenizer}")
        print(f"  ASR backend:     {self.asr_backend}")
        print(f"  ASR model:       {self.asr_model_name}")
        if self.asr_backend == "whisper":
            print(f"  Whisper size:    {self.whisper_model_size}")
            print(f"  Whisper lang:    {self.whisper_language}")
        print(f"  Adapt subset h:  {self.adapt_subset_hours}")
        print(f"  Num jobs:        {self.num_jobs}")
        print(f"{'=' * 60}\n")

        for d in (self.data_dir, self.manifests_dir, self.mls_manifests_dir):
            os.makedirs(d, exist_ok=True)

        if self.start_stage <= 1:
            self._stage1_download_mls()

        self._validate()

        if self.start_stage <= 2:
            self._stage2_prepare_lhotse_manifests()
        if self.start_stage <= 3:
            self._stage3_download_mfa_models()

        for split in self.splits:
            print(f"\n{'─' * 60}")
            print(f"  Processing {self.language} / {split}")
            print(f"{'─' * 60}")
            self._process_split(split)

        print(f"\n{'=' * 60}")
        print(f"  MLS preparation finished (lang: {self.language})")
        print(f"{'=' * 60}\n")

    # ── Stage 1: Download MLS data ────────────────────────────────────────

    def _stage1_download_mls(self) -> None:
        if os.path.isdir(self.mls_lang_dir) and any(Path(self.mls_lang_dir).iterdir()):
            logger.info("[Stage 1] MLS data already exists at %s. Skipping.", self.mls_lang_dir)
            return

        suffix = f"mls_{self.language}_opus.tar.gz" if self.opus else f"mls_{self.language}.tar.gz"
        url = f"{MLS_BASE_URL}/{suffix}"
        archive_path = os.path.join(self.mls_src_dir, suffix)

        os.makedirs(self.mls_src_dir, exist_ok=True)

        if not os.path.isfile(archive_path):
            logger.info("[Stage 1] Downloading MLS %s from %s ...", self.language, url)
            self._download_with_progress(url, archive_path)
        else:
            logger.info("[Stage 1] Archive already downloaded: %s", archive_path)

        logger.info("[Stage 1] Extracting %s → %s ...", archive_path, self.mls_src_dir)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.mls_src_dir)

        logger.info("[Stage 1] Done. MLS data at %s", self.mls_lang_dir)

    @staticmethod
    def _download_with_progress(url: str, dest: str) -> None:
        partial_file = dest + ".partial"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as resp, open(partial_file, "wb") as f:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 8 * 1024 * 1024
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 / total
                        mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        logger.info("  %.1f / %.1f MB  (%.0f%%)", mb, total_mb, pct)
            os.rename(partial_file, dest)
        except Exception:
            if os.path.isfile(partial_file):
                os.remove(partial_file)
            raise

    # ── validation ────────────────────────────────────────────────────────

    def _validate(self) -> None:
        if not os.path.isdir(self.mls_lang_dir):
            logger.error(
                "MLS source directory not found: %s\n"
                "Download from https://www.openslr.org/94/ and extract so that\n"
                "  %s/train/transcripts.txt exists.",
                self.mls_lang_dir,
                self.mls_lang_dir,
            )
            sys.exit(1)

    # ── Stage 2: Prepare Lhotse recordings + supervisions ─────────────────

    def _stage2_prepare_lhotse_manifests(self) -> None:
        need_prepare = False
        for split in self.splits:
            rec = os.path.join(self.mls_manifests_dir, f"{self.manifest_prefix}_recordings_{split}.jsonl.gz")
            sup = os.path.join(self.mls_manifests_dir, f"{self.manifest_prefix}_supervisions_{split}.jsonl.gz")
            if not os.path.isfile(rec) or not os.path.isfile(sup):
                need_prepare = True
                break

        if not need_prepare:
            logger.info("Lhotse manifests already exist at %s. Skipping stage 2.", self.mls_manifests_dir)
            return

        logger.info("[Stage 2] Preparing Lhotse manifests for MLS %s...", self.language)

        from lhotse import CutSet
        from lhotse.recipes.mls import prepare_mls

        manifests = prepare_mls(
            corpus_dir=Path(self.mls_src_dir),
            output_dir=Path(self.mls_manifests_dir),
            opus=self.opus,
            num_jobs=self.num_jobs,
        )

        if self.language not in manifests:
            logger.error("Language '%s' was not returned by lhotse prepare_mls. Aborting.", self.language)
            sys.exit(1)

        for split, data in manifests[self.language].items():
            cutset_path = Path(self.mls_manifests_dir) / f"{self.manifest_prefix}_cutset_{split}.jsonl.gz"
            if cutset_path.exists():
                logger.info("  [%s/%s] cutset already exists, skipping", self.language, split)
                continue

            recordings = data["recordings"]
            supervisions = data["supervisions"]
            cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
            cuts.to_file(cutset_path)

            total_hours = sum(c.duration for c in cuts) / 3600
            n_speakers = len({s.speaker for c in cuts for s in c.supervisions})
            logger.info(
                "  [%s/%s] %d cuts, %.1fh, %d speakers -> %s",
                self.language, split, len(cuts), total_hours, n_speakers, cutset_path,
            )

        logger.info("[Stage 2] Done.")

    # ── Stage 3: Download MFA models ──────────────────────────────────────

    def _stage3_download_mfa_models(self) -> None:
        logger.info("[Stage 3] Ensuring MFA models are available...")

        for model_type, model_name in [
            ("acoustic", self.mfa_acoustic),
            ("dictionary", self.mfa_dict),
        ]:
            self._mfa_model_download(model_type, model_name)

        if self.mfa_g2p:
            self._mfa_model_download("g2p", self.mfa_g2p)
        if self.mfa_tokenizer:
            self._mfa_model_download("tokenizer", self.mfa_tokenizer)

        logger.info("[Stage 3] Done.")

    @staticmethod
    def _mfa_model_download(model_type: str, model_name: str) -> None:
        cmd = ["mfa", "model", "download", model_type, model_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                "Could not download/refresh %s model '%s' (may already be cached): %s",
                model_type, model_name, result.stderr.strip()[-200:],
            )
        else:
            logger.info("  MFA %s model '%s' OK", model_type, model_name)

    # ── Per-split processing (Stages 4-11) ────────────────────────────────

    def _process_split(self, split: str) -> None:
        rec_file = os.path.join(self.mls_manifests_dir, f"{self.manifest_prefix}_recordings_{split}.jsonl.gz")
        cuts_file = os.path.join(self.mls_manifests_dir, f"{self.manifest_prefix}_cuts_{split}.jsonl.gz")

        state_dir = self._pipeline_state_dir(split)
        stage5_ckpt = os.path.join(state_dir, "stage5_tasks.jsonl")
        stage6_ckpt = os.path.join(state_dir, "stage6_tasks.jsonl")
        stage7_ckpt = os.path.join(state_dir, "stage7_tasks.jsonl")
        stage9_ckpt = os.path.join(state_dir, "stage9_tasks.jsonl")

        mfa_corpus_dir = os.path.join(self.data_dir, "mls", "mfa_corpus", f"{self.language}_{split}")
        mfa_textgrid_dir = os.path.join(self.data_dir, "mls", "mfa_textgrids", f"{self.language}_{split}")
        resampled_dir = os.path.join(self.data_dir, "mls", "resampled", f"{self.language}_{split}")

        cuts_aligned = os.path.join(
            self.mls_manifests_dir,
            f"{self.manifest_prefix}_{self.target_sample_rate}_aligned_cuts_{split}.jsonl.gz",
        )

        if not os.path.isfile(rec_file):
            logger.warning("%s not found. Skipping split '%s'.", rec_file, split)
            return

        if self.start_stage <= 4:
            self._stage4_create_cutset(split)
        if self.start_stage <= 5:
            self._stage5_resample_and_checkpoint(cuts_file, resampled_dir, stage5_ckpt, split)
        if self.start_stage <= 6:
            self._stage6_transcript_normalize(stage5_ckpt, stage6_ckpt, split)
        if self.start_stage <= 7:
            self._stage7_asr_wer_filter_and_checkpoint(stage6_ckpt, stage7_ckpt, split)
        if self.start_stage <= 8:
            self._stage8_mfa_adapt(stage7_ckpt, split)
        if self.start_stage <= 9:
            self._stage9_mfa_alignment(stage7_ckpt, mfa_textgrid_dir, stage9_ckpt, split)
        if self.start_stage <= 10:
            self._stage10_merge_textgrids(cuts_file, mfa_textgrid_dir, cuts_aligned, split)
        if self.start_stage <= 11:
            self._stage11_rewrite_paths(cuts_aligned, resampled_dir, split)

    # ── Stage 4: Create cutset ────────────────────────────────────────────

    def _stage4_create_cutset(self, split: str) -> None:
        rec_file = os.path.join(self.mls_manifests_dir, f"{self.manifest_prefix}_recordings_{split}.jsonl.gz")
        sup_file = os.path.join(self.mls_manifests_dir, f"{self.manifest_prefix}_supervisions_{split}.jsonl.gz")
        cuts_file = os.path.join(self.mls_manifests_dir, f"{self.manifest_prefix}_cuts_{split}.jsonl.gz")

        if os.path.isfile(cuts_file):
            logger.info("  [Stage 4] Cutset already exists for %s. Skipping.", split)
            return

        logger.info("  [Stage 4] Creating cutset for %s...", split)
        from lhotse import CutSet, fix_manifests, load_manifest

        recordings = load_manifest(rec_file)
        supervisions = load_manifest(sup_file)
        cuts = CutSet.from_manifests(*fix_manifests(recordings, supervisions))
        cuts.to_file(cuts_file)
        logger.info("  [Stage 4] Wrote %s", cuts_file)

    # ── Stage 5: Resample + create AudioTask checkpoints ──────────────────

    def _stage5_resample_and_checkpoint(
        self, cuts_file: str, resampled_dir: str, stage5_ckpt: str, split: str,
    ) -> None:
        if os.path.isfile(stage5_ckpt):
            logger.info("  [Stage 5] Checkpoint exists for %s. Skipping.", split)
            return

        logger.info("  [Stage 5] Resampling + building AudioTasks for %s...", split)
        from lhotse import CutSet
        from tqdm.auto import tqdm

        from nemo_curator.tasks import AudioTask

        cuts = CutSet.from_file(cuts_file)
        resampled_path = Path(resampled_dir)
        resampled_path.mkdir(parents=True, exist_ok=True)

        tasks: list[AudioTask] = []
        exported, skipped = 0, 0

        for cut in tqdm(cuts, desc=f"Resample {split}"):
            text = " ".join(s.text for s in cut.supervisions if s.text).strip()
            if not text:
                skipped += 1
                continue

            speakers = {s.speaker for s in cut.supervisions if s.speaker}
            speaker = sorted(speakers)[0] if speakers else "unknown"
            spk_dir = resampled_path / speaker
            spk_dir.mkdir(parents=True, exist_ok=True)

            src_path = cut.recording.sources[0].source
            already_ok = (cut.recording.sampling_rate == self.target_sample_rate
                          and os.path.isfile(src_path))
            if already_ok:
                audio_path = src_path
            else:
                wav_path = spk_dir / f"{cut.id}.wav"
                audio_path = str(wav_path)
                if not wav_path.exists():
                    cmd = [
                        "ffmpeg", "-y", "-i", src_path,
                        "-ar", str(self.target_sample_rate), "-ac", "1",
                        "-acodec", "pcm_s16le",
                        "-af", "aresample=dither_method=none",
                        str(wav_path),
                    ]
                    result = subprocess.run(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                        text=True, errors="replace",
                    )
                    if result.returncode != 0:
                        logger.warning("ffmpeg failed for '%s': %s", cut.id, result.stderr[-200:])
                        skipped += 1
                        continue

            tasks.append(
                AudioTask(
                    task_id=cut.id,
                    dataset_name=f"mls_{self.language}",
                    data={
                        "audio_filepath": audio_path,
                        "text": text,
                        "speaker": speaker,
                        "duration": float(cut.duration),
                    },
                )
            )
            exported += 1

        _write_audio_tasks_jsonl(tasks, stage5_ckpt)
        logger.info("  [Stage 5] Exported %d tasks, skipped %d", exported, skipped)

    # ── Stage 6: Transcript normalization + num2words ─────────────────────

    def _stage6_transcript_normalize(
        self, stage5_ckpt: str, stage6_ckpt: str, split: str,
    ) -> None:
        if os.path.isfile(stage6_ckpt):
            logger.info("  [Stage 6] Checkpoint exists for %s. Skipping.", split)
            return

        logger.info("  [Stage 6] Normalizing transcripts for %s...", split)

        from nemo_curator.stages.audio.preprocessing.transcript_normalization import (
            AudioTranscriptNormalizeStage,
            resolve_alphabet,
        )

        tasks = _read_audio_tasks_jsonl(stage5_ckpt)

        num2words_lang = _MLS_LANGUAGE_TO_NUM2WORDS.get(self.language, "")
        alphabet = resolve_alphabet(self.language, None)

        norm_stage = AudioTranscriptNormalizeStage(
            language=self.language,
            alphabet=alphabet,
            num2words_lang=num2words_lang,
            text_key="text",
        )

        normalized = []
        for task in tasks:
            out = norm_stage.process(task)
            if out is not None:
                normalized.append(out)

        logger.info("  [Stage 6] %d -> %d tasks after normalization", len(tasks), len(normalized))
        _write_audio_tasks_jsonl(normalized, stage6_ckpt)

    # ── Stage 7: ASR + WER filter ─────────────────────────────────────────

    def _stage7_asr_wer_filter_and_checkpoint(
        self, stage6_ckpt: str, stage7_ckpt: str, split: str,
    ) -> None:
        if os.path.isfile(stage7_ckpt):
            logger.info("  [Stage 7] Checkpoint exists for %s. Skipping.", split)
            return

        logger.info("  [Stage 7] ASR transcription + WER filtering for %s (backend=%s)...",
                     split, self.asr_backend)

        audio_tasks = _read_audio_tasks_jsonl(stage6_ckpt)
        total_before = len(audio_tasks)

        if self.asr_backend == "whisper":
            self._transcribe_whisper(audio_tasks, total_before)
        else:
            self._transcribe_nemo(audio_tasks, total_before)

        # Normalize texts for WER
        for task in audio_tasks:
            task.data["text_norm"] = _normalize_for_wer(task.data.get("text", ""))
            task.data["pred_text_norm"] = _normalize_for_wer(task.data.get("pred_text", ""))

        # Compute WER on normalized texts
        from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage

        wer_stage = GetPairwiseWerStage(
            text_key="text_norm",
            pred_text_key="pred_text_norm",
            wer_key="wer",
        )
        for task in audio_tasks:
            wer_stage.process(task)

        # Log transcription details + write TSV
        state_dir = os.path.dirname(os.path.join(
            self.data_dir, "mls", "pipeline_state", f"{self.language}_{split}", "",
        ))
        tsv_path = os.path.join(state_dir, f"stage7_transcriptions_{split}.tsv")
        os.makedirs(state_dir, exist_ok=True)

        wer_values = []
        with open(tsv_path, "w", encoding="utf-8") as tsv:
            tsv.write("task_id\twer\tref_text\tpred_text\tref_norm\tpred_norm\n")
            for task in audio_tasks:
                wer_val = task.data.get("wer", -1)
                wer_values.append(wer_val)
                tsv.write(
                    f"{task.task_id}\t{wer_val}\t"
                    f"{task.data.get('text', '')}\t{task.data.get('pred_text', '')}\t"
                    f"{task.data.get('text_norm', '')}\t{task.data.get('pred_text_norm', '')}\n"
                )

        if wer_values:
            avg_wer = sum(wer_values) / len(wer_values)
            zero_wer = sum(1 for w in wer_values if w == 0.0)
            logger.info(
                "  [Stage 7] WER stats: avg=%.2f%%, exact_match=%d/%d (%.1f%%)",
                avg_wer, zero_wer, len(wer_values), 100.0 * zero_wer / len(wer_values),
            )
        logger.info("  [Stage 7] TSV written to %s", tsv_path)

        # Filter: keep only WER == 0
        from nemo_curator.stages.audio.common import PreserveByValueStage

        wer_filter = PreserveByValueStage(input_value_key="wer", target_value=0.0, operator="eq")
        kept = [t for t in audio_tasks if wer_filter.process(t) is not None]

        logger.info(
            "  [Stage 7] WER filter: %d -> %d tasks (kept %.1f%%)",
            total_before, len(kept), 100.0 * len(kept) / max(total_before, 1),
        )
        _write_audio_tasks_jsonl(kept, stage7_ckpt)

    def _transcribe_nemo(self, audio_tasks: list, total_before: int) -> None:
        """Transcribe using a NeMo ASR model."""
        import torch

        import nemo.collections.asr as nemo_asr

        logger.info("  [Stage 7] Loading NeMo model: %s ...", self.asr_model_name)

        model = None
        model_classes = [
            nemo_asr.models.EncDecRNNTBPEModel,
            nemo_asr.models.EncDecCTCModelBPE,
            nemo_asr.models.EncDecHybridRNNTCTCBPEModel,
        ]
        for cls in model_classes:
            try:
                model = cls.from_pretrained(
                    model_name=self.asr_model_name,
                    map_location=torch.device("cuda"),
                )
                logger.info("  [Stage 7] Loaded model via %s", cls.__name__)
                break
            except Exception:
                continue

        if model is None:
            raise RuntimeError(
                f"Could not load NeMo model '{self.asr_model_name}' "
                f"with any of: {[c.__name__ for c in model_classes]}"
            )

        from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage

        asr_stage = InferenceAsrNemoStage(
            model_name=self.asr_model_name,
            asr_model=model,
            pred_text_key="pred_text",
        )

        batch_size = 16
        for i in range(0, len(audio_tasks), batch_size):
            batch = audio_tasks[i : i + batch_size]
            asr_stage.process_batch(batch)
            done = min(i + batch_size, len(audio_tasks))
            if done % 1000 < batch_size or done == len(audio_tasks):
                logger.info("  [Stage 7] ASR progress: %d / %d", done, total_before)

    def _transcribe_whisper(self, audio_tasks: list, total_before: int) -> None:
        """Transcribe using OpenAI Whisper."""
        import whisper

        logger.info("  [Stage 7] Loading Whisper model: %s ...", self.whisper_model_size)
        model = whisper.load_model(self.whisper_model_size)

        for i, task in enumerate(audio_tasks):
            audio_path = task.data.get("audio_filepath", "")
            result = model.transcribe(audio_path, language=self.whisper_language)
            task.data["pred_text"] = result.get("text", "").strip()

            if (i + 1) % 500 == 0 or (i + 1) == total_before:
                logger.info("  [Stage 7] Whisper progress: %d / %d", i + 1, total_before)

    # ── Stage 8: MFA adapt ────────────────────────────────────────────────

    def _stage8_mfa_adapt(self, stage7_ckpt: str, split: str) -> None:
        adapted_model_dir = os.path.join(self.data_dir, "mls", "mfa_adapted_model", self.language)
        adapted_acoustic = os.path.join(adapted_model_dir, f"{self.mfa_acoustic}_adapted.zip")

        if os.path.isfile(adapted_acoustic):
            logger.info("  [Stage 8] Adapted model already exists: %s. Skipping.", adapted_acoustic)
            return

        logger.info("  [Stage 8] MFA model adaptation for %s...", split)

        all_tasks = _read_audio_tasks_jsonl(stage7_ckpt)

        # Determine adaptation subset
        if self.adapt_subset_hours > 0:
            total_hours = sum(t.data.get("duration", 0.0) for t in all_tasks) / 3600
            if total_hours > self.adapt_subset_hours:
                logger.info(
                    "  [Stage 8] Selecting %.0f h subset (full set: %.1f h)...",
                    self.adapt_subset_hours, total_hours,
                )
                adapt_tasks = self._select_adapt_subset(all_tasks, self.adapt_subset_hours)
            else:
                logger.info("  [Stage 8] Full set (%.1f h) <= target (%.0f h), using all.",
                            total_hours, self.adapt_subset_hours)
                adapt_tasks = all_tasks
        else:
            adapt_tasks = all_tasks

        adapt_hours = sum(t.data.get("duration", 0.0) for t in adapt_tasks) / 3600
        adapt_speakers = len({t.data.get("speaker", "?") for t in adapt_tasks})
        logger.info(
            "  [Stage 8] Adapt subset: %d tasks, %.1f h, %d speakers",
            len(adapt_tasks), adapt_hours, adapt_speakers,
        )

        # Build dictionary from ALL tasks (covers full alignment vocabulary)
        logger.info(
            "  [Stage 8] Building dictionary from ALL %d tasks (covers full alignment vocab)...",
            len(all_tasks),
        )
        corpus_dict_path = self._build_corpus_dictionary(all_tasks, split)

        # Build MFA adapt corpus from subset
        suffix = f"_{int(self.adapt_subset_hours)}h" if self.adapt_subset_hours > 0 else ""
        adapt_corpus_dir = os.path.join(
            self.data_dir, "mls", "mfa_adapt_corpus", f"{self.language}_{split}{suffix}",
        )

        # Clean adapt corpus before rebuilding
        if os.path.isdir(adapt_corpus_dir):
            shutil.rmtree(adapt_corpus_dir)
        os.makedirs(adapt_corpus_dir, exist_ok=True)

        for task in adapt_tasks:
            spk = task.data.get("speaker", "unknown")
            spk_dir = Path(adapt_corpus_dir) / spk
            spk_dir.mkdir(parents=True, exist_ok=True)

            wav_src = task.data["audio_filepath"]
            wav_dst = spk_dir / f"{task.task_id}.wav"
            txt_dst = spk_dir / f"{task.task_id}.txt"

            if not wav_dst.exists():
                os.symlink(os.path.abspath(wav_src), str(wav_dst))
            if not txt_dst.exists():
                txt_dst.write_text(task.data.get("text", ""), encoding="utf-8")

        # Run mfa adapt
        os.makedirs(adapted_model_dir, exist_ok=True)
        adapt_cmd = [
            "mfa", "adapt",
            adapt_corpus_dir,
            corpus_dict_path,
            self.mfa_acoustic,
            adapted_acoustic,
            "-j", str(self.num_jobs),
            "--beam", str(self.beam),
            "--retry_beam", str(self.retry_beam),
            "--output_format", self.output_format,
        ]

        logger.info("  [Stage 8] Running: %s", " ".join(adapt_cmd))
        result = subprocess.run(adapt_cmd, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"mfa adapt failed (exit {result.returncode})")

        logger.info("  [Stage 8] Adapted model saved to %s", adapted_acoustic)

    def _build_corpus_dictionary(self, tasks: list, split: str) -> str:
        """Build a combined dictionary (base MFA dict + G2P pronunciations for OOV words)."""
        dict_dir = os.path.join(self.data_dir, "mls", "mfa_adapt_dict", f"{self.language}_{split}")

        # Clean before rebuilding
        if os.path.isdir(dict_dir):
            shutil.rmtree(dict_dir)
        os.makedirs(dict_dir, exist_ok=True)

        # Collect all words from tasks
        all_words = set()
        for task in tasks:
            text = task.data.get("text", "")
            all_words.update(text.split())

        # Load base MFA dictionary to find OOV words
        base_dict_words = set()
        mfa_root = os.environ.get("MFA_ROOT_DIR", os.path.expanduser("~/.mfa"))
        dict_path = os.path.join(mfa_root, "pretrained_models", "dictionary", f"{self.mfa_dict}.dict")
        if not os.path.isfile(dict_path):
            dict_path_alt = dict_path.replace(".dict", ".txt")
            if os.path.isfile(dict_path_alt):
                dict_path = dict_path_alt

        if os.path.isfile(dict_path):
            with open(dict_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        base_dict_words.add(parts[0].lower())
            logger.info("  [Stage 8] Base dictionary has %d words", len(base_dict_words))

        oov_words = {w for w in all_words if w.lower() not in base_dict_words}
        logger.info("  [Stage 8] OOV words: %d / %d total", len(oov_words), len(all_words))

        oov_list_path = os.path.join(dict_dir, "oov_words.txt")
        with open(oov_list_path, "w", encoding="utf-8") as f:
            for w in sorted(oov_words):
                f.write(w + "\n")

        if oov_words and self.mfa_g2p:
            oov_pron_path = os.path.join(dict_dir, "oov_pronunciations.txt")
            g2p_cmd = [
                "mfa", "g2p",
                self.mfa_g2p,
                oov_list_path,
                oov_pron_path,
            ]
            logger.info("  [Stage 8] Running G2P: %s", " ".join(g2p_cmd))
            result = subprocess.run(g2p_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("  mfa g2p failed: %s", result.stderr[-500:])
            else:
                logger.info("  [Stage 8] G2P generated pronunciations at %s", oov_pron_path)

        # Combine base dict + OOV pronunciations
        combined_dict_path = os.path.join(dict_dir, "combined_dictionary.txt")
        with open(combined_dict_path, "w", encoding="utf-8") as out:
            if os.path.isfile(dict_path):
                with open(dict_path, encoding="utf-8") as f:
                    out.write(f.read())

            oov_pron_path = os.path.join(dict_dir, "oov_pronunciations.txt")
            if os.path.isfile(oov_pron_path):
                with open(oov_pron_path, encoding="utf-8") as f:
                    out.write("\n")
                    out.write(f.read())

        logger.info("  [Stage 8] Combined dictionary: %s", combined_dict_path)
        return combined_dict_path

    def _select_adapt_subset(self, tasks: list, target_hours: float) -> list:
        """Select a time-limited subset ensuring all speakers are represented (round-robin)."""
        target_seconds = target_hours * 3600
        by_speaker: dict[str, list] = defaultdict(list)
        for t in tasks:
            by_speaker[t.data.get("speaker", "unknown")].append(t)

        logger.info("  [Stage 8] %d speakers total, target %.0f h", len(by_speaker), target_hours)

        selected: list = []
        total_dur = 0.0

        speaker_iters = {spk: iter(tlist) for spk, tlist in by_speaker.items()}
        active_speakers = set(speaker_iters.keys())

        while active_speakers and total_dur < target_seconds:
            exhausted = set()
            for spk in sorted(active_speakers):
                if total_dur >= target_seconds:
                    break
                try:
                    task = next(speaker_iters[spk])
                    dur = task.data.get("duration", 0.0)
                    selected.append(task)
                    total_dur += dur
                except StopIteration:
                    exhausted.add(spk)
            active_speakers -= exhausted

        selected_hours = total_dur / 3600
        selected_speakers = len({t.data.get("speaker", "?") for t in selected})
        logger.info(
            "  [Stage 8] Subset selected: %d tasks, %.1f h, %d speakers",
            len(selected), selected_hours, selected_speakers,
        )
        return selected

    # ── Stage 9: MFA alignment ────────────────────────────────────────────

    def _stage9_mfa_alignment(
        self, stage7_ckpt: str, mfa_textgrid_dir: str, stage9_ckpt: str, split: str,
    ) -> None:
        if os.path.isfile(stage9_ckpt):
            logger.info("  [Stage 9] Checkpoint exists for %s. Skipping.", split)
            return

        logger.info("  [Stage 9] MFA alignment for %s...", split)

        # Use adapted model if available
        adapted_model_dir = os.path.join(self.data_dir, "mls", "mfa_adapted_model", self.language)
        adapted_acoustic = os.path.join(adapted_model_dir, f"{self.mfa_acoustic}_adapted.zip")
        acoustic_model = adapted_acoustic if os.path.isfile(adapted_acoustic) else self.mfa_acoustic

        if acoustic_model == adapted_acoustic:
            logger.info("  [Stage 9] Using adapted model: %s", adapted_acoustic)
        else:
            logger.info("  [Stage 9] Using base model: %s", acoustic_model)

        # Use combined dictionary if available
        dict_dir = os.path.join(self.data_dir, "mls", "mfa_adapt_dict", f"{self.language}_{split}")
        combined_dict = os.path.join(dict_dir, "combined_dictionary.txt")
        dictionary = combined_dict if os.path.isfile(combined_dict) else self.mfa_dict

        if dictionary == combined_dict:
            logger.info("  [Stage 9] Using combined dictionary: %s", combined_dict)

        audio_tasks = _read_audio_tasks_jsonl(stage7_ckpt)

        from nemo_curator.stages.audio.alignment import MFAAlignmentStage

        mfa_root = os.environ.get("MFA_ROOT_DIR", "")
        mfa_stage = MFAAlignmentStage(
            acoustic_model=acoustic_model,
            dictionary=dictionary,
            g2p_model=self.mfa_g2p,
            output_dir=mfa_textgrid_dir,
            audio_filepath_key="audio_filepath",
            text_key="text",
            speaker_key="speaker",
            num_jobs=self.num_jobs,
            beam=self.beam,
            retry_beam=self.retry_beam,
            output_format=self.output_format,
            mfa_root_dir=mfa_root,
            copy_models_to_local=False,
            single_speaker=False,
            clean=True,
            use_mp=True,
            create_rttm=False,
            create_ctm=False,
        )

        mfa_stage.setup(None)

        batch_size = 5000
        all_results = []
        for i in range(0, len(audio_tasks), batch_size):
            batch = audio_tasks[i : i + batch_size]
            results = mfa_stage.process_batch(batch)
            all_results.extend(results)
            logger.info("  [Stage 9] Aligned %d / %d tasks", len(all_results), len(audio_tasks))

        mfa_stage.teardown()

        _write_audio_tasks_jsonl(all_results, stage9_ckpt)
        logger.info("  [Stage 9] Done. %d tasks aligned.", len(all_results))

    # ── Stage 10: Merge TextGrids into Lhotse aligned cuts ────────────────

    def _stage10_merge_textgrids(
        self, cuts_file: str, mfa_textgrid_dir: str, cuts_aligned: str, split: str,
    ) -> None:
        if os.path.isfile(cuts_aligned):
            logger.info("  [Stage 10] Aligned cutset already exists for %s. Skipping.", split)
            return

        logger.info("  [Stage 10] Merging MFA TextGrids into Lhotse cuts for %s...", split)
        from lhotse import CutSet

        if not os.path.isfile(cuts_file):
            logger.warning("  [Stage 10] Cutset file not found: %s", cuts_file)
            return

        cuts = CutSet.from_file(cuts_file)
        total_count = len(cuts)

        tg_dir = os.path.join(mfa_textgrid_dir, "textgrids") if os.path.isdir(
            os.path.join(mfa_textgrid_dir, "textgrids")
        ) else mfa_textgrid_dir

        tg_index = _build_textgrid_index(tg_dir)
        logger.info("  Found %d TextGrid files", len(tg_index))

        align_fn = partial(
            _align_single_cut,
            textgrid_dir=tg_dir,
            tg_index=tg_index,
            silence_markers=self.silence_markers,
        )

        if self.num_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.num_jobs) as executor:
                mapped = list(executor.map(align_fn, cuts))
            aligned_cuts = CutSet.from_items(c for c in mapped if c is not None)
        else:
            aligned_cuts = cuts.map(align_fn).filter(lambda c: c is not None)

        aligned_cuts.to_file(cuts_aligned)
        logger.info(
            "  [Stage 10] Wrote %s  (%d/%d cuts matched)",
            cuts_aligned, len(aligned_cuts), total_count,
        )

    # ── Stage 11: Rewrite recording paths ─────────────────────────────────

    @staticmethod
    def _stage11_rewrite_paths(cuts_aligned: str, resampled_dir: str, split: str) -> None:
        if not os.path.isfile(cuts_aligned):
            logger.warning("  [Stage 11] Aligned cutset not found for %s. Skipping.", split)
            return

        logger.info("  [Stage 11] Updating recording paths to resampled WAVs...")
        import soundfile as sf
        from lhotse import CutSet

        wav_index = {p.stem: str(p) for p in Path(resampled_dir).rglob("*.wav")}

        cuts = CutSet.from_file(cuts_aligned).to_eager()
        updated = 0
        for cut in cuts:
            wav_path = wav_index.get(cut.id)
            if wav_path:
                info = sf.info(wav_path)
                cut.recording.sources[0].source = wav_path
                cut.recording.sampling_rate = info.samplerate
                cut.recording.num_samples = info.frames
                cut.recording.duration = info.frames / info.samplerate
                cut.recording.transforms = None
                updated += 1

        cuts.to_file(cuts_aligned)
        logger.info("  [Stage 11] Updated %d/%d recording paths", updated, len(cuts))


# ═══════════════════════════════════════════════════════════════════════════════
# Hydra entry point
# ═══════════════════════════════════════════════════════════════════════════════


def _pipeline_from_cfg(cfg) -> MLSMFAPipeline:
    """Construct :class:`MLSMFAPipeline` from a Hydra / OmegaConf config."""
    splits = list(cfg.get("splits", ["train", "dev", "test"]))

    return MLSMFAPipeline(
        language=cfg.language,
        mfa_acoustic=cfg.mfa_acoustic,
        mfa_dict=cfg.mfa_dict,
        mls_src_dir=cfg.mls_src_dir,
        data_dir=cfg.data_dir,
        manifests_dir=cfg.manifests_dir,
        mfa_g2p=cfg.get("mfa_g2p", ""),
        mfa_tokenizer=cfg.get("mfa_tokenizer", ""),
        mfa_root_dir=cfg.get("mfa_root_dir", ""),
        num_jobs=int(cfg.get("num_jobs", 16)),
        splits=splits,
        start_stage=int(cfg.get("stage", 1)),
        target_sample_rate=int(cfg.get("target_sample_rate", DEFAULT_TARGET_SR)),
        beam=int(cfg.get("beam", 100)),
        retry_beam=int(cfg.get("retry_beam", 400)),
        output_format=cfg.get("output_format", "long_textgrid"),
        opus=bool(cfg.get("opus", False)),
        silence_markers=list(cfg.get("silence_markers", DEFAULT_SILENCE_MARKERS)),
        asr_model_name=cfg.get("asr_model_name", "nvidia/stt_fr_conformer_transducer_large"),
        asr_backend=cfg.get("asr_backend", "nemo"),
        whisper_model_size=cfg.get("whisper_model_size", "large-v3"),
        whisper_language=cfg.get("whisper_language", "french"),
        adapt_subset_hours=float(cfg.get("adapt_subset_hours", 0)),
    )


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Run MLS + MFA pipeline (WER+Adapt) from YAML configuration."""
    from omegaconf import OmegaConf

    logger.info("MLS + MFA Preparation Pipeline (WER + Adapt)")
    logger.info("=" * 60)
    logger.info("Effective config:\n%s", OmegaConf.to_yaml(cfg))

    pipeline = _pipeline_from_cfg(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
