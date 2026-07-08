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

"""MLS dataset preparation with MFA alignment + NFA cross-validation filtering.

Runs MFA alignment, then independently runs NFA (NeMo Forced Aligner) on the
same data.  Utterances where the two aligners disagree by more than a
configurable threshold (default 100 ms) are filtered out.

Stages:
   1. (reserved -- download / extract)
   2. Prepare Lhotse recordings + supervisions
   3. Download MFA models
   4. Create Lhotse cutsets
   5. Resample audio to 16 kHz WAV; write stage5_tasks.jsonl
   6. Transcript normalization + num2words; write stage6_tasks.jsonl
   7. MFA forced alignment; write stage7_tasks.jsonl
   8. NFA forced alignment (NeMo CTC); produce NFA TextGrids
   9. Compare NFA vs MFA, filter diff > threshold; write stage9_tasks.jsonl
  10. Merge MFA TextGrids into Lhotse aligned cutsets
  11. Rewrite recording paths to resampled WAV files

Usage::

    python prepare_mls_with_mfa_for_fastmss_nfa_filter.py \\
        --config-path . --config-name input_french_nfa_filter \\
        mls_src_dir=~/multilingual_librispeech \\
        data_dir=mls_workdir \\
        manifests_dir=mls_manifests
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tarfile
import urllib.request
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
    "french": "fr", "english": "en", "german": "de",
    "spanish": "es", "italian": "it", "portuguese": "pt",
}


# ---------------------------------------------------------------------------
# TextGrid helpers
# ---------------------------------------------------------------------------

def _build_textgrid_index(textgrid_dir: str) -> dict[str, str]:
    return {p.stem: str(p) for p in Path(textgrid_dir).rglob("*.TextGrid")}


def _textgrid_key_for_cut(cut) -> str:
    """TextGrid stems match the source audio basename, not the cut id suffix."""
    return Path(cut.recording.sources[0].source).stem


_STAGE10_TG_INDEX: dict[str, str] | None = None
_STAGE10_SILENCE_MARKERS: list | None = None


def _init_stage10_worker(textgrid_dir: str, silence_markers) -> None:
    global _STAGE10_TG_INDEX, _STAGE10_SILENCE_MARKERS
    _STAGE10_TG_INDEX = _build_textgrid_index(textgrid_dir)
    _STAGE10_SILENCE_MARKERS = list(silence_markers) if silence_markers is not None else None


def _align_single_cut_worker(cut):
    return _align_single_cut(
        cut,
        textgrid_dir=None,
        tg_index=_STAGE10_TG_INDEX,
        silence_markers=_STAGE10_SILENCE_MARKERS,
    )


def _parse_textgrid_words(path: str) -> list[tuple[float, float, str]]:
    """Return ``[(start, end, word_lower), ...]`` from the *words* tier."""
    text = Path(path).read_text()
    m = re.search(
        r'name\s*=\s*"words"\s*\n\s*xmin\s*=\s*[\d.]+\s*\n\s*xmax\s*=\s*[\d.]+\s*\n'
        r'\s*intervals:\s*size\s*=\s*(\d+)', text,
    )
    if not m:
        return []
    tier_text = text[m.start():]
    intervals = []
    for iv in re.finditer(
        r'intervals\s*\[\d+\]:\s*\n\s*xmin\s*=\s*([\d.]+)\s*\n'
        r'\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*"([^"]*)"', tier_text,
    ):
        xmin, xmax, word = float(iv.group(1)), float(iv.group(2)), iv.group(3).strip().lower()
        if word and word not in {"<b>", "<blank>"}:
            intervals.append((xmin, xmax, word))
    return intervals


def _align_word_sequences(mfa_words, nfa_words):
    """Greedy ordered alignment of two word sequences by matching text."""
    pairs, j = [], 0
    for mfa_item in mfa_words:
        for k in range(j, min(j + 3, len(nfa_words))):
            if nfa_words[k][2] == mfa_item[2]:
                pairs.append((mfa_item, nfa_words[k]))
                j = k + 1
                break
    return pairs


def _align_single_cut(cut, *, textgrid_dir, tg_index, silence_markers=None):
    import textgrid as tg_mod
    from lhotse.supervision import AlignmentItem

    tg_key = _textgrid_key_for_cut(cut)
    tg_path = tg_index.get(tg_key) if tg_index else os.path.join(textgrid_dir, f"{tg_key}.TextGrid")
    if not tg_path or not os.path.exists(tg_path):
        return None
    try:
        tg = tg_mod.TextGrid.fromFile(tg_path)
    except Exception:
        return None
    skip = set(silence_markers) if silence_markers is not None else set(DEFAULT_SILENCE_MARKERS)
    try:
        words_tier = tg.getFirst("words")
    except ValueError:
        return None
    words = [
        AlignmentItem(symbol=iv.mark, start=round(iv.minTime, 6) + cut.start,
                       duration=round(iv.maxTime - iv.minTime, 6))
        for iv in words_tier.intervals if iv.mark and iv.mark not in skip
    ]
    if not words:
        return None
    for sup in cut.supervisions:
        if sup.alignment is None:
            sup.alignment = {}
        sup.alignment["word"] = words
    return cut


# ---------------------------------------------------------------------------
# TextGrid writer
# ---------------------------------------------------------------------------

def _write_textgrid(words, output_path, audio_duration=None):
    if not words:
        return
    xmax = audio_duration if audio_duration else words[-1][1] + 0.01
    intervals, prev = [], 0.0
    for s, e, w in words:
        if s > prev + 0.001:
            intervals.append((prev, s, ""))
        intervals.append((s, e, w))
        prev = e
    if prev < xmax:
        intervals.append((prev, xmax, ""))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write('File type = "ooTextFile"\nObject class = "TextGrid"\n\n')
        f.write(f"xmin = 0.0\nxmax = {xmax}\ntiers? <exists>\nsize = 1\nitem []:\n")
        f.write('    item [1]:\n        class = "IntervalTier"\n        name = "words"\n')
        f.write(f"        xmin = 0.0\n        xmax = {xmax}\n        intervals: size = {len(intervals)}\n")
        for i, (s, e, t) in enumerate(intervals, 1):
            f.write(f'        intervals [{i}]:\n            xmin = {s}\n            xmax = {e}\n            text = "{t}"\n')


# ---------------------------------------------------------------------------
# AudioTask JSONL helpers
# ---------------------------------------------------------------------------

def _write_audio_tasks_jsonl(tasks, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps({"task_id": t.task_id, "dataset_name": t.dataset_name,
                                "data": dict(t.data)}, ensure_ascii=False) + "\n")
    logger.info("  Wrote %d tasks to %s", len(tasks), path)


def _read_audio_tasks_jsonl(path):
    from nemo_curator.tasks import AudioTask
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tasks.append(AudioTask(task_id=r["task_id"], dataset_name=r.get("dataset_name", ""), data=r["data"]))
    logger.info("  Loaded %d tasks from %s", len(tasks), path)
    return tasks


# ===========================================================================
# Pipeline class
# ===========================================================================

class MLSMFAPipeline:
    """MLS preparation with MFA alignment + NFA cross-validation filtering."""

    def __init__(
        self, *, language, mfa_acoustic, mfa_dict, mls_src_dir, data_dir, manifests_dir,
        mfa_g2p="", mfa_tokenizer="", mfa_root_dir="", num_jobs=16,
        splits=None, start_stage=1, target_sample_rate=DEFAULT_TARGET_SR,
        beam=100, retry_beam=400, output_format="long_textgrid",
        opus=False, silence_markers=None,
        nfa_model_name="nvidia/stt_fr_conformer_ctc_large",
        nfa_batch_size=32, nfa_max_diff_ms=100.0,
        enable_nfa_filter=True,
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
        self.nfa_model_name = nfa_model_name
        self.nfa_batch_size = nfa_batch_size
        self.nfa_max_diff_ms = nfa_max_diff_ms
        self.enable_nfa_filter = enable_nfa_filter

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

    def _pipeline_state_dir(self, split):
        d = os.path.join(self.data_dir, "mls", "pipeline_state", f"{self.language}_{split}")
        os.makedirs(d, exist_ok=True)
        return d

    # -- public API --------------------------------------------------------

    def run(self):
        print(f"\n{'=' * 60}")
        print("  MLS + MFA Pipeline (NFA cross-validation filter)")
        print(f"  Language:        {self.language}")
        print(f"  Source:          {self.mls_lang_dir}")
        print(f"  Data dir:        {self.data_dir}")
        print(f"  Splits:          {' '.join(self.splits)}")
        print(f"  Start stage:     {self.start_stage}")
        print(f"  MFA acoustic:    {self.mfa_acoustic}")
        print(f"  MFA dict:        {self.mfa_dict}")
        if self.mfa_g2p:
            print(f"  MFA G2P:         {self.mfa_g2p}")
        print(f"  NFA filter:      {self.enable_nfa_filter}")
        if self.enable_nfa_filter:
            print(f"  NFA model:       {self.nfa_model_name}")
            print(f"  NFA max diff:    {self.nfa_max_diff_ms} ms")
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
            print(f"\n{'_' * 60}\n  Processing {self.language} / {split}\n{'_' * 60}")
            self._process_split(split)

        print(f"\n{'=' * 60}\n  Done (lang: {self.language})\n{'=' * 60}\n")

    # -- Stage 1 -----------------------------------------------------------

    def _stage1_download_mls(self):
        if os.path.isdir(self.mls_lang_dir) and any(Path(self.mls_lang_dir).iterdir()):
            logger.info("[Stage 1] MLS data exists. Skipping.")
            return
        sfx = f"mls_{self.language}_opus.tar.gz" if self.opus else f"mls_{self.language}.tar.gz"
        url = f"{MLS_BASE_URL}/{sfx}"
        arc = os.path.join(self.mls_src_dir, sfx)
        os.makedirs(self.mls_src_dir, exist_ok=True)
        if not os.path.isfile(arc):
            logger.info("[Stage 1] Downloading MLS %s ...", self.language)
            self._download_with_progress(url, arc)
        logger.info("[Stage 1] Extracting ...")
        with tarfile.open(arc, "r:gz") as tar:
            tar.extractall(path=self.mls_src_dir)
        logger.info("[Stage 1] Done.")

    @staticmethod
    def _download_with_progress(url, dest):
        pf = dest + ".partial"
        try:
            with urllib.request.urlopen(urllib.request.Request(url)) as resp, open(pf, "wb") as f:
                total = int(resp.headers.get("Content-Length", 0))
                dl = 0
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    dl += len(chunk)
                    if total:
                        logger.info("  %.1f / %.1f MB  (%.0f%%)", dl / 1e6, total / 1e6, dl * 100 / total)
            os.rename(pf, dest)
        except Exception:
            if os.path.isfile(pf):
                os.remove(pf)
            raise

    def _validate(self):
        if not os.path.isdir(self.mls_lang_dir):
            logger.error("MLS source not found: %s", self.mls_lang_dir)
            sys.exit(1)

    # -- Stage 2 -----------------------------------------------------------

    def _stage2_prepare_lhotse_manifests(self):
        need = any(
            not os.path.isfile(os.path.join(self.mls_manifests_dir, f"{self.manifest_prefix}_{t}_{s}.jsonl.gz"))
            for s in self.splits for t in ("recordings", "supervisions")
        )
        if not need:
            logger.info("Lhotse manifests exist. Skipping stage 2.")
            return
        logger.info("[Stage 2] Preparing Lhotse manifests (lang=%s) ...", self.language)
        import shutil
        from lhotse import CutSet
        from lhotse.recipes.mls import prepare_mls

        out_dir = Path(self.mls_manifests_dir)
        existing = {p.name for p in out_dir.glob("*.jsonl.gz")}
        backup = {}
        for name in existing:
            if not name.startswith(self.manifest_prefix):
                bak = out_dir / (name + ".bak")
                shutil.copy2(str(out_dir / name), str(bak))
                backup[name] = bak

        manifests = prepare_mls(corpus_dir=Path(self.mls_src_dir), output_dir=out_dir,
                                opus=self.opus, num_jobs=self.num_jobs)

        for name, bak in backup.items():
            shutil.move(str(bak), str(out_dir / name))
            logger.info("  Restored %s", name)

        if self.language not in manifests:
            logger.error("Language '%s' not in manifests.", self.language); sys.exit(1)
        for split, data in manifests[self.language].items():
            cp = out_dir / f"{self.manifest_prefix}_cutset_{split}.jsonl.gz"
            if cp.exists():
                continue
            cuts = CutSet.from_manifests(recordings=data["recordings"], supervisions=data["supervisions"])
            cuts.to_file(cp)
            h = sum(c.duration for c in cuts) / 3600
            ns = len({s.speaker for c in cuts for s in c.supervisions})
            logger.info("  [%s/%s] %d cuts, %.1fh, %d spk", self.language, split, len(cuts), h, ns)
        logger.info("[Stage 2] Done.")

    # -- Stage 3 -----------------------------------------------------------

    def _stage3_download_mfa_models(self):
        logger.info("[Stage 3] MFA models ...")
        for mt, mn in [("acoustic", self.mfa_acoustic), ("dictionary", self.mfa_dict)]:
            self._mfa_dl(mt, mn)
        if self.mfa_g2p:
            self._mfa_dl("g2p", self.mfa_g2p)
        if self.mfa_tokenizer:
            self._mfa_dl("tokenizer", self.mfa_tokenizer)
        logger.info("[Stage 3] Done.")

    @staticmethod
    def _mfa_dl(mt, mn):
        r = subprocess.run(["mfa", "model", "download", mt, mn], capture_output=True, text=True)
        if r.returncode != 0:
            logger.warning("  MFA %s '%s' download issue: %s", mt, mn, r.stderr[-200:])
        else:
            logger.info("  MFA %s '%s' OK", mt, mn)

    # -- Per-split processing ----------------------------------------------

    def _process_split(self, split):
        rec = os.path.join(self.mls_manifests_dir, f"{self.manifest_prefix}_recordings_{split}.jsonl.gz")
        cuts_file = os.path.join(self.mls_manifests_dir, f"{self.manifest_prefix}_cuts_{split}.jsonl.gz")
        sd = self._pipeline_state_dir(split)
        s5 = os.path.join(sd, "stage5_tasks.jsonl")
        s6 = os.path.join(sd, "stage6_tasks.jsonl")
        s7 = os.path.join(sd, "stage7_tasks.jsonl")
        s9 = os.path.join(sd, "stage9_tasks.jsonl")
        resampled = os.path.join(self.data_dir, "mls", "resampled", f"{self.language}_{split}")
        mfa_tg = os.path.join(self.data_dir, "mls", "mfa_textgrids", f"{self.language}_{split}")
        nfa_out = os.path.join(self.data_dir, "mls", "nfa_output", f"{self.language}_{split}")
        aligned = os.path.join(self.mls_manifests_dir,
                               f"{self.manifest_prefix}_{self.target_sample_rate}_aligned_cuts_{split}.jsonl.gz")
        if not os.path.isfile(rec):
            logger.warning("%s not found. Skipping '%s'.", rec, split); return

        if self.start_stage <= 4:
            self._stage4_create_cutset(split)
        if self.start_stage <= 5:
            self._stage5_resample(cuts_file, resampled, s5, split)
        if self.start_stage <= 6:
            self._stage6_normalize(s5, s6, split)
        if self.start_stage <= 7:
            self._stage7_mfa(s6, mfa_tg, s7, split)
        if self.enable_nfa_filter and self.start_stage <= 8:
            self._stage8_nfa(s7, nfa_out, split)
        if self.enable_nfa_filter and self.start_stage <= 9:
            self._stage9_compare_filter(s7, mfa_tg, nfa_out, s9, split)
        if self.start_stage <= 10:
            self._stage10_merge(cuts_file, mfa_tg, aligned, split)
        if self.start_stage <= 11:
            self._stage11_rewrite(aligned, resampled, split)

    # -- Stage 4 -----------------------------------------------------------

    def _stage4_create_cutset(self, split):
        rf = os.path.join(self.mls_manifests_dir, f"{self.manifest_prefix}_recordings_{split}.jsonl.gz")
        sf = os.path.join(self.mls_manifests_dir, f"{self.manifest_prefix}_supervisions_{split}.jsonl.gz")
        cf = os.path.join(self.mls_manifests_dir, f"{self.manifest_prefix}_cuts_{split}.jsonl.gz")
        if os.path.isfile(cf):
            logger.info("  [Stage 4] Cutset exists. Skipping."); return
        logger.info("  [Stage 4] Creating cutset for %s ...", split)
        from lhotse import CutSet, fix_manifests, load_manifest
        cuts = CutSet.from_manifests(*fix_manifests(load_manifest(rf), load_manifest(sf)))
        cuts.to_file(cf)
        logger.info("  [Stage 4] Wrote %s", cf)

    # -- Stage 5 -----------------------------------------------------------

    def _stage5_resample(self, cuts_file, resampled_dir, ckpt, split):
        if os.path.isfile(ckpt):
            logger.info("  [Stage 5] Checkpoint exists. Skipping."); return
        logger.info("  [Stage 5] Resampling for %s ...", split)
        from lhotse import CutSet
        from tqdm.auto import tqdm
        from nemo_curator.tasks import AudioTask

        cuts = CutSet.from_file(cuts_file)
        rp = Path(resampled_dir); rp.mkdir(parents=True, exist_ok=True)
        tasks, exported, skipped = [], 0, 0
        for cut in tqdm(cuts, desc=f"Resample {split}"):
            text = " ".join(s.text for s in cut.supervisions if s.text).strip()
            if not text:
                skipped += 1; continue
            spks = {s.speaker for s in cut.supervisions if s.speaker}
            spk = sorted(spks)[0] if spks else "unknown"
            sd = rp / spk; sd.mkdir(parents=True, exist_ok=True)
            src = cut.recording.sources[0].source
            already_ok = (cut.recording.sampling_rate == self.target_sample_rate
                          and os.path.isfile(src))
            if already_ok:
                audio_path = src
            else:
                wav = sd / f"{cut.id}.wav"
                audio_path = str(wav)
                if not wav.exists():
                    r = subprocess.run(["ffmpeg", "-y", "-i", src, "-ar", str(self.target_sample_rate),
                                        "-ac", "1", "-acodec", "pcm_s16le", "-af",
                                        "aresample=dither_method=none", str(wav)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
                    if r.returncode != 0:
                        skipped += 1; continue
            tasks.append(AudioTask(task_id=cut.id, dataset_name=f"mls_{self.language}",
                                   data={"audio_filepath": audio_path, "text": text,
                                         "speaker": spk, "duration": float(cut.duration)}))
            exported += 1
        _write_audio_tasks_jsonl(tasks, ckpt)
        logger.info("  [Stage 5] %d exported, %d skipped", exported, skipped)

    # -- Stage 6 -----------------------------------------------------------

    def _stage6_normalize(self, s5, s6, split):
        if os.path.isfile(s6):
            logger.info("  [Stage 6] Checkpoint exists. Skipping."); return
        logger.info("  [Stage 6] Normalizing transcripts for %s ...", split)
        from nemo_curator.stages.audio.preprocessing.transcript_normalization import (
            AudioTranscriptNormalizeStage, resolve_alphabet)
        tasks = _read_audio_tasks_jsonl(s5)
        n2w = _MLS_LANGUAGE_TO_NUM2WORDS.get(self.language, "")
        alph = resolve_alphabet(self.language, None)
        ns = AudioTranscriptNormalizeStage(language=self.language, alphabet=alph, num2words_lang=n2w, text_key="text")
        out = [o for t in tasks if (o := ns.process(t)) is not None]
        logger.info("  [Stage 6] %d -> %d", len(tasks), len(out))
        _write_audio_tasks_jsonl(out, s6)

    # -- Stage 7: MFA alignment --------------------------------------------

    def _stage7_mfa(self, s6, mfa_tg_dir, s7, split):
        if os.path.isfile(s7):
            logger.info("  [Stage 7] Checkpoint exists. Skipping."); return
        logger.info("  [Stage 7] MFA alignment for %s ...", split)
        tasks = _read_audio_tasks_jsonl(s6)
        from nemo_curator.stages.audio.alignment import MFAAlignmentStage
        mfa_root = os.environ.get("MFA_ROOT_DIR", "")
        stage = MFAAlignmentStage(
            acoustic_model=self.mfa_acoustic, dictionary=self.mfa_dict, g2p_model=self.mfa_g2p,
            output_dir=mfa_tg_dir, audio_filepath_key="audio_filepath", text_key="text",
            speaker_key="speaker", num_jobs=self.num_jobs, beam=self.beam,
            retry_beam=self.retry_beam, output_format=self.output_format,
            mfa_root_dir=mfa_root, copy_models_to_local=False,
            single_speaker=False,
            clean=True, use_mp=True, create_rttm=False, create_ctm=False)
        stage.setup(None)
        results = []
        bs = 5000
        for i in range(0, len(tasks), bs):
            results.extend(stage.process_batch(tasks[i:i + bs]))
            logger.info("  [Stage 7] MFA %d / %d", len(results), len(tasks))
        stage.teardown()
        _write_audio_tasks_jsonl(results, s7)
        logger.info("  [Stage 7] Done. %d aligned.", len(results))

    # -- Stage 8: NFA alignment (via NeMo Python API) ------------------------

    def _stage8_nfa(self, s7, nfa_out_dir, split):
        nfa_tg = os.path.join(nfa_out_dir, "textgrids")
        if os.path.isdir(nfa_tg) and any(Path(nfa_tg).rglob("*.TextGrid")):
            logger.info("  [Stage 8] NFA TextGrids exist. Skipping."); return
        logger.info("  [Stage 8] NFA alignment for %s (model=%s) ...", split, self.nfa_model_name)

        import torch
        from nemo.collections.asr.models.ctc_models import EncDecCTCModel
        from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
        from nemo.collections.asr.parts.utils.aligner_utils import (
            Segment, Word, add_t_start_end_to_utt_obj, get_batch_variables, viterbi_decoding,
        )

        tasks = _read_audio_tasks_jsonl(s7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("  [Stage 8] Loading CTC model: %s", self.nfa_model_name)
        model = EncDecCTCModel.from_pretrained(model_name=self.nfa_model_name)
        if isinstance(model, EncDecHybridRNNTCTCModel):
            model.change_decoding_strategy(decoder_type="ctc")
        model = model.to(device)
        model.eval()
        try:
            model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[64, 64])
        except Exception:
            logger.warning("  [Stage 8] Could not switch to local attention; using default.")

        os.makedirs(nfa_tg, exist_ok=True)
        output_timestep_duration = None
        n_written = 0

        for i in range(0, len(tasks), self.nfa_batch_size):
            batch = tasks[i:i + self.nfa_batch_size]
            audio_paths = [t.data["audio_filepath"] for t in batch]
            gt_texts = [t.data.get("text", "") for t in batch]

            (
                log_probs_batch, y_batch, T_batch, U_batch,
                utt_obj_batch, output_timestep_duration,
            ) = get_batch_variables(
                audio=audio_paths,
                model=model,
                gt_text_batch=gt_texts,
                output_timestep_duration=output_timestep_duration,
            )

            alignments_batch = viterbi_decoding(
                log_probs_batch, y_batch, T_batch, U_batch, device,
            )

            for task, utt_obj, alignment in zip(batch, utt_obj_batch, alignments_batch):
                utt_obj = add_t_start_end_to_utt_obj(utt_obj, alignment, output_timestep_duration)

                nfa_pause = {"<b>", "<blank>", ""}
                words = []
                for seg_or_tok in utt_obj.segments_and_tokens:
                    if isinstance(seg_or_tok, Segment):
                        for wt in seg_or_tok.words_and_tokens:
                            if isinstance(wt, Word):
                                w = wt.text.strip().lower()
                                if w not in nfa_pause:
                                    words.append((wt.t_start, wt.t_end, w))
                    elif hasattr(seg_or_tok, "t_start"):
                        w = seg_or_tok.text.strip().lower()
                        if w not in nfa_pause:
                            words.append((seg_or_tok.t_start, seg_or_tok.t_end, w))

                if words:
                    audio_stem = Path(task.data["audio_filepath"]).stem
                    tg_path = Path(nfa_tg) / f"{audio_stem}.TextGrid"
                    _write_textgrid(words, tg_path, audio_duration=task.data.get("duration"))
                    n_written += 1

            done = min(i + self.nfa_batch_size, len(tasks))
            if done % (self.nfa_batch_size * 10) < self.nfa_batch_size or done >= len(tasks):
                logger.info("  [Stage 8] NFA progress: %d / %d", done, len(tasks))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("  [Stage 8] NFA done: %d TextGrids written", n_written)

    # -- Stage 9: Compare NFA vs MFA + filter ------------------------------

    def _stage9_compare_filter(self, s7, mfa_tg_dir, nfa_out_dir, s9, split):
        if os.path.isfile(s9):
            logger.info("  [Stage 9] Checkpoint exists. Skipping."); return
        logger.info("  [Stage 9] NFA vs MFA comparison (threshold=%d ms) ...", int(self.nfa_max_diff_ms))

        tasks = _read_audio_tasks_jsonl(s7)
        mfa_dir = os.path.join(mfa_tg_dir, "textgrids") if os.path.isdir(
            os.path.join(mfa_tg_dir, "textgrids")) else mfa_tg_dir
        nfa_dir = os.path.join(nfa_out_dir, "textgrids")

        mfa_idx = _build_textgrid_index(mfa_dir)
        nfa_idx = _build_textgrid_index(nfa_dir)
        logger.info("  MFA TGs: %d, NFA TGs: %d", len(mfa_idx), len(nfa_idx))

        thresh = self.nfa_max_diff_ms / 1000.0
        kept, skip_tg, skip_match, skip_diff = [], 0, 0, 0
        all_diffs = []

        tsv = os.path.join(self._pipeline_state_dir(split), f"stage9_nfa_comparison_{split}.tsv")
        with open(tsv, "w", encoding="utf-8") as f:
            f.write("task_id\tmax_diff_ms\tn_mfa\tn_nfa\tn_matched\tresult\n")
            for task in tasks:
                tid = task.task_id
                audio_stem = Path(task.data["audio_filepath"]).stem
                mt = mfa_idx.get(tid) or mfa_idx.get(audio_stem)
                nt = nfa_idx.get(tid) or nfa_idx.get(audio_stem)
                if not mt or not nt:
                    skip_tg += 1; f.write(f"{tid}\t-1\t0\t0\t0\tno_tg\n"); continue
                mw = _parse_textgrid_words(mt)
                nw = _parse_textgrid_words(nt)
                if not mw or not nw:
                    skip_tg += 1; f.write(f"{tid}\t-1\t{len(mw)}\t{len(nw)}\t0\tempty\n"); continue
                pairs = _align_word_sequences(mw, nw)
                if not pairs:
                    skip_match += 1; f.write(f"{tid}\t-1\t{len(mw)}\t{len(nw)}\t0\tno_match\n"); continue

                mx = max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in pairs)
                mx_ms = round(mx * 1000, 1)
                all_diffs.append(mx_ms)
                task.data["nfa_max_diff_ms"] = mx_ms
                ok = mx <= thresh
                f.write(f"{tid}\t{mx_ms}\t{len(mw)}\t{len(nw)}\t{len(pairs)}\t{'pass' if ok else 'fail'}\n")
                if ok:
                    kept.append(task)
                else:
                    skip_diff += 1

        if all_diffs:
            import numpy as np
            a = np.array(all_diffs)
            logger.info("  [Stage 9] Diff stats (ms): mean=%.1f median=%.1f p95=%.1f p99=%.1f",
                        np.mean(a), np.median(a), np.percentile(a, 95), np.percentile(a, 99))
            logger.info("  [Stage 9] %.1f%% within %d ms", 100.0 * np.mean(a <= self.nfa_max_diff_ms),
                        int(self.nfa_max_diff_ms))

        logger.info("  [Stage 9] %d -> %d kept, %d diff>%dms, %d no_tg, %d no_match",
                     len(tasks), len(kept), skip_diff, int(self.nfa_max_diff_ms), skip_tg, skip_match)
        logger.info("  [Stage 9] TSV: %s", tsv)
        _write_audio_tasks_jsonl(kept, s9)

    # -- Stage 10: Merge TextGrids -----------------------------------------

    def _stage10_merge(self, cuts_file, mfa_tg_dir, aligned, split):
        if os.path.isfile(aligned):
            logger.info("  [Stage 10] Aligned cutset exists. Skipping."); return
        logger.info("  [Stage 10] Merging TextGrids for %s ...", split)
        from lhotse import CutSet
        if not os.path.isfile(cuts_file):
            logger.warning("  Cutset not found: %s", cuts_file); return
        cuts = CutSet.from_file(cuts_file)
        tc = len(cuts)
        tg_dir = os.path.join(mfa_tg_dir, "textgrids") if os.path.isdir(
            os.path.join(mfa_tg_dir, "textgrids")) else mfa_tg_dir
        idx = _build_textgrid_index(tg_dir)
        logger.info("  Found %d TextGrids in %s", len(idx), tg_dir)
        chunk_size = 5000
        shard_dir = aligned + ".shards"
        if os.path.isdir(shard_dir):
            import shutil
            shutil.rmtree(shard_dir)
        os.makedirs(shard_dir, exist_ok=True)
        shard_paths: list[str] = []
        matched_total = 0
        fn = partial(
            _align_single_cut,
            textgrid_dir=tg_dir,
            tg_index=idx,
            silence_markers=self.silence_markers,
        )
        worker_fn = _align_single_cut_worker
        if self.num_jobs > 1:
            pool_ctx = ProcessPoolExecutor(
                max_workers=self.num_jobs,
                initializer=_init_stage10_worker,
                initargs=(tg_dir, self.silence_markers),
            )
        else:
            from contextlib import nullcontext
            pool_ctx = nullcontext(None)
        with pool_ctx as ex:
            for chunk_idx, start in enumerate(range(0, tc, chunk_size)):
                batch = list(cuts[start:start + chunk_size])
                if ex is not None:
                    mapped = list(ex.map(worker_fn, batch, chunksize=64))
                else:
                    mapped = [fn(c) for c in batch]
                chunk_matched = [c for c in mapped if c is not None]
                matched_total += len(chunk_matched)
                done = min(start + chunk_size, tc)
                logger.info(
                    "  [Stage 10] progress: %d/%d processed, %d matched so far",
                    done, tc, matched_total,
                )
                if chunk_matched:
                    shard_path = os.path.join(shard_dir, f"part_{chunk_idx:05d}.jsonl.gz")
                    CutSet.from_items(chunk_matched).to_file(shard_path)
                    shard_paths.append(shard_path)
        if matched_total == 0:
            import shutil
            shutil.rmtree(shard_dir, ignore_errors=True)
            logger.error(
                "  [Stage 10] 0/%d matched — TextGrid lookup keys likely wrong", tc,
            )
            return
        if len(shard_paths) == 1:
            os.replace(shard_paths[0], aligned)
        else:
            CutSet.from_files(shard_paths).to_file(aligned)
        import shutil
        shutil.rmtree(shard_dir, ignore_errors=True)
        logger.info("  [Stage 10] %s (%d/%d matched)", aligned, matched_total, tc)

    # -- Stage 11: Rewrite paths -------------------------------------------

    @staticmethod
    def _stage11_rewrite(aligned, resampled_dir, split):
        if not os.path.isfile(aligned):
            logger.warning("  [Stage 11] Not found. Skipping."); return
        logger.info("  [Stage 11] Rewriting paths ...")
        import soundfile as sf
        from lhotse import CutSet
        wi = {p.stem: str(p) for p in Path(resampled_dir).rglob("*.wav")}
        cuts = CutSet.from_file(aligned)
        if len(cuts) == 0:
            logger.warning("  [Stage 11] Empty aligned cutset. Skipping."); return
        cuts = cuts.to_eager()
        u = 0
        for c in cuts:
            wp = wi.get(c.id)
            if wp:
                i = sf.info(wp)
                c.recording.sources[0].source = wp
                c.recording.sampling_rate = i.samplerate
                c.recording.num_samples = i.frames
                c.recording.duration = i.frames / i.samplerate
                c.recording.transforms = None
                u += 1
        cuts.to_file(aligned)
        logger.info("  [Stage 11] Updated %d/%d paths", u, len(cuts))


# ===========================================================================
# Hydra entry point
# ===========================================================================

def _pipeline_from_cfg(cfg):
    return MLSMFAPipeline(
        language=cfg.language, mfa_acoustic=cfg.mfa_acoustic, mfa_dict=cfg.mfa_dict,
        mls_src_dir=cfg.mls_src_dir, data_dir=cfg.data_dir, manifests_dir=cfg.manifests_dir,
        mfa_g2p=cfg.get("mfa_g2p", ""), mfa_tokenizer=cfg.get("mfa_tokenizer", ""),
        mfa_root_dir=cfg.get("mfa_root_dir", ""),
        num_jobs=int(cfg.get("num_jobs", 16)),
        splits=list(cfg.get("splits", ["train", "dev", "test"])),
        start_stage=int(cfg.get("stage", 1)),
        target_sample_rate=int(cfg.get("target_sample_rate", DEFAULT_TARGET_SR)),
        beam=int(cfg.get("beam", 100)), retry_beam=int(cfg.get("retry_beam", 400)),
        output_format=cfg.get("output_format", "long_textgrid"),
        opus=bool(cfg.get("opus", False)),
        silence_markers=list(cfg.get("silence_markers", DEFAULT_SILENCE_MARKERS)),
        nfa_model_name=cfg.get("nfa_model_name", "nvidia/stt_fr_conformer_ctc_large"),
        nfa_batch_size=int(cfg.get("nfa_batch_size", 32)),
        nfa_max_diff_ms=float(cfg.get("nfa_max_diff_ms", 100.0)),
        enable_nfa_filter=bool(cfg.get("enable_nfa_filter", True)),
    )


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    logger.info("MLS + MFA Pipeline (NFA filter)")
    logger.info("=" * 60)
    logger.info("Effective config:\n%s", OmegaConf.to_yaml(cfg))
    _pipeline_from_cfg(cfg).run()


if __name__ == "__main__":
    main()
