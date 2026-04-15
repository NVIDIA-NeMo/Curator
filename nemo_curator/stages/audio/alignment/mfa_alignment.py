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

"""
MFA Batch Alignment Stage for NeMo Curator.

A ``ProcessingStage`` that runs `Montreal Forced Aligner (MFA)
<https://montreal-forced-aligner.readthedocs.io>`_ in batch mode on a set
of ``AudioTask`` entries, producing TextGrid, RTTM, and/or CTM output files.

The stage operates via ``process_batch``: it collects all tasks in a batch,
prepares a temporary MFA corpus (symlinked WAVs + ``.txt`` transcript files),
runs a single ``mfa align`` subprocess, and converts the resulting TextGrid
files to RTTM and/or CTM format depending on configuration.

Node-level isolation
    ``setup_on_node()`` copies MFA models from shared storage to a node-local
    directory.  This avoids NFS/Lustre race conditions and Kaldi errors when
    multiple distributed nodes share the same model directory.

Worker scheduling
    ``xenna_stage_spec()`` returns ``{"num_workers_per_node": 1}`` to
    guarantee exactly one MFA worker per node.
"""

from __future__ import annotations

import os
import shlex
import shutil
import socket
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import soundfile as sf
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

_DEFAULT_SILENCE_MARKERS = ("", "sp", "sil", "spn", "<eps>")


@dataclass
class MFAAlignmentStage(ProcessingStage[AudioTask, AudioTask]):
    """Batch forced alignment using the Montreal Forced Aligner (MFA).

    This stage only supports :meth:`process_batch`; calling :meth:`process`
    raises ``NotImplementedError``.  Use ``.with_(batch_size=N)`` to control
    how many tasks are aligned per ``mfa align`` invocation.

    Args:
        mfa_command: Shell command (or absolute path) to the ``mfa`` binary.
        acoustic_model: MFA acoustic model name or path.
        dictionary: MFA dictionary name or path.
        g2p_model: MFA G2P model for out-of-vocabulary words.  Set to ``""``
            to disable G2P.
        output_dir: Root directory for all output files.  Sub-directories
            ``textgrids/``, ``rttms/``, and ``ctms/`` are created beneath it.
        audio_filepath_key: Key in ``task.data`` pointing to the WAV file.
        text_key: Key in ``task.data`` containing the transcript text.
        speaker_key: Key in ``task.data`` for the speaker label (used in
            RTTM output).
        duration_key: Key in ``task.data`` for audio duration.  Computed
            automatically if missing.
        max_gap_for_merge: Maximum gap (seconds) between speech intervals
            before they are merged in the RTTM output.
        num_jobs: Number of parallel MFA jobs.  ``0`` means ``os.cpu_count()``.
        beam: MFA beam size for alignment search.
        retry_beam: MFA retry beam when initial alignment fails.
        single_speaker: Pass ``--single_speaker`` to MFA.
        clean: Pass ``--clean`` to MFA (remove temp files after alignment).
        use_mp: Pass ``--use_mp`` to MFA (use multiprocessing).
        output_format: MFA output format (``long_textgrid`` or
            ``short_textgrid``).
        mfa_root_dir: MFA root directory containing pretrained models.
            Defaults to ``MFA_ROOT_DIR`` env var or ``~/.mfa``.
        local_mfa_base_dir: Base directory for node-local model copies.
            Defaults to ``tempfile.gettempdir()`` (typically ``/tmp``).
        copy_models_to_local: Whether ``setup_on_node`` should copy models
            to node-local storage.
        silence_markers: Labels to treat as silence when converting TextGrids.
        create_rttm: Whether to convert TextGrids to RTTM files.
        create_ctm: Whether to convert TextGrids to CTM files.
    """

    name: str = "MFAAlignmentStage"
    mfa_command: str = "mfa"
    acoustic_model: str = "english_us_arpa"
    dictionary: str = "english_us_arpa"
    g2p_model: str = "english_us_arpa"
    output_dir: str = ""
    audio_filepath_key: str = "audio_filepath"
    text_key: str = "text"
    speaker_key: str = "speaker"
    duration_key: str = "duration"
    max_gap_for_merge: float = 0.3
    num_jobs: int = 0
    beam: int = 100
    retry_beam: int = 400
    single_speaker: bool = True
    clean: bool = True
    use_mp: bool = True
    output_format: str = "long_textgrid"
    mfa_root_dir: str = ""
    local_mfa_base_dir: str = ""
    copy_models_to_local: bool = True
    silence_markers: tuple[str, ...] = _DEFAULT_SILENCE_MARKERS
    create_rttm: bool = True
    create_ctm: bool = True

    # Set during lifecycle hooks -- not user-configurable
    _mfa_root: str = field(default="", init=False, repr=False)
    _textgrid_mod: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.output_dir:
            msg = "output_dir is required for MFAAlignmentStage"
            raise ValueError(msg)
        self._effective_num_jobs = self.num_jobs or os.cpu_count()
        self._effective_mfa_root = self.mfa_root_dir or os.environ.get(
            "MFA_ROOT_DIR", os.path.expanduser("~/.mfa")
        )
        self._effective_local_base = (
            self.local_mfa_base_dir or tempfile.gettempdir()
        )
        self._textgrid_dir = Path(self.output_dir) / "textgrids"
        self._rttm_dir = Path(self.output_dir) / "rttms"
        self._ctm_dir = Path(self.output_dir) / "ctms"

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, self.text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        data_keys = ["textgrid_filepath"]
        if self.create_rttm:
            data_keys.append("rttm_filepath")
        if self.create_ctm:
            data_keys.append("ctm_filepath")
        return [], data_keys

    # ------------------------------------------------------------------
    # Xenna scheduling
    # ------------------------------------------------------------------

    def xenna_stage_spec(self) -> dict[str, Any]:
        # Current implementation is meant to run with one worker per node. because the MFA library has issues when running in parallel.
        # We are copying the MFA models to node-local storage to avoid race conditions and Kaldi errors when multiple distributed nodes share the same model directory.
        return {"num_workers_per_node": 1}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup_on_node(
        self,
        node_info: Any = None,  # noqa: ARG002, ANN401
        worker_metadata: Any = None,  # noqa: ARG002, ANN401
    ) -> None:
        """Copy MFA models from shared storage to node-local directory."""
        if not self.copy_models_to_local:
            self._mfa_root = self._effective_mfa_root
            return
        hostname = socket.gethostname()
        self._mfa_root = self._setup_local_mfa(
            self._effective_mfa_root, hostname
        )
        logger.info(
            f"[setup_on_node] MFA root set to {self._mfa_root} on {hostname}"
        )

    def setup(
        self,
        worker_metadata: Any = None,  # noqa: ARG002, ANN401
    ) -> None:
        """Import praatio and verify MFA root is available."""
        try:
            from praatio import textgrid as tg_mod

            self._textgrid_mod = tg_mod
        except ImportError:
            raise ImportError(
                "praatio is required for MFA alignment. "
                "Install with: pip install 'praatio>=6.0'"
            ) from None

        if not self._mfa_root:
            if self.copy_models_to_local:
                hostname = socket.gethostname()
                local_candidate = (
                    Path(self._effective_local_base) / f"mfa_models_{hostname}"
                )
                if local_candidate.exists():
                    self._mfa_root = str(local_candidate)
                    logger.info(
                        f"[setup] Re-using local MFA root: {self._mfa_root}"
                    )
                else:
                    self._mfa_root = self._effective_mfa_root
                    logger.info(
                        f"[setup] Local copy not found; using shared MFA root: "
                        f"{self._mfa_root}"
                    )
            else:
                self._mfa_root = self._effective_mfa_root

        self._textgrid_dir.mkdir(parents=True, exist_ok=True)
        if self.create_rttm:
            self._rttm_dir.mkdir(parents=True, exist_ok=True)
        if self.create_ctm:
            self._ctm_dir.mkdir(parents=True, exist_ok=True)

    def teardown(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, task: AudioTask) -> AudioTask:
        msg = "MFAAlignmentStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        """Align all tasks in a single ``mfa align`` invocation."""
        if not tasks:
            return []

        stem_to_task: dict[str, AudioTask] = {}
        for task in tasks:
            if not self.validate_input(task):
                msg = f"Task {task!s} failed validation for stage {self}"
                raise ValueError(msg)
            audio_filepath = task.data[self.audio_filepath_key]
            text = task.data[self.text_key].strip()
            if not text:
                msg = (
                    f"Empty text for {audio_filepath} "
                    f"(key={self.text_key!r})"
                )
                raise ValueError(msg)
            audio_path = Path(audio_filepath)
            if not audio_path.exists():
                msg = f"Audio file not found: {audio_path}"
                raise FileNotFoundError(msg)

            file_stem = audio_path.stem
            if not task.data.get(self.duration_key):
                task.data[self.duration_key] = self._get_audio_duration(
                    str(audio_path)
                )
            stem_to_task[file_stem] = task

        batch_uuid = uuid.uuid4().hex[:12]
        tg_out_path = self._textgrid_dir / batch_uuid
        tg_out_path.mkdir(parents=True, exist_ok=True)

        results: list[AudioTask] = []

        with tempfile.TemporaryDirectory(prefix="mfa_corpus_") as corpus_dir:
            corpus_path = Path(corpus_dir)
            for file_stem, task in stem_to_task.items():
                audio_path = Path(task.data[self.audio_filepath_key])
                corpus_wav = corpus_path / f"{file_stem}.wav"
                if not corpus_wav.exists() and not corpus_wav.is_symlink():
                    try:
                        corpus_wav.symlink_to(audio_path.resolve())
                    except OSError:
                        shutil.copy2(audio_path, corpus_wav)
                corpus_txt = corpus_path / f"{file_stem}.txt"
                corpus_txt.write_text(
                    task.data[self.text_key].strip(), encoding="utf-8"
                )

            self._run_mfa_align(corpus_path, tg_out_path)

            all_tg = {
                tg.stem: tg for tg in tg_out_path.rglob("*.TextGrid")
            }
            missing = {s for s in stem_to_task if s not in all_tg}

            if missing:
                logger.warning(
                    f"MFA silently dropped {len(missing)}/{len(stem_to_task)} "
                    f"files (exit code was 0). Creating fallback outputs."
                )

            for file_stem, task in stem_to_task.items():
                if file_stem in missing:
                    self._handle_missing_textgrid(file_stem, task)
                else:
                    self._handle_successful_textgrid(
                        file_stem, task, all_tg[file_stem]
                    )
                results.append(task)

        return results

    # ------------------------------------------------------------------
    # Result handlers
    # ------------------------------------------------------------------

    def _handle_successful_textgrid(
        self, file_stem: str, task: AudioTask, tg_path: Path
    ) -> None:
        task.data["textgrid_filepath"] = str(tg_path)
        speaker = task.data.get(self.speaker_key, "unknown")

        if self.create_rttm:
            rttm_path = self._rttm_dir / f"{file_stem}.rttm"
            self._textgrid_to_rttm(tg_path, file_stem, speaker, rttm_path)
            task.data["rttm_filepath"] = str(rttm_path)

        if self.create_ctm:
            ctm_path = self._ctm_dir / f"{file_stem}.ctm"
            self._textgrid_to_ctm(tg_path, file_stem, ctm_path)
            task.data["ctm_filepath"] = str(ctm_path)

    def _handle_missing_textgrid(
        self, file_stem: str, task: AudioTask
    ) -> None:
        duration = task.data.get(self.duration_key, 0.0)
        text = task.data.get(self.text_key, "").strip()
        speaker = task.data.get(self.speaker_key, "unknown")

        logger.warning(
            f"  MFA dropped '{file_stem}': duration={duration:.2f}s, "
            f"text='{text[:120]}'"
        )

        task.data["textgrid_filepath"] = ""
        task.data["mfa_skipped"] = True

        if self.create_rttm:
            rttm_path = self._rttm_dir / f"{file_stem}.rttm"
            self._create_duration_fallback_rttm(
                file_stem, speaker, duration, rttm_path
            )
            task.data["rttm_filepath"] = str(rttm_path)

        if self.create_ctm:
            ctm_path = self._ctm_dir / f"{file_stem}.ctm"
            self._create_duration_fallback_ctm(
                file_stem, text, duration, ctm_path
            )
            task.data["ctm_filepath"] = str(ctm_path)

    # ------------------------------------------------------------------
    # MFA subprocess
    # ------------------------------------------------------------------

    def _run_mfa_align(
        self, corpus_dir: Path, textgrid_output_dir: Path
    ) -> None:
        env = os.environ.copy()
        env["MFA_ROOT_DIR"] = self._mfa_root

        mfa_cmd_parts = shlex.split(self.mfa_command)
        mfa_bin_dir = (
            os.path.dirname(mfa_cmd_parts[0])
            if os.path.isabs(mfa_cmd_parts[0])
            else None
        )
        if mfa_bin_dir:
            env["PATH"] = f"{mfa_bin_dir}:{env.get('PATH', '')}"

        history_file = Path(self._mfa_root) / "command_history.yaml"
        if history_file.exists():
            try:
                history_file.unlink()
            except Exception:  # noqa: BLE001
                pass

        cmd = mfa_cmd_parts + [
            "align",
            str(corpus_dir),
            self.dictionary,
            self.acoustic_model,
            str(textgrid_output_dir),
            "--output_format",
            self.output_format,
            "-j",
            str(self._effective_num_jobs),
            "--beam",
            str(self.beam),
            "--retry_beam",
            str(self.retry_beam),
        ]
        if self.single_speaker:
            cmd.append("--single_speaker")
        if self.use_mp:
            cmd.append("--use_mp")
        if self.clean:
            cmd.append("--clean")

        if self.g2p_model:
            g2p_path = (
                Path(self._mfa_root)
                / "pretrained_models"
                / "g2p"
                / f"{self.g2p_model}.zip"
            )
            if g2p_path.exists():
                cmd.extend(["--g2p_model_path", str(g2p_path)])
            else:
                g2p_alt = (
                    Path(self._mfa_root)
                    / "pretrained_models"
                    / "g2p"
                    / self.g2p_model
                )
                if g2p_alt.exists():
                    cmd.extend(["--g2p_model_path", str(g2p_alt)])

        logger.info(f"Running MFA align: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.stdout and result.stdout.strip():
            logger.info(
                f"MFA stdout (last 5000 chars):\n{result.stdout[-5000:]}"
            )
        if result.stderr and result.stderr.strip():
            logger.warning(
                f"MFA stderr (last 5000 chars):\n{result.stderr[-5000:]}"
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"mfa align failed (exit code {result.returncode}).\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

    # ------------------------------------------------------------------
    # TextGrid -> RTTM
    # ------------------------------------------------------------------

    def _parse_textgrid_words(self, textgrid_path: Path) -> list[tuple]:
        """Return ``[(start, end, label), ...]`` from the words tier."""
        tg = self._textgrid_mod.openTextgrid(
            str(textgrid_path), includeEmptyIntervals=False
        )
        tier_name = "words"
        tier = (
            tg.getTier(tier_name)
            if tier_name in tg.tierNames
            else tg.getTier(tg.tierNames[0])
        )
        return [(e.start, e.end, e.label) for e in tier.entries]

    def _textgrid_to_rttm(
        self,
        textgrid_path: Path,
        file_stem: str,
        speaker: str,
        rttm_path: Path,
    ) -> None:
        intervals = self._parse_textgrid_words(textgrid_path)
        silence = set(self.silence_markers)
        speech_intervals: list[dict] = []
        for start, end, label in intervals:
            if label.strip() and label.strip() not in silence:
                speech_intervals.append(
                    {"start": start, "duration": end - start}
                )

        merged = self._merge_intervals(speech_intervals)

        with open(rttm_path, "w", encoding="utf-8") as f:
            for iv in merged:
                f.write(
                    f"SPEAKER {file_stem} 1 "
                    f"{iv['start']:.3f} {iv['duration']:.3f} "
                    f"<NA> <NA> {speaker} <NA> <NA>\n"
                )

    # ------------------------------------------------------------------
    # TextGrid -> CTM
    # ------------------------------------------------------------------

    def _textgrid_to_ctm(
        self,
        textgrid_path: Path,
        file_stem: str,
        ctm_path: Path,
    ) -> None:
        intervals = self._parse_textgrid_words(textgrid_path)
        silence = set(self.silence_markers)

        with open(ctm_path, "w", encoding="utf-8") as f:
            for start, end, label in intervals:
                word = label.strip()
                if word and word not in silence:
                    f.write(
                        f"{file_stem} 1 {start:.3f} {end - start:.3f} {word}\n"
                    )

    # ------------------------------------------------------------------
    # Interval merging
    # ------------------------------------------------------------------

    def _merge_intervals(self, intervals: list[dict]) -> list[dict]:
        if not intervals:
            return []
        sorted_ivs = sorted(intervals, key=lambda x: x["start"])
        merged: list[dict] = []
        cur_start = sorted_ivs[0]["start"]
        cur_end = cur_start + sorted_ivs[0]["duration"]

        for iv in sorted_ivs[1:]:
            iv_start = iv["start"]
            iv_end = iv_start + iv["duration"]
            if iv_start - cur_end <= self.max_gap_for_merge:
                cur_end = max(cur_end, iv_end)
            else:
                merged.append(
                    {"start": cur_start, "duration": cur_end - cur_start}
                )
                cur_start = iv_start
                cur_end = iv_end

        merged.append({"start": cur_start, "duration": cur_end - cur_start})
        return merged

    # ------------------------------------------------------------------
    # Fallback helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        with sf.SoundFile(audio_path) as f:
            return len(f) / f.samplerate

    @staticmethod
    def _create_duration_fallback_rttm(
        file_stem: str, speaker: str, duration: float, rttm_path: Path
    ) -> None:
        with open(rttm_path, "w", encoding="utf-8") as f:
            f.write(
                f"SPEAKER {file_stem} 1 0.000 {duration:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )

    @staticmethod
    def _create_duration_fallback_ctm(
        file_stem: str, text: str, duration: float, ctm_path: Path
    ) -> None:
        words = text.strip().split()
        if not words:
            ctm_path.write_text("", encoding="utf-8")
            return
        word_dur = duration / len(words)
        with open(ctm_path, "w", encoding="utf-8") as f:
            for i, word in enumerate(words):
                f.write(
                    f"{file_stem} 1 {i * word_dur:.3f} {word_dur:.3f} {word}\n"
                )

    # ------------------------------------------------------------------
    # Node-local MFA setup
    # ------------------------------------------------------------------

    def _setup_local_mfa(self, shared_mfa_root: str, hostname: str) -> str:
        local_mfa_root = Path(self._effective_local_base) / f"mfa_models_{hostname}"

        if local_mfa_root.exists():
            has_models = (
                (local_mfa_root / "pretrained_models").exists()
                or (local_mfa_root / "extracted_models").exists()
            )
            if has_models:
                logger.info(
                    f"Using existing local MFA root: {local_mfa_root}"
                )
                return str(local_mfa_root)

        logger.info(f"Copying MFA models to local storage: {local_mfa_root}")
        local_mfa_root.mkdir(parents=True, exist_ok=True)

        src = Path(shared_mfa_root)
        for subdir in ("pretrained_models", "extracted_models"):
            src_path = src / subdir
            dst_path = local_mfa_root / subdir
            if src_path.exists() and not dst_path.exists():
                logger.info(f"  Copying {subdir}...")
                shutil.copytree(src_path, dst_path)

        logger.info(f"Local MFA setup complete: {local_mfa_root}")
        return str(local_mfa_root)
