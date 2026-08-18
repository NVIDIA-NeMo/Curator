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
    multiple distributed nodes share the same model directory.  The node-local
    cache directory is namespaced by a digest of the resolved shared source
    root plus the requested acoustic/dictionary/g2p model identity, and is
    only ever populated atomically (staged, then published via ``os.replace``)
    under a per-cache file lock -- so a stale, wrong-source, or interrupted
    copy is never silently reused.

Worker scheduling
    MFA/Kaldi is not safe to run concurrently against a shared model directory,
    so each backend is constrained to avoid overlapping workers:

    * Xenna: ``xenna_stage_spec()`` returns ``{"num_workers_per_node": 1}`` to
      guarantee exactly one MFA worker per node.
    * Ray Data: the backend has no per-node worker cap, so ``num_workers()``
      returns ``1`` and ``ray_stage_spec()`` marks this as an actor stage,
      yielding a single MFA actor cluster-wide (concurrency=1).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import soundfile as sf
from loguru import logger
from praatio import textgrid as praatio_textgrid

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_DEFAULT_SILENCE_MARKERS = ("", "sp", "sil", "spn", "<eps>")
_WORD_TIER_NAMES = ("words", "word")
_PHONE_TIER_NAMES = frozenset({
    "phones",
    "phone",
    "phonemes",
    "phoneme",
    "phons",
})
# Written into a local model cache directory only after it has been fully and
# successfully populated (see ``_setup_local_mfa``). Its presence -- plus a
# matching identity -- is the sole signal that a cache is safe to reuse.
_MFA_CACHE_MARKER_NAME = ".mfa_cache_complete.json"


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
        num_jobs: Number of parallel MFA jobs (``-j`` flag passed to MFA).
            Must be positive. Also determines the stage's default CPU
            reservation (see ``resources``) unless ``resources`` is set
            explicitly.
        beam: MFA beam size for alignment search.
        retry_beam: MFA retry beam when initial alignment fails.
        single_speaker: Pass ``--single_speaker`` to MFA.
        clean: Pass ``--clean`` to MFA (remove temp files after alignment).
        use_mp: Pass ``--use_mp`` to MFA (use multiprocessing).
        output_format: MFA output format (``long_textgrid`` or
            ``short_textgrid``).
        align_timeout_seconds: Hard timeout (seconds) for each ``mfa align``
            subprocess invocation. Must be positive. On expiry, the entire
            MFA process group is killed (not just the immediate process) so
            no orphaned Kaldi/MFA worker processes survive, and a bounded
            ``TimeoutError`` is raised. This stage runs as a single worker
            cluster-wide (see module docstring), so a wedged MFA process
            would otherwise stall the whole pipeline indefinitely.
        mfa_root_dir: MFA root directory containing pretrained models, or
            ``None`` to use ``MFA_ROOT_DIR`` / ``~/.mfa``.
        local_mfa_base_dir: Base directory for node-local model copies, or
            ``None`` to use ``tempfile.gettempdir()`` (typically ``/tmp``).
        copy_models_to_local: Whether ``setup_on_node`` should copy models
            to node-local storage.
        silence_markers: Labels to treat as silence when converting TextGrids.
        create_rttm: Whether to convert TextGrids to RTTM files.
        create_ctm: Whether to convert TextGrids to CTM files.
        resources: Executor resource reservation for this stage. Defaults to
            ``Resources(cpus=num_jobs)`` so the executor reserves CPUs
            proportional to the ``-j`` jobs MFA actually forks; pass an
            explicit value to override.
    """

    output_dir: str
    name: str = "MFAAlignmentStage"
    mfa_command: str = "mfa"
    acoustic_model: str = "english_us_arpa"
    dictionary: str = "english_us_arpa"
    g2p_model: str = "english_us_arpa"
    audio_filepath_key: str = "audio_filepath"
    text_key: str = "text"
    speaker_key: str = "speaker"
    duration_key: str = "duration"
    max_gap_for_merge: float = 0.3
    num_jobs: int = 1
    beam: int = 100
    retry_beam: int = 400
    single_speaker: bool = True
    clean: bool = True
    use_mp: bool = True
    output_format: str = "long_textgrid"
    align_timeout_seconds: float = 3600.0
    mfa_root_dir: str | None = None
    local_mfa_base_dir: str | None = None
    copy_models_to_local: bool = True
    silence_markers: tuple[str, ...] = _DEFAULT_SILENCE_MARKERS
    create_rttm: bool = True
    create_ctm: bool = True
    batch_size: int = 256
    resources: Resources | None = None

    # Set during lifecycle hooks -- not user-configurable
    _mfa_root: str = field(default="", init=False, repr=False)
    _textgrid_mod: Any = field(default=praatio_textgrid, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_jobs <= 0:
            msg = f"num_jobs must be positive, got {self.num_jobs}"
            raise ValueError(msg)
        if self.align_timeout_seconds <= 0:
            msg = f"align_timeout_seconds must be positive, got {self.align_timeout_seconds}"
            raise ValueError(msg)

        # Reserve CPUs proportional to the -j jobs MFA forks, so Ray/Xenna
        # don't schedule this as a one-CPU task while MFA itself uses
        # num_jobs cores. Only applied when the caller hasn't already
        # supplied an explicit override.
        if self.resources is None:
            self.resources = Resources(cpus=float(self.num_jobs))

        self._effective_mfa_root = self.mfa_root_dir or os.environ.get(
            "MFA_ROOT_DIR", os.path.expanduser("~/.mfa")
        )
        self._effective_local_base = (
            self.local_mfa_base_dir or tempfile.gettempdir()
        )
        self._textgrid_dir = Path(self.output_dir) / "textgrids"
        self._rttm_dir = Path(self.output_dir) / "rttms"
        self._ctm_dir = Path(self.output_dir) / "ctms"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.audio_filepath_key, self.text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        data_keys = ["textgrid_filepath", "mfa_skipped", self.duration_key]
        if self.create_rttm:
            data_keys.append("rttm_filepath")
        if self.create_ctm:
            data_keys.append("ctm_filepath")
        return [], data_keys

    def xenna_stage_spec(self) -> dict[str, Any]:
        # Current implementation is meant to run with one worker per node. because the MFA library has issues when running in parallel.
        # We are copying the MFA models to node-local storage to avoid race conditions and Kaldi errors when multiple distributed nodes share the same model directory.
        return {"num_workers_per_node": 1}

    def ray_stage_spec(self) -> dict[str, Any]:
        # Ray Data has no per-node worker cap, so run MFA as a single actor
        # (see num_workers() -> 1). Without this the backend would launch
        # multiple MFA workers per node, re-introducing the Kaldi/NFS races and
        # shared-model corruption that this stage is designed to avoid.
        return {RayStageSpecKeys.IS_ACTOR_STAGE: True}

    def num_workers(self) -> int | None:
        # Force a single MFA worker cluster-wide on the Ray Data backend
        # (Xenna instead honours xenna_stage_spec()'s num_workers_per_node=1).
        # MFA/Kaldi is not safe to run concurrently against a shared model dir.
        return 1

    def setup_on_node(
        self,
        node_info: Any = None,  # noqa: ARG002, ANN401
        worker_metadata: Any = None,  # noqa: ARG002, ANN401
    ) -> None:
        """Copy MFA models from shared storage to node-local directory."""
        if not self.copy_models_to_local:
            self._mfa_root = self._effective_mfa_root
            return
        self._mfa_root = self._setup_local_mfa()
        logger.info(
            f"[setup_on_node] MFA root set to {self._mfa_root} on "
            f"{socket.gethostname()}"
        )

    def setup(
        self,
        worker_metadata: Any = None,  # noqa: ARG002, ANN401
    ) -> None:
        """Resolve the MFA root and create output directories."""
        if not self._mfa_root:
            if self.copy_models_to_local:
                local_candidate = self._local_cache_dir()
                if self._cache_is_valid(local_candidate):
                    self._mfa_root = str(local_candidate)
                    logger.info(
                        f"[setup] Re-using local MFA cache: {self._mfa_root}"
                    )
                else:
                    self._mfa_root = self._effective_mfa_root
                    logger.info(
                        "[setup] Valid local MFA cache not found; using shared "
                        f"MFA root: {self._mfa_root}"
                    )
            else:
                self._mfa_root = self._effective_mfa_root

        self._textgrid_dir.mkdir(parents=True, exist_ok=True)
        if self.create_rttm:
            self._rttm_dir.mkdir(parents=True, exist_ok=True)
        if self.create_ctm:
            self._ctm_dir.mkdir(parents=True, exist_ok=True)

    def process(self, task: AudioTask) -> AudioTask:
        msg = "MFAAlignmentStage only supports process_batch"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:  # noqa: C901
        """Align all tasks in a single ``mfa align`` invocation.

        Per-task pre-flight failures (failed validation, empty text, or a
        missing audio file) degrade gracefully: the offending task is marked
        ``mfa_skipped=True`` with empty outputs and kept in the returned list,
        so a single malformed row never discards the rest of the batch. This
        mirrors the graceful handling of files MFA silently drops (see
        ``_handle_missing_textgrid``).  Batch-level failures (the ``mfa align``
        subprocess itself failing) still propagate.

        The returned list has the same length and order as ``tasks``; every
        task is mutated in place, so cardinality is preserved.
        """
        if len(tasks) == 0:
            return []

        stem_to_task: dict[str, AudioTask] = {}
        for task in tasks:
            try:
                file_stem = self._preflight_task(task, stem_to_task)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Skipping task that failed MFA pre-flight: {exc}")
                self._mark_task_skipped(task)
                continue
            stem_to_task[file_stem] = task

        # All tasks failed pre-flight; nothing to align.
        if not stem_to_task:
            return list(tasks)

        batch_uuid = uuid.uuid4().hex[:12]
        tg_out_path = self._textgrid_dir / batch_uuid
        tg_out_path.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="mfa_corpus_") as corpus_dir:
            corpus_path = Path(corpus_dir)
            for corpus_stem, task in stem_to_task.items():
                audio_path = Path(task.data[self.audio_filepath_key])
                corpus_wav = corpus_path / f"{corpus_stem}.wav"
                if not corpus_wav.exists() and not corpus_wav.is_symlink():
                    try:
                        corpus_wav.symlink_to(audio_path.resolve())
                    except OSError:
                        shutil.copy2(audio_path, corpus_wav)
                corpus_txt = corpus_path / f"{corpus_stem}.txt"
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

        return list(tasks)

    def _preflight_task(
        self, task: AudioTask, stem_to_task: dict[str, AudioTask]
    ) -> str:
        """Validate one task and return its unique corpus stem.

        Raises ``ValueError``/``FileNotFoundError`` if the task cannot be
        aligned (failed validation, empty text, or a missing audio file);
        :meth:`process_batch` treats any such failure as a per-row skip.
        """
        if not self.validate_input(task):
            msg = f"Task {task!s} failed validation for stage {self}"
            raise ValueError(msg)
        audio_filepath = task.data[self.audio_filepath_key]
        text = task.data[self.text_key].strip()
        if not text:
            msg = f"Empty text for {audio_filepath} (key={self.text_key!r})"
            raise ValueError(msg)
        audio_path = Path(audio_filepath)
        if not audio_path.exists():
            msg = f"Audio file not found: {audio_path}"
            raise FileNotFoundError(msg)

        file_stem = audio_path.stem
        if file_stem in stem_to_task:
            original_stem = file_stem
            file_stem = f"{file_stem}_{uuid.uuid4().hex[:8]}"
            logger.warning(
                f"Duplicate stem '{original_stem}' — renamed to "
                f"'{file_stem}' to avoid silent data loss"
            )
        if not task.data.get(self.duration_key):
            task.data[self.duration_key] = self._get_audio_duration(
                str(audio_path)
            )
        return file_stem

    def _mark_task_skipped(self, task: AudioTask) -> None:
        """Mark a task as skipped with empty outputs.

        Used for tasks that fail pre-flight so the batch can continue while
        preserving cardinality and the declared output keys.
        """
        task.data["textgrid_filepath"] = ""
        task.data["mfa_skipped"] = True
        if self.create_rttm:
            task.data["rttm_filepath"] = ""
        if self.create_ctm:
            task.data["ctm_filepath"] = ""

    def _handle_successful_textgrid(
        self, file_stem: str, task: AudioTask, tg_path: Path
    ) -> None:
        task.data["textgrid_filepath"] = str(tg_path)
        task.data["mfa_skipped"] = False
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

    def _run_mfa_align(  # noqa: C901, PLR0912
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
        if history_file.exists() and self._is_node_local_mfa_root():
            try:
                history_file.unlink()
            except OSError:
                logger.debug(
                    f"Could not remove MFA history file: {history_file}"
                )

        cmd = [
            *mfa_cmd_parts,
            "align",
            str(corpus_dir),
            self.dictionary,
            self.acoustic_model,
            str(textgrid_output_dir),
            "--output_format",
            self.output_format,
            "-j",
            str(self.num_jobs),
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
                else:
                    logger.warning(
                        f"G2P model '{self.g2p_model}' not found at "
                        f"{g2p_path} or {g2p_alt}. MFA will run without "
                        f"G2P — OOV words may fail alignment."
                    )

        logger.info(f"Running MFA align: {' '.join(cmd)}")

        result = self._run_mfa_subprocess(cmd, env)

        if result.stdout and result.stdout.strip():
            logger.info(
                f"MFA stdout (last 5000 chars):\n{result.stdout[-5000:]}"
            )
        if result.stderr and result.stderr.strip():
            logger.warning(
                f"MFA stderr (last 5000 chars):\n{result.stderr[-5000:]}"
            )

        if result.returncode != 0:
            msg = (
                f"mfa align failed (exit code {result.returncode}).\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
            raise RuntimeError(msg)

    def _run_mfa_subprocess(
        self, cmd: list[str], env: dict[str, str]
    ) -> subprocess.CompletedProcess:
        """Run the ``mfa align`` command with a hard timeout.

        Unlike ``subprocess.run(..., timeout=...)`` -- whose own internal
        timeout handling only kills the immediate child process -- this
        starts ``cmd`` in its own process group (``start_new_session=True``)
        and, on expiry, kills the *entire* group via ``os.killpg``. That
        covers any Kaldi/MFA worker processes MFA itself forked (e.g. via
        ``-j``), which would otherwise be orphaned and keep running. This
        stage runs as a single worker cluster-wide, so a wedged MFA process
        would otherwise block every remaining row indefinitely.
        """
        process = subprocess.Popen(  # noqa: S603
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=self.align_timeout_seconds)
        except subprocess.TimeoutExpired:
            logger.error(
                f"mfa align exceeded timeout of {self.align_timeout_seconds:.0f}s "
                f"(pid={process.pid}); killing its process group."
            )
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            msg = (
                f"mfa align timed out after {self.align_timeout_seconds:.0f}s "
                "and its process group was killed.\n"
                f"STDOUT (last 2000 chars):\n{(stdout or '')[-2000:]}\n"
                f"STDERR (last 2000 chars):\n{(stderr or '')[-2000:]}"
            )
            raise TimeoutError(msg) from None

        return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)

    def _get_word_alignment_tier(self, tg: Any, textgrid_path: Path) -> Any:  # noqa: ANN401
        """Select the word-level tier, avoiding phone-level tiers when possible."""
        for tier_name in _WORD_TIER_NAMES:
            if tier_name in tg.tierNames:
                return tg.getTier(tier_name)

        if not tg.tierNames:
            msg = f"No tiers found in TextGrid: {textgrid_path}"
            raise ValueError(msg)

        non_phone_tiers = [
            name
            for name in tg.tierNames
            if name.lower() not in _PHONE_TIER_NAMES
        ]
        if non_phone_tiers:
            fallback_name = non_phone_tiers[0]
            logger.warning(
                f"No 'words' tier in {textgrid_path}; "
                f"available tiers: {list(tg.tierNames)}. "
                f"Using '{fallback_name}'."
            )
            return tg.getTier(fallback_name)

        msg = (
            f"No word alignment tier in {textgrid_path}; "
            f"available tiers: {list(tg.tierNames)}. "
            "Refusing to parse phone-level tiers as words."
        )
        raise ValueError(msg)

    def _parse_textgrid_words(self, textgrid_path: Path) -> list[tuple]:
        """Return ``[(start, end, label), ...]`` from the word alignment tier."""
        tg = self._textgrid_mod.openTextgrid(
            str(textgrid_path), includeEmptyIntervals=False
        )
        tier = self._get_word_alignment_tier(tg, textgrid_path)
        return [(e.start, e.end, e.label) for e in tier.entries]

    def _is_node_local_mfa_root(self) -> bool:
        """True when MFA root is the per-node local copy (safe to mutate)."""
        if self.copy_models_to_local:
            return True
        try:
            mfa_root = Path(self._mfa_root).resolve()
            local_mfa = self._local_cache_dir().resolve()
        except OSError:
            return False
        return mfa_root == local_mfa or local_mfa in mfa_root.parents

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
            f.writelines(
                f"SPEAKER {file_stem} 1 "
                f"{iv['start']:.3f} {iv['duration']:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
                for iv in merged
            )

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
            f.writelines(
                f"{file_stem} 1 {i * word_dur:.3f} {word_dur:.3f} {word}\n"
                for i, word in enumerate(words)
            )

    def _cache_identity(self) -> dict[str, str]:
        """Identity that must match for a node-local model cache to be reused.

        Includes the *resolved* shared source root (so two distinct sources
        never collide on a shared hostname/local-base) plus the specific
        acoustic/dictionary/g2p model identity requested by this stage
        instance (so a cache built for one model set is never silently
        handed to a run that asked for a different one).
        """
        resolved_source = str(Path(self._effective_mfa_root).resolve())
        return {
            "source": resolved_source,
            "acoustic_model": self.acoustic_model,
            "dictionary": self.dictionary,
            "g2p_model": self.g2p_model or "",
        }

    def _cache_digest(self) -> str:
        canonical = json.dumps(self._cache_identity(), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def _local_cache_dir(self) -> Path:
        """Node-local cache directory, namespaced by hostname + source/model identity."""
        digest = self._cache_digest()
        return (
            Path(self._effective_local_base)
            / f"mfa_models_{socket.gethostname()}_{digest}"
        )

    def _read_cache_marker(self, cache_dir: Path) -> dict[str, Any] | None:
        marker_path = cache_dir / _MFA_CACHE_MARKER_NAME
        try:
            return json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_is_valid(self, cache_dir: Path) -> bool:
        """True only if ``cache_dir`` was fully populated for our exact identity.

        A missing/corrupt marker means either the cache was never fully built
        (e.g. an interrupted previous copy) or it belongs to a different
        source/model identity, either way it must not be trusted.
        """
        marker = self._read_cache_marker(cache_dir)
        return marker is not None and marker.get("complete") is True and marker.get("identity") == self._cache_identity()

    @contextlib.contextmanager
    def _cache_lock(self, cache_dir: Path):  # noqa: ANN202
        """Serialize concurrent check-and-populate of the same cache directory.

        Guards against two stage instances on the same node (e.g. separate
        actors, or a retried ``setup_on_node``) racing to copy into -- and
        publish -- the same node-local cache simultaneously.
        """
        lock_path = Path(f"{cache_dir}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _setup_local_mfa(self) -> str:
        """Copy MFA models from shared storage to a node-local, identity-namespaced cache.

        The cache directory name embeds a digest of the resolved source root
        plus the requested acoustic/dictionary/g2p model identity (see
        :meth:`_cache_identity`), so two different sources -- or two
        different requested models -- sharing a hostname/local base never
        collide. The copy is staged in a sibling temp directory and a
        completeness marker recording that identity is written *inside* the
        staging directory before it is atomically published via
        ``os.replace``. A reader therefore only ever observes the final
        cache directory in a fully-populated, marker-validated state; a
        crash or interruption mid-copy simply leaves an orphaned staging
        directory behind rather than a partially-populated cache that could
        be silently (and incorrectly) reused. A per-cache file lock
        serializes concurrent populators on the same node.
        """
        cache_dir = self._local_cache_dir()
        with self._cache_lock(cache_dir):
            if cache_dir.exists() and self._cache_is_valid(cache_dir):
                logger.info(f"Using existing local MFA cache: {cache_dir}")
                return str(cache_dir)

            if cache_dir.exists():
                logger.warning(
                    f"Local MFA cache at {cache_dir} has no valid completeness "
                    "marker (e.g. from an interrupted previous copy); rebuilding."
                )
                shutil.rmtree(cache_dir, ignore_errors=True)

            logger.info(f"Copying MFA models to local cache: {cache_dir}")
            staging_dir = cache_dir.with_name(f"{cache_dir.name}.tmp-{uuid.uuid4().hex[:8]}")
            shutil.rmtree(staging_dir, ignore_errors=True)
            staging_dir.mkdir(parents=True)
            try:
                src = Path(self._effective_mfa_root)
                for subdir in ("pretrained_models", "extracted_models"):
                    src_path = src / subdir
                    if src_path.exists():
                        logger.info(f"  Copying {subdir}...")
                        shutil.copytree(src_path, staging_dir / subdir)

                marker = {"identity": self._cache_identity(), "complete": True}
                marker_tmp = staging_dir / f"{_MFA_CACHE_MARKER_NAME}.tmp"
                marker_tmp.write_text(json.dumps(marker), encoding="utf-8")
                marker_tmp.replace(staging_dir / _MFA_CACHE_MARKER_NAME)

                os.replace(staging_dir, cache_dir)
            except BaseException:
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise

            logger.info(f"Local MFA cache populated: {cache_dir}")
            return str(cache_dir)
