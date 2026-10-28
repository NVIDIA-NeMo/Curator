# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invoke the standard Montreal Forced Aligner 3.x CLI on transcript/audio pairs."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_curator.tasks import AudioTask


class MontrealAligner:
    def __init__(  # noqa: PLR0913
        self,
        output_root: Path,
        *,
        executable: str,
        dictionary: str,
        acoustic_model: str,
        jobs: int = 4,
        model_root: Path | None = None,
    ) -> None:
        self.output_root = output_root
        self.executable = executable
        self.dictionary = dictionary
        self.acoustic_model = acoustic_model
        self.jobs = jobs
        self.model_root = model_root or Path(os.environ.get("MFA_ROOT_DIR", "~/.mfa")).expanduser()
        self._state = tempfile.TemporaryDirectory(prefix="curator-montreal-")

    def close(self) -> None:
        self._state.cleanup()

    def _model_path(self, value: str, category: str, extension: str) -> str:
        direct = Path(value).expanduser()
        if direct.is_file():
            return str(direct.resolve())
        for name in (value, value + extension):
            candidate = self.model_root / "pretrained_models" / category / name
            if candidate.is_file():
                return str(candidate.resolve())
        return value

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if not tasks:
            return []
        self.output_root.mkdir(parents=True, exist_ok=True)
        batch = Path(tempfile.mkdtemp(prefix="batch-", dir=self.output_root))
        corpus = batch / "corpus"
        output = batch / "textgrids"
        seen: set[str] = set()
        for task in tasks:
            data = task.data
            row_id = str(data["segment_id"])
            if row_id in seen or not row_id.isalnum():
                msg = f"Invalid or duplicate MFA clip identifier: {row_id!r}"
                raise ValueError(msg)
            seen.add(row_id)
            text = data.get("text")
            if not isinstance(text, str) or not text.strip():
                msg = f"MFA requires a nonempty supplied transcript for {row_id}"
                raise ValueError(msg)
            speaker = str(data["speaker_id"])
            if not speaker or any(c.isspace() or c in "/\\" for c in speaker) or speaker in {".", ".."}:
                msg = f"Invalid MFA speaker identifier: {speaker!r}"
                raise ValueError(msg)
            directory = corpus / speaker
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(data["audio_filepath"], directory / f"{row_id}.wav")
            (directory / f"{row_id}.lab").write_text(text, encoding="utf-8")
        executable_path = Path(shutil.which(self.executable) or self.executable).expanduser()
        child_env = {**os.environ, "MFA_ROOT_DIR": self._state.name}
        if executable_path.is_absolute():
            child_env["PATH"] = str(executable_path.parent) + os.pathsep + child_env.get("PATH", "")
        command = [
            str(executable_path),
            "align",
            str(corpus),
            self._model_path(self.dictionary, "dictionary", ".dict"),
            self._model_path(self.acoustic_model, "acoustic", ".zip"),
            str(output),
            "--output_format",
            "long_textgrid",
            "--num_jobs",
            str(self.jobs),
            "--temporary_directory",
            str(batch / "work"),
            "--clean",
        ]
        log = batch / "mfa.log"
        with log.open("w", encoding="utf-8") as stream:
            try:
                subprocess.run(  # noqa: S603
                    command,
                    env=child_env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                msg = f"MFA failed with exit code {exc.returncode}; see {log}"
                raise RuntimeError(msg) from exc
        grids = {path.stem: path for path in output.rglob("*.TextGrid")}
        for task in tasks:
            row_id = str(task.data["segment_id"])
            path = grids.get(row_id)
            if path is None:
                msg = f"MFA returned no TextGrid for {row_id}; see {log}"
                raise ValueError(msg)
            task.data["mfa_textgrid"] = str(path.resolve())
        return tasks
