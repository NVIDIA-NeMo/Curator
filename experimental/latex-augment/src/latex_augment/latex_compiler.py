# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import subprocess
import tempfile
import time


class LatexCompileError(ValueError):
    def __init__(self, returncode, cmd, output=b"", verbose=False):
        try:
            output = output.decode("utf8")
        except UnicodeDecodeError:
            # best effort
            output = output.decode("latin1")
        self.returncode = returncode
        self.cmd = cmd
        self.output = output
        msg = ""
        for line in output.split("\n"):
            if msg or line.startswith("!") or line.startswith("Error") or ":fatal" in line:
                msg += line + "\n"
        if not msg:
            msg = f"Compilation failed with return code {returncode}: {cmd}"
        if verbose:
            msg += f"\nFull output:\n{output}\nSee /tmp/main.tex and /tmp/main.log for more details."
        super().__init__(msg)


def compile_latex(
    source: bytes,
    *,
    dirname: str = None,
    latex: str = "pdflatex",
    bibtex: str = "bibtex",
    quick: bool = False,
    timeout: int | float | None = None,
    verbose: bool = False,
) -> bytes:
    """Compile LaTeX to PDF using pdflatex.
    :param timeout: compilation timeout in seconds, or None for no timeout
    """
    cmds = [
        (latex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"),
        (bibtex, "main"),
        (latex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"),
        (latex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"),
    ]
    if not bibtex:
        del cmds[1]
    if quick:
        cmds = cmds[:1]
    deadline = time.time() + timeout if timeout else None
    with tempfile.TemporaryDirectory() as tempdir:
        if dirname:
            shutil.copytree(dirname, tempdir, dirs_exist_ok=True)
        with open(f"{tempdir}/main.tex", "wb") as f:
            f.write(source)
        for cmd in cmds:
            try:
                # cmd = ("strace", "-e", "trace=%file", "-f") + cmd,
                subprocess.run(
                    cmd,
                    cwd=tempdir,
                    check=True,
                    # move tex temp folders away from NFS
                    # also, slurm mounts $HOME on NFS
                    env={
                        **os.environ,
                        "HOME": "/tmp/home",
                        "TEXMFHOME": "/tmp/texmf",
                        "TEXMFCONFIG": "/tmp/texmf-config",
                        "TEXMFVAR": "/tmp/texmf-var",
                    },
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=max(1, deadline - time.time()) if deadline else None,
                )
            except subprocess.CalledProcessError as e:
                if b"! Interruption." in e.stdout:
                    raise KeyboardInterrupt(f"Interrupted: {cmd}") from None

                if b":fatal" in e.stdout:
                    shutil.copy(f"{tempdir}/main.tex", "/tmp")
                    shutil.copy(f"{tempdir}/main.log", "/tmp")
                    raise LatexCompileError(
                        e.returncode, cmd, output=e.stdout, verbose=verbose
                    ) from None

                if cmd[0] != bibtex:  # Ignore bibtex errors
                    shutil.copy(f"{tempdir}/main.tex", "/tmp")
                    shutil.copy(f"{tempdir}/main.log", "/tmp")
                    raise LatexCompileError(
                        e.returncode, cmd, output=e.stdout, verbose=verbose
                    ) from None
            except subprocess.TimeoutExpired:
                raise LatexCompileError(1, cmd, output=b"Timed out") from None

        with open(f"{tempdir}/main.pdf", "rb") as f:
            return f.read()
