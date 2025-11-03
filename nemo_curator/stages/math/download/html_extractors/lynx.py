# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import shutil
import subprocess

import ftfy


class LynxExtractor:
    """Extract text from HTML using the lynx command-line browser."""

    def __init__(self, timeout_sec: int = 20):
        self.timeout_sec = timeout_sec
        # Validate lynx executable exists at initialization
        lynx_path = shutil.which("lynx")
        if not lynx_path:
            error_msg = "lynx executable not found in PATH"
            raise RuntimeError(error_msg)

    def extract_text(self, html: str) -> str:
        """Extract text from HTML content.

        Returns empty string on any failure (timeout, encoding errors, etc).
        """
        if not html:
            return ""

        try:
            proc = subprocess.run(  # noqa: S603,UP022
                [  # noqa: S607
                    "lynx",
                    "-dump",
                    "-stdin",
                    "-nolist",
                    "-width=10000",
                    "-assume_charset=utf-8",
                    "-display_charset=utf-8",
                    "-localhost",
                    "-force_html",
                ],
                input=html.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=self.timeout_sec,
            )
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError, UnicodeEncodeError):
            return ""

        if proc.returncode == 0:
            try:
                return proc.stdout.decode("utf-8")
            except (UnicodeDecodeError, UnicodeError):
                return ftfy.fix_text(proc.stdout.decode("utf-8", errors="replace"))

        return ""
