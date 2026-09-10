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

from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

VENDORED_TUTORIAL_FILES = (
    (
        Path("../tutorials/video/getting-started/video_split_clip_example.py"),
        Path("curator_benchmarking/_tutorials/video/getting-started/video_split_clip_example.py"),
    ),
    (
        Path("../tutorials/interleaved/nemotron_parse_pdf/pipeline_utils.py"),
        Path("curator_benchmarking/_tutorials/interleaved/nemotron_parse_pdf/pipeline_utils.py"),
    ),
    (
        Path("../tutorials/math/datasets.json"),
        Path("curator_benchmarking/_tutorials/math/datasets.json"),
    ),
)
_TUTORIAL_PACKAGE_DIR = Path("curator_benchmarking/_tutorials")


class BuildPy(build_py):
    """Copy selected tutorial files needed by tutorial-backed benchmarks."""

    def run(self) -> None:
        super().run()
        self._copy_vendored_tutorial_files()

    def _copy_vendored_tutorial_files(self) -> None:
        package_root = Path(__file__).resolve().parent
        build_root = Path(self.build_lib)
        destination_root = build_root / _TUTORIAL_PACKAGE_DIR
        if destination_root.exists():
            shutil.rmtree(destination_root)
        destination_root.mkdir(parents=True)
        shutil.copy2(package_root / _TUTORIAL_PACKAGE_DIR / "__init__.py", destination_root / "__init__.py")

        for source_relative_path, package_relative_path in VENDORED_TUTORIAL_FILES:
            source_path = package_root / source_relative_path
            if not source_path.exists():
                msg = (
                    "nemo-curator-benchmarking must be built from a full Curator "
                    f"checkout; missing required tutorial file: {source_path}"
                )
                raise FileNotFoundError(msg)

            destination_path = build_root / package_relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)


setup(cmdclass={"build_py": BuildPy})
