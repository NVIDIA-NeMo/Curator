# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Resolve per-stage pip_specs into virtualenvs and set _resolved_site_packages_path on stages.

Uses the `uv` CLI to create venvs (no Python dependency on uv). When a stage defines
pip_specs (e.g. ["vllm==0.6.0"]), the resolver creates a venv with those packages
and sets the stage's _resolved_site_packages_path so executors can inject PYTHONPATH.
"""

import atexit
import shutil
import sysconfig
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage


def _site_packages_for_venv(venv_dir: Path) -> Path:
    # Windows uses "Lib" (capital), Unix uses "lib"
    for lib_name in ("lib", "Lib"):
        lib = venv_dir / lib_name
        if not lib.exists():
            continue
        # On Windows, site-packages may live directly under Lib\ (no python3.x subdir)
        site_direct = lib / "site-packages"
        if site_direct.exists():
            return site_direct.resolve()
        for p in lib.iterdir():
            if p.is_dir() and p.name.startswith("python"):
                site = p / "site-packages"
                if site.exists():
                    return site.resolve()
    msg = f"no site-packages found under {venv_dir}"
    raise RuntimeError(msg)


def _venv_python_exe(venv_path: Path) -> Path:
    """Path to the venv Python interpreter (portable: Windows Scripts/python.exe, Unix bin/python)."""
    if sysconfig.get_platform().startswith("win"):
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _create_venv_for_specs(
    specs: list[str], base_dir: Path, subdir_name: str, uv_exe: str
) -> Path:
    import subprocess

    venv_root = base_dir / subdir_name
    venv_root.mkdir(parents=True, exist_ok=True)
    venv_path = venv_root / ".venv"
    python_path = _venv_python_exe(venv_path)
    subprocess.run(  # noqa: S603
        [uv_exe, "venv", str(venv_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    # specs come from stage pip_specs (pipeline author configuration)
    try:
        subprocess.run(  # noqa: S603
            [uv_exe, "pip", "install", "--python", str(python_path), *specs],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("uv pip install failed for specs %s:\n%s", specs, e.stderr)
        raise

    return _site_packages_for_venv(venv_path)


def resolve_stage_pip_envs(
    stages: list["ProcessingStage"],
    base_dir: Path | None = None,
) -> None:
    """Create venvs for stages that have pip_specs and set _resolved_site_packages_path.

    Stages with the same pip_specs (order-independent) share one venv. Uses the `uv`
    CLI; must be on PATH. Modifies each stage instance in place.

    Args:
        stages: Flat list of execution stages (e.g. after pipeline build).
        base_dir: Directory for venv roots. If None, a temp directory is used.
    """
    import subprocess
    import tempfile

    stages_with_pip = [s for s in stages if getattr(s, "pip_specs", None)]
    uv_exe = shutil.which("uv")
    if not uv_exe:
        if stages_with_pip:
            names = ", ".join(getattr(s, "name", s.__class__.__name__) for s in stages_with_pip)
            msg = (
                "Stages with pip_specs require `uv` on PATH to resolve dependencies, "
                f"but `uv` was not found. Affected stages: {names}. "
                "Install uv (e.g. pip install uv) or add it to PATH."
            )
            raise RuntimeError(msg)
        return
    try:
        subprocess.run(  # noqa: S603
            [uv_exe, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        if stages_with_pip:
            names = ", ".join(getattr(s, "name", s.__class__.__name__) for s in stages_with_pip)
            msg = (
                f"Stages with pip_specs require a working `uv` CLI; uv check failed. "
                f"Affected stages: {names}. Error: {e}"
            )
            raise RuntimeError(msg) from e
        return

    if base_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="curator_pip_envs_")
        atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
        base = Path(tmp_dir)
    else:
        base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    # Dedupe by normalized specs
    unique_specs: dict[tuple[str, ...], Path] = {}
    for stage in stages:
        if getattr(stage, "_resolved_site_packages_path", None):
            continue  # already resolved; skip
        specs = getattr(stage, "pip_specs", None)
        if not specs or not isinstance(specs, list):
            continue
        key = tuple(sorted(s.strip().lower() for s in specs))
        if key not in unique_specs:
            subdir = f"env_{len(unique_specs)}"
            unique_specs[key] = _create_venv_for_specs(list(specs), base, subdir, uv_exe)
            logger.info("Created venv for pip_specs %s at %s", list(specs), unique_specs[key])
        stage._resolved_site_packages_path = unique_specs[key]
