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

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage


def _site_packages_for_venv(venv_dir: Path) -> Path:
    lib = venv_dir / "lib"
    if not lib.exists():
        msg = f"venv has no lib/: {venv_dir}"
        raise RuntimeError(msg)
    for p in lib.iterdir():
        if p.is_dir() and p.name.startswith("python"):
            site = p / "site-packages"
            if site.exists():
                return site.resolve()
    msg = f"no site-packages found under {venv_dir}"
    raise RuntimeError(msg)


def _create_venv_for_specs(specs: list[str], base_dir: Path, subdir_name: str) -> Path:
    import subprocess

    venv_root = base_dir / subdir_name
    venv_root.mkdir(parents=True, exist_ok=True)
    venv_path = venv_root / ".venv"
    python_path = venv_path / "bin" / "python"
    subprocess.run(
        ["uv", "venv", str(venv_path)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python_path), *specs],
        check=True,
        capture_output=True,
    )
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

    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(
            "resolve_stage_pip_envs requires `uv` on PATH; skipping. Error: %s",
            e,
        )
        return

    base = Path(base_dir) if base_dir else Path(tempfile.mkdtemp(prefix="curator_pip_envs_"))
    base.mkdir(parents=True, exist_ok=True)

    # Dedupe: same sorted specs -> one venv
    unique_specs: dict[tuple[str, ...], Path] = {}
    for i, stage in enumerate(stages):
        specs = getattr(stage, "pip_specs", None)
        if not specs or not isinstance(specs, list):
            continue
        key = tuple(sorted(specs))
        if key not in unique_specs:
            subdir = f"env_{len(unique_specs)}"
            unique_specs[key] = _create_venv_for_specs(list(specs), base, subdir)
            logger.info("Created venv for pip_specs %s at %s", list(key), unique_specs[key])
        setattr(stage, "_resolved_site_packages_path", unique_specs[key])
