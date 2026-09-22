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

"""Provision and locate the isolated Indic Canary TensorRT-LLM runtime.

TensorRT-LLM 1.2.1 is a CPython 3.12/CUDA 13 native stack whose dependency
and ABI requirements cannot share Curator's main environment.  This module
keeps that stack in a second, independently locked virtual environment.  The
Curator worker only launches its Python executable and communicates with it
over local IPC.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

_RUNTIME_ENV_VAR = "NEMO_CURATOR_INDIC_CANARY_RUNTIME_PYTHON"
_SYSTEM_RUNTIME_ROOT = Path("/opt/nemo-curator-runtimes/indic-canary-trtllm")
_LOCK_MARKER = ".nemo_curator_runtime_lock_sha256"
_PYTHON_VERSION = "3.12"
_MPI_ENV_PREFIXES = (
    "HYDRA_",
    "I_MPI_",
    "MPI_",
    "MPICH_",
    "MV2_",
    "OMPI_",
    "PMI_",
    "PMIX_",
    "SLURM_",
)
_DISTRIBUTED_ENV_VARS = {
    "GROUP_RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "RANK",
    "ROLE_RANK",
    "WORLD_SIZE",
}
_RUNTIME_DISTRIBUTIONS = {
    "cuda-python": "13.3.1",
    "cuda-toolkit": "13.3.1",
    "nvidia-cublas": "13.3.0.5",
    "numpy": "1.26.4",
    "openmpi": "4.1.8",
    "tensorrt": "10.14.1.48.post1",
    "tensorrt-llm": "1.2.1",
    "torch": "2.9.1+cu128",
    "transformers": "4.57.3",
}


def runtime_project_dir() -> Path:
    """Return the packaged, independently locked runtime project."""
    return Path(__file__).with_name("runtimes") / "indic_canary_trtllm"


def runtime_lock_digest() -> str:
    """Return a content identity for the runtime specification and lock."""
    digest = hashlib.sha256()
    for filename in ("pyproject.toml", "uv.lock"):
        path = runtime_project_dir() / filename
        digest.update(filename.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def default_runtime_root() -> Path:
    """Return the user-cache location for this exact runtime lock."""
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "nemo_curator" / "runtimes" / "indic_canary_trtllm" / runtime_lock_digest()[:16]


def _runtime_python(root: Path) -> Path:
    return root / "bin" / "python"


def _runtime_prefix(runtime_python: Path) -> Path:
    return runtime_python.parent.parent


def _validate_runtime_layout(runtime_python: Path) -> None:
    """Require the native libraries needed by the packaged worker."""
    prefix = _runtime_prefix(runtime_python)
    site_packages = prefix / "lib" / "python3.12" / "site-packages"
    required_libraries = (
        prefix / "lib" / "libmpi.so.40",
        site_packages / "nvidia" / "cu13" / "lib" / "libcublasLt.so.13",
        site_packages / "tensorrt_libs" / "libnvinfer.so.10",
        site_packages / "tensorrt_llm" / "libs" / "libtensorrt_llm.so",
        site_packages / "torch" / "lib" / "libtorch.so",
    )
    missing = [str(path) for path in required_libraries if not path.is_file()]
    if missing:
        msg = f"Indic Canary runtime is missing required native libraries: {missing}"
        raise RuntimeError(msg)


def _validate_runtime_python(path: Path) -> Path:
    path = path.expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        msg = f"Indic Canary runtime Python is not an executable file: {path}"
        raise FileNotFoundError(msg)
    validation_script = f"""
import importlib.metadata
import platform
import sys

expected = {_RUNTIME_DISTRIBUTIONS!r}
problems = []
if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 12):
    problems.append(
        f"expected CPython 3.12, found {{platform.python_implementation()}} "
        f"{{sys.version_info.major}}.{{sys.version_info.minor}}"
    )
for distribution, expected_version in expected.items():
    try:
        actual_version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        problems.append(f"{{distribution}} is not installed")
    else:
        if actual_version != expected_version:
            problems.append(f"{{distribution}}=={{actual_version}} (expected {{expected_version}})")
if problems:
    sys.stderr.write("; ".join(problems))
    raise SystemExit(1)
"""
    completed = subprocess.run(  # noqa: S603
        [str(path), "-I", "-c", validation_script],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        msg = f"Indic Canary TensorRT-LLM runtime does not match its locked stack: {path}"
        if detail:
            msg = f"{msg} ({detail})"
        raise RuntimeError(msg)
    _validate_runtime_layout(path)
    return path


def resolve_runtime_python(configured: str | None = None) -> Path:
    """Resolve a pre-provisioned isolated runtime Python executable.

    Resolution order is an explicit stage value, the environment variable,
    the opt-in Curator Docker target, and the lock-keyed user cache created by
    :func:`install_runtime`.
    """
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    elif os.environ.get(_RUNTIME_ENV_VAR):
        candidates.append(Path(os.environ[_RUNTIME_ENV_VAR]))
    else:
        candidates.extend((_runtime_python(_SYSTEM_RUNTIME_ROOT), _runtime_python(default_runtime_root())))

    for candidate in candidates:
        if candidate.expanduser().is_file():
            return _validate_runtime_python(candidate)

    locations = ", ".join(str(path.expanduser()) for path in candidates)
    msg = (
        "Indic Canary requires its isolated TensorRT-LLM runtime, but no runtime Python was found "
        f"at: {locations}. Pre-provision it with `python -m "
        "nemo_curator.stages.audio.inference.scripts.install_indic_canary_trtllm_runtime`, "
        f"or set {_RUNTIME_ENV_VAR} to that environment's bin/python."
    )
    raise FileNotFoundError(msg)


def ensure_runtime_python(configured: str | None = None) -> Path:
    """Resolve the runtime, provisioning the lock-keyed cache when implicit.

    Explicit paths remain strict: a misspelled stage value or environment
    override must fail instead of silently downloading and selecting another
    interpreter.  With no override, audio-stage node prefetch installs the
    frozen runtime once and subsequent workers reuse it.
    """
    if configured is not None or os.environ.get(_RUNTIME_ENV_VAR):
        return resolve_runtime_python(configured)

    system_python = _runtime_python(_SYSTEM_RUNTIME_ROOT)
    if system_python.is_file():
        return _validate_runtime_python(system_python)

    # Always pass the implicit cache through the marker-aware installer. It
    # validates and reuses a complete environment, and repairs one left
    # partial by an interrupted node-preparation attempt.
    return install_runtime()


def _filtered_inherited_paths(value: str, parent_prefixes: list[Path]) -> list[Path]:
    """Drop paths belonging to Curator's venv or its CUDA toolkit."""
    filtered: list[Path] = []
    for entry in value.split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry)
        absolute_candidate = candidate.expanduser().absolute()
        if any(absolute_candidate.is_relative_to(prefix) for prefix in parent_prefixes):
            continue
        if str(absolute_candidate).startswith("/usr/local/cuda"):
            continue
        filtered.append(candidate)
    return filtered


def _deduplicate_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def runtime_subprocess_environment(runtime_python: Path) -> dict[str, str]:
    """Build the worker environment without exposing Curator's site-packages."""
    runtime_python = _validate_runtime_python(runtime_python)
    prefix = _runtime_prefix(runtime_python)
    site_packages = prefix / "lib" / "python3.12" / "site-packages"
    runtime_library_dirs = [
        site_packages / "tensorrt_llm" / "libs",
        site_packages / "tensorrt_libs",
        site_packages / "nvidia" / "cu13" / "lib",
        site_packages / "torch" / "lib",
        site_packages / "nvidia" / "nccl" / "lib",
        site_packages / "nvidia" / "cudnn" / "lib",
        prefix / "lib",
        prefix / "lib" / "openmpi",
    ]
    runtime_library_dirs.extend(sorted((site_packages / "nvidia").glob("*/lib")))

    # The worker is intentionally a one-process TensorRT-LLM runtime.  Do not
    # let an enclosing Slurm/mpirun/torchrun job make its bundled OpenMPI join
    # the parent's communicator or report a rank incompatible with world_size=1.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in _DISTRIBUTED_ENV_VARS and not key.startswith(_MPI_ENV_PREFIXES)
    }
    parent_venv = environment.get("VIRTUAL_ENV")
    parent_prefixes = [Path(parent_venv).expanduser().absolute()] if parent_venv else []
    if sys.prefix != sys.base_prefix:
        parent_prefixes.append(Path(sys.prefix).expanduser().absolute())
    inherited_library_dirs = _filtered_inherited_paths(
        environment.get("LD_LIBRARY_PATH", ""),
        parent_prefixes,
    )
    unique_library_dirs = _deduplicate_paths([*runtime_library_dirs, *inherited_library_dirs])

    runtime_bin = prefix / "bin"
    inherited_path = _filtered_inherited_paths(environment.get("PATH", ""), parent_prefixes)
    path_entries = _deduplicate_paths([runtime_bin, *inherited_path])

    environment["LD_LIBRARY_PATH"] = os.pathsep.join(str(path) for path in unique_library_dirs)
    environment["PATH"] = os.pathsep.join(str(path) for path in path_entries)
    environment["VIRTUAL_ENV"] = str(prefix)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUDA_HOME"] = str(site_packages / "nvidia" / "cu13")
    environment["OPAL_PREFIX"] = str(prefix)
    for variable in (
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "CONDA_PROMPT_MODIFIER",
        "PYTHONHOME",
        "PYTHONPATH",
        "UV_PROJECT_ENVIRONMENT",
    ):
        environment.pop(variable, None)
    return environment


def install_runtime(runtime_root: Path | None = None) -> Path:
    """Synchronize the exact locked runtime and return its Python executable.

    This is intentionally an explicit image/setup operation, never a hidden
    first-inference download.  A file lock makes concurrent node preparation
    safe when multiple launchers target the same cache.
    """
    if sys.platform != "linux" or platform.machine().lower() not in {"amd64", "x86_64"}:
        msg = "Indic Canary TensorRT-LLM runtime is supported only on Linux x86_64"
        raise RuntimeError(msg)

    # Import lazily because fcntl is unavailable on Windows, whose imports must remain healthy.
    import fcntl

    adjacent_uv = Path(sys.executable).with_name("uv")
    uv = str(adjacent_uv) if adjacent_uv.is_file() and os.access(adjacent_uv, os.X_OK) else shutil.which("uv")
    if uv is None:
        msg = "uv>=0.12,<0.13 is required; install nemo_curator[audio_trt] first"
        raise RuntimeError(msg)

    root = (runtime_root or default_runtime_root()).expanduser().absolute()
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{root.name}.install.lock"
    expected_digest = runtime_lock_digest()
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        python = _runtime_python(root)
        marker = root / _LOCK_MARKER
        reinstall = False
        if python.is_file() and marker.is_file() and marker.read_text().strip() == expected_digest:
            try:
                return _validate_runtime_python(python)
            except (FileNotFoundError, RuntimeError):
                # A killed node-preparation process or external cache cleanup
                # can invalidate files after the completion marker was written.
                # Force uv to restore every locked distribution in that case.
                reinstall = True

        environment = dict(os.environ)
        environment["UV_PROJECT_ENVIRONMENT"] = str(root)
        environment.setdefault("UV_LINK_MODE", "copy")
        command = [
            uv,
            "sync",
            "--frozen",
            "--no-dev",
            "--no-install-project",
            "--python",
            _PYTHON_VERSION,
            "--project",
            str(runtime_project_dir()),
        ]
        if reinstall:
            command.append("--reinstall")
        subprocess.run(command, check=True, env=environment)  # noqa: S603
        subprocess.run(  # noqa: S603
            [uv, "pip", "check", "--python", str(python), "--no-config"],
            check=True,
            env=environment,
        )
        validated_python = _validate_runtime_python(python)
        marker.write_text(f"{expected_digest}\n")
        return validated_python


def main() -> int:
    """CLI entry point used by image builders and node preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=None,
        help="Virtual-environment directory (default: lock-keyed user cache)",
    )
    args = parser.parse_args()
    print(install_runtime(args.runtime_root))
    return 0


__all__ = [
    "default_runtime_root",
    "ensure_runtime_python",
    "install_runtime",
    "resolve_runtime_python",
    "runtime_lock_digest",
    "runtime_project_dir",
    "runtime_subprocess_environment",
]
