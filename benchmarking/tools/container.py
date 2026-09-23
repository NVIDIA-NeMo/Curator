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

"""Run the source-tree benchmark tools against a Curator Docker container."""

from __future__ import annotations

import argparse
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

THIS_SCRIPT = Path(__file__).resolve()
BENCHMARKING_DIR = THIS_SCRIPT.parents[1]
HOST_REPO_ROOT = THIS_SCRIPT.parents[2]
CONTAINER_BENCHMARK_SOURCE_MOUNT_DIR = Path("/tmp/.curator-benchmark-source")  # noqa: S108
CONTAINER_BENCHMARK_SOURCE_DIR = Path("/opt/curator-benchmark-source")
CONTAINER_CURATOR_UNDER_TEST_DIR = Path("/opt/Curator")
CONTAINER_SETUP_SCRIPT_CANDIDATES = (
    CONTAINER_BENCHMARK_SOURCE_MOUNT_DIR / "benchmarking" / "tools" / "setup_benchmark_env.sh",
    CONTAINER_BENCHMARK_SOURCE_DIR / "benchmarking" / "tools" / "setup_benchmark_env.sh",
)
BENCHMARK_SOURCE_SUBDIRS = ("benchmarking", "tutorials")

# The container helper imports only runner modules that are intentionally light:
# YAML config merging, path resolution, and cgroup-aware memory sizing.
sys.path.insert(0, str(BENCHMARKING_DIR))

# ruff: noqa: E402
from nemo_curator_benchmarking.config import assert_valid_config_dict, merge_config_files
from runner.path_resolver import DEFAULT_CONTAINER_PATH_PREFIX, PathResolver
from runner.utils import get_total_memory_bytes

PASSTHROUGH_ENV_VARS = (
    "SLACK_BOT_TOKEN",
    "SLACK_CHANNEL_ID",
    "NVIDIA_API_KEY",
)
DEFAULT_IMAGE = os.environ.get("CURATOR_IMAGE", "nemo_curator:latest")
DEFAULT_CONTAINER_NAME = os.environ.get("CURATOR_BENCHMARK_CONTAINER", "curator-benchmark")
DEFAULT_GPUS = os.environ.get("GPUS", "all")
MAX_CONTAINER_MEMORY_BYTES = 2 * 1024**4
SHM_MEMORY_FRACTION = 0.5


def _run(cmd: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True)  # noqa: S603


def _capture(cmd: Sequence[str]) -> str:
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)  # noqa: S603
    return result.stdout.strip()


def _mount_path_for_host_path(host_path: Path) -> Path:
    if not host_path.is_absolute():
        msg = f"Path '{host_path}' must be absolute before it can be mounted."
        raise ValueError(msg)
    return Path(DEFAULT_CONTAINER_PATH_PREFIX) / host_path.relative_to("/")


def _container_path_for_config(config_path: Path) -> Path:
    resolved = config_path.expanduser().resolve()
    try:
        relative = resolved.relative_to(HOST_REPO_ROOT)
    except ValueError:
        return _mount_path_for_host_path(resolved)
    if relative.parts and relative.parts[0] in BENCHMARK_SOURCE_SUBDIRS:
        return CONTAINER_BENCHMARK_SOURCE_DIR / relative
    return _mount_path_for_host_path(resolved)


def _config_args_for_container(configs: Sequence[Path]) -> list[str]:
    args: list[str] = []
    for config in configs:
        args.extend(["--config", str(_container_path_for_config(config))])
    return args


def _read_config_mounts(configs: Sequence[Path]) -> list[tuple[Path, Path]]:
    if not configs:
        return []

    config = merge_config_files([config.expanduser().resolve() for config in configs])
    assert_valid_config_dict(config)
    resolver = PathResolver(config)
    return [(host.expanduser().resolve(), container) for host, container in resolver.volume_mount_pairs()]


def _docker_volume_args(configs: Sequence[Path], *, use_host_curator: bool) -> list[str]:
    mounts = [(HOST_REPO_ROOT, CONTAINER_BENCHMARK_SOURCE_MOUNT_DIR, True)]

    if use_host_curator:
        mounts.append((HOST_REPO_ROOT, CONTAINER_CURATOR_UNDER_TEST_DIR, False))

    for config in configs:
        resolved = config.expanduser().resolve()
        if not resolved.exists():
            msg = f"Config file does not exist: {resolved}"
            raise FileNotFoundError(msg)
        try:
            relative = resolved.relative_to(HOST_REPO_ROOT)
        except ValueError:
            relative = None
        if relative is None or not relative.parts or relative.parts[0] not in BENCHMARK_SOURCE_SUBDIRS:
            mounts.append((resolved, _mount_path_for_host_path(resolved), False))

    mounts.extend((host, container, False) for host, container in _read_config_mounts(configs))

    seen: set[tuple[Path, Path, bool]] = set()
    args: list[str] = []
    for host, container, read_only in mounts:
        pair = (host, container, read_only)
        if pair in seen:
            continue
        seen.add(pair)
        mode = ":ro" if read_only else ""
        args.extend(["--volume", f"{host}:{container}{mode}"])
    return args


def _default_memory_args() -> tuple[int, int]:
    container_memory = min(get_total_memory_bytes(), MAX_CONTAINER_MEMORY_BYTES)
    shm_size = int(container_memory * SHM_MEMORY_FRACTION)
    return container_memory, shm_size


def _image_digest(image: str) -> str:
    digest = _capture(["docker", "image", "inspect", image, "--format", "{{.Digest}}"])
    if digest and digest != "<none>":
        return digest
    image_id = _capture(["docker", "image", "inspect", image, "--format", "{{.ID}}"])
    return image_id if image_id and image_id != "<none>" else "<unknown>"


def _docker_env_args(image: str) -> list[str]:
    env = {
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video",
        "IMAGE_DIGEST": _image_digest(image),
        "HOST_HOSTNAME": socket.gethostname(),
        "CURATOR_BENCHMARK_PATH_MODE": "container",
        "CURATOR_BENCHMARK_CURATOR_REPO_DIR": str(CONTAINER_CURATOR_UNDER_TEST_DIR),
    }
    for name in PASSTHROUGH_ENV_VARS:
        if name in os.environ:
            env[name] = os.environ[name]

    args: list[str] = []
    for name, value in env.items():
        args.extend(["--env", f"{name}={value}"])
    return args


def _docker_exec(container: str, command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["docker", "exec", container, *command], check=check)


def _docker_exec_bash(container: str, script: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _docker_exec(container, ["bash", "-lc", script], check=check)


def _strip_arg_separator(args: Sequence[str]) -> list[str]:
    return list(args[1:] if args and args[0] == "--" else args)


def _run_setup_script(
    container: str,
    action: str,
    *,
    curator_extras: Sequence[str] = (),
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    args: list[str] = []
    if action == "check":
        args.append("--check")
    elif action != "install":
        msg = f"Unknown setup action: {action}"
        raise ValueError(msg)
    for extra in curator_extras:
        args.extend(["--curator-extra", extra])
    quoted_args = " ".join(shlex.quote(arg) for arg in args)
    candidates = " ".join(shlex.quote(str(path)) for path in CONTAINER_SETUP_SCRIPT_CANDIDATES)
    script = f"""
set -u
for setup_script in {candidates}; do
  if [ -f "$setup_script" ]; then
    exec bash "$setup_script" {quoted_args}
  fi
done
echo "ERROR: could not find setup_benchmark_env.sh" >&2
exit 1
"""
    return _docker_exec_bash(container, script, check=check)


def start_container(args: argparse.Namespace) -> int:
    container_memory, shm_size = _default_memory_args()
    if args.container_memory is not None:
        container_memory = args.container_memory
    if args.shm_size is not None:
        shm_size = args.shm_size

    cmd = [
        "docker",
        "run",
        "--detach",
        "--name",
        args.name,
        "--net=host",
        "--memory",
        str(container_memory),
        "--shm-size",
        str(shm_size),
        *_docker_volume_args(args.config, use_host_curator=args.use_host_curator),
        *_docker_env_args(args.image),
    ]
    if args.gpus != "none":
        cmd.extend(["--gpus", args.gpus])
    cmd.extend([args.image, "sleep", "infinity"])

    _run(cmd)
    if args.setup_benchmark_env == "yes":
        _run_setup_script(args.name, "install", curator_extras=args.curator_extra)
    print(args.name)
    return 0


def check_container(args: argparse.Namespace) -> int:
    return _run_setup_script(args.container, "check", check=False).returncode


def run_in_container(args: argparse.Namespace) -> int:
    command = [
        "python",
        str(CONTAINER_BENCHMARK_SOURCE_DIR / "benchmarking" / "run.py"),
        *_config_args_for_container(args.config),
        *_strip_arg_separator(args.runner_args),
    ]
    return _docker_exec(args.container, command, check=False).returncode


def shell_in_container(args: argparse.Namespace) -> int:
    if args.shell_command:
        command = " ".join(shlex.quote(part) for part in _strip_arg_separator(args.shell_command))
        return _docker_exec_bash(args.container, command, check=False).returncode
    return _run(["docker", "exec", "--interactive", "--tty", args.container, "bash"], check=False).returncode


def add_common_container_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", action="append", type=Path, default=[], help="Benchmark YAML config file.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help=f"Curator image to use. Default: {DEFAULT_IMAGE}")
    parser.add_argument(
        "--name",
        default=DEFAULT_CONTAINER_NAME,
        help=f"Container name. Default: {DEFAULT_CONTAINER_NAME}",
    )
    parser.add_argument("--gpus", default=DEFAULT_GPUS, help=f"Value passed to docker --gpus. Default: {DEFAULT_GPUS}")
    parser.add_argument("--container-memory", type=int, help="Docker memory limit in bytes.")
    parser.add_argument("--shm-size", type=int, help="Docker /dev/shm size in bytes.")
    parser.add_argument("--use-host-curator", action="store_true", help="Mount this checkout at /opt/Curator.")
    parser.add_argument(
        "--setup-benchmark-env",
        choices=("yes", "no"),
        default="yes",
        help="Install benchmark runtime dependencies in the container. Default: yes.",
    )
    parser.add_argument(
        "--curator-extra",
        action="append",
        default=[],
        help="Install a Curator-under-test Python extra, e.g. video_cuda12. Can be repeated.",
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start and optionally set up a reusable benchmark container.")
    add_common_container_args(start)
    start.set_defaults(func=start_container)

    run = subparsers.add_parser("run", help="Run benchmarks in an existing container.")
    run.add_argument("--container", required=True, help="Existing container name or ID.")
    run.add_argument("--config", action="append", type=Path, default=[], help="Benchmark YAML config file.")
    run.add_argument("runner_args", nargs=argparse.REMAINDER, help="Additional args passed to benchmarking/run.py.")
    run.set_defaults(func=run_in_container)

    check = subparsers.add_parser("check", help="Check a running benchmark container.")
    check.add_argument("--container", required=True, help="Existing container name or ID.")
    check.set_defaults(func=check_container)

    shell = subparsers.add_parser("shell", help="Open a shell, or run a command, in a running benchmark container.")
    shell.add_argument("--container", required=True, help="Existing container name or ID.")
    shell.add_argument(
        "shell_command",
        nargs=argparse.REMAINDER,
        help="Optional command to run. Use -- before commands with flags.",
    )
    shell.set_defaults(func=shell_in_container)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
