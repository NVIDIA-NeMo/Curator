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

import os
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from curator_benchmarking.paths import resolve_benchmark_suite_dir
from runner.path_resolver import (
    CONTAINER_CURATOR_DIR,
    CURATOR_BENCHMARK_PATH_MODE_ENV,
    DEFAULT_CONTAINER_PATH_PREFIX,
    PathResolver,
)
from runner.utils import (
    assert_valid_config_dict,
    get_total_memory_bytes,
    merge_config_files,
    resolve_env_vars,
)

_KB = 1024
_MB = 1024 * _KB
_GB = 1024 * _MB
_TB = 1024 * _GB
_MAX_CONTAINER_MEMORY_BYTES = 2 * _TB
_SHM_SIZE_CONTAINER_MEMORY_PERCENTAGE = 0.5
DEFAULT_IMAGE = os.environ.get(
    "CURATOR_BENCHMARK_IMAGE",
    os.environ.get("CURATOR_BENCHMARKING_IMAGE", "nemo_curator:latest"),
)
DEFAULT_SUITE_CONTAINER_DIR = Path("/opt/curator-benchmark-suite")


@dataclass
class DockerTarget:
    image: str | None = None
    container: str | None = None
    name: str | None = None
    benchmark_setup: str = "auto"
    benchmark_suite_dir: Path | None = None
    benchmark_suite_container_dir: Path = DEFAULT_SUITE_CONTAINER_DIR
    benchmark_extras: list[str] = field(default_factory=lambda: ["all"])
    use_host_curator: bool = False
    use_host_curator_benchmarking: bool = False
    gpus: str = field(default_factory=lambda: os.environ.get("GPUS", "all"))
    memory: str | None = None
    shm_size: str | None = None
    network: str = "host"
    tty: bool = True

    @property
    def uses_docker(self) -> bool:
        return self.image is not None or self.container is not None

    def validate(self) -> None:
        if self.image is not None and self.container is not None:
            msg = "--image and --container are mutually exclusive"
            raise ValueError(msg)
        if self.name is not None and self.container is not None:
            msg = "--name cannot be combined with --container"
            raise ValueError(msg)
        if self.benchmark_setup not in {"auto", "always", "never"}:
            msg = "--benchmark-setup must be one of: auto, always, never"
            raise ValueError(msg)
        if self.use_host_curator and self.use_host_curator_benchmarking:
            msg = "--use-host-curator and --use-host-curator-benchmarking cannot be combined"
            raise ValueError(msg)


def default_container_memory_bytes() -> int:
    return min(get_total_memory_bytes(), _MAX_CONTAINER_MEMORY_BYTES)


def default_shm_size_bytes() -> int:
    return int(default_container_memory_bytes() * _SHM_SIZE_CONTAINER_MEMORY_PERCENTAGE)


def add_target_arguments(parser) -> None:  # noqa: ANN001
    parser.add_argument(
        "--image",
        nargs="?",
        const=DEFAULT_IMAGE,
        default=None,
        help=f"Start a new Docker container from IMAGE. If omitted, uses {DEFAULT_IMAGE}.",
    )
    parser.add_argument(
        "--container",
        default=None,
        help="Run inside an existing Docker container with docker exec.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name for a detached container started by `curator-benchmark start`.",
    )
    parser.add_argument(
        "--benchmark-setup",
        choices=["auto", "always", "never"],
        default="auto",
        help="Install/check the benchmark package in Docker targets before running.",
    )
    parser.add_argument(
        "--benchmark-suite-dir",
        type=Path,
        default=None,
        help="Curator checkout that provides the benchmark package, scripts, and configs.",
    )
    parser.add_argument(
        "--benchmark-suite-container-dir",
        type=Path,
        default=DEFAULT_SUITE_CONTAINER_DIR,
        help="Container path where --benchmark-suite-dir is mounted or already available.",
    )
    parser.add_argument(
        "--benchmark-extra",
        action="append",
        dest="benchmark_extras",
        default=None,
        help="Benchmark package extra to install. Can be specified multiple times. Defaults to all.",
    )
    parser.add_argument(
        "--use-host-curator",
        action="store_true",
        help="Mount the host Curator checkout as the Curator package under test.",
    )
    parser.add_argument(
        "--use-host-curator-benchmarking",
        action="store_true",
        help="Compatibility alias for selecting the host checkout as the benchmark suite.",
    )
    parser.add_argument(
        "--gpus",
        default=os.environ.get("GPUS", "all"),
        help="Value passed to docker run --gpus. Use 'none' to disable GPU access.",
    )
    parser.add_argument(
        "--container-memory",
        default=None,
        help="Value passed to docker run --memory. Defaults to host memory capped at 2 TiB.",
    )
    parser.add_argument(
        "--shm-size",
        default=None,
        help="Value passed to docker run --shm-size. Defaults to 50%% of container memory.",
    )
    parser.add_argument(
        "--network",
        default="host",
        help="Docker network mode for --image targets. Defaults to host.",
    )
    parser.add_argument(
        "--no-tty",
        action="store_true",
        help="Do not allocate a TTY for docker run/exec.",
    )


def docker_target_from_args(args) -> DockerTarget:  # noqa: ANN001
    return DockerTarget(
        image=args.image,
        container=args.container,
        name=args.name,
        benchmark_setup=args.benchmark_setup,
        benchmark_suite_dir=args.benchmark_suite_dir,
        benchmark_suite_container_dir=args.benchmark_suite_container_dir,
        benchmark_extras=args.benchmark_extras or ["all"],
        use_host_curator=args.use_host_curator,
        use_host_curator_benchmarking=args.use_host_curator_benchmarking,
        gpus=args.gpus,
        memory=args.container_memory,
        shm_size=args.shm_size,
        network=args.network,
        tty=not args.no_tty,
    )


def run_in_target(target: DockerTarget, command_args: list[str]) -> int:
    """Run curator-benchmark with command_args in the selected Docker target."""
    target.validate()
    if not target.uses_docker:
        msg = "A Docker target requires --image or --container"
        raise ValueError(msg)

    container_args = _containerize_command_args(target, command_args)
    inner_command = _join_shell_commands(
        [
            _setup_command(target),
            "curator-benchmark " + " ".join(shlex.quote(arg) for arg in container_args),
        ]
    )
    return _run_shell_command(target, inner_command, command_args=command_args)


def run_start(target: DockerTarget, command_args: list[str]) -> int:
    """Start a named, detached benchmark container for later docker exec runs."""
    target.validate()
    if target.container is not None:
        msg = "`curator-benchmark start` starts a new container; use --image, not --container"
        raise ValueError(msg)
    if target.name is None:
        msg = "`curator-benchmark start` requires --name"
        raise ValueError(msg)

    if target.image is None:
        target.image = DEFAULT_IMAGE

    cmd = _docker_run_base_args(target, command_args, remove=False, detach=True)
    cmd.extend(["--entrypoint", "bash", target.image, "-lc", "sleep infinity"])
    status = subprocess.call(cmd)  # noqa: S603
    if status != 0:
        return status

    setup_target = DockerTarget(
        container=target.name,
        benchmark_setup=target.benchmark_setup,
        benchmark_suite_container_dir=target.benchmark_suite_container_dir,
        benchmark_extras=target.benchmark_extras,
        tty=False,
    )
    setup_command = _setup_command(setup_target)
    if setup_command:
        status = _run_shell_command(setup_target, setup_command, command_args=[])
        if status != 0:
            return status

    run_args = _containerize_command_args(target, ["run", *command_args])
    followup_args = [
        "curator-benchmark",
        run_args[0],
        "--container",
        target.name,
        "--benchmark-setup",
        "never",
        *run_args[1:],
    ]
    shell_args = [
        "curator-benchmark",
        "shell",
        "--container",
        target.name,
        "--benchmark-setup",
        "never",
    ]
    print("Started benchmark container.")
    print("Run benchmarks with:")
    print("  " + " ".join(shlex.quote(arg) for arg in followup_args))
    print("Open a shell with:")
    print("  " + " ".join(shlex.quote(arg) for arg in shell_args))
    print("Stop and remove it with:")
    print("  " + " ".join(shlex.quote(arg) for arg in ["docker", "rm", "--force", target.name]))
    return 0


def run_shell(target: DockerTarget, shell_args: list[str]) -> int:
    """Open or run a shell in the selected Docker target."""
    target.validate()
    if not target.uses_docker:
        command = (
            [os.environ.get("SHELL", "bash")]
            if not shell_args
            else ["bash", "-lc", _shell_args_to_command(shell_args)]
        )
        return subprocess.call(command)  # noqa: S603

    setup_command = _setup_command(target)
    if shell_args:
        requested = _shell_args_to_command(shell_args)
        command = _join_shell_commands([setup_command, requested])
        return _run_shell_command(target, command, command_args=[])

    if setup_command:
        return _run_shell_command(target, _join_shell_commands([setup_command, "exec bash"]), command_args=[])
    return _open_interactive_shell(target)


def run_setup(target: DockerTarget) -> int:
    """Install or verify the benchmark package in the selected Docker target."""
    target.validate()
    if not target.uses_docker:
        msg = "Docker setup requires --image or --container"
        raise ValueError(msg)
    command = _setup_command(target, force=True)
    if not command:
        return 0
    return _run_shell_command(target, command, command_args=[])


def _run_shell_command(target: DockerTarget, shell_command: str, command_args: list[str]) -> int:
    if target.container:
        cmd = ["docker", "exec"]
        cmd.extend(_tty_args(target))
        cmd.extend(_docker_env_args(target, image_digest="<existing-container>"))
        cmd.extend([target.container, "bash", "-lc", shell_command])
    else:
        cmd = _docker_run_base_args(target, command_args)
        cmd.extend(["--entrypoint", "bash", target.image or DEFAULT_IMAGE, "-lc", shell_command])

    return subprocess.call(cmd)  # noqa: S603


def _open_interactive_shell(target: DockerTarget) -> int:
    if target.container:
        cmd = ["docker", "exec"]
        cmd.extend(_tty_args(target))
        cmd.extend(_docker_env_args(target, image_digest="<existing-container>"))
        cmd.extend([target.container, "bash"])
    else:
        cmd = _docker_run_base_args(target, [])
        cmd.extend(["--entrypoint", "bash", target.image or DEFAULT_IMAGE])
    return subprocess.call(cmd)  # noqa: S603


def _docker_run_base_args(
    target: DockerTarget,
    command_args: list[str],
    *,
    remove: bool = True,
    detach: bool = False,
) -> list[str]:
    suite_dir = resolve_benchmark_suite_dir(target.benchmark_suite_dir)
    cmd = ["docker", "run"]
    if remove:
        cmd.append("--rm")
    if detach:
        cmd.append("--detach")
    if target.name:
        cmd.extend(["--name", target.name])
    cmd.extend(["--net", target.network])
    if not detach:
        cmd.extend(_tty_args(target))
    if target.gpus != "none":
        cmd.extend(["--gpus", target.gpus])
    cmd.extend(["--memory", target.memory or str(default_container_memory_bytes())])
    cmd.extend(["--shm-size", target.shm_size or str(default_shm_size_bytes())])
    cmd.extend(["--volume", f"{suite_dir}:{target.benchmark_suite_container_dir}"])
    if target.use_host_curator:
        cmd.extend(["--volume", f"{suite_dir}:{CONTAINER_CURATOR_DIR}"])
    elif target.use_host_curator_benchmarking:
        cmd.extend(
            [
                "--volume",
                f"{suite_dir / 'benchmarking'}:{Path(CONTAINER_CURATOR_DIR) / 'benchmarking'}",
            ]
        )

    config_paths = _host_config_paths_from_args(command_args)
    for host_path, container_path in _volume_mount_pairs_from_configs(config_paths):
        cmd.extend(["--volume", f"{host_path}:{container_path}"])

    for host_path, container_path in _config_file_mount_pairs(config_paths):
        cmd.extend(["--volume", f"{host_path}:{container_path}"])

    cmd.extend(_docker_env_args(target, image_digest=_image_digest(target.image or DEFAULT_IMAGE)))
    return cmd


def _tty_args(target: DockerTarget) -> list[str]:
    if target.tty and sys.stdin.isatty() and sys.stdout.isatty():
        return ["--interactive", "--tty"]
    return []


def _docker_env_args(target: DockerTarget, image_digest: str) -> list[str]:  # noqa: ARG001
    env_values = {
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video",
        "IMAGE_DIGEST": image_digest,
        "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
        "SLACK_BOT_TOKEN": os.environ.get("SLACK_BOT_TOKEN", ""),
        "SLACK_CHANNEL_ID": os.environ.get("SLACK_CHANNEL_ID", ""),
        "GDRIVE_FOLDER_ID": os.environ.get("GDRIVE_FOLDER_ID", ""),
        "GDRIVE_SERVICE_ACCOUNT_FILE": os.environ.get("GDRIVE_SERVICE_ACCOUNT_FILE", ""),
        "CURATOR_BENCHMARKING_DEBUG": os.environ.get("CURATOR_BENCHMARKING_DEBUG", "0"),
        CURATOR_BENCHMARK_PATH_MODE_ENV: "container",
        "CURATOR_REPO_DIR": CONTAINER_CURATOR_DIR,
        "HOST_HOSTNAME": socket.gethostname(),
        "NVIDIA_API_KEY": os.environ.get("NVIDIA_API_KEY", ""),
    }
    args = []
    for key, value in env_values.items():
        args.extend(["--env", f"{key}={value}"])
    return args


_EXTRA_IMPORT_PROBES = {
    "audio": ["pyloudnorm", "wget"],
    "nemotron_parse": ["albumentations", "cv2"],
    "sinks": ["mlflow", "oauth2client", "pydrive2", "slack_sdk"],
}


def _requested_import_probes(extras: list[str]) -> list[str] | None:
    probes = {"curator_benchmarking", "runner"}
    requested = set(extras)
    if "all" in requested:
        requested.update(_EXTRA_IMPORT_PROBES)
    unknown = requested - set(_EXTRA_IMPORT_PROBES) - {"all"}
    if unknown:
        return None
    for extra in requested:
        probes.update(_EXTRA_IMPORT_PROBES.get(extra, []))
    return sorted(probes)


def _setup_command(target: DockerTarget, force: bool = False) -> str:
    if target.benchmark_setup == "never" and not force:
        return ""

    package_path = target.benchmark_suite_container_dir / "benchmarking"
    extras = ",".join(target.benchmark_extras)
    package_spec = f"{package_path}[{extras}]" if extras else str(package_path)
    install_command = (
        "if command -v uv >/dev/null 2>&1; then "
        f"uv pip install {shlex.quote(package_spec)}; "
        "else "
        f"python -m pip install {shlex.quote(package_spec)}; "
        "fi"
    )
    if target.benchmark_setup == "always" or force:
        return install_command
    probes = _requested_import_probes(target.benchmark_extras)
    if probes is None:
        return install_command
    probe_script = (
        "import importlib.util, sys; "
        f"modules = {probes!r}; "
        "missing = [module for module in modules if importlib.util.find_spec(module) is None]; "
        "sys.exit(1 if missing else 0)"
    )
    return (
        f"python -c {shlex.quote(probe_script)} >/dev/null 2>&1 "
        f"|| (echo Installing missing benchmark package && {install_command})"
    )


def _join_shell_commands(commands: list[str]) -> str:
    return " && ".join(command for command in commands if command)


def _shell_args_to_command(shell_args: list[str]) -> str:
    if len(shell_args) == 1:
        return shell_args[0]
    return " ".join(shlex.quote(arg) for arg in shell_args)


def _containerize_command_args(target: DockerTarget, args: list[str]) -> list[str]:
    container_args = []
    config_paths = _host_config_paths_from_args(args)
    config_path_map = {} if target.container else dict(_config_file_mount_pairs(config_paths))

    suite_dir = _suite_dir_for_config_rewrite(target)
    for index, arg in enumerate(args):
        if arg == "--config" and index + 1 < len(args):
            original = Path(args[index + 1]).expanduser().resolve()
            container_args.append(arg)
            container_args.append(
                str(
                    _container_path_for_host_path(
                        original,
                        config_path_map,
                        suite_dir,
                        target,
                    )
                )
            )
        elif arg.startswith("--config="):
            original = Path(arg.split("=", 1)[1]).expanduser().resolve()
            container_args.append(
                "--config="
                + str(
                    _container_path_for_host_path(
                        original,
                        config_path_map,
                        suite_dir,
                        target,
                    )
                )
            )
        elif index > 0 and args[index - 1] == "--config":
            continue
        else:
            container_args.append(arg)
    return container_args


def _suite_dir_for_config_rewrite(target: DockerTarget) -> Path | None:
    if target.container and target.benchmark_suite_dir is None:
        try:
            return resolve_benchmark_suite_dir()
        except ValueError:
            return None
    return resolve_benchmark_suite_dir(target.benchmark_suite_dir)


def _container_path_for_host_path(
    host_path: Path,
    config_path_map: dict[Path, Path],
    suite_dir: Path | None,
    target: DockerTarget,
) -> Path:
    if target.container:
        if suite_dir is None:
            return host_path
        try:
            return target.benchmark_suite_container_dir / host_path.relative_to(suite_dir)
        except ValueError:
            return host_path
    return config_path_map.get(host_path, _mount_path_for_host_path(host_path))


def _host_config_paths_from_args(args: list[str]) -> list[Path]:
    config_paths: list[Path] = []
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--config" and index + 1 < len(args):
            config_paths.append(Path(args[index + 1]).expanduser().resolve())
            skip_next = True
        elif arg.startswith("--config="):
            config_paths.append(Path(arg.split("=", 1)[1]).expanduser().resolve())
    return config_paths


def _volume_mount_pairs_from_configs(config_paths: list[Path]) -> list[tuple[Path, Path]]:
    if not config_paths:
        return []
    config_dict = resolve_env_vars(merge_config_files(config_paths))
    assert_valid_config_dict(config_dict)
    path_resolver = PathResolver(config_dict)
    pairs = []
    for host_path, container_path in path_resolver.volume_mount_pairs():
        if not host_path.is_absolute():
            msg = f"Configured host path must be absolute: {host_path}"
            raise ValueError(msg)
        pairs.append((host_path, container_path))
    return pairs


def _config_file_mount_pairs(config_paths: list[Path]) -> list[tuple[Path, Path]]:
    pairs = []
    for config_path in config_paths:
        if not config_path.exists():
            msg = f"Config file does not exist: {config_path}"
            raise ValueError(msg)
        pairs.append((config_path, _mount_path_for_host_path(config_path)))
    return pairs


def _mount_path_for_host_path(host_path: Path) -> Path:
    return Path(f"{DEFAULT_CONTAINER_PATH_PREFIX}/{host_path}")


def _image_digest(image: str) -> str:
    for image_format in ("{{.Digest}}", "{{.ID}}"):
        try:
            completed = subprocess.run(  # noqa: S603
                ["docker", "image", "inspect", image, "--format", image_format],  # noqa: S607
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        value = completed.stdout.strip()
        if completed.returncode == 0 and value and value != "<none>":
            return value
    return "<unknown>"
