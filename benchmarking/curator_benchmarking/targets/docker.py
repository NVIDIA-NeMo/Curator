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
from dataclasses import dataclass
from pathlib import Path

from curator_benchmarking.config import load_benchmark_config
from curator_benchmarking.dependencies import (
    dependency_groups_from_config,
    python_extras_for_dependency_groups,
    validate_dependency_groups,
)
from curator_benchmarking.paths import (
    BENCHMARK_SUITE_DIR_ENV,
    resolve_benchmark_suite_dir,
    volume_mount_pairs_from_configs,
)
from curator_benchmarking.system_tools import system_dependency_install_command
from runner.path_resolver import (
    CONTAINER_CURATOR_DIR,
    CURATOR_BENCHMARK_PATH_MODE_ENV,
    DEFAULT_CONTAINER_PATH_PREFIX,
)
from runner.utils import get_total_memory_bytes

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
DEFAULT_BENCHMARK_CONFIG_ENV = "CURATOR_BENCHMARK_CONFIG"


@dataclass
class DockerTarget:
    image: str | None = None
    container: str | None = None
    name: str | None = None
    setup_benchmark_env: str = "auto"
    benchmark_suite_dir: Path | None = None
    benchmark_suite_container_dir: Path = DEFAULT_SUITE_CONTAINER_DIR
    use_host_curator: bool = False
    use_host_curator_benchmarking: bool = False
    gpus: str | None = None
    memory: str | None = None
    shm_size: str | None = None
    network: str | None = None
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
        invalid_exec_options = [
            option_name
            for option_name, value in [
                ("--gpus", self.gpus),
                ("--container-memory", self.memory),
                ("--shm-size", self.shm_size),
                ("--network", self.network),
            ]
            if value is not None
        ]
        if self.container is not None and invalid_exec_options:
            options = ", ".join(invalid_exec_options)
            msg = (
                f"{options} cannot be combined with --container because they "
                "configure docker run and cannot be changed with docker exec"
            )
            raise ValueError(msg)
        if self.setup_benchmark_env not in {"auto", "yes", "no"}:
            msg = "--setup-benchmark-env must be one of: auto, yes, no"
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
        "--setup-benchmark-env",
        choices=["auto", "yes", "no"],
        dest="setup_benchmark_env",
        default="auto",
        help="Install/check benchmark dependencies in Docker targets before running.",
    )
    parser.add_argument(
        "--benchmark-suite-dir",
        type=Path,
        default=None,
        help="Benchmarking package directory. A Curator checkout root is also accepted.",
    )
    parser.add_argument(
        "--benchmark-suite-container-dir",
        type=Path,
        default=DEFAULT_SUITE_CONTAINER_DIR,
        help="Container path where the benchmark suite package is mounted or already available.",
    )
    parser.add_argument(
        "--use-host-curator",
        action="store_true",
        help="Mount the host Curator checkout as the Curator package under test.",
    )
    parser.add_argument(
        "--use-host-curator-benchmarking",
        action="store_true",
        help="Compatibility alias for mounting the host benchmark suite into /opt/Curator/benchmarking.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Value passed to docker run --gpus. Only valid with --image/start. Use 'none' to disable GPU access.",
    )
    parser.add_argument(
        "--container-memory",
        default=None,
        help="Value passed to docker run --memory. Only valid with --image/start. Defaults to host memory capped at 2 TiB.",
    )
    parser.add_argument(
        "--shm-size",
        default=None,
        help="Value passed to docker run --shm-size. Only valid with --image/start. Defaults to 50%% of container memory.",
    )
    parser.add_argument(
        "--network",
        default=None,
        help="Docker network mode for --image/start targets. Defaults to host.",
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
        setup_benchmark_env=args.setup_benchmark_env,
        benchmark_suite_dir=args.benchmark_suite_dir,
        benchmark_suite_container_dir=args.benchmark_suite_container_dir,
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
            _setup_command(target, command_args=command_args),
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
        setup_benchmark_env=target.setup_benchmark_env,
        benchmark_suite_dir=target.benchmark_suite_dir,
        benchmark_suite_container_dir=target.benchmark_suite_container_dir,
        tty=False,
    )
    setup_command = _setup_command(setup_target, command_args=["run", *command_args])
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
        "--setup-benchmark-env",
        "no",
        *run_args[1:],
    ]
    shell_args = [
        "curator-benchmark",
        "shell",
        "--container",
        target.name,
        "--setup-benchmark-env",
        "no",
    ]
    print("Started benchmark container.")
    _print_shell_context_summary(target, command_args)
    print("Run benchmarks with:")
    print("  " + " ".join(shlex.quote(arg) for arg in followup_args))
    print("Open a shell with:")
    print("  " + " ".join(shlex.quote(arg) for arg in shell_args))
    print("Stop and remove it with:")
    print("  " + " ".join(shlex.quote(arg) for arg in ["docker", "rm", "--force", target.name]))
    return 0


def run_shell(target: DockerTarget, shell_args: list[str]) -> int:
    """Open or run a shell in the selected Docker target."""
    context_args, requested_command_args = _split_shell_args(shell_args)
    target.validate()
    if not target.uses_docker:
        if context_args:
            msg = "shell context arguments such as --config require --image"
            raise ValueError(msg)
        command = (
            [os.environ.get("SHELL", "bash")]
            if not requested_command_args
            else ["bash", "-lc", _shell_args_to_command(requested_command_args)]
        )
        return subprocess.call(command)  # noqa: S603

    if target.container is not None and context_args:
        msg = (
            "shell context arguments such as --config cannot be used with "
            "--container because docker exec cannot add mounts to an existing container"
        )
        raise ValueError(msg)

    setup_command = _setup_command(target, command_args=["shell", *context_args])
    if requested_command_args:
        requested = _shell_args_to_command(requested_command_args)
        command = _join_shell_commands(
            [
                _cd_to_benchmark_suite_command(target),
                setup_command,
                requested,
            ]
        )
        return _run_shell_command(target, command, command_args=context_args)

    if setup_command:
        return _run_shell_command(
            target,
            _interactive_shell_command(target, context_args, setup_command=setup_command),
            command_args=context_args,
        )
    return _open_interactive_shell(target, command_args=context_args)


def run_setup(target: DockerTarget, setup_args: list[str] | None = None) -> int:
    """Install or verify the benchmark package in the selected Docker target."""
    target.validate()
    if not target.uses_docker:
        msg = "Docker setup requires --image or --container"
        raise ValueError(msg)
    command = _setup_command(target, command_args=["setup", *(setup_args or [])], force=True)
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


def _open_interactive_shell(target: DockerTarget, command_args: list[str] | None = None) -> int:
    shell_command = _interactive_shell_command(target, command_args or [])
    if target.container:
        cmd = ["docker", "exec"]
        cmd.extend(_tty_args(target))
        cmd.extend(_docker_env_args(target, image_digest="<existing-container>"))
        cmd.extend([target.container, "bash", "-lc", shell_command])
    else:
        cmd = _docker_run_base_args(target, command_args or [])
        cmd.extend(["--entrypoint", "bash", target.image or DEFAULT_IMAGE, "-lc", shell_command])
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
    cmd.extend(["--net", target.network or "host"])
    if not detach:
        cmd.extend(_tty_args(target))
    gpus = target.gpus if target.gpus is not None else os.environ.get("GPUS", "all")
    if gpus != "none":
        cmd.extend(["--gpus", gpus])
    cmd.extend(["--memory", target.memory or str(default_container_memory_bytes())])
    cmd.extend(["--shm-size", target.shm_size or str(default_shm_size_bytes())])
    cmd.extend(["--volume", f"{suite_dir}:{target.benchmark_suite_container_dir}"])
    if target.use_host_curator:
        cmd.extend(["--volume", f"{suite_dir.parent}:{CONTAINER_CURATOR_DIR}"])
    elif target.use_host_curator_benchmarking:
        cmd.extend(
            [
                "--volume",
                f"{suite_dir}:{Path(CONTAINER_CURATOR_DIR) / 'benchmarking'}",
            ]
        )

    config_paths = _host_config_paths_from_args(command_args)
    for host_path, container_path in volume_mount_pairs_from_configs(config_paths):
        cmd.extend(["--volume", f"{host_path}:{container_path}"])

    for host_path, container_path in _config_file_mount_pairs(config_paths):
        cmd.extend(["--volume", f"{host_path}:{container_path}"])

    cmd.extend(_docker_env_args(target, image_digest=_image_digest(target.image or DEFAULT_IMAGE)))
    return cmd


def _tty_args(target: DockerTarget) -> list[str]:
    if target.tty and sys.stdin.isatty() and sys.stdout.isatty():
        return ["--interactive", "--tty"]
    return []


def _docker_env_args(target: DockerTarget, image_digest: str) -> list[str]:
    env_values = {
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video",
        "IMAGE_DIGEST": image_digest,
        "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
        "SLACK_BOT_TOKEN": os.environ.get("SLACK_BOT_TOKEN", ""),
        "SLACK_CHANNEL_ID": os.environ.get("SLACK_CHANNEL_ID", ""),
        "GDRIVE_FOLDER_ID": os.environ.get("GDRIVE_FOLDER_ID", ""),
        "GDRIVE_SERVICE_ACCOUNT_FILE": os.environ.get("GDRIVE_SERVICE_ACCOUNT_FILE", ""),
        "CURATOR_BENCHMARKING_DEBUG": os.environ.get("CURATOR_BENCHMARKING_DEBUG", "0"),
        BENCHMARK_SUITE_DIR_ENV: str(target.benchmark_suite_container_dir),
        DEFAULT_BENCHMARK_CONFIG_ENV: str(_default_benchmark_config_path(target)),
        CURATOR_BENCHMARK_PATH_MODE_ENV: "container",
        "CURATOR_REPO_DIR": CONTAINER_CURATOR_DIR,
        "HOST_HOSTNAME": socket.gethostname(),
        "NVIDIA_API_KEY": os.environ.get("NVIDIA_API_KEY", ""),
    }
    args = []
    for key, value in env_values.items():
        args.extend(["--env", f"{key}={value}"])
    return args


def _setup_command(
    target: DockerTarget,
    *,
    command_args: list[str] | None = None,
    force: bool = False,
) -> str:
    if target.setup_benchmark_env == "no" and not force:
        return ""

    dependency_groups = _dependency_groups_from_command_args(command_args or [])
    host_suite_dir = _host_suite_dir_for_setup(target)
    validate_dependency_groups(dependency_groups, suite_dir=host_suite_dir)
    python_extras = python_extras_for_dependency_groups(
        dependency_groups,
        suite_dir=host_suite_dir,
    )

    package_path = target.benchmark_suite_container_dir
    extras = ",".join(python_extras)
    package_spec = f"{package_path}[{extras}]" if extras else str(package_path)
    install_command = (
        "if command -v uv >/dev/null 2>&1; then "
        f"uv --no-config pip install {shlex.quote(package_spec)}; "
        "else "
        f"python -m pip install {shlex.quote(package_spec)}; "
        "fi"
    )
    if target.setup_benchmark_env == "yes" or force:
        return _join_shell_commands(
            [
                install_command,
                _system_dependency_setup_command(target, dependency_groups, command_args or []),
            ]
        )
    return _join_shell_commands(
        [
            install_command,
            _system_dependency_setup_command(target, dependency_groups, command_args or []),
        ]
    )


def _system_dependency_setup_command(
    target: DockerTarget,
    dependency_groups: list[str],
    command_args: list[str],
) -> str:
    if not _should_install_system_dependencies(command_args):
        return ""
    return system_dependency_install_command(
        dependency_groups,
        suite_dir=target.benchmark_suite_container_dir,
        source_suite_dir=_host_suite_dir_for_setup(target),
    )


def _dependency_groups_from_command_args(command_args: list[str]) -> list[str]:
    command_groups = _dependency_groups_from_inner_command(command_args)
    if command_groups is None:
        return ["all"] if command_args[:1] == ["setup"] else []
    return sorted(set(command_groups))


def _dependency_groups_from_inner_command(command_args: list[str]) -> list[str] | None:
    if not command_args:
        return None

    command = command_args[0]
    args = command_args[1:]
    if command in {"check", "list", "shell"} or "--list" in args:
        return []
    if command == "setup":
        return _dependency_groups_from_setup_args(args)
    if command == "run":
        return _dependency_groups_from_run_args(args)
    return None


def _dependency_groups_from_setup_args(args: list[str]) -> list[str] | None:
    config_paths = _host_config_paths_from_args(args)
    if not config_paths:
        return None
    return _dependency_groups_from_configs(config_paths, entry_names=_setup_entry_names_from_args(args))


def _dependency_groups_from_run_args(args: list[str]) -> list[str] | None:
    config_paths = _host_config_paths_from_args(args)
    if not config_paths or any(not path.exists() for path in config_paths):
        return None
    return _dependency_groups_from_configs(config_paths, entry_names=_run_entry_names_from_args(args))


def _dependency_groups_from_configs(config_paths: list[Path], *, entry_names: list[str] | None) -> list[str]:
    if any(not path.exists() for path in config_paths):
        return []
    try:
        config = load_benchmark_config(config_paths, drop_disabled=True)
        return list(dependency_groups_from_config(config, entry_names=entry_names))
    except FileNotFoundError:
        return []


def _should_install_system_dependencies(command_args: list[str]) -> bool:
    if not command_args:
        return False
    command = command_args[0]
    args = command_args[1:]
    return command in {"run", "setup"} and "--list" not in args


def _run_entry_names_from_args(args: list[str]) -> list[str] | None:
    entries_exact = _option_value(args, "--entries-exact")
    if entries_exact is None:
        return None
    return _comma_separated_values([entries_exact])


def _setup_entry_names_from_args(args: list[str]) -> list[str] | None:
    entry_names = _comma_separated_values(_option_values(args, "--entry-name"))
    return entry_names or None


def _host_suite_dir_for_setup(target: DockerTarget) -> Path:
    return resolve_benchmark_suite_dir(target.benchmark_suite_dir)


def _option_value(args: list[str], option_name: str) -> str | None:
    values = _option_values(args, option_name)
    return values[0] if values else None


def _option_values(args: list[str], option_name: str) -> list[str]:
    values = []
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == option_name and index + 1 < len(args):
            values.append(args[index + 1])
            skip_next = True
            continue
        if arg.startswith(f"{option_name}="):
            values.append(arg.split("=", 1)[1])
            continue
        if arg in {"--config", "--entries", "--entries-exact", "--entry-name"}:
            skip_next = True
    return values


def _comma_separated_values(values: list[str] | None) -> list[str]:
    result = []
    for value in values or []:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def _split_shell_args(args: list[str]) -> tuple[list[str], list[str]]:
    """Split Docker shell context args from the command to run inside bash.

    Args before ``--`` are interpreted by ``curator-benchmark`` so image-based
    shells can use configs to create the same mounts as a benchmark run. Args
    after ``--`` are passed to the shell as the command to execute.
    """
    if "--" in args:
        separator_index = args.index("--")
        return args[:separator_index], args[separator_index + 1 :]
    if args and args[0].startswith("-"):
        return args, []
    return [], args


def _interactive_shell_command(
    target: DockerTarget,
    context_args: list[str],
    *,
    setup_command: str = "",
) -> str:
    return _join_shell_commands(
        [
            _cd_to_benchmark_suite_command(target),
            setup_command,
            _shell_banner_command(target, context_args),
            "exec bash",
        ]
    )


def _cd_to_benchmark_suite_command(target: DockerTarget) -> str:
    suite_dir = shlex.quote(str(target.benchmark_suite_container_dir))
    return f"[ ! -d {suite_dir} ] || cd {suite_dir}"


def _shell_banner_command(target: DockerTarget, context_args: list[str]) -> str:
    lines = _shell_banner_lines(target, context_args)
    return "printf '%s\\n' " + " ".join(shlex.quote(line) for line in lines)


def _shell_banner_lines(target: DockerTarget, context_args: list[str]) -> list[str]:
    config_paths = _container_config_paths_for_shell_context(target, context_args)
    lines = [
        "Curator benchmark shell",
        f"  Benchmark suite: {target.benchmark_suite_container_dir}",
        f"  Default config: {_default_benchmark_config_path(target)}",
    ]
    if config_paths:
        lines.append("  Context configs:")
        lines.extend(f"    --config {config_path}" for config_path in config_paths)
        check_config_args = " ".join(f"--config {config_path}" for config_path in config_paths)
    else:
        check_config_args = f"--config ${DEFAULT_BENCHMARK_CONFIG_ENV}"

    if target.container is not None:
        lines.append("  Existing containers must already have required mounts.")

    lines.append(f"  Try: curator-benchmark check {check_config_args}")
    return lines


def _print_shell_context_summary(target: DockerTarget, context_args: list[str]) -> None:
    for line in _shell_banner_lines(target, context_args):
        print(line)


def _default_benchmark_config_path(target: DockerTarget) -> Path:
    return target.benchmark_suite_container_dir / "benchmarks.yaml"


def _container_config_paths_for_shell_context(target: DockerTarget, context_args: list[str]) -> tuple[Path, ...]:
    config_paths = _host_config_paths_from_args(context_args)
    if not config_paths:
        return ()

    suite_dir = resolve_benchmark_suite_dir(target.benchmark_suite_dir)
    container_paths = []
    for config_path in config_paths:
        try:
            container_path = target.benchmark_suite_container_dir / config_path.relative_to(suite_dir)
        except ValueError:
            container_path = _mount_path_for_host_path(config_path)
        container_paths.append(container_path)
    return tuple(container_paths)


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
    if suite_dir is not None:
        try:
            return target.benchmark_suite_container_dir / host_path.relative_to(suite_dir)
        except ValueError:
            pass
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
