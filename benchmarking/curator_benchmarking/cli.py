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

import argparse
import sys

from curator_benchmarking.targets.docker import (
    DockerTarget,
    add_target_arguments,
    docker_target_from_args,
    run_in_target,
    run_setup,
    run_shell,
    run_start,
)

_COMMANDS = {"run", "list", "check", "setup", "shell", "start"}
_DOCKER_COMMAND_ARGS = {
    "run": lambda args: ["run", *args],
    "list": lambda args: ["run", "--list", *args],
    "check": lambda args: ["check", *args],
}


def _print_help() -> None:
    print(
        """usage: curator-benchmark <command> [options]

commands:
  run     Run benchmarks in the current environment, image, or container.
  list    List benchmark entries from one or more configs.
  check   Check the benchmark environment.
  setup   Install benchmark package dependencies.
  start   Start a named, detached benchmark container.
  shell   Open or run a shell in the current environment, image, or container.

target options for run/list/check/setup/start/shell:
  --image [IMAGE]              start a new Docker container from IMAGE
  --container NAME             exec into an existing Docker container
  --name NAME                  name for a container created by start
  --run-benchmark-setup MODE   auto, yes, or no
  --benchmark-suite-dir PATH   checkout that provides benchmark code/configs
  --benchmark-extra EXTRA      benchmark package extra to install
  --use-host-curator           mount the host checkout as Curator under test
  --gpus VALUE                 value passed to docker run --gpus
  --container-memory VALUE     value passed to docker run --memory
  --shm-size VALUE             value passed to docker run --shm-size

When no command is provided, arguments are treated as `run` arguments for
compatibility with the previous `benchmarking/run.py` interface.
"""
    )


def _parse_target(args: list[str]) -> tuple[DockerTarget, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    add_target_arguments(parser)
    target_args, remainder = parser.parse_known_args(args)
    return docker_target_from_args(target_args), remainder


def _wants_help(args: list[str]) -> bool:
    return "-h" in args or "--help" in args


def _print_command_help(command: str, args: list[str]) -> int:
    if not _wants_help(args):
        return -1
    if command in {"run", "list"}:
        from curator_benchmarking.commands.run import main as run_main

        return run_main(["--help"])
    if command == "check":
        from curator_benchmarking.commands.check import main as check_main

        return check_main(["--help"])
    if command == "setup":
        from curator_benchmarking.commands.setup import main as setup_main

        return setup_main(["--help"])
    _print_help()
    return 0


def _run_in_docker_target(target: DockerTarget, command: str, args: list[str]) -> int:
    if command == "setup":
        return run_setup(target)
    if command == "start":
        return run_start(target, args)
    if command == "shell":
        return run_shell(target, args)
    command_args_factory = _DOCKER_COMMAND_ARGS.get(command)
    if command_args_factory is None:
        msg = f"unknown command {command}"
        raise ValueError(msg)
    return run_in_target(target, command_args_factory(args))


def _run_in_current_env(command: str, args: list[str]) -> int:
    if command in {"run", "list"}:
        from curator_benchmarking.commands.run import main as run_main

        return run_main(args if command == "run" else ["--list", *args])
    if command == "check":
        from curator_benchmarking.commands.check import main as check_main

        return check_main(args)
    if command == "setup":
        from curator_benchmarking.commands.setup import main as setup_main

        return setup_main(args)
    if command == "shell":
        return run_shell(DockerTarget(), args)
    msg = f"unknown command {command}"
    raise ValueError(msg)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0

    command = args.pop(0) if args[0] in _COMMANDS else "run"

    try:
        target, remainder = _parse_target(args)
        help_status = _print_command_help(command, remainder)
        if help_status >= 0:
            return help_status

        if target.uses_docker or command == "start":
            return _run_in_docker_target(target, command, remainder)
        return _run_in_current_env(command, remainder)
    except ValueError as exc:
        print(f"curator-benchmark: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
