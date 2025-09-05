from __future__ import annotations

import shlex
import subprocess
from typing import Any

from loguru import logger


def run_command_with_timeout(
    command: str,
    timeout_seconds: int,
    stdout_path: str,
    stderr_path: str,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a shell command with timeout, streaming to log files.

    Returns a dict with returncode and timed_out flag.
    """
    cmd_list = command if isinstance(command, list) else shlex.split(command)
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        proc = subprocess.Popen(cmd_list, stdout=out, stderr=err, env=env)  # noqa: S603
        try:
            proc.wait(timeout=timeout_seconds)
        except Exception:  # noqa: BLE001
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception as e:  # noqa: BLE001
                    logger.exception(f"Failed to kill process: {e}")
            return {"returncode": 124, "timed_out": True}
        else:
            return {"returncode": proc.returncode, "timed_out": False}
