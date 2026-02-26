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

from __future__ import annotations

import shlex
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
from collections import deque
from pathlib import Path
from typing import Any

from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# Create a translation table that maps all control characters to None for deletion in order to safely print subprocess output to the scrolling live window.
# This includes characters in the Unicode category 'Cc' (Control).
_control_chars = {c: None for c in range(sys.maxunicode) if unicodedata.category(chr(c)) == "Cc"}


def run_command_with_timeout(  # noqa: PLR0913
    command: str,
    timeout: int,
    stdouterr_path: Path = Path("stdouterr.log"),
    env: dict[str, str] | None = None,
    run_id: str | None = None,
    fancy: bool = True,
    collapse_on_success: bool = True,
) -> dict[str, Any]:
    """Run a shell command with an optional timeout, streaming output to a log file.

    If running in an interactive terminal, displays subprocess output in a live, scrolling window.
    Otherwise, prints output to the console and saves it to a log file.

    Args:
        command: The shell command to run (as a string or list of arguments).
        timeout: Timeout (in seconds) to terminate the command.
        stdouterr_path: Path to the file for writing combined stdout and stderr.
        env: Optional dictionary of environment variables.
        run_id: Optional run ID to identify the run.
        fancy: If True, displays subprocess output in a live, scrolling window.
        collapse_on_success: If True and command succeeds, collapses live window output (only for fancy=True and interactive mode).

    Returns:
        dict: Contains 'returncode' and 'timed_out' fields.
    """
    cmd_list = command if isinstance(command, list) else shlex.split(command)

    if sys.stdout.isatty() and fancy:
        return display_scrolling_subprocess(
            cmd_list,
            timeout=timeout,
            stdouterr_path=stdouterr_path,
            env=env,
            window_height=6,
            collapse_on_success=collapse_on_success,
            run_id=run_id,
        )
    else:
        return display_simple_subprocess(
            cmd_list, timeout=timeout, stdouterr_path=stdouterr_path, env=env, run_id=run_id
        )


def display_simple_subprocess(
    cmd_list: list[str],
    timeout: int,
    stdouterr_path: Path = Path("stdouterr.log"),
    env: dict[str, str] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run a shell command with an optional timeout, streaming both stdout and stderr to a log file.

    This function runs the given command using subprocess, writes all combined output (stdout and stderr)
    to the provided log file, and streams it live to sys.stdout. Output is processed line by line in a dedicated
    thread to ensure real-time updates. If the command does not complete within the specified timeout, it is terminated
    and, if necessary, force killed. In case of timeout, a message is written to both the log file and stdout.

    Args:
        cmd_list: List of command arguments to execute.
        timeout: Maximum allowed time in seconds before the process is terminated.
        stdouterr_path: Destination file to save all subprocess output.
        env: Optional dictionary of environment variables to use.
        run_id: Optional run ID to identify the run.

    Returns:
        dict: Contains 'returncode' (process exit code or 124 if timed out) and 'timed_out' (True if killed on timeout).
    """
    return_code = 0
    timed_out = False
    msg = ""
    run_id_msg = f" for run ID: {run_id}" if run_id else ""

    with open(stdouterr_path, "w") as outfile:
        start_time = time.time()
        try:
            process = subprocess.Popen(  # noqa: S603
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            def reader() -> None:
                """Reads process output line by line and updates both the file and stdout."""
                for line in process.stdout:
                    outfile.write(line)
                    outfile.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()

            reader_thread = threading.Thread(target=reader)
            reader_thread.start()
            reader_thread.join(timeout=timeout)

            if reader_thread.is_alive():
                # Timeout occurred
                process.terminate()
                try:
                    process.wait(timeout=1)  # Give it a second to terminate gracefully
                except subprocess.TimeoutExpired:
                    process.kill()  # Force kill if it doesn't respond

                reader_thread.join()  # Wait for the reader thread to finish
                msg = f"\n--- Subprocess TIMED OUT after {timeout}s{run_id_msg} ---\n"
                return_code = 124
                timed_out = True

            else:
                # If here, the process completed within the timeout
                return_code = process.wait()
                timed_out = False
                # Determine the final message based on success/failure
                if return_code == 0:
                    msg = (
                        f"\n--- Subprocess completed successfully in {time.time() - start_time:.2f}s{run_id_msg} ---\n"
                    )
                else:
                    msg = f"\n--- Subprocess failed (Exit Code: {return_code}){run_id_msg} ---\n"

        except Exception as e:
            tb = traceback.format_exc()
            msg = f"\n--- An error occurred:\n{e}\n{tb}{run_id_msg} ---\n"

        finally:
            outfile.write(msg)
            outfile.flush()
            sys.stdout.write(msg)
            sys.stdout.flush()

    return {"returncode": return_code, "timed_out": timed_out}


def display_scrolling_subprocess(  # noqa: PLR0913,PLR0915
    cmd_list: list[str],
    timeout: int,
    stdouterr_path: Path = Path("stdouterr.log"),
    env: dict[str, str] | None = None,
    window_height: int = 6,
    run_id: str | None = None,
    collapse_on_success: bool = True,
) -> dict[str, Any]:
    """
    Runs the given shell command in a subprocess, streaming combined stdout and stderr
    both to a log file (stdouterr_path) and to a live scrolling window in the terminal.
    The process output is displayed in a limited-height ("window_height") panel that updates live.

    If the process runs longer than 'timeout' seconds, it is terminated and the function
    returns with a special timeout code.

    Args:
        cmd_list (list[str]): Command and arguments to execute.
        timeout (int): Timeout in seconds.
        stdouterr_path (Path): Log file path to write stdout/stderr.
        env (dict[str, str] | None): Environment variables for the subprocess.
        window_height (int): Number of output lines to display in the live panel.
        run_id (str | None): Optional run ID to identify the run.
        collapse_on_success (bool): If True, collapse panel after successful completion.

    Returns:
        dict: {
            "returncode": int (the exit code, or 124 if timed out),
            "timed_out": bool (True if timeout occurred)
        }
    """
    output_buffer = deque(maxlen=window_height)
    return_code = 0
    timed_out = False
    msg = ""
    run_id_msg = f" for run ID: {run_id}" if run_id else ""

    with (
        Live(auto_refresh=False, vertical_overflow="visible") as live,
        open(stdouterr_path, "w") as outfile,
    ):
        start_time = time.time()
        final_panel = None
        try:
            process = subprocess.Popen(  # noqa: S603
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            def reader() -> None:
                """Reads process output line by line and updates both the file and live display."""
                last_line_not_blank = True
                for line in process.stdout:
                    outfile.write(line)
                    outfile.flush()
                    # Filter out chars that might break the scrolling live window before adding to the buffer.
                    line = line.translate(_control_chars).strip()  # noqa: PLW2901
                    # Do not allow multiple blank lines, it's a waste of already limited space.
                    if line or last_line_not_blank:
                        output_buffer.append(line)
                        display_text = Text("\n".join(output_buffer), no_wrap=True)
                        panel = Panel(
                            display_text,
                            title=f"[bold blue]Subprocess Output{run_id_msg}, elapsed time: {time.time() - start_time:.2f}s[/]",
                            border_style="green",
                            height=window_height + 2,  # +2 for top/bottom borders
                        )
                        live.update(panel)
                        live.refresh()
                    last_line_not_blank = True if line else False  # noqa: SIM210

            reader_thread = threading.Thread(target=reader)
            reader_thread.start()
            reader_thread.join(timeout=timeout)

            if reader_thread.is_alive():
                # Timeout occurred
                process.terminate()
                try:
                    process.wait(timeout=1)  # Give it a second to terminate gracefully
                except subprocess.TimeoutExpired:
                    process.kill()  # Force kill if it doesn't respond

                reader_thread.join()  # Wait for the reader thread to finish
                msg = f"Subprocess TIMED OUT after {timeout}s{run_id_msg}"
                final_panel = Panel(
                    Text("\n".join(output_buffer), no_wrap=True),
                    title=f"[bold red]{msg}[/]",
                    border_style="red",
                    height=window_height + 2,
                )
                timed_out = True
                return_code = 124

            else:
                # If here, the process completed within the timeout
                return_code = process.wait()
                runtime = time.time() - start_time
                timed_out = False

                # Determine the final state of the panel based on success/failure
                if return_code == 0:
                    msg = f"Subprocess completed successfully in {runtime:.2f}s{run_id_msg}"
                    if collapse_on_success:
                        final_panel = Panel(
                            Text(msg),
                            title=f"[bold blue]{msg}[/]",
                            border_style="green",
                            height=3,  # A smaller height for the collapsed view
                        )
                    else:
                        final_panel = Panel(
                            Text("\n".join(output_buffer), no_wrap=True),
                            title=f"[bold blue]{msg}[/]",
                            border_style="green",
                            height=window_height + 2,
                        )
                else:
                    msg = f"Subprocess failed (Exit Code: {return_code}){run_id_msg}"
                    final_panel = Panel(
                        Text("\n".join(output_buffer), no_wrap=True),
                        title=f"[bold red]{msg}[/]",
                        border_style="red",
                        height=window_height + 2,
                    )

        except Exception as e:
            tb = traceback.format_exc()
            msg = f"An error occurred:\n{e}\n{tb}{run_id_msg}"
            final_panel = Panel(f"[bold red]{msg}[/]", title="[bold red]Error[/]")

        finally:
            live.update(final_panel)
            live.refresh()
            outfile.write(f"\n--- {msg} ---\n")
            outfile.flush()

    return {"returncode": return_code, "timed_out": timed_out}
