from __future__ import annotations

import unicodedata
import traceback
import subprocess
import time
import shlex
from collections import deque
import sys
import threading
from typing import Any
from pathlib import Path

#from loguru import logger
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# Create a translation table that maps all control characters to None for deletion in order to safely print subprocess output to the scrolling live window.
# This includes characters in the Unicode category 'Cc' (Control).
_control_chars = {c: None for c in range(sys.maxunicode) if unicodedata.category(chr(c)) == 'Cc'}

def run_command_with_timeout(
    command: str,
    timeout: int,
    stdouterr_path: Path = Path("stdouterr.log"),
    env: dict[str, str] | None = None,
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
        collapse_on_success: If True and command succeeds, collapses live window output (only for interactive mode).

    Returns:
        dict: Contains 'returncode' and 'timed_out' fields.
    """
    cmd_list = command if isinstance(command, list) else shlex.split(command)

    if sys.stdout.isatty():
        return display_scrolling_subprocess(cmd_list, timeout=timeout, stdouterr_path=stdouterr_path, window_height=6, collapse_on_success=collapse_on_success)
    else:
        return display_simple_subprocess(cmd_list, timeout=timeout, stdouterr_path=stdouterr_path)


def display_simple_subprocess(
    cmd_list: list[str],
    timeout: int,
    stdouterr_path: Path = Path("stdouterr.log"),
    env: dict[str, str] | None = None,
    collapse_on_success: bool = False,
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
        collapse_on_success: Unused in this function.

    Returns:
        dict: Contains 'returncode' (process exit code or 124 if timed out) and 'timed_out' (True if killed on timeout).
    """
    return_code = 0
    timed_out = False
    msg = ""

    with open(stdouterr_path, "w") as outfile:
        start_time = time.time()
        try:
            process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, bufsize=1, universal_newlines=True)
            def reader():
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
                    process.wait(timeout=1) # Give it a second to terminate gracefully
                except subprocess.TimeoutExpired:
                    process.kill() # Force kill if it doesn't respond

                reader_thread.join() # Wait for the reader thread to finish
                msg = f"\n--- Subprocess TIMED OUT after {timeout}s ---\n"
                return_code = 124
                timed_out = True
            
            else:
                # If here, the process completed within the timeout
                return_code = process.wait()
                timed_out = False
                # Determine the final message based on success/failure
                if return_code == 0:
                    msg = f"\n--- Subprocess completed successfully in {time.time() - start_time:.2f}s ---\n"
                else:
                    msg = f"\n--- Subprocess failed (Exit Code: {return_code}) ---\n"
        
        except Exception as e:
            tb = traceback.format_exc()
            msg = f"\n--- An error occurred:\n{e}\n{tb} ---\n"

        finally:
            outfile.write(msg)
            outfile.flush()
            sys.stdout.write(msg)
            sys.stdout.flush()

    return {"returncode": return_code, "timed_out": timed_out}


def display_scrolling_subprocess(
    cmd_list: list[str],
    timeout: int,
    stdouterr_path: Path = Path("stdouterr.log"),
    env: dict[str, str] | None = None,
    window_height: int = 6,
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
    
    with Live(auto_refresh=False, vertical_overflow="visible") as live, open(stdouterr_path, "w") as outfile:
        start_time = time.time()
        final_panel = None
        try:
            process = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            def reader():
                """Reads process output line by line and updates both the file and live display."""
                last_line_not_blank = True
                for line in process.stdout:
                    outfile.write(line)
                    outfile.flush()
                    # Filter out chars that might break the scrolling live window before adding to the buffer.
                    line = line.translate(_control_chars).strip()
                    # Do not allow multiple blank lines, waste of already limited space.
                    if line or last_line_not_blank:
                        output_buffer.append(line)
                        display_text = Text("\n".join(output_buffer), no_wrap=True)
                        panel = Panel(
                            display_text,
                            title="[bold blue]Subprocess Output[/]",
                            border_style="green",
                            height=window_height + 2,  # +2 for top/bottom borders
                        )
                        live.update(panel)
                        live.refresh()
                    if line:
                        last_line_not_blank = True
                    else:
                        last_line_not_blank = False
                    
            reader_thread = threading.Thread(target=reader)
            reader_thread.start()
            reader_thread.join(timeout=timeout)

            if reader_thread.is_alive():
                # Timeout occurred
                process.terminate()
                try:
                    process.wait(timeout=1) # Give it a second to terminate gracefully
                except subprocess.TimeoutExpired:
                    process.kill() # Force kill if it doesn't respond

                reader_thread.join() # Wait for the reader thread to finish
                msg = f"Subprocess TIMED OUT after {timeout}s"
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
                    msg = f"Subprocess completed successfully in {runtime:.2f}s"
                    if collapse_on_success:
                        final_panel = Panel(
                            Text(msg),
                            title=f"[bold blue]{msg}[/]",
                            border_style="green",
                            height=3, # A smaller height for the collapsed view
                        )
                    else:
                        final_panel = Panel(
                            Text("\n".join(output_buffer), no_wrap=True),
                            title=f"[bold blue]{msg}[/]",
                            border_style="green",
                            height=window_height + 2,
                        )
                else:
                    msg = f"Subprocess failed (Exit Code: {return_code})"
                    final_panel = Panel(
                        Text("\n".join(output_buffer), no_wrap=True),
                        title=f"[bold red]{msg}[/]",
                        border_style="red",
                        height=window_height + 2,
                    )

        except Exception as e:
            tb = traceback.format_exc()
            msg = f"An error occurred:\n{e}\n{tb}"
            final_panel = Panel(f"[bold red]{msg}[/]", title="[bold red]Error[/]")

        finally:
            live.update(final_panel)
            live.refresh()
            outfile.write(f"\n--- {msg} ---\n")
            outfile.flush()

    return {"returncode": return_code, "timed_out": timed_out}
