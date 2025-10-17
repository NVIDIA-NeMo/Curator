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
    timeout: int | None = None,
    stdouterr_path: Path = Path("stdouterr.log"),
    env: dict[str, str] | None = None,
    collapse_on_success: bool = True,
) -> dict[str, Any]:
    """Run a shell command with timeout, streaming to log files.

    Returns a dict with returncode and timed_out flag.
    """
    cmd_list = command if isinstance(command, list) else shlex.split(command)

    if sys.stdout.isatty():
        return display_scrolling_subprocess(cmd_list, timeout=timeout, stdouterr_path=stdouterr_path, window_height=6, collapse_on_success=collapse_on_success)
    else:
        return display_simple_subprocess(cmd_list, timeout=timeout, stdouterr_path=stdouterr_path)


def display_simple_subprocess(
    cmd_list: list[str],
    timeout: int | None = None,
    stdouterr_path: Path = Path("stdouterr.log"),
    env: dict[str, str] | None = None,
    collapse_on_success: bool = False,
) -> dict[str, Any]:
    """Run a shell command with timeout, streaming to log files.

    Returns a dict with returncode and timed_out flag.
    If sys.stdout is a TTY, it will display the output in a scrolling live window.
    Otherwise, it will print the output to the log file.
    """
    start_time = time.time()

    with open(stdouterr_path, "w") as outfile:
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

            # Helper function to read output in a separate thread
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
                outfile.write(msg)
                outfile.flush()
                sys.stdout.write(msg)
                sys.stdout.flush()
                return {"returncode": 124, "timed_out": True}

            # If we get here, the process finished within the timeout
            return_code = process.wait()
            runtime = time.time() - start_time
            final_panel = None

            # Determine the final state of the panel based on success/failure
            if return_code == 0:
                msg = f"\n--- Subprocess completed successfully in {runtime:.2f}s ---\n"
                outfile.write(msg)
                outfile.flush()
                sys.stdout.write(msg)
                sys.stdout.flush()
            else:
                # If the process failed, show the final output with a red border
                msg = f"\n--- Subprocess failed (Exit Code: {return_code}) ---\n"
                outfile.write(msg)
                outfile.flush()
                sys.stdout.write(msg)
                sys.stdout.flush()

        except Exception as e:
            tb = traceback.format_exc()
            msg = f"\n--- An error occurred:\n{e}\n{tb} ---\n"
            outfile.write(msg)
            outfile.flush()
            sys.stdout.write(msg)
            sys.stdout.flush()

    return {"returncode": return_code, "timed_out": False}


def display_scrolling_subprocess(
    cmd_list: list[str],
    timeout: int | None = None,
    stdouterr_path: Path = Path("stdouterr.log"),
    env: dict[str, str] | None = None,
    window_height: int = 6,
    collapse_on_success: bool = True,
) -> dict[str, Any]:
    """Run a shell command with timeout, streaming to log files.

    Returns a dict with returncode and timed_out flag.
    """
    output_buffer = deque(maxlen=window_height)
    start_time = time.time()
    with Live(auto_refresh=False, vertical_overflow="visible") as live, open(stdouterr_path, "w") as outfile:
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

            # Helper function to read output in a separate thread
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
                live.update(final_panel)
                live.refresh()
                outfile.write(f"\n--- {msg} ---\n")
                outfile.flush()
                return {"returncode": 124, "timed_out": True}

            # If we get here, the process finished within the timeout
            return_code = process.wait()
            runtime = time.time() - start_time

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
                    # If not collapsing, show the final state in the expanded window
                    final_panel = Panel(
                        Text("\n".join(output_buffer), no_wrap=True),
                        title=f"[bold blue]{msg}[/]",
                        border_style="green",
                        height=window_height + 2,
                    )
            else:
                # If the process failed, show the final output with a red border
                msg = f"Subprocess failed (Exit Code: {return_code})"
                final_panel = Panel(
                    Text("\n".join(output_buffer), no_wrap=True),
                    title=f"[bold red]{msg}[/]",
                    border_style="red",
                    height=window_height + 2,
                )

            # Update the live display with the final panel
            live.update(final_panel)
            live.refresh()
            outfile.write(f"\n--- {msg} ---\n")
            outfile.flush()

        except Exception as e:
            tb = traceback.format_exc()
            msg = f"An error occurred:\n{e}\n{tb}"
            live.update(Panel(f"[bold red]{msg}[/]", title="[bold red]Error[/]"))
            live.refresh()
            outfile.write(f"\n--- {msg} ---\n")
            outfile.flush()

    return {"returncode": return_code, "timed_out": False}
