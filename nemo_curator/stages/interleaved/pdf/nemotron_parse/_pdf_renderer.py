# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process isolation for the reusable PDF renderer."""

from __future__ import annotations

import atexit
import contextlib
import os
import pickle
import socket
import struct
import subprocess
import sys
import threading
import time

from PIL import Image

from nemo_curator.stages.interleaved.pdf.nemotron_parse.utils import render_pdf_pages

DEFAULT_RASTER_MAX_SIZE = (1664, 2048)
_FRAME = struct.Struct("!Q")
_COUNT = struct.Struct("!q")
_DIMENSIONS = struct.Struct("!II")
_MAX_FRAME_BYTES = 256 * 1024**2
_POLL_SECONDS = 0.1
_RGB_CHANNELS = 3

RenderOptions = tuple[int, int, tuple[int, int], bool, list[int] | None, bool]
RenderResult = tuple[list[Image.Image | None], bool, int | None]


class RenderCancelledError(RuntimeError):
    """The owner cancelled a range before it could be delivered."""


class RenderWorkerUnavailableError(RuntimeError):
    """The worker cannot render this range through its bounded connection."""


def check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        msg = "PDF rendering was cancelled"
        raise RenderCancelledError(msg)


def _transfer(
    sock: socket.socket,
    data: bytes | bytearray,
    *,
    send: bool = False,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    with memoryview(data) as chunk:
        offset = 0
        while offset < len(chunk):
            check_cancel(cancel_event)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                sock.settimeout(min(_POLL_SECONDS, remaining))
            try:
                count = sock.send(chunk[offset:]) if send else sock.recv_into(chunk[offset:])
            except TimeoutError:
                continue
            if count == 0:
                raise EOFError
            offset += count


def _read_frame(
    sock: socket.socket, deadline: float | None = None, cancel_event: threading.Event | None = None
) -> bytearray:
    header = bytearray(_FRAME.size)
    _transfer(sock, header, deadline=deadline, cancel_event=cancel_event)
    (size,) = _FRAME.unpack(header)
    if size > _MAX_FRAME_BYTES:
        msg = "PDF renderer frame exceeds the transport limit"
        raise RenderWorkerUnavailableError(msg)
    body = bytearray(size)
    _transfer(sock, body, deadline=deadline, cancel_event=cancel_event)
    return body


def _read_count(sock: socket.socket, deadline: float, cancel_event: threading.Event | None) -> int:
    header = bytearray(_COUNT.size)
    _transfer(sock, header, deadline=deadline, cancel_event=cancel_event)
    (count,) = _COUNT.unpack(header)
    if count < -1:
        msg = "Invalid PDF renderer count"
        raise RenderWorkerUnavailableError(msg)
    return count


def _read_pages(
    sock: socket.socket,
    deadline: float,
    cancel_event: threading.Event | None,
    max_pages: int,
    max_size: tuple[int, int],
) -> list[Image.Image | None]:
    count = _read_count(sock, deadline, cancel_event)
    if not 0 <= count <= max_pages:
        msg = "PDF renderer page count exceeds the requested range or response limit"
        raise RenderWorkerUnavailableError(msg)
    pages: list[Image.Image | None] = []
    remaining = _MAX_FRAME_BYTES
    header = bytearray(_DIMENSIONS.size)
    for _ in range(count):
        _transfer(sock, header, deadline=deadline, cancel_event=cancel_event)
        width, height = _DIMENSIONS.unpack(header)
        if not width and not height:
            pages.append(None)
            continue
        # PDFium rounds fitted dimensions up to a whole pixel.
        size = width * height * _RGB_CHANNELS
        if not (0 < width <= max_size[0] + 1 and 0 < height <= max_size[1] + 1 and size <= remaining):
            msg = "PDF renderer returned invalid dimensions or exceeded the response limit"
            raise RenderWorkerUnavailableError(msg)
        pixels = bytearray(size)
        _transfer(sock, pixels, deadline=deadline, cancel_event=cancel_event)
        pages.append(Image.frombytes("RGB", (width, height), bytes(pixels)))
        remaining -= size
    return pages


def _send_pages(sock: socket.socket, pages: list[Image.Image | None]) -> None:
    # The response is raw RGB and fixed-width headers. The owner never unpickles
    # data sent by the process parsing the source PDF.
    images = [page for page in pages if page is not None]
    size = sum(page.width * page.height * _RGB_CHANNELS for page in images)
    if size > _MAX_FRAME_BYTES or any(page.mode != "RGB" for page in images):
        # Fall back to a fresh renderer for ranges outside the reuse limit.
        sock.sendall(_COUNT.pack(-1))
        return
    sock.sendall(_COUNT.pack(len(pages)))
    for page in pages:
        sock.sendall(_DIMENSIONS.pack(*(page.size if page is not None else (0, 0))))
        if page is not None:
            sock.sendall(page.tobytes())


def _owner_watch(read_fd: int) -> None:
    with os.fdopen(read_fd, "rb", buffering=0) as owner:
        # Only the parent holds the write end. EOF also detects abrupt actor exit
        # while the renderer is inside a native call, without a Linux PID API.
        owner.read(1)
    os._exit(0)


def _page_count(pdf: bytes) -> int:
    import pypdfium2 as pdfium

    with contextlib.closing(pdfium.PdfDocument(pdf)) as document:
        return len(document)


def _serve(sock_fd: int, owner_fd: int) -> None:
    threading.Thread(target=_owner_watch, args=(owner_fd,), daemon=True).start()
    with socket.socket(fileno=sock_fd) as sock:
        while payload := _read_frame(sock):
            # The socket is inherited directly from our owner.
            pdf, options = pickle.loads(payload)  # noqa: S301
            dpi, max_pages, max_size, keep_failed, indexes, count_pages = options
            count = None
            pages: list[Image.Image | None] = []
            try:
                if count_pages:
                    count = _page_count(pdf)
            except Exception:  # noqa: BLE001
                sock.sendall(_COUNT.pack(-1))
            else:
                sock.sendall(_COUNT.pack(-1 if count is None else count))
                pages = render_pdf_pages(
                    pdf, dpi=dpi, max_pages=max_pages, max_size=max_size, keep_failed=keep_failed, page_indices=indexes
                )
            _send_pages(sock, pages)
            del payload, pdf, pages


class PersistentPdfRenderer:
    """Own one rendering subprocess for a stage and discard interrupted connections."""

    def __init__(self) -> None:
        self._worker: tuple[subprocess.Popen[bytes], socket.socket, int] | None = None
        self._lock = threading.Lock()
        self._failed_reap = False

    @property
    def worker_pid(self) -> int | None:
        return None if self._worker is None else self._worker[0].pid

    def _start(self) -> socket.socket:
        parent, child = socket.socketpair()
        try:
            read_fd, write_fd = os.pipe()
            try:
                process = subprocess.Popen(  # noqa: S603
                    [
                        sys.executable,
                        "-c",
                        "from nemo_curator.stages.interleaved.pdf.nemotron_parse._pdf_renderer import _main; _main()",
                        str(child.fileno()),
                        str(read_fd),
                    ],
                    pass_fds=(child.fileno(), read_fd),
                    stdin=subprocess.DEVNULL,
                )
            except BaseException:
                os.close(write_fd)
                raise
            finally:
                os.close(read_fd)
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        self._worker = process, parent, write_fd
        atexit.register(self.close)
        return parent

    def close(self) -> None:
        with self._lock:
            self._close()

    def _close(self) -> None:
        if self._worker is None:
            return
        process, sock, write_fd = self._worker
        self._worker = None
        atexit.unregister(self.close)
        with contextlib.suppress(OSError):
            sock.close()
        with contextlib.suppress(OSError):
            os.close(write_fd)
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired as error:
                self._failed_reap = True
                msg = "Owned PDF renderer failed to exit after termination"
                raise RuntimeError(msg) from error

    def render(
        self, pdf: bytes, options: RenderOptions, deadline: float, cancel_event: threading.Event | None
    ) -> RenderResult:
        with self._lock:
            return self._render(pdf, options, deadline, cancel_event)

    def _render(
        self, pdf: bytes, options: RenderOptions, deadline: float, cancel_event: threading.Event | None
    ) -> RenderResult:
        if self._failed_reap:
            msg = "PDF renderer is closed because a previous worker could not be reaped"
            raise RuntimeError(msg)
        check_cancel(cancel_event)
        # Do not copy oversized input into a request the worker cannot accept.
        if len(pdf) > _MAX_FRAME_BYTES:
            msg = "PDF input exceeds the persistent renderer transport limit"
            raise RenderWorkerUnavailableError(msg)
        request = pickle.dumps((pdf, options), protocol=pickle.HIGHEST_PROTOCOL)
        if len(request) > _MAX_FRAME_BYTES:
            msg = "PDF render request exceeds the persistent renderer transport limit"
            raise RenderWorkerUnavailableError(msg)

        count = None
        try:
            if self._worker is not None and self._worker[0].poll() is not None:
                self._close()
            sock = self._start() if self._worker is None else self._worker[1]
            _transfer(sock, _FRAME.pack(len(request)), send=True, deadline=deadline, cancel_event=cancel_event)
            _transfer(sock, request, send=True, deadline=deadline, cancel_event=cancel_event)
            reported = _read_count(sock, deadline, cancel_event)
            count = None if reported == -1 else reported
            _, max_pages, max_size, _, indexes, _ = options
            max_pages = max(0, max_pages) if indexes is None else len(indexes)
            pages = _read_pages(sock, deadline, cancel_event, max_pages, max_size)
            check_cancel(cancel_event)
        except TimeoutError:
            self._close()
            return [], True, count
        except (OSError, EOFError) as error:
            self._close()
            msg = "PDF renderer exited before completing its response"
            raise RenderWorkerUnavailableError(msg) from error
        except BaseException:
            self._close()
            raise
        return pages, False, count


def _main() -> None:
    with contextlib.suppress(EOFError, BrokenPipeError, ConnectionResetError):
        _serve(int(sys.argv[1]), int(sys.argv[2]))
