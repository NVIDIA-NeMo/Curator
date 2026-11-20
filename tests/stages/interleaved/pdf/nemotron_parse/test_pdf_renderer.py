# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise renderer reuse, fresh-process parity, and bounded process cleanup."""

from __future__ import annotations

import io
import os
import signal
import socket
import threading
import time

import pytest

from nemo_curator.stages.interleaved.pdf.nemotron_parse import _pdf_renderer
from nemo_curator.stages.interleaved.pdf.nemotron_parse.preprocess import PDFPreprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.utils import image_to_bytes

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Reusable worker passes POSIX file descriptors")


def _pdf(page_sizes: list[tuple[int, int]]) -> bytes:
    pdfium = pytest.importorskip("pypdfium2")
    with io.BytesIO() as stream:
        pdf = pdfium.PdfDocument.new()
        try:
            for width, height in page_sizes:
                pdf.new_page(width, height).close()
            pdf.save(stream)
        finally:
            pdf.close()
        return stream.getvalue()


def _signature(result: _pdf_renderer.RenderResult) -> tuple[list[bytes | None], bool, int | None]:
    pages, timed_out, page_count = result
    return [None if page is None else image_to_bytes(page) for page in pages], timed_out, page_count


def test_reused_worker_matches_fresh_rasters_after_valid_and_adverse_ranges() -> None:
    sizes = (160, 192)
    options = {"retain_failures": True, "raster_max_size": sizes}
    fresh = PDFPreprocessStage(**options)
    persistent = PDFPreprocessStage(**options, persistent_rendering=True)
    wide = _pdf([(800, 200), (400, 800)])
    tall = _pdf([(220, 600), (380, 140), (270, 320)])
    inputs = [(wide, [1, 0]), (b"invalid PDF", None), (tall, [2, 99, 0]), (wide, [1, 0])]
    try:
        pids = set()
        for pdf, indexes in inputs:
            assert _signature(persistent._render_once(pdf, indexes)) == _signature(fresh._render_once(pdf, indexes))
            pids.add(persistent._persistent_renderer.worker_pid)
        (worker_pid,) = pids
        assert worker_pid is not None
        assert worker_pid != os.getpid()
    finally:
        persistent.teardown()
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)


def test_reusable_worker_restarts_after_exit_and_releases_oversized_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = PDFPreprocessStage(persistent_rendering=True, retain_failures=True, raster_max_size=(64, 64))
    pdf = _pdf([(240, 330)])
    try:
        before = _signature(stage._render_once(pdf, [0]))
        assert before[0][0] is not None
        renderer = stage._persistent_renderer
        first_pid = renderer.worker_pid
        os.kill(first_pid, signal.SIGKILL)
        renderer._worker[0].wait(timeout=5)
        assert _signature(stage._render_once(pdf, [0])) == before
        second_pid = renderer.worker_pid
        assert second_pid != first_pid
        monkeypatch.setattr(_pdf_renderer, "_MAX_FRAME_BYTES", 1)
        assert _signature(stage._render_once(pdf, [0])) == before
        assert renderer.worker_pid is None
        with pytest.raises(ProcessLookupError):
            os.kill(second_pid, 0)
    finally:
        stage.teardown()


def test_cancelled_range_does_not_create_a_worker() -> None:
    stage = PDFPreprocessStage(persistent_rendering=True)
    event = threading.Event()
    event.set()
    with pytest.raises(_pdf_renderer.RenderCancelledError, match="cancelled"):
        stage._render_once(b"unused", [0], cancel_event=event)
    assert stage._persistent_renderer is None


def test_cancellation_reaps_an_inflight_worker() -> None:
    stage = PDFPreprocessStage(persistent_rendering=True, raster_max_size=(64, 64))
    pdf = _pdf([(240, 330)])
    event = threading.Event()
    try:
        stage._render_once(pdf, [0])
        worker_pid = stage._persistent_renderer.worker_pid
        os.kill(worker_pid, signal.SIGSTOP)
        trigger = threading.Timer(0.2, event.set)
        trigger.start()
        try:
            with pytest.raises(_pdf_renderer.RenderCancelledError, match="cancelled"):
                stage._render_once(pdf, [0], timeout_s=5, cancel_event=event)
        finally:
            trigger.cancel()
            trigger.join()
        assert stage._persistent_renderer.worker_pid is None
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
    finally:
        stage.teardown()


def test_renderer_rejects_invalid_page_dimensions_before_reading_pixels() -> None:
    receiver, sender = socket.socketpair()
    with receiver, sender:
        sender.sendall(_pdf_renderer._COUNT.pack(1) + _pdf_renderer._DIMENSIONS.pack(0, 1))
        with pytest.raises(_pdf_renderer.RenderWorkerUnavailableError, match="invalid dimensions"):
            _pdf_renderer._read_pages(receiver, time.monotonic() + 1, None, 1, (64, 64))
