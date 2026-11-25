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
import weakref
from types import SimpleNamespace

import cloudpickle
import pytest
from loguru import logger

from nemo_curator.stages.interleaved.pdf.nemotron_parse import _pdf_renderer
from nemo_curator.stages.interleaved.pdf.nemotron_parse.preprocess import PDFPreprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.utils import image_to_bytes

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Reusable worker passes POSIX file descriptors")


@pytest.fixture
def reusable_pdfium() -> None:
    if not _pdf_renderer._supports_pdfium_reset():
        pytest.skip("Same-process PDFium reset is unavailable")


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


@pytest.mark.usefixtures("reusable_pdfium")
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


@pytest.mark.usefixtures("reusable_pdfium")
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


def test_unqualified_pdfium_uses_fresh_renderer_without_starting_a_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = _pdf([(240, 330)])
    options = {"retain_failures": True, "raster_max_size": (64, 64)}
    fresh = PDFPreprocessStage(**options)
    stage = PDFPreprocessStage(**options, persistent_rendering=True)
    monkeypatch.setattr(_pdf_renderer, "version", lambda _distribution: "unqualified")

    def unexpected_start(_renderer: _pdf_renderer.PersistentPdfRenderer) -> None:
        pytest.fail("An unqualified runtime started a persistent helper")

    monkeypatch.setattr(_pdf_renderer.PersistentPdfRenderer, "_start", unexpected_start)
    try:
        assert _signature(stage._render_once(pdf, [0])) == _signature(fresh._render_once(pdf, [0]))
        assert stage._persistent_renderer.worker_pid is None
    finally:
        stage.teardown()


@pytest.mark.usefixtures("reusable_pdfium")
def test_reset_requires_supported_hooks_and_closed_pdfium_handles(monkeypatch: pytest.MonkeyPatch) -> None:
    from pypdfium2 import PdfDocument, _library_scope

    calls = []
    monkeypatch.setattr(_library_scope, "destroy_lib", lambda: calls.append("destroy"))
    monkeypatch.setattr(_library_scope, "init_lib", None)
    with pytest.raises(_pdf_renderer.RenderWorkerUnavailableError, match="reset hooks are unavailable"):
        _pdf_renderer._reset_pdfium_library()
    assert not calls

    monkeypatch.setattr(_library_scope, "init_lib", lambda: calls.append("init"))
    document = PdfDocument.new()
    try:
        with pytest.raises(_pdf_renderer.RenderWorkerUnavailableError, match="Live PDFium handles"):
            _pdf_renderer._reset_pdfium_library()
        assert not calls
    finally:
        document.close()

    _pdf_renderer._reset_pdfium_library()
    assert calls == ["destroy", "init"]


def test_reset_declines_an_unqualified_release_a_missing_tracker_and_a_live_font_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdfium = pytest.importorskip("pypdfium2")
    from pypdfium2 import _library_scope, internal
    from pypdfium2 import version as wrapper_version
    from pypdfium2._helpers.sysfontinfo import _DefaultSysfontInfo

    calls = []
    monkeypatch.setattr(_library_scope, "destroy_lib", lambda: calls.append("destroy"))
    monkeypatch.setattr(_library_scope, "init_lib", lambda: calls.append("init"))
    monkeypatch.setattr(wrapper_version, "PYPDFIUM_INFO", SimpleNamespace(tag="5.8.0"))
    with pytest.raises(_pdf_renderer.RenderWorkerUnavailableError, match="not qualified"):
        _pdf_renderer._reset_pdfium_library()
    # PyPDFium2 5.15.0 records a font handle under ObjectTracker[None] alone. An older
    # wrapper is held to the same rule so the check runs on every qualified release.
    # A build from source reports its origin in the version and keeps the release tag.
    monkeypatch.setattr(wrapper_version, "PYPDFIUM_INFO", SimpleNamespace(tag="5.15.0", version="5.15.0:given"))
    font = pdfium.PdfSysfontBase()
    try:
        with monkeypatch.context() as untracked:
            untracked.delattr(internal, "ObjectTracker", raising=False)
            with pytest.raises(_pdf_renderer.RenderWorkerUnavailableError, match="object tracker is unavailable"):
                _pdf_renderer._reset_pdfium_library()
        if not hasattr(internal, "ObjectTracker"):
            monkeypatch.setattr(internal, "ObjectTracker", {None: {weakref.ref(font)}}, raising=False)
        with pytest.raises(_pdf_renderer.RenderWorkerUnavailableError, match="Live PDFium handles"):
            _pdf_renderer._reset_pdfium_library()
        assert not calls
        assert _DefaultSysfontInfo._is_loaded
    finally:
        _DefaultSysfontInfo.close()


@pytest.mark.parametrize("worker_release", [None, "5.8.0"])
def test_worker_reports_its_own_renderer_choice_once(
    monkeypatch: pytest.MonkeyPatch, worker_release: str | None
) -> None:
    pdfium = pytest.importorskip("pypdfium2")
    pdf = _pdf([(240, 330)])
    options = {"retain_failures": True, "raster_max_size": (64, 64)}
    expected = _signature(PDFPreprocessStage(**options)._render_once(pdf, [0]))
    built_on_driver = cloudpickle.dumps(PDFPreprocessStage(**options, persistent_rendering=True))
    if worker_release is not None:
        monkeypatch.setattr(_pdf_renderer, "version", lambda _distribution: worker_release)
    reused = _pdf_renderer._supports_pdfium_reset()
    stage = cloudpickle.loads(built_on_driver)
    reports = []
    sink = logger.add(reports.append, filter=lambda record: record["message"].startswith("PDF rendering:"))
    try:
        for _ in range(2):
            assert _signature(stage._render_once(pdf, [0])) == expected
            assert (stage._persistent_renderer.worker_pid is not None) == reused
    finally:
        logger.remove(sink)
        stage.teardown()
    (report,) = reports
    message = report.record["message"]
    assert report.record["level"].name == ("INFO" if reused else "WARNING")
    assert f"requested=persistent effective={'persistent' if reused else 'fresh'} " in message
    assert f"pypdfium2={worker_release or pdfium.PYPDFIUM_INFO.version} pdfium={pdfium.PDFIUM_INFO.tag}" in message
    assert ("reason=" in message) == (not reused)


@pytest.mark.usefixtures("reusable_pdfium")
def test_helper_death_during_a_range_recovers_through_a_fresh_child_and_a_new_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = PDFPreprocessStage(persistent_rendering=True, retain_failures=True, raster_max_size=(64, 64))
    pdf = _pdf([(240, 330)])
    try:
        before = _signature(stage._render_once(pdf, [0]))
        assert before[0][0] is not None
        helper = stage._persistent_renderer._worker[0]
        helper.send_signal(signal.SIGSTOP)
        read_count = _pdf_renderer._read_count

        def die_before_responding(*args: object) -> int:
            helper.kill()
            return read_count(*args)

        with monkeypatch.context() as inflight:
            inflight.setattr(_pdf_renderer, "_read_count", die_before_responding)
            assert _signature(stage._render_once(pdf, [0])) == before
        assert helper.returncode == -signal.SIGKILL
        assert stage._persistent_renderer.worker_pid is None
        assert _signature(stage._render_once(pdf, [0])) == before
        assert stage._persistent_renderer.worker_pid not in (None, helper.pid)
    finally:
        stage.teardown()


def test_cancelled_range_does_not_create_a_worker() -> None:
    stage = PDFPreprocessStage(persistent_rendering=True)
    event = threading.Event()
    event.set()
    with pytest.raises(_pdf_renderer.RenderCancelledError, match="cancelled"):
        stage._render_once(b"unused", [0], cancel_event=event)
    assert stage._persistent_renderer is None


@pytest.mark.usefixtures("reusable_pdfium")
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
