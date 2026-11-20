# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overlap a page-budgeted PDF parent's rendering and served parsing."""

from __future__ import annotations

import asyncio
import contextlib
import pickle
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse._image_transport import PNG_METADATA_KEY, png_batch_provenance
from nemo_curator.tasks import FileGroupTask, InterleavedBatch

if TYPE_CHECKING:
    from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import NemotronParseHTTPClientStage
    from nemo_curator.stages.interleaved.pdf.nemotron_parse.preprocess import PDFPreprocessStage

_INPUT_BYTES = 32 * 1024**2
_HELD_BYTES = 256 * 1024**2
_RENDER_JOIN_SECONDS = 5
_Row = dict[str, Any]


def _page_key(row: _Row) -> tuple[str, str, int]:
    return row["document_id"], row["pdf_name"], row["page_number"]


class _RangeBuffer:
    """Thread-to-event-loop handoff with credits held through the parser call."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.ready = asyncio.Event()
        self.stopped = asyncio.Event()
        self.cancelled = threading.Event()
        self.condition = threading.Condition()
        self.pending: deque[tuple[_Row, int]] = deque()
        self.input_bytes = 0
        self.finished = False
        self.error: BaseException | None = None

    def submit(self, rows: list[_Row]) -> None:
        for row in rows:
            payload = row.get("binary_content")
            if payload is not None and not isinstance(payload, bytes):
                msg = "Rendered page data must be immutable bytes"
                raise TypeError(msg)
            size = len(payload or b"")
            with self.condition:
                # One page larger than the credit may use the buffer alone.
                self.condition.wait_for(
                    lambda size=size: (
                        self.cancelled.is_set() or not self.input_bytes or self.input_bytes + size <= _INPUT_BYTES
                    )
                )
                if self.cancelled.is_set():
                    msg = "Render handoff cancelled"
                    raise RuntimeError(msg)
                self.pending.append((row, size))
                self.input_bytes += size
            self.loop.call_soon_threadsafe(self.ready.set)

    def finish(self, error: BaseException | None = None) -> None:
        with self.condition:
            self.error = error
            self.finished = True
        # A blocked source read may finish after the cancelled parent exits.
        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(self.ready.set)

    async def receive(self) -> tuple[_Row, int] | None:
        while True:
            with self.condition:
                if self.error is not None:
                    raise self.error
                if self.pending:
                    return self.pending.popleft()
                if self.finished or self.cancelled.is_set():
                    return None
                self.ready.clear()
            await self.ready.wait()

    def release(self, size: int) -> None:
        with self.condition:
            self.input_bytes -= size
            self.condition.notify_all()

    def cancel(self) -> None:
        with self.condition:
            self.cancelled.set()
            self.condition.notify_all()
        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(self.stopped.set)
            self.loop.call_soon_threadsafe(self.ready.set)


class EarlyAdmissionStage(ProcessingStage[FileGroupTask, InterleavedBatch]):
    """Start serving complete render ranges while their siblings render."""

    name = "pdf_render_parse_overlap"

    def __init__(self, renderer: PDFPreprocessStage, inference: NemotronParseHTTPClientStage) -> None:
        self.renderer = renderer
        self.inference = inference

    def inputs(self) -> tuple[list[str], list[str]]:
        return self.renderer.inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self.inference.outputs()

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_actor_stage": True}

    def setup(self, worker_metadata: dict | None = None) -> None:
        self._parent_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._buffer: _RangeBuffer | None = None
        self._closed = False
        self.renderer.setup(worker_metadata)
        try:
            self.inference.setup(worker_metadata)
        except BaseException:
            self.renderer.teardown()
            raise

    def teardown(self) -> None:
        with self._state_lock:
            self._closed = True
            if self._buffer is not None:
                self._buffer.cancel()
        with self._parent_lock:
            try:
                self.inference.teardown()
            finally:
                self.renderer.teardown()

    def process(self, task: FileGroupTask) -> InterleavedBatch:
        with self._parent_lock:
            return asyncio.run(self._process_parent(task))

    async def _process_parent(self, task: FileGroupTask) -> InterleavedBatch:
        chunks = task._metadata.get("pdf_page_chunks", {})
        expected = {
            (chunk["document_id"], name, page) for name, chunk in chunks.items() for page in chunk["page_indices"]
        }
        if not expected:
            msg = "Overlapped rendering requires a page-budgeted PDF parent"
            raise ValueError(msg)
        buffer = _RangeBuffer(asyncio.get_running_loop())
        with self._state_lock:
            if self._closed:
                msg = "The render and parse worker is closed"
                raise RuntimeError(msg)
            self.inference._check_parse_task(task._metadata)
            self._buffer = buffer
        rendered: InterleavedBatch | None = None
        started = time.perf_counter()

        def produce() -> None:
            nonlocal rendered
            try:
                rendered = self.renderer._process(task, range_sink=buffer.submit, cancel_event=buffer.cancelled)
            except BaseException as error:  # noqa: BLE001 - Preserve the producer's error on the owning thread.
                buffer.finish(error)
            else:
                buffer.finish()

        producer = threading.Thread(target=produce, name="curator-pdf-render", daemon=True)
        producer.start()
        try:
            parsed = await self._parse_available(buffer, expected)
        finally:
            buffer.cancel()
            producer.join(_RENDER_JOIN_SECONDS)
            with self._state_lock:
                # The producer still owns the renderer after a stuck source read.
                self._closed |= producer.is_alive()
                self._buffer = None
        if rendered is None or parsed.keys() != expected:
            msg = "Render handoff did not produce all physical pages in the parent"
            raise RuntimeError(msg)
        renderer_order = [_page_key(row) for row in rendered.to_pyarrow().to_pylist()]
        if len(renderer_order) != len(expected) or set(renderer_order) != expected:
            msg = "Rendered parent metadata disagrees with its admitted physical pages"
            raise RuntimeError(msg)
        rows = [parsed[key] for key in renderer_order]
        if self.renderer.reuse_encoded_images:
            rendered._metadata[PNG_METADATA_KEY] = png_batch_provenance(row.get("binary_content") for row in rows)
        result = self.inference._finish_parse_task(rendered, rows, started, includes_rendering=True)
        self._log_metrics(self.inference._consume_custom_metrics())
        self._log_metrics({f"render.{name}": value for name, value in self.renderer._consume_custom_metrics().items()})
        return result

    async def _parse_available(
        self, buffer: _RangeBuffer, expected: set[tuple[str, str, int]]
    ) -> dict[tuple[str, str, int], _Row]:
        seen: set[tuple[str, str, int]] = set()
        parsed: dict[tuple[str, str, int], _Row] = {}
        held_bytes = 0
        reuse = self.renderer.reuse_encoded_images and self.inference.reuse_encoded_images

        async with self.inference._page_parser(settle_on_cancel=True) as parse_page:

            async def consume() -> None:
                nonlocal held_bytes
                while (entry := await buffer.receive()) is not None:
                    row, size = entry
                    try:
                        key = _page_key(row)
                        if key not in expected or key in seen:
                            msg = f"Unexpected or duplicate physical page in render handoff: {key}"
                            raise ValueError(msg)
                        seen.add(key)
                        result = await parse_page(row, reuse)
                        held_bytes += len(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
                        if held_bytes > _HELD_BYTES:
                            msg = "Completed PDF parent exceeds the held-result byte allowance"
                            raise RuntimeError(msg)
                        parsed[key] = result
                    finally:
                        buffer.release(size)

            tasks = [
                asyncio.create_task(consume()) for _ in range(min(len(expected), self.inference.inference_batch_size))
            ]
            workers = asyncio.gather(*tasks)
            stopped = asyncio.create_task(buffer.stopped.wait())
            try:
                completed, _ = await asyncio.wait((workers, stopped), return_when=asyncio.FIRST_COMPLETED)
                if stopped in completed:
                    msg = "Render and parse worker cancelled"
                    raise RuntimeError(msg)
                await workers
            finally:
                for task in (*tasks, stopped):
                    task.cancel()
                await asyncio.gather(workers, *tasks, stopped, return_exceptions=True)
        return parsed
