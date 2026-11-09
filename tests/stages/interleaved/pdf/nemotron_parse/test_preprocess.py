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

"""Retain rendering failures only for the opted-in served pipeline."""

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cloudpickle
import pyarrow as pa
import pytest
from PIL import Image

from nemo_curator.stages.interleaved.pdf.nemotron_parse import preprocess, utils
from nemo_curator.stages.interleaved.pdf.nemotron_parse._image_transport import (
    PNG_METADATA_KEY,
    has_png_batch_provenance,
    png_batch_provenance,
)
from nemo_curator.tasks import FileGroupTask


def pdf_bytes():
    pdfium = pytest.importorskip("pypdfium2")
    doc = pdfium.PdfDocument.new()
    for _ in range(3):
        doc.new_page(220, 170).close()
    stream = io.BytesIO()
    doc.save(stream)
    doc.close()
    return stream.getvalue()


@pytest.mark.parametrize("retain", [False, True])
def test_unreadable_document_failure_is_opt_in(tmp_path: Path, retain: bool):
    stage = preprocess.PDFPreprocessStage(pdf_dir=str(tmp_path), retain_failures=retain)
    task = FileGroupTask(dataset_name="test", data=[json.dumps({"file_name": "missing.pdf"})])
    result = stage.process(task)
    if not retain:
        assert result is None
    else:
        rows = result.to_pyarrow().to_pylist()
        assert len(rows) == 1
        assert rows[0]["page_number"] is None
        assert json.loads(rows[0]["materialize_error"])["stage"] == "read_pdf"


def test_unopened_document_keeps_integer_page_numbers(tmp_path: Path):
    (tmp_path / "three.pdf").write_bytes(pdf_bytes())
    stage = preprocess.PDFPreprocessStage(pdf_dir=str(tmp_path), retain_failures=True)
    entries = [json.dumps({"file_name": name}) for name in ("missing.pdf", "three.pdf")]
    table = stage.process(FileGroupTask(dataset_name="test", data=entries)).to_pyarrow()
    assert table.schema.field("page_number").type == pa.int64()
    assert table.column("page_number").to_pylist() == [None, 0, 1, 2]


def test_partial_render_preserves_physical_indexes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def render(_document: object, index: int, scale: float, max_size: tuple[int, int]) -> Image.Image | None:
        assert scale == 300 / 72.0
        assert max_size == (1664, 2048)
        return None if index == 1 else Image.new("RGB", (220, 170), "red")

    monkeypatch.setattr(utils, "_render_page", render)
    (tmp_path / "three.pdf").write_bytes(pdf_bytes())
    stage = preprocess.PDFPreprocessStage(pdf_dir=str(tmp_path), retain_failures=True)
    task = FileGroupTask(dataset_name="test", data=[json.dumps({"file_name": "three.pdf"})])
    rows = stage.process(task).to_pyarrow().to_pylist()
    assert [(row["position"], row["page_number"]) for row in rows] == [(0, 0), (1, 1), (2, 2)]
    assert rows[0]["binary_content"]
    assert rows[2]["binary_content"]
    assert rows[1]["binary_content"] is None
    assert json.loads(rows[1]["materialize_error"])["stage"] == "render_pdf"


def test_manifest_selected_pages_only_preserve_physical_indexes(tmp_path: Path):
    (tmp_path / "three.pdf").write_bytes(pdf_bytes())
    stage = preprocess.PDFPreprocessStage(pdf_dir=str(tmp_path), retain_failures=True)
    task = FileGroupTask(
        dataset_name="test",
        data=[json.dumps({"file_name": "three.pdf", "selected_page_indices": [2, 0]})],
    )
    rows = stage.process(task).to_pyarrow().to_pylist()
    assert [(row["position"], row["page_number"]) for row in rows] == [(2, 2), (0, 0)]


@pytest.mark.parametrize("retain", [False, True])
def test_selected_pages_keep_successful_indexes_after_render_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, retain: bool
):
    def render(_document: object, index: int, _scale: float, _max_size: tuple[int, int]) -> Image.Image | None:
        return None if index == 1 else Image.new("RGB", (10, 10), "red")

    monkeypatch.setattr(utils, "_render_page", render)
    (tmp_path / "three.pdf").write_bytes(pdf_bytes())
    stage = preprocess.PDFPreprocessStage(pdf_dir=str(tmp_path), retain_failures=retain)
    task = FileGroupTask(
        dataset_name="test",
        data=[json.dumps({"file_name": "three.pdf", "selected_page_indices": [2, 1, 0]})],
    )
    rows = stage.process(task).to_pyarrow().to_pylist()
    assert [row["position"] for row in rows] == ([2, 1, 0] if retain else [2, 0])
    if retain:
        assert json.loads(rows[1]["materialize_error"])["stage"] == "render_pdf"


def test_timed_out_render_window_is_retried_in_smaller_groups(monkeypatch: pytest.MonkeyPatch):
    stage = preprocess.PDFPreprocessStage(retain_failures=True)
    calls = []

    def render_once(
        _pdf_bytes: bytes, indexes: list[int] | None, **_kwargs
    ) -> tuple[list[Image.Image | None], bool, int | None]:
        calls.append(indexes)
        if indexes is None:
            return [], True, 4
        return [Image.new("RGB", (1, 1), "red") for _ in indexes], False, None

    monkeypatch.setattr(stage, "_render_once", render_once)
    pages = stage._render_with_timeout(b"pdf", "slow.pdf")

    assert calls == [None, [0, 1], [2, 3]]
    assert len(pages) == 4


def test_single_page_timeout_does_not_drop_successful_siblings(monkeypatch: pytest.MonkeyPatch):
    stage = preprocess.PDFPreprocessStage(retain_failures=True)

    def render_once(
        _pdf_bytes: bytes, indexes: list[int] | None, **_kwargs
    ) -> tuple[list[Image.Image | None], bool, int | None]:
        assert indexes is not None
        if len(indexes) > 1:
            return [], True, None
        if indexes == [1]:
            return [], True, None
        return [Image.new("RGB", (1, 1), "red")], False, None

    monkeypatch.setattr(stage, "_render_once", render_once)
    pages = stage._render_with_timeout(b"pdf", "slow.pdf", [0, 1, 2])

    assert [page is not None for page in pages] == [True, False, True]


@pytest.mark.parametrize("indexes", [[], [-1], [1, 1], [True]])
def test_invalid_manifest_selected_pages_are_retained_as_failures(tmp_path: Path, indexes: list[int]):
    (tmp_path / "three.pdf").write_bytes(pdf_bytes())
    stage = preprocess.PDFPreprocessStage(pdf_dir=str(tmp_path), retain_failures=True)
    task = FileGroupTask(
        dataset_name="test",
        data=[json.dumps({"file_name": "three.pdf", "selected_page_indices": indexes})],
    )
    rows = stage.process(task).to_pyarrow().to_pylist()
    assert len(rows) == 1
    assert json.loads(rows[0]["materialize_error"])["stage"] == "manifest"


@pytest.mark.parametrize("indexes", [None, [8, 2, 5, 1]])
def test_crashed_recovery_window_preserves_all_physical_slots(
    monkeypatch: pytest.MonkeyPatch, indexes: list[int] | None
):
    stage = preprocess.PDFPreprocessStage(retain_failures=True)
    requested = list(range(4)) if indexes is None else indexes

    def render_once(_data: bytes, window: list[int] | None, **_kwargs) -> tuple[list, bool, int | None]:
        if window is None or len(window) == 4:
            return [], True, 4
        if all(index in requested[:2] for index in window):
            return [], False, None
        return [Image.new("RGB", (1, 1)) for _ in window], False, None

    monkeypatch.setattr(stage, "_render_once", render_once)
    pages = stage._render_with_timeout(b"pdf", "crashed.pdf", indexes)
    assert len(pages) == 4
    assert [page is None for page in pages] == [True, True, False, False]


@pytest.mark.parametrize("bad_result", [[], [None], [None] * 3, ["not an image"] * 2])
def test_malformed_render_window_never_reassigns_page_identity(monkeypatch: pytest.MonkeyPatch, bad_result: list):
    stage = preprocess.PDFPreprocessStage(retain_failures=True)
    monkeypatch.setattr(stage, "_render_once", lambda *_args, **_kwargs: (bad_result, False, None))
    assert stage._render_with_timeout(b"pdf", "bad.pdf", [5, 2]) == [None, None]


def test_recovery_attempt_budget_is_shared_across_siblings(monkeypatch: pytest.MonkeyPatch):
    stage = preprocess.PDFPreprocessStage(retain_failures=True)
    calls = []

    def render_once(_data: bytes, indexes: list[int] | None, **kwargs) -> tuple[list, bool, int]:
        calls.append((indexes, kwargs))
        return [], True, 20

    monkeypatch.setattr(stage, "_render_once", render_once)
    pages = stage._render_with_timeout(b"pdf", "slow.pdf")
    assert pages == [None] * 20
    assert len(calls) == stage._RENDER_MAX_ATTEMPTS
    assert all(0 < kwargs["timeout_s"] <= 60 for _, kwargs in calls[1:])


def test_recovery_deadline_does_not_launch_more_children(monkeypatch: pytest.MonkeyPatch):
    stage = preprocess.PDFPreprocessStage(retain_failures=True)
    times = iter([0, 121, 122])
    monkeypatch.setattr(preprocess.time, "monotonic", lambda: next(times))
    calls = []
    monkeypatch.setattr(stage, "_render_once", lambda *_a, **_k: (calls.append(True) or [], True, 4))
    assert stage._render_with_timeout(b"pdf", "slow.pdf") == [None] * 4
    assert calls == [True]


def test_default_parser_does_not_enter_render_recovery(monkeypatch: pytest.MonkeyPatch):
    stage = preprocess.PDFPreprocessStage()
    calls = []
    monkeypatch.setattr(stage, "_render_once", lambda *_a, **_k: (calls.append(True) or [], True, 4))
    assert stage._render_with_timeout(b"pdf", "slow.pdf") == []
    assert calls == [True]


def test_unknown_page_count_remains_document_error(monkeypatch: pytest.MonkeyPatch):
    stage = preprocess.PDFPreprocessStage(retain_failures=True)
    monkeypatch.setattr(stage, "_render_once", lambda *_a, **_k: ([], True, None))
    assert stage._render_with_timeout(b"pdf", "unopened.pdf") == []


def test_render_child_crash_returns_without_waiting_full_timeout(monkeypatch: pytest.MonkeyPatch):
    import os
    import time

    stage = preprocess.PDFPreprocessStage(retain_failures=True)
    monkeypatch.setattr(preprocess, "render_pdf_pages", lambda *_a, **_k: os._exit(2))
    started = time.monotonic()
    pages, timed_out, count = stage._render_once(pdf_bytes(), [2])
    assert pages == []
    assert not timed_out
    assert count is None
    assert time.monotonic() - started < 5


@pytest.mark.parametrize("failure", ["start", "join", "kill", "surviving_child"])
def test_render_queues_close_even_when_process_cleanup_fails(monkeypatch: pytest.MonkeyPatch, failure: str):
    queues = []

    def create_queue() -> SimpleNamespace:
        result = SimpleNamespace(get=Mock(return_value=[]), close=Mock())
        queues.append(result)
        return result

    process = SimpleNamespace(
        pid=None if failure == "start" else 42,
        start=Mock(side_effect=RuntimeError("start failed") if failure == "start" else None),
        join=Mock(side_effect=RuntimeError("join failed") if failure == "join" else None),
        kill=Mock(side_effect=RuntimeError("kill failed") if failure == "kill" else None),
        is_alive=Mock(return_value=True),
    )
    context = SimpleNamespace(Queue=create_queue, Process=lambda **_kwargs: process)
    monkeypatch.setattr(preprocess.multiprocessing, "get_context", lambda *_args: context)
    with pytest.raises(RuntimeError):
        preprocess.PDFPreprocessStage(retain_failures=True)._render_once(b"pdf", [2])
    assert len(queues) == 2
    for result in queues:
        result.close.assert_called_once()


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true"])
def test_encoded_png_reuse_rejects_nonboolean_options(value: object):
    with pytest.raises(TypeError, match="reuse_encoded_images must be a boolean"):
        preprocess.PDFPreprocessStage(reuse_encoded_images=value)


def test_encoded_png_provenance_binds_actual_producer_and_survives_serialization(monkeypatch: pytest.MonkeyPatch):
    stage = preprocess.PDFPreprocessStage(retain_failures=True, reuse_encoded_images=True)
    images = [Image.new("RGB", (13, 9), "red"), None, Image.new("RGB", (7, 11), "blue")]
    monkeypatch.setattr(stage, "_get_pdf_bytes", lambda *_args, **_kwargs: b"pdf")
    monkeypatch.setattr(stage, "_render_with_timeout", lambda *_args, **_kwargs: images)
    metadata = {"source": {"name": "test"}, PNG_METADATA_KEY: {"stale": "marker"}}
    task = FileGroupTask(
        dataset_name="test",
        data=[json.dumps({"file_name": "three.pdf", "selected_page_indices": [5, 1, 9]})],
        _metadata=metadata,
    )

    result = stage.process(task)
    rows = result.to_pyarrow().to_pylist()
    assert [row["position"] for row in rows] == [5, 1, 9]
    assert has_png_batch_provenance(result._metadata, (row["binary_content"] for row in rows))
    assert result._metadata is not metadata
    assert metadata == {"source": {"name": "test"}, PNG_METADATA_KEY: {"stale": "marker"}}
    assert set(rows[0]) == set(stage._page_row("three", "three.pdf", "", 5, images[0]))
    assert json.loads(rows[1]["materialize_error"])["stage"] == "render_pdf"
    with Image.open(io.BytesIO(rows[0]["binary_content"])) as restored_image:
        assert restored_image.mode == "RGB"
        assert restored_image.size == images[0].size
        assert restored_image.tobytes() == images[0].tobytes()
    restored = cloudpickle.loads(cloudpickle.dumps(result))
    assert restored.to_pyarrow().equals(result.to_pyarrow())
    assert has_png_batch_provenance(restored._metadata, restored.to_pyarrow().column("binary_content").to_pylist())


@pytest.mark.parametrize("stale_marker", [False, True])
def test_disabled_png_reuse_preserves_content_and_drops_stale_provenance(
    monkeypatch: pytest.MonkeyPatch, stale_marker: bool
):
    stage = preprocess.PDFPreprocessStage(retain_failures=True)
    image = Image.new("RGB", (5, 3), "green")
    monkeypatch.setattr(stage, "_get_pdf_bytes", lambda *_args, **_kwargs: b"pdf")
    monkeypatch.setattr(stage, "_render_with_timeout", lambda *_args, **_kwargs: [image])
    metadata = {"source": "test"}
    if stale_marker:
        metadata[PNG_METADATA_KEY] = png_batch_provenance([b"old payload"])
    original_metadata = metadata.copy()
    task = FileGroupTask(dataset_name="test", data=[json.dumps({"file_name": "one.pdf"})], _metadata=metadata)

    result = stage.process(task)
    assert PNG_METADATA_KEY not in result._metadata
    assert result._metadata == {"source": "test"}
    assert result.to_pyarrow().column("binary_content").to_pylist() == [utils.image_to_bytes(image)]
    assert task._metadata == original_metadata


@pytest.mark.parametrize(("failure_stage", "rendered"), [("read_pdf", []), ("render_pdf", []), ("render_pdf", [None])])
@pytest.mark.parametrize("retain_failures", [False, True])
def test_png_provenance_keeps_empty_and_failed_document_behavior(
    monkeypatch: pytest.MonkeyPatch, retain_failures: bool, failure_stage: str, rendered: list[None]
):
    stage = preprocess.PDFPreprocessStage(retain_failures=retain_failures, reuse_encoded_images=True)
    monkeypatch.setattr(
        stage, "_get_pdf_bytes", lambda *_args, **_kwargs: None if failure_stage == "read_pdf" else b"pdf"
    )
    monkeypatch.setattr(stage, "_render_with_timeout", lambda *_args, **_kwargs: rendered)
    task = FileGroupTask(dataset_name="test", data=[json.dumps({"file_name": "missing.pdf"})])

    result = stage.process(task)
    assert task._metadata == {}
    if retain_failures:
        assert result.to_pyarrow().column("binary_content").to_pylist() == [None]
        assert has_png_batch_provenance(result._metadata, [None])
        error = json.loads(result.to_pyarrow().to_pylist()[0]["materialize_error"])
        assert error["stage"] == failure_stage
    else:
        assert result is None
