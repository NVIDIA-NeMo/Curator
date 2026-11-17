# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import json
import sys
import threading
from collections.abc import Iterator
from contextlib import nullcontext
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking/scripts"))
import common_crawl_benchmark as benchmark

from tests.benchmarking.scripts.test_modifier_benchmark import SerialExecutor


@pytest.fixture(scope="session")
def shared_ray_cluster() -> None:
    """Tiny WARC files execute actual extraction and writing without Ray."""


@pytest.mark.parametrize("change", ["oversize", "remote_size", "remote_etag", "duplicate", "snapshot"])
def test_manifest_rejects_unbounded_or_changed_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    obj = {
        "url": "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-18/test.warc.gz",
        "size_bytes": 100,
        "etag": "original",
    }
    manifest = {"snapshot": "CC-MAIN-2024-18", "objects": [obj]}
    if change == "oversize":
        obj["size_bytes"] = 5 * 1024**3 + 1
    if change == "duplicate":
        manifest["objects"].append(obj)
    if change == "snapshot":
        manifest["snapshot"] = "CC-MAIN-2023-01"
    headers = {
        "Content-Length": "99" if change == "remote_size" else "100",
        "ETag": "changed" if change == "remote_etag" else "original",
    }
    monkeypatch.setattr(
        benchmark.requests,
        "head",
        lambda *_, **__: SimpleNamespace(status_code=200, headers=headers, raise_for_status=lambda: None),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"WARC|Expected|Invalid"):
        benchmark.verify_warc_manifest(path)


@pytest.mark.parametrize("interrupted", [False, True])
def test_fresh_complete_warc_download_extract_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted: bool
) -> None:
    text = (
        "This is a comprehensive test paragraph with many important words and meaningful content. "
        "We are testing the HTML extraction functionality to ensure that it works properly and efficiently "
        "with various types of content. The system should be able to process this text correctly. Café."
    )
    warc = tmp_path / "sample.warc.gz"
    with warc.open("wb") as stream:
        writer = WARCWriter(stream, gzip=True)
        for i in range(2):
            record = writer.create_warc_record(
                f"http://example.com/{i}",
                "response",
                payload=io.BytesIO(f"<html><body><p>{text}</p></body></html>".encode()),
                http_headers=StatusAndHeaders(
                    "200 OK", [("Content-Type", "text/html; charset=utf-8")], protocol="HTTP/1.0"
                ),
            )
            writer.write_record(record)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    manifest = {
        "snapshot": "CC-MAIN-2024-18",
        "objects": [{"url": f"http://127.0.0.1:{server.server_port}/{warc.name}", "size_bytes": warc.stat().st_size}],
    }
    monkeypatch.setattr(benchmark, "verify_warc_manifest", lambda _: manifest)
    monkeypatch.setattr(benchmark, "setup_executor", lambda _: SerialExecutor())
    if interrupted:
        iterate = benchmark.CompletedWarcIterator.iterate

        def fail_iteration(self: benchmark.CompletedWarcIterator, path: str) -> Iterator[dict]:
            for record in iterate(self, path):
                yield record
                msg = "interrupted before EOF"
                raise ValueError(msg)

        monkeypatch.setattr(benchmark.CompletedWarcIterator, "iterate", fail_iteration)
    args = argparse.Namespace(
        warc_manifest=str(tmp_path / "manifest.json"),
        download_path=str(tmp_path / "downloads"),
        output_path=str(tmp_path / "output"),
        output_format="parquet",
        html_extraction="justext",
        executor="xenna",
    )
    try:
        result = benchmark.run_manifest_benchmark(args)
        m = result["metrics"]
        if interrupted:
            assert not m["is_success"]
            assert m["input_compressed_bytes"] is None
            assert m["time_taken_s"] is None
            return
        assert m["is_success"], result["params"].get("failure")
        assert m["num_output_documents"] == m["input_response_records"] == 2
        assert m["input_compressed_bytes"] == m["downloaded_compressed_bytes"] == warc.stat().st_size
        assert m["num_completed_warc_files"] == 1
        assert m["cached_files_before"] == 0
        assert m["output_text_bytes"] == 2 * len(text.encode())
        assert m["time_taken_s"] == m["acquisition_time_s"] + m["extraction_write_time_s"] > 0
        assert sum(pq.read_table(p).num_rows for t in result["tasks"] for p in t.data) == 2
        with pytest.raises(FileExistsError):
            benchmark.run_manifest_benchmark(args)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.parametrize("chunks", [[b"a" * 60, b"b" * 60], [b"a" * 99]])
def test_download_rejects_oversized_and_short_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunks: list[bytes]
) -> None:
    url = "https://data.commoncrawl.org/sample.warc.gz"
    response = SimpleNamespace(
        status_code=200,
        headers={"Content-Length": "100"},
        raise_for_status=lambda: None,
        iter_content=lambda **_: iter(chunks),
    )
    monkeypatch.setattr(benchmark.requests, "get", lambda *_, **__: nullcontext(response))
    downloader = benchmark.BoundedWarcDownloader(str(tmp_path), [{"url": url, "size_bytes": 100}])
    with pytest.raises(ValueError, match=r"WARC|Incomplete"):
        downloader.download(url)
    assert not (tmp_path / "sample.warc.gz").exists()
    assert (tmp_path / "sample.warc.gz.tmp").stat().st_size <= 100
