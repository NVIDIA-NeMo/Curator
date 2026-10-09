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

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastwarc.stream_io import BrotliStream, BufferedReader, BytesIOStream, GZipStream, StreamError
from fastwarc.warc import ArchiveIterator, WarcRecord, WarcRecordType
from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.stages.text.download import DocumentIterator

_CONTENT_DECODERS = {
    "gzip": GZipStream,
    "x-gzip": GZipStream,
    "deflate": lambda stream: GZipStream(stream, zlib=True),
    "br": BrotliStream,
}


def _dechunk(body: bytes) -> bytes:
    """Undo HTTP chunked transfer-encoding.

    Like warcio, a body that stops parsing as chunked is kept raw from that point on, and a
    truncated chunk is returned as far as it goes.
    """
    chunks, pos = [], 0
    while pos < len(body):
        eol = body.find(b"\r\n", pos)
        try:
            size = int(body[pos:eol].split(b";")[0], 16) if eol != -1 else -1
        except ValueError:
            size = -1
        if size < 0:
            chunks.append(body[pos:])
            break
        if size == 0:
            break
        chunks.append(body[eol + 2 : eol + 2 + size])
        pos = eol + 2 + size + 2
    return b"".join(chunks)


def read_http_body(record: WarcRecord) -> bytes:
    """Return a response record's HTTP body with Transfer- and Content-Encoding undone.

    The record must come from an ArchiveIterator with auto_decode="none". fastwarc 0.x's own
    auto_decode does not de-chunk, which older Common Crawl crawls (e.g. CC-MAIN-2016-07) still
    need, and raises on a body whose Content-Encoding header is wrong. warcio did both: it
    de-chunked, and fell back to the raw body when decoding failed. This does the same.
    """
    body = record.reader.read()
    headers = record.http_headers
    if headers is None:
        return body
    if "chunked" in (headers.get("Transfer-Encoding") or "").lower():
        body = _dechunk(body)
    decoder = _CONTENT_DECODERS.get((headers.get("Content-Encoding") or "").strip().lower())
    if decoder is None:
        return body
    try:
        return BufferedReader(decoder(BytesIOStream(body))).read()
    except StreamError:
        return body


class CommonCrawlWarcIterator(DocumentIterator):
    """Processes WARC files from local or fsspec-compatible storage."""

    def __init__(self, storage_options: dict[str, Any] | None = None):
        """Create a WARC iterator.

        Args:
            storage_options: Options forwarded to the fsspec filesystem inferred
                from each input path. For example, S3 credentials, a profile, or
                an endpoint URL can be supplied here.
        """
        self.storage_options = storage_options or {}

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        """Process a task containing WARC files and extract their contents."""
        file_path_str = str(file_path)
        filename = file_path.name if isinstance(file_path, Path) else file_path_str.rsplit("/", 1)[-1]

        num_records = 0
        fs, fs_path = url_to_fs(file_path_str, **self.storage_options)
        with fs.open(fs_path, "rb") as file_pointer:
            # fastwarc wraps any file-like object and sniffs gzip itself, so the fsspec
            # handle can be passed straight through. Non-response records are discarded
            # in C++ before their headers reach Python, and read_http_body() decodes the
            # HTTP body's Transfer- and Content-Encoding, as this iterator did with warcio.
            # strict_mode=False resynchronizes past a record with an unparseable WARC
            # header instead of silently ending the file there, which is the default.
            #
            # Gaps that come with fastwarc 0.x: a record resynchronized past is dropped
            # silently -- the parser exposes no skip counter or callback and does not surface
            # the record even with record_types=any_type, so there is nothing this loop can
            # log, and a short record count is the only symptom; and the HTTP preamble is
            # only stripped from records that declare Content-Type: application/http, which
            # Common Crawl emits. ARC input, which the previous arc2warc=True accepted, is not
            # supported.
            archive_iterator = ArchiveIterator(
                file_pointer, record_types=WarcRecordType.response, auto_decode="none", strict_mode=False
            )
            while True:
                try:
                    rec = next(archive_iterator)
                except StopIteration:
                    # End of file reached normally
                    break
                except Exception as e:  # noqa: BLE001
                    # next() has its own try because a stream the parser cannot open at all
                    # fails here (fastwarc raises StreamError) rather than while reading a
                    # record, and leaves nothing to resynchronize to. Report it once and stop
                    # instead of calling next() again on a stream that is already dead.
                    logger.error(f"Error processing record {num_records} in {filename}: {e!s}")
                    break

                try:
                    content = read_http_body(rec)
                    warc_id = rec.headers.get("WARC-Record-ID")[10:-1]
                    url = rec.headers.get("WARC-Target-URI")
                    yield {"url": url, "warc_id": warc_id, "source_id": filename, "content": content}
                    num_records += 1
                except Exception as e:  # noqa: BLE001
                    # Handle corruption or other errors
                    logger.error(f"Error processing record {num_records} in {filename}: {e!s}")
                    # Try to continue with next record
                    continue

    def output_columns(self) -> list[str]:
        return ["url", "warc_id", "source_id", "content"]
