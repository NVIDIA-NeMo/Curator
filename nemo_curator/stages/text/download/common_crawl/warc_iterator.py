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

from fastwarc.warc import ArchiveIterator, WarcRecordType
from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.stages.text.download import DocumentIterator


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
            # in C++ before their headers reach Python, and auto_decode="all" keeps the
            # HTTP body decoded from its Content-Encoding, as this iterator did before.
            # strict_mode=False resynchronizes past a record with an unparseable WARC
            # header instead of silently ending the file there, which is the default.
            #
            # Gaps that come with fastwarc 0.x, none of them reachable from Common Crawl's
            # own files: a record resynchronized past is dropped silently -- the parser
            # exposes no skip counter or callback and does not surface the record even
            # with record_types=any_type, so there is nothing this loop can log, and a
            # short record count is the only symptom; chunked transfer-encoding is not
            # decoded even with auto_decode="all"; and the HTTP preamble is only stripped
            # from records that declare Content-Type: application/http, which Common Crawl
            # emits. ARC input, which the previous arc2warc=True accepted, is not supported.
            archive_iterator = ArchiveIterator(
                file_pointer, record_types=WarcRecordType.response, auto_decode="all", strict_mode=False
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
                    content = rec.reader.read()
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
