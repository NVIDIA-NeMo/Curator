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

import bz2
import codecs
import os
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

from loguru import logger

from nemo_curator.stages.text.download import DocumentIterator


class WikipediaIterator(DocumentIterator):
    """Processes downloaded Wikipedia dump files and extracts article content."""

    def __init__(self, language: str = "en", log_frequency: int = 1000):
        """
        Initialize the Wikipedia iterator.

        Args:
            language: Language code for the Wikipedia dump
            log_frequency: How often to log progress (every N articles)
        """
        super().__init__()
        self._language = language
        self._log_frequency = log_frequency
        self._counter = 0

    def _should_log_progress(self, _: str) -> bool:
        """Check if progress should be logged based on counter."""
        return self._counter > 0 and self._counter % self._log_frequency == 0

    def _extract_element_text(self, elem: ET.Element, namespace: str, tag: str) -> str | None:
        """Extract text from an XML element."""
        element = elem.find(f"./{namespace}{tag}")
        return element.text if element is not None else None

    def _get_article_metadata(self, elem: ET.Element, namespace: str) -> dict[str, Any] | None:
        """Extract metadata from a Wikipedia article element."""
        title = self._extract_element_text(elem, namespace, "title")
        ns = self._extract_element_text(elem, namespace, "ns")
        id_ = self._extract_element_text(elem, namespace, "id")

        if not all([title, ns, id_]):
            return None

        return {"title": title, "ns": ns, "id": id_, "redirect": elem.find(f"./{namespace}redirect")}

    def _get_article_content(self, elem: ET.Element, namespace: str) -> str | None:
        """Extract raw content from Wikipedia article element."""
        revision_elem = elem.find(f"./{namespace}revision")
        if revision_elem is None:
            return None

        text_elem = revision_elem.find(f"./{namespace}text")
        if text_elem is None:
            return None

        return text_elem.text

    def _should_skip_article(self, metadata: dict[str, Any], raw_content: str | None) -> bool:
        """Check if article should be skipped based on metadata and content."""
        # Skip non-main namespace articles
        if metadata["ns"] != "0":
            return True

        # Skip redirects
        if metadata["redirect"] is not None:
            return True

        # Skip empty content
        return raw_content is None

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        """Process a Wikipedia dump file and extract article content.

        Args:
            file_path: Path to the downloaded .bz2 file

        Yields:
            Dict containing article metadata and raw content
        """
        self._counter = 0
        filename = file_path.name if hasattr(file_path, "name") else os.path.basename(file_path)

        try:
            with bz2.BZ2File(filename=file_path) as input_file:
                utf_f = codecs.getreader("utf-8")(input_file)
                context = ET.iterparse(utf_f, events=("end",))  # noqa: S314

                for _i, (_unused_event, elem) in enumerate(context):
                    if not elem.tag.endswith("page"):
                        continue

                    if self._should_log_progress(file_path):
                        logger.debug(f"Extracted {self._counter} articles from {file_path}")
                    self._counter += 1

                    namespace = elem.tag[:-4]

                    # Extract metadata
                    metadata = self._get_article_metadata(elem, namespace)
                    if metadata is None:
                        elem.clear()
                        continue

                    # Extract content
                    raw_content = self._get_article_content(elem, namespace)
                    elem.clear()

                    # Check if article should be skipped
                    if self._should_skip_article(metadata, raw_content):
                        continue

                    # Create URL
                    url = f"https://{self._language}.wikipedia.org/wiki/{quote(metadata['title'])}"

                    yield {
                        "title": metadata["title"],
                        "id": metadata["id"],
                        "url": url,
                        "language": self._language,
                        "source_id": filename,
                        "raw_content": raw_content,
                    }

        except Exception as e:
            logger.error(f"Error processing Wikipedia dump file {file_path}: {e}")
            raise

    def output_columns(self) -> list[str]:
        """Define the output columns produced by this iterator."""
        return ["title", "id", "url", "language", "source_id", "raw_content"]
