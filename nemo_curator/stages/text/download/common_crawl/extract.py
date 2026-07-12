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

from typing import Any

from loguru import logger

from nemo_curator.stages.text.download import DocumentExtractor
from nemo_curator.stages.text.download.html_extractors import HTMLExtractorAlgorithm
from nemo_curator.stages.text.download.html_extractors.justext import JusTextExtractor
from nemo_curator.stages.text.download.html_extractors.resiliparse import ResiliparseExtractor
from nemo_curator.stages.text.download.html_extractors.trafilatura import TrafilaturaExtractor
from nemo_curator.stages.text.download.html_extractors.utils import get_stop_list_dict
from nemo_curator.stages.text.download.utils import decode_html, lang_detect


class CommonCrawlHTMLExtractor(DocumentExtractor):
    def __init__(
        self,
        algorithm: HTMLExtractorAlgorithm | str | None = None,
        algorithm_kwargs: dict | None = None,
        stop_lists: dict[str, frozenset[str]] | None = None,
    ):
        super().__init__()
        algorithm_kwargs = algorithm_kwargs or {}
        if algorithm is None:
            logger.warning("No algorithm provided, using justext with default parameters")
            algorithm = JusTextExtractor()
        elif isinstance(algorithm, str):
            if algorithm == "justext":
                algorithm = JusTextExtractor(**algorithm_kwargs)
            elif algorithm == "resiliparse":
                algorithm = ResiliparseExtractor(**algorithm_kwargs)
            elif algorithm == "trafilatura":
                algorithm = TrafilaturaExtractor(**algorithm_kwargs)
            else:
                msg = f"Invalid algorithm: {algorithm}"
                raise ValueError(msg)
        elif isinstance(algorithm, HTMLExtractorAlgorithm):
            if algorithm_kwargs:
                logger.warning("Algorithm kwargs provided are ignored when an HTMLExtractorAlgorithm is provided")
        else:
            msg = f"Invalid algorithm: {algorithm}"
            raise ValueError(msg)

        if stop_lists is not None:
            self._stop_lists = stop_lists
        else:
            self._stop_lists = get_stop_list_dict()

        self.algorithm = algorithm

    def extract(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Extract text from HTML content in the record.

        Takes a record dict containing "content" field with HTML and returns
        a new dict with only the output columns: url, warc_id, source_id, language, text.
        """
        # Extract the HTML content from the record
        html_content = record.get("content")
        if not html_content:
            return None

        # Content from WARC records is bytes, even though type annotation suggests str
        html = decode_html(html_content)

        if html is not None:
            # Language detection and HTML extraction
            lang = lang_detect(html)

            text = None
            if lang in self._stop_lists:
                text = self.algorithm.extract_text(html, self._stop_lists[lang], lang)

            if text is not None:
                if len(text) > 0:
                    text = "\n\n".join(text)
                    return {
                        "url": record["url"],
                        "warc_id": record["warc_id"],
                        "source_id": record["source_id"],
                        "language": lang,
                        "text": text,
                    }
                else:
                    return None
        return None

    def input_columns(self) -> list[str]:
        return ["url", "warc_id", "source_id", "content"]

    def output_columns(self) -> list[str]:
        return ["url", "warc_id", "source_id", "language", "text"]
