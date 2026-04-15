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

from typing import Literal

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.download import DocumentDownloadExtractStage
from nemo_curator.stages.text.download.html_extractors import HTMLExtractorAlgorithm
from nemo_curator.stages.text.download.html_extractors.justext import JusTextExtractor
from nemo_curator.stages.text.download.html_extractors.model_based import (
    AssembleModelBasedHTMLExtractionStage,
    ModelBasedHTMLExtractionStage,
    ModelBasedHTMLInferenceStage,
)
from nemo_curator.stages.text.models.tokenizer import TokenizerStage
from nemo_curator.utils.column_utils import resolve_filename_column

from .download import CommonCrawlWARCDownloader
from .extract import CommonCrawlHTMLExtractor, CommonCrawlModelBasedCandidateExtractor
from .url_generation import MainCommonCrawlUrlGenerator, NewsCommonCrawlUrlGenerator
from .warc_iterator import CommonCrawlWarcIterator


class CommonCrawlDownloadExtractStage(DocumentDownloadExtractStage):
    """Composite stage for downloading and processing Common Crawl data.

    This pipeline:
    1. Generates WARC URLs (either from main or news crawls)
    2. Downloads WARC files
    3. Extracts content from WARC files
    4. Extracts text from HTML content
    """

    def __init__(  # noqa: PLR0913
        self,
        start_snapshot: str,
        end_snapshot: str,
        download_dir: str,
        crawl_type: Literal["main", "news"] = "main",
        html_extraction: HTMLExtractorAlgorithm | str | None = None,
        html_extraction_kwargs: dict | None = None,
        stop_lists: dict[str, frozenset[str]] | None = None,
        use_aws_to_download: bool = False,
        verbose: bool = False,
        url_limit: int | None = None,
        record_limit: int | None = None,
        add_filename_column: bool | str = True,
        extractor_max_calls_per_worker: int | None = None,
    ):
        self.crawl_type = crawl_type
        self.start_snapshot = start_snapshot
        self.end_snapshot = end_snapshot

        if crawl_type == "main":
            self.url_generator = MainCommonCrawlUrlGenerator(
                start_snapshot_str=start_snapshot, end_snapshot_str=end_snapshot, limit=url_limit
            )
        else:
            self.url_generator = NewsCommonCrawlUrlGenerator(
                start_snapshot_str=start_snapshot, end_snapshot_str=end_snapshot, limit=url_limit
            )

        self.downloader = CommonCrawlWARCDownloader(
            download_dir=download_dir, use_aws_to_download=use_aws_to_download, verbose=verbose
        )
        self.iterator = CommonCrawlWarcIterator()
        html_extraction_kwargs = html_extraction_kwargs or {}
        model_based_algorithm = self._resolve_model_based_algorithm(html_extraction, html_extraction_kwargs)
        if model_based_algorithm is not None:
            self.extractor = CommonCrawlModelBasedCandidateExtractor(
                algorithm=model_based_algorithm,
                stop_lists=stop_lists,
            )
            self.stages = self._build_model_based_stages(
                algorithm=model_based_algorithm,
                stop_lists=stop_lists,
                url_limit=url_limit,
                record_limit=record_limit,
                add_filename_column=add_filename_column,
                extractor_max_calls_per_worker=extractor_max_calls_per_worker,
            )
            self._with_operations = []
        else:
            self.extractor = CommonCrawlHTMLExtractor(
                algorithm=html_extraction,
                algorithm_kwargs=html_extraction_kwargs,
                stop_lists=stop_lists,
            )
            if extractor_max_calls_per_worker is None and isinstance(self.extractor.algorithm, JusTextExtractor):
                extractor_max_calls_per_worker = 2
                logger.info(
                    "jusText extraction can cause memory fragmentation and lead to OOM errors. "
                    "Setting extractor_max_calls_per_worker=2 for the iterate-extract stage. "
                    "Pass extractor_max_calls_per_worker explicitly to override."
                )
            super().__init__(
                url_generator=self.url_generator,
                downloader=self.downloader,
                iterator=self.iterator,
                extractor=self.extractor,
                url_limit=url_limit,
                record_limit=record_limit,
                add_filename_column=add_filename_column,
                extractor_max_calls_per_worker=extractor_max_calls_per_worker,
            )
        self.name = f"common_crawl_{self.crawl_type}_pipeline"

    def decompose(self) -> list[ProcessingStage]:
        """Decompose this composite stage into its constituent stages."""
        return self.stages

    def get_description(self) -> str:
        """Get a description of this composite stage."""
        return f"Common Crawl {self.crawl_type} pipeline: {self.start_snapshot} to {self.end_snapshot}"

    @staticmethod
    def _resolve_model_based_algorithm(
        html_extraction: HTMLExtractorAlgorithm | str | None,
        html_extraction_kwargs: dict,
    ) -> ModelBasedHTMLExtractionStage | None:
        if isinstance(html_extraction, ModelBasedHTMLExtractionStage):
            return html_extraction
        if html_extraction in {"model", "model_based"}:
            return ModelBasedHTMLExtractionStage(**html_extraction_kwargs)
        return None

    def _build_model_based_stages(  # noqa: PLR0913
        self,
        algorithm: ModelBasedHTMLExtractionStage,
        stop_lists: dict[str, frozenset[str]] | None,
        url_limit: int | None,
        record_limit: int | None,
        add_filename_column: bool | str,
        extractor_max_calls_per_worker: int | None,
    ) -> list[ProcessingStage]:
        filename_column = resolve_filename_column(add_filename_column) if add_filename_column else None

        base_stage = DocumentDownloadExtractStage(
            url_generator=self.url_generator,
            downloader=self.downloader,
            iterator=self.iterator,
            extractor=self.extractor,
            url_limit=url_limit,
            record_limit=record_limit,
            add_filename_column=add_filename_column,
            extractor_max_calls_per_worker=extractor_max_calls_per_worker,
        )
        iterate_stage = base_stage.decompose()[-1]
        tokenizer_stage = TokenizerStage(
            model_identifier=algorithm.model_identifier,
            cache_dir=algorithm.cache_dir,
            text_field="candidate_model_input",
            max_seq_length=algorithm.max_length,
            sort_by_length=True,
            transformers_init_kwargs=algorithm.transformers_init_kwargs,
        )
        inference_stage = ModelBasedHTMLInferenceStage(
            model_identifier=algorithm.model_identifier,
            cache_dir=algorithm.cache_dir,
            model_inference_batch_size=algorithm.model_inference_batch_size,
            max_seq_length=None,
            transformers_init_kwargs=algorithm.transformers_init_kwargs,
        )
        assemble_stage = AssembleModelBasedHTMLExtractionStage(
            stop_lists=stop_lists or self.extractor._stop_lists,
            output_format=algorithm.output_format,
            fallback_threshold=algorithm.fallback_threshold,
            fallback_extractor=algorithm.fallback_extractor,
            filename_column=filename_column,
        )
        return [base_stage.decompose()[0], base_stage.decompose()[1], iterate_stage, tokenizer_stage, inference_stage, assemble_stage]
