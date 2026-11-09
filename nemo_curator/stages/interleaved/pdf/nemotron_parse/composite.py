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

"""Composite stage that bundles the full PDF -> Nemotron-Parse -> interleaved pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_PATH,
    NemotronParseHTTPClientStage,
    NemotronParseInferenceStage,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.page_budget import validate_extraction_config
from nemo_curator.stages.interleaved.pdf.nemotron_parse.partitioning import PDFPartitioningStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocess import NemotronParsePostprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.preprocess import PDFPreprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.reassembly import PDFDocumentMergeStage
from nemo_curator.tasks import EmptyTask, InterleavedBatch


@dataclass
class NemotronParsePDFReader(CompositeStage[EmptyTask, InterleavedBatch]):
    """Composite reader: partition -> preprocess -> infer -> postprocess.

    Decomposes into four execution stages:

    1. :class:`PDFPartitioningStage` — read manifest, create FileGroupTasks
    2. :class:`PDFPreprocessStage` — extract PDFs, render pages to images
    3. :class:`NemotronParseInferenceStage` or :class:`NemotronParseHTTPClientStage` — model inference
    4. :class:`NemotronParsePostprocessStage` — parse output, align, crop

    Parameters
    ----------
    manifest_path
        Path to JSONL manifest listing PDFs.
    zip_base_dir
        Root of CC-MAIN zip archive hierarchy.
    pdf_dir
        Directory containing PDF files.
    jsonl_base_dir
        Root directory for JSONL-based PDF datasets (e.g. GitHub PDFs).
    model_path
        HuggingFace model ID or local path.
    backend
        Inference backend: ``"vllm"`` (recommended) or ``"hf"``.
    pdfs_per_task
        Number of PDFs per task in legacy mode; ignored when pages_per_task is set.
    pages_per_task
        Opt-in page-only task budget. Pack across documents and split long PDFs
        without dropping pages. Requires served single mode and RayDataExecutor.
    max_pdfs
        Maximum PDFs to process (for testing).
    dpi
        PDF rendering resolution.
    max_pages
        Legacy per-PDF page limit; page mode processes the full manifest selection.
    inference_batch_size
        Pages per GPU forward pass for HF, or maximum concurrent page requests
        from each HTTP client worker for an inference server. Start with 32;
        tune on the target hardware and corpus.
    max_num_seqs
        Maximum concurrent sequences (vLLM only).
    text_in_pic
        Whether to predict text inside pictures (v1.2+ prompt control).
    min_crop_px
        Minimum pixel dimension for image crops.
    dataset_name
        Name assigned to output tasks.
    file_name_field
        JSONL field containing a single PDF filename.
    file_names_field
        JSONL field containing a list of PDF filenames (CC-MAIN style).
    url_field
        JSONL field containing the source URL.
    parse_mode
        Opt in to the model package's served Parse 3.5 single, double or triple
        pipeline. Requires inference_server_endpoint. Unset preserves v1.2.
    triple_regions
        Combine detector and full-page regions (union), or use detector alone.
    inference_server_endpoint
        OpenAI-compatible inference server endpoint. A Dynamo-backed server is
        the recommended production topology; when omitted, inference runs in
        process.
    inference_server_client_num_workers
        Fixed number of concurrent HTTP client stage workers. Use four times
        the number of inference GPUs.
    inference_server_reuse_encoded_images
        Reuse verified unchanged producer PNGs in served single mode. Disabled
        by default; unverified images keep the ordinary encoding path.

    page_budget_extraction_config
        Resolved effective parser, model, request, rendering and projection
        identity for page mode. This record verifies settings; it does not
        override them. Required when pages_per_task is set.
    """

    manifest_path: str | None = None
    zip_base_dir: str | None = None
    pdf_dir: str | None = None
    jsonl_base_dir: str | None = None
    model_path: str = DEFAULT_MODEL_PATH
    backend: str = "vllm"
    pdfs_per_task: int = 10
    pages_per_task: int | None = field(default=None, kw_only=True)
    page_budget_extraction_config: dict | None = field(default=None, kw_only=True)
    max_pdfs: int | None = None
    dpi: int = 300
    max_pages: int = 50
    inference_batch_size: int = 4
    max_num_seqs: int = 64
    max_tokens: int = DEFAULT_MAX_TOKENS
    text_in_pic: bool = False
    enforce_eager: bool = False
    min_crop_px: int = 10
    dataset_name: str = "pdf_dataset"
    file_name_field: str = "file_name"
    file_names_field: str = "cc_pdf_file_names"
    url_field: str = "url"
    parse_mode: str | None = field(default=None, kw_only=True)
    triple_regions: str = field(default="union", kw_only=True)
    inference_server_endpoint: str | None = None
    inference_server_model_name: str | None = None
    inference_server_client_num_workers: int = 4
    inference_server_request_timeout_s: float = 300.0
    inference_server_max_retries: int = 3
    inference_server_reuse_encoded_images: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        super().__init__()
        if type(self.inference_server_reuse_encoded_images) is not bool:
            msg = "inference_server_reuse_encoded_images must be a boolean"
            raise TypeError(msg)
        if self.inference_server_reuse_encoded_images and (
            self.parse_mode != "single" or self.inference_server_endpoint is None
        ):
            msg = "Encoded image reuse requires served Parse 3.5 single mode"
            raise ValueError(msg)
        if self.manifest_path is None:
            msg = "manifest_path is required"
            raise ValueError(msg)
        if self.pages_per_task is not None and (self.parse_mode != "single" or self.inference_server_endpoint is None):
            msg = "pages_per_task requires served Parse 3.5 single mode and RayDataExecutor"
            raise ValueError(msg)
        if self.pages_per_task is not None:
            self.page_budget_extraction_config = validate_extraction_config(self.page_budget_extraction_config)
        if self.parse_mode is not None and self.inference_server_endpoint is None:
            msg = "parse_mode requires inference_server_endpoint"
            raise ValueError(msg)
        self._partitioner = PDFPartitioningStage(
            manifest_path=self.manifest_path,
            pdfs_per_task=self.pdfs_per_task,
            pages_per_task=self.pages_per_task,
            max_pdfs=self.max_pdfs,
            dataset_name=self.dataset_name,
            file_name_field=self.file_name_field,
            file_names_field=self.file_names_field,
            url_field=self.url_field,
        )
        self._preprocessor = PDFPreprocessStage(
            zip_base_dir=self.zip_base_dir,
            pdf_dir=self.pdf_dir,
            jsonl_base_dir=self.jsonl_base_dir,
            dpi=self.dpi,
            max_pages=self.max_pages,
            retain_failures=self.parse_mode is not None,
            reuse_encoded_images=self.inference_server_reuse_encoded_images,
        )
        if self.inference_server_endpoint is None:
            self._inference = NemotronParseInferenceStage(
                model_path=self.model_path,
                text_in_pic=self.text_in_pic,
                backend=self.backend,
                inference_batch_size=self.inference_batch_size,
                max_num_seqs=self.max_num_seqs,
                max_tokens=self.max_tokens,
                enforce_eager=self.enforce_eager,
            )
        else:
            if self.inference_server_client_num_workers < 1:
                msg = "inference_server_client_num_workers must be at least 1"
                raise ValueError(msg)
            self._inference = NemotronParseHTTPClientStage(
                endpoint=self.inference_server_endpoint,
                parse_mode=self.parse_mode,
                triple_regions=self.triple_regions,
                model_name=self.inference_server_model_name or self.model_path,
                model_path=self.model_path,
                text_in_pic=self.text_in_pic,
                request_timeout_s=self.inference_server_request_timeout_s,
                max_retries=self.inference_server_max_retries,
                inference_batch_size=self.inference_batch_size,
                max_tokens=self.max_tokens,
                reuse_encoded_images=self.inference_server_reuse_encoded_images,
            ).with_(num_workers=self.inference_server_client_num_workers)
        self._postprocessor = NemotronParsePostprocessStage(
            min_crop_px=self.min_crop_px,
        )
        self._postprocessor._page_budget = self.pages_per_task is not None
        self._merger = PDFDocumentMergeStage() if self.pages_per_task is not None else None
        if self._merger is not None:
            self._configure_page_budget()

    def _configure_page_budget(self) -> None:
        self._inference.extraction_config = self.page_budget_extraction_config
        self._partitioner._extraction_config = self.page_budget_extraction_config
        self._preprocessor._extraction_config = self.page_budget_extraction_config
        if self.page_budget_extraction_config["render"].get("dpi") != self.dpi or self.page_budget_extraction_config[
            "render"
        ].get("proc_size") != list(self._inference.proc_size):
            msg = "Page-budget extraction configuration does not match rendering settings"
            raise ValueError(msg)
        if self.page_budget_extraction_config["projection"] != {
            "schema": "pdf-page-budget.v1",
            "min_crop_px": self.min_crop_px,
        }:
            msg = "Page-budget extraction configuration does not match projection settings"
            raise ValueError(msg)
        self._partitioner._parser_config = {
            "model_path": self.model_path,
            "parse_mode": self.parse_mode,
            "triple_regions": self.triple_regions,
            "dpi": self.dpi,
            "max_tokens": self.max_tokens,
            "text_in_pic": self.text_in_pic,
            "min_crop_px": self.min_crop_px,
        }

    def inputs(self) -> tuple[list[str], list[str]]:
        return self._partitioner.inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self._postprocessor.outputs()

    def decompose(self) -> list[ProcessingStage]:
        stages = [self._partitioner, self._preprocessor, self._inference, self._postprocessor]
        return stages if self._merger is None else [*stages, self._merger]
