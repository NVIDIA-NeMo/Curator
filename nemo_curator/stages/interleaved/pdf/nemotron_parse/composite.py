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
    DEFAULT_MODEL_PATH,
    NemotronParseInferenceStage,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.partitioning import PDFPartitioningStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocess import NemotronParsePostprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.preprocess import PDFPreprocessStage
from nemo_curator.tasks import InterleavedBatch, _EmptyTask


@dataclass
class NemotronParsePDFReader(CompositeStage[_EmptyTask, InterleavedBatch]):
    """Composite reader: partition -> preprocess -> infer -> postprocess.

    Decomposes into four execution stages:

    1. :class:`PDFPartitioningStage` — read manifest, create FileGroupTasks
    2. :class:`PDFPreprocessStage` — extract PDFs, render pages to images
    3. :class:`NemotronParseInferenceStage` — GPU model inference
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
        Number of PDFs per processing task.
    max_pdfs
        Maximum PDFs to process (for testing).
    dpi
        PDF rendering resolution.
    max_pages
        Maximum pages to render per PDF.
    inference_batch_size
        Pages per GPU forward pass (HF only).
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
    """

    manifest_path: str | None = None
    zip_base_dir: str | None = None
    pdf_dir: str | None = None
    jsonl_base_dir: str | None = None
    completed_ids: set[str] = field(default_factory=set)
    model_path: str = DEFAULT_MODEL_PATH
    backend: str = "vllm"
    pdfs_per_task: int = 10
    max_pdfs: int | None = None
    dpi: int = 300
    max_pages: int = 50
    inference_batch_size: int = 4
    max_num_seqs: int = 64
    text_in_pic: bool = False
    enforce_eager: bool = False
    min_crop_px: int = 10
    dataset_name: str = "pdf_dataset"
    file_name_field: str = "file_name"
    file_names_field: str = "cc_pdf_file_names"
    url_field: str = "url"

    def __post_init__(self) -> None:
        super().__init__()
        if self.manifest_path is None:
            msg = "manifest_path is required"
            raise ValueError(msg)
        self._partitioner = PDFPartitioningStage(
            manifest_path=self.manifest_path,
            pdfs_per_task=self.pdfs_per_task,
            max_pdfs=self.max_pdfs,
            completed_ids=self.completed_ids,
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
        )
        self._inference = NemotronParseInferenceStage(
            model_path=self.model_path,
            text_in_pic=self.text_in_pic,
            backend=self.backend,
            inference_batch_size=self.inference_batch_size,
            max_num_seqs=self.max_num_seqs,
            enforce_eager=self.enforce_eager,
        )
        self._postprocessor = NemotronParsePostprocessStage(
            min_crop_px=self.min_crop_px,
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return self._partitioner.inputs()

    def outputs(self) -> tuple[list[str], list[str]]:
        return self._postprocessor.outputs()

    def decompose(self) -> list[ProcessingStage]:
        return [self._partitioner, self._preprocessor, self._inference, self._postprocessor]
