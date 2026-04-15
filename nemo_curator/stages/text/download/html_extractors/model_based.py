# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from bs4.element import Tag
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.models.model import ModelStage
from nemo_curator.tasks import DocumentBatch

from .base import HTMLExtractorAlgorithm
from .trafilatura import TrafilaturaExtractor

if TYPE_CHECKING:
    from pandas import Series

    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

ModelBasedOutputFormat = Literal["markdown", "plain", "plain_text"]

MODEL_INPUT_FIELD = "candidate_model_input"
PREDICTION_LABEL_FIELD = "candidate_label"
PREDICTION_CONFIDENCE_FIELD = "candidate_confidence"
CANDIDATE_INDEX_FIELD = "candidate_index"
CANDIDATE_TAG_NAME_FIELD = "candidate_tag_name"
CANDIDATE_TEXT_FIELD = "candidate_text"
CANDIDATE_HTML_FIELD = "candidate_html"
CANDIDATE_ATTRIBUTES_FIELD = "candidate_attributes"
HTML_FIELD = "html"
MIN_CANDIDATE_INDEX = 0
PLACEHOLDER_CANDIDATE_INDEX = -1

MAIN_CONTENT_LABELS = frozenset({"main", "main_content", "content", "paragraph", "heading", "list_item"})
STRUCTURED_LABELS = frozenset({"code", "code_block", "table", "formula", "math"})
DROP_LABELS = frozenset({"boilerplate", "navigation", "nav", "ad", "ads", "advertisement", "footer", "header"})
BLOCK_TAGS = frozenset(
    {
        "article",
        "blockquote",
        "code",
        "dd",
        "div",
        "dt",
        "figcaption",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "math",
        "p",
        "pre",
        "section",
        "table",
    }
)
SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg"})
HEADING_TAG_NAME_LENGTH = 2
MAX_HEADING_LEVEL = 6
MIN_MARKDOWN_FENCE_LENGTH = 3
MIN_FENCED_BLOCK_LINES = 2


@dataclass(frozen=True)
class HTMLElement:
    """A candidate HTML element prepared for model classification."""

    index: int
    tag_name: str
    text: str
    html: str
    attributes: dict[str, str]


@dataclass(frozen=True)
class HTMLElementPrediction:
    """The model label and confidence for a candidate HTML element."""

    label: str
    confidence: float


class HTMLElementClassifier(ABC):
    """Predict semantic labels for a batch of HTML elements."""

    @abstractmethod
    def predict(self, elements: list[HTMLElement]) -> list[HTMLElementPrediction]:
        ...


def format_html_element_for_model(element: HTMLElement) -> str:
    attributes = " ".join(f'{key}="{value}"' for key, value in element.attributes.items())
    return f"<{element.tag_name} {attributes}> {element.text}"


def serialize_html_element(element: HTMLElement) -> dict[str, Any]:
    return {
        CANDIDATE_INDEX_FIELD: element.index,
        CANDIDATE_TAG_NAME_FIELD: element.tag_name,
        CANDIDATE_TEXT_FIELD: element.text,
        CANDIDATE_HTML_FIELD: element.html,
        CANDIDATE_ATTRIBUTES_FIELD: element.attributes,
        MODEL_INPUT_FIELD: format_html_element_for_model(element),
    }


def deserialize_html_element(row: Series) -> HTMLElement | None:
    candidate_html = row.get(CANDIDATE_HTML_FIELD)
    candidate_tag_name = row.get(CANDIDATE_TAG_NAME_FIELD)
    if not isinstance(candidate_html, str) or not isinstance(candidate_tag_name, str):
        return None

    attributes = row.get(CANDIDATE_ATTRIBUTES_FIELD, {})
    if not isinstance(attributes, dict):
        attributes = {}

    return HTMLElement(
        index=int(row.get(CANDIDATE_INDEX_FIELD, PLACEHOLDER_CANDIDATE_INDEX)),
        tag_name=candidate_tag_name,
        text=str(row.get(CANDIDATE_TEXT_FIELD, "")),
        html=candidate_html,
        attributes={str(key): str(value) for key, value in attributes.items()},
    )


class _TransformersHTMLElementClassifier(HTMLElementClassifier):
    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        cache_dir: str | None,
        device: Literal["cuda", "cpu"],
        model_inference_batch_size: int,
        max_length: int,
        local_files_only: bool,
        transformers_init_kwargs: dict[str, Any],
    ):
        self.model_identifier = model_identifier
        self.cache_dir = cache_dir
        self.device = device
        self.model_inference_batch_size = model_inference_batch_size
        self.max_length = max_length
        self.local_files_only = local_files_only
        self.transformers_init_kwargs = transformers_init_kwargs
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def _setup(self, local_files_only: bool = True) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        if self.device == "cuda" and not torch.cuda.is_available():
            msg = "CUDA requested for model-based HTML extraction, but CUDA is unavailable."
            raise RuntimeError(msg)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_identifier,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
            **self.transformers_init_kwargs,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_identifier,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
            **self.transformers_init_kwargs,
        )
        self._model.to(self.device)
        self._model.eval()

    def predict(self, elements: list[HTMLElement]) -> list[HTMLElementPrediction]:
        if self._model is None or self._tokenizer is None:
            self._setup(local_files_only=self.local_files_only)

        model = self._model
        tokenizer = self._tokenizer

        predictions: list[HTMLElementPrediction] = []
        for start in range(0, len(elements), self.model_inference_batch_size):
            batch = elements[start : start + self.model_inference_batch_size]
            model_inputs = [format_html_element_for_model(element) for element in batch]
            encoded = tokenizer(
                model_inputs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1)
                confidences, label_ids = torch.max(probabilities, dim=-1)

            id2label = getattr(model.config, "id2label", {})
            for label_id, confidence in zip(label_ids.cpu().tolist(), confidences.cpu().tolist(), strict=True):
                label = str(id2label.get(label_id, label_id)).lower()
                predictions.append(HTMLElementPrediction(label=label, confidence=float(confidence)))

        return predictions


class ModelBasedHTMLInferenceStage(ModelStage):
    def __init__(
        self,
        model_identifier: str,
        cache_dir: str | None = None,
        model_inference_batch_size: int = 64,
        max_seq_length: int | None = 512,
        transformers_init_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(
            model_identifier=model_identifier,
            cache_dir=cache_dir,
            model_inference_batch_size=model_inference_batch_size,
            has_seq_order=True,
            padding_side="right",
            max_seq_length=max_seq_length,
            unpack_inference_batch=True,
            autocast=True,
        )
        self.transformers_init_kwargs = transformers_init_kwargs or {}

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [PREDICTION_LABEL_FIELD, PREDICTION_CONFIDENCE_FIELD]

    def _setup(self, local_files_only: bool = True) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_identifier,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
            **self.transformers_init_kwargs,
        ).cuda()
        self.model.eval()
        self.labels = {
            int(label_id): str(label).lower()
            for label_id, label in getattr(self.model.config, "id2label", {}).items()
        }

    def process_model_output(
        self, outputs: torch.Tensor, _model_input_batch: dict[str, torch.Tensor] | None = None
    ) -> dict[str, np.ndarray]:
        probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        label_ids = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        labels = np.array([self.labels.get(int(label_id), str(label_id).lower()) for label_id in label_ids])
        return {
            PREDICTION_LABEL_FIELD: labels,
            PREDICTION_CONFIDENCE_FIELD: confidences,
        }

    def create_output_dataframe(self, df_cpu: pd.DataFrame, collected_output: dict[str, np.ndarray]) -> pd.DataFrame:
        df_cpu = df_cpu.copy()
        df_cpu[PREDICTION_LABEL_FIELD] = collected_output[PREDICTION_LABEL_FIELD]
        df_cpu[PREDICTION_CONFIDENCE_FIELD] = collected_output[PREDICTION_CONFIDENCE_FIELD]
        return df_cpu


class AssembleModelBasedHTMLExtractionStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    name = "assemble_model_based_html_extraction"
    resources = Resources(cpus=1.0)

    def __init__(
        self,
        stop_lists: dict[str, frozenset[str]],
        output_format: ModelBasedOutputFormat = "markdown",
        fallback_threshold: float = 0.65,
        fallback_extractor: HTMLExtractorAlgorithm | None = None,
        filename_column: str | None = None,
    ):
        self.stop_lists = stop_lists
        self.output_format = output_format
        self.fallback_threshold = fallback_threshold
        self.fallback_extractor = fallback_extractor or TrafilaturaExtractor()
        self.filename_column = filename_column

    def inputs(self) -> tuple[list[str], list[str]]:
        columns = [
            "url",
            "warc_id",
            "source_id",
            "language",
            HTML_FIELD,
            CANDIDATE_INDEX_FIELD,
            CANDIDATE_TAG_NAME_FIELD,
            CANDIDATE_TEXT_FIELD,
            CANDIDATE_HTML_FIELD,
            CANDIDATE_ATTRIBUTES_FIELD,
            PREDICTION_LABEL_FIELD,
            PREDICTION_CONFIDENCE_FIELD,
        ]
        if self.filename_column is not None:
            columns.append(self.filename_column)
        return ["data"], columns

    def outputs(self) -> tuple[list[str], list[str]]:
        columns = ["url", "warc_id", "source_id", "language", "text"]
        if self.filename_column is not None:
            columns.append(self.filename_column)
        return ["data"], columns

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas()
        output_records = []

        group_columns = ["url", "warc_id", "source_id", "language", HTML_FIELD]
        if self.filename_column is not None:
            group_columns.append(self.filename_column)

        for _, group in df.groupby(group_columns, sort=False, dropna=False):
            first_row = group.iloc[0]
            html = str(first_row[HTML_FIELD])
            language = str(first_row["language"])
            stop_words = self.stop_lists.get(language)
            if stop_words is None:
                continue

            rendered_blocks = self._render_blocks(group)
            if rendered_blocks is None:
                rendered_blocks = self.fallback_extractor.extract_text(html, stop_words, language)

            if not rendered_blocks:
                continue

            record = {
                "url": first_row["url"],
                "warc_id": first_row["warc_id"],
                "source_id": first_row["source_id"],
                "language": language,
                "text": "\n\n".join(rendered_blocks),
            }
            if self.filename_column is not None:
                record[self.filename_column] = first_row[self.filename_column]
            output_records.append(record)

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=pd.DataFrame(output_records),
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _render_blocks(self, group: pd.DataFrame) -> list[str] | None:
        candidates = group[group[CANDIDATE_INDEX_FIELD] >= MIN_CANDIDATE_INDEX].sort_values(CANDIDATE_INDEX_FIELD)
        if candidates.empty:
            return None

        accepted: list[tuple[HTMLElement, str]] = []
        accepted_confidences: list[float] = []
        for _, row in candidates.iterrows():
            label = str(row[PREDICTION_LABEL_FIELD]).lower()
            if label in DROP_LABELS:
                continue
            if label not in MAIN_CONTENT_LABELS and label not in STRUCTURED_LABELS:
                continue

            element = deserialize_html_element(row)
            if element is None:
                continue

            accepted.append((element, label))
            accepted_confidences.append(float(row[PREDICTION_CONFIDENCE_FIELD]))

        if not accepted:
            return None

        mean_confidence = sum(accepted_confidences) / len(accepted_confidences)
        if mean_confidence < self.fallback_threshold:
            return None

        rendered_blocks = [ModelBasedHTMLExtractionStage._render_element(element, label) for element, label in accepted]
        rendered_blocks = [block for block in rendered_blocks if block]
        if not rendered_blocks:
            return None

        if self.output_format in {"plain", "plain_text"}:
            rendered_blocks = [ModelBasedHTMLExtractionStage._markdown_block_to_plain_text(block) for block in rendered_blocks]

        return rendered_blocks


class ModelBasedHTMLExtractionStage(HTMLExtractorAlgorithm):
    """Model-based HTML extraction with semantic Markdown conversion."""

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str = "opendatalab/MinerU-HTML-0.6B",
        output_format: ModelBasedOutputFormat = "markdown",
        fallback_threshold: float = 0.65,
        device: Literal["cuda"] = "cuda",
        model_inference_batch_size: int = 64,
        max_length: int = 512,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        classifier: HTMLElementClassifier | None = None,
        fallback_extractor: HTMLExtractorAlgorithm | None = None,
        transformers_init_kwargs: dict[str, Any] | None = None,
        batch_size: int | None = None,
    ):
        if output_format not in {"markdown", "plain", "plain_text"}:
            msg = f"Invalid output_format: {output_format}"
            raise ValueError(msg)
        if not 0 <= fallback_threshold <= 1:
            msg = "fallback_threshold must be between 0 and 1"
            raise ValueError(msg)

        if batch_size is not None:
            model_inference_batch_size = batch_size

        self.model_identifier = model_identifier
        self.output_format = output_format
        self.fallback_threshold = fallback_threshold
        self.device = device
        self.model_inference_batch_size = model_inference_batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.classifier = classifier
        self.fallback_extractor = fallback_extractor or TrafilaturaExtractor()
        self.transformers_init_kwargs = transformers_init_kwargs or {}
        self.resources = Resources(cpus=1.0, gpus=1.0)

    def extract_text(self, html: str, stop_words: frozenset[str], language: str) -> list[str] | None:
        soup = BeautifulSoup(html, "lxml")
        elements = self._extract_candidate_elements(soup)
        if not elements:
            return self.fallback_extractor.extract_text(html, stop_words, language)

        classifier = self._get_classifier()
        if isinstance(classifier, _TransformersHTMLElementClassifier):
            classifier._setup(local_files_only=self.local_files_only)
        predictions = classifier.predict(elements)
        return self._select_and_render_blocks(html, stop_words, language, elements, predictions)

    def _select_and_render_blocks(
        self,
        html: str,
        stop_words: frozenset[str],
        language: str,
        elements: list[HTMLElement],
        predictions: list[HTMLElementPrediction],
    ) -> list[str] | None:
        if len(predictions) != len(elements):
            msg = "HTML element classifier returned a different number of predictions than inputs"
            raise RuntimeError(msg)

        accepted: list[tuple[HTMLElement, str]] = []
        accepted_confidences: list[float] = []
        for element, prediction in zip(elements, predictions, strict=True):
            label = prediction.label.lower()
            if label in DROP_LABELS:
                continue
            if label in MAIN_CONTENT_LABELS or label in STRUCTURED_LABELS:
                accepted.append((element, label))
                accepted_confidences.append(prediction.confidence)

        if not accepted:
            return self.fallback_extractor.extract_text(html, stop_words, language)

        mean_confidence = sum(accepted_confidences) / len(accepted_confidences)
        if mean_confidence < self.fallback_threshold:
            return self.fallback_extractor.extract_text(html, stop_words, language)

        rendered_blocks = [self._render_element(element, label) for element, label in accepted]
        rendered_blocks = [block for block in rendered_blocks if block]
        if not rendered_blocks:
            return self.fallback_extractor.extract_text(html, stop_words, language)

        if self.output_format in {"plain", "plain_text"}:
            rendered_blocks = [self._markdown_block_to_plain_text(block) for block in rendered_blocks]

        return rendered_blocks

    def _get_classifier(self) -> HTMLElementClassifier:
        if self.classifier is None:
            self.classifier = _TransformersHTMLElementClassifier(
                model_identifier=self.model_identifier,
                cache_dir=self.cache_dir,
                device=self.device,
                model_inference_batch_size=self.model_inference_batch_size,
                max_length=self.max_length,
                local_files_only=self.local_files_only,
                transformers_init_kwargs=self.transformers_init_kwargs,
            )
        return self.classifier

    def setup_on_node(self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None) -> None:
        snapshot_download(repo_id=self.model_identifier, cache_dir=self.cache_dir, local_files_only=False)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        classifier = self._get_classifier()
        if isinstance(classifier, _TransformersHTMLElementClassifier):
            classifier._setup(local_files_only=True)

    def teardown(self) -> None:
        classifier = self.classifier
        if isinstance(classifier, _TransformersHTMLElementClassifier):
            classifier._model = None
            classifier._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def ray_stage_spec() -> dict[str, Any]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: True}

    @staticmethod
    def _extract_candidate_elements(soup: BeautifulSoup) -> list[HTMLElement]:
        body = soup.body or soup
        candidates: list[HTMLElement] = []
        seen_text: set[str] = set()
        for tag in body.find_all(list(BLOCK_TAGS)):
            if not isinstance(tag, Tag) or ModelBasedHTMLExtractionStage._has_skip_parent(tag):
                continue
            if tag.name == "code" and tag.find_parent("pre"):
                continue
            if tag.name != "table" and tag.find_parent("table"):
                continue
            if tag.name not in {"pre", "code", "table", "math"} and tag.find(list(BLOCK_TAGS - {"code"})):
                continue

            text = tag.get_text("\n" if tag.name in {"pre", "code", "table"} else " ", strip=True)
            if not text:
                continue
            dedupe_key = f"{tag.name}:{text}"
            if dedupe_key in seen_text:
                continue
            seen_text.add(dedupe_key)
            candidates.append(
                HTMLElement(
                    index=len(candidates),
                    tag_name=tag.name or "",
                    text=text,
                    html=str(tag),
                    attributes={
                        key: " ".join(value) if isinstance(value, list) else str(value)
                        for key, value in tag.attrs.items()
                    },
                )
            )
        return candidates

    @staticmethod
    def _has_skip_parent(tag: Tag) -> bool:
        return any(parent.name in SKIP_TAGS for parent in tag.parents if isinstance(parent, Tag))

    @staticmethod
    def _render_element(element: HTMLElement, label: str) -> str:
        tag = BeautifulSoup(element.html, "lxml").find(element.tag_name)
        if tag is None:
            return element.text

        if label in {"code", "code_block"} or tag.name in {"pre", "code"}:
            rendered = ModelBasedHTMLExtractionStage._render_code(tag)
        elif label == "table" or tag.name == "table":
            rendered = ModelBasedHTMLExtractionStage._render_table(tag)
        elif label in {"formula", "math"} or tag.name == "math":
            rendered = ModelBasedHTMLExtractionStage._render_formula(tag)
        elif (
            tag.name
            and tag.name.startswith("h")
            and len(tag.name) == HEADING_TAG_NAME_LENGTH
            and tag.name[1].isdigit()
        ):
            level = min(int(tag.name[1]), MAX_HEADING_LEVEL)
            rendered = f"{'#' * level} {element.text}"
        elif tag.name == "li":
            rendered = f"- {element.text}"
        elif tag.name == "blockquote":
            rendered = "\n".join(f"> {line}" for line in element.text.splitlines())
        else:
            rendered = element.text
        return rendered

    @staticmethod
    def _render_code(tag: Tag) -> str:
        code_tag = tag.find("code") if tag.name == "pre" else tag
        code = code_tag.get_text("\n", strip=False) if isinstance(code_tag, Tag) else tag.get_text("\n", strip=False)
        language = ""
        class_values = code_tag.get("class", []) if isinstance(code_tag, Tag) else []
        if isinstance(class_values, str):
            class_values = [class_values]
        for class_value in class_values:
            if class_value.startswith("language-"):
                language = class_value.removeprefix("language-")
                break
        max_backtick_run = max((len(match.group()) for match in re.finditer(r"`+", code)), default=0)
        fence = "`" * max(MIN_MARKDOWN_FENCE_LENGTH, max_backtick_run + 1)
        return f"{fence}{language}\n{code.strip()}\n{fence}"

    @staticmethod
    def _render_formula(tag: Tag) -> str:
        formula = tag.get_text(" ", strip=True)
        if formula.startswith("$"):
            return formula
        return f"$$\n{formula}\n$$"

    @staticmethod
    def _render_table(tag: Tag) -> str:
        rows: list[list[str]] = []
        for row in tag.find_all("tr"):
            cells = [cell.get_text(" ", strip=True).replace("|", "\\|") for cell in row.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if not rows:
            return tag.get_text(" ", strip=True)

        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        header = rows[0]
        separator = ["---"] * width
        markdown_rows = [
            f"| {' | '.join(header)} |",
            f"| {' | '.join(separator)} |",
        ]
        markdown_rows.extend(f"| {' | '.join(row)} |" for row in rows[1:])
        return "\n".join(markdown_rows)

    @staticmethod
    def _markdown_block_to_plain_text(block: str) -> str:
        if block.startswith("```"):
            lines = block.splitlines()
            text = "\n".join(lines[1:-1]).strip() if len(lines) >= MIN_FENCED_BLOCK_LINES else ""
        elif block.startswith("$$"):
            text = "\n".join(line for line in block.splitlines() if line != "$$").strip()
        elif block.startswith("#"):
            text = block.lstrip("#").strip()
        elif block.startswith("- "):
            text = block[2:].strip()
        elif block.startswith("> "):
            text = "\n".join(line.removeprefix("> ") for line in block.splitlines())
        elif block.startswith("|"):
            rows = []
            for line in block.splitlines():
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if cells and not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    rows.append("\t".join(cells))
            text = "\n".join(rows)
        else:
            text = block
        return text
