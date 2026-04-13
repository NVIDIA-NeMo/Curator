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
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from bs4 import BeautifulSoup
from bs4.element import Tag
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import HTMLExtractorAlgorithm
from .trafilatura import TrafilaturaExtractor

ModelBasedOutputFormat = Literal["markdown", "plain", "plain_text"]

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


class HTMLElementClassifier(Protocol):
    """Predict semantic labels for a batch of HTML elements."""

    def predict(self, elements: list[HTMLElement]) -> list[HTMLElementPrediction]:
        ...


class _TransformersHTMLElementClassifier:
    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        cache_dir: str | None,
        device: Literal["cuda", "cpu"],
        batch_size: int,
        max_length: int,
        local_files_only: bool,
        transformers_init_kwargs: dict[str, Any],
    ):
        self.model_identifier = model_identifier
        self.cache_dir = cache_dir
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.local_files_only = local_files_only
        self.transformers_init_kwargs = transformers_init_kwargs
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def _setup(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        if self.device == "cuda" and not torch.cuda.is_available():
            msg = "CUDA requested for model-based HTML extraction, but CUDA is unavailable."
            raise RuntimeError(msg)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_identifier,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            **self.transformers_init_kwargs,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_identifier,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            **self.transformers_init_kwargs,
        )
        self._model.to(self.device)
        self._model.eval()

    def predict(self, elements: list[HTMLElement]) -> list[HTMLElementPrediction]:
        self._setup()

        if self._model is None or self._tokenizer is None:
            msg = "Model-based HTML classifier was not initialized"
            raise RuntimeError(msg)

        model = self._model
        tokenizer = self._tokenizer

        predictions: list[HTMLElementPrediction] = []
        for start in range(0, len(elements), self.batch_size):
            batch = elements[start : start + self.batch_size]
            model_inputs = [self._format_element(element) for element in batch]
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

    @staticmethod
    def _format_element(element: HTMLElement) -> str:
        attributes = " ".join(f'{key}="{value}"' for key, value in element.attributes.items())
        return f"<{element.tag_name} {attributes}> {element.text}"


class ModelBasedHTMLExtractionStage(HTMLExtractorAlgorithm):
    """Model-based HTML extraction with semantic Markdown conversion.

    This extractor classifies candidate HTML elements as main content, boilerplate, navigation, ads,
    code, tables, and math/formulas, then converts the accepted structure to Markdown or plain text.
    The Hugging Face model is loaded lazily on first use so constructing Common Crawl stages remains cheap.
    Tests and downstream users can inject any object implementing ``HTMLElementClassifier``.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str = "opendatalab/MinerU-HTML-0.6B",
        output_format: ModelBasedOutputFormat = "markdown",
        fallback_threshold: float = 0.65,
        device: Literal["cuda", "cpu"] = "cuda",
        batch_size: int = 64,
        max_length: int = 512,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        classifier: HTMLElementClassifier | None = None,
        fallback_extractor: HTMLExtractorAlgorithm | None = None,
        transformers_init_kwargs: dict[str, Any] | None = None,
    ):
        if output_format not in {"markdown", "plain", "plain_text"}:
            msg = f"Invalid output_format: {output_format}"
            raise ValueError(msg)
        if not 0 <= fallback_threshold <= 1:
            msg = "fallback_threshold must be between 0 and 1"
            raise ValueError(msg)

        self.model_identifier = model_identifier
        self.output_format = output_format
        self.fallback_threshold = fallback_threshold
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.classifier = classifier
        self.fallback_extractor = fallback_extractor or TrafilaturaExtractor()
        self.transformers_init_kwargs = transformers_init_kwargs or {}

    def extract_text(self, html: str, stop_words: frozenset[str], language: str) -> list[str] | None:
        soup = BeautifulSoup(html, "lxml")
        elements = self._extract_candidate_elements(soup)
        if not elements:
            return self.fallback_extractor.extract_text(html, stop_words, language)

        predictions = self._get_classifier().predict(elements)
        if len(predictions) != len(elements):
            msg = "HTML element classifier returned a different number of predictions than inputs"
            raise RuntimeError(msg)

        accepted: list[tuple[HTMLElement, HTMLElementPrediction]] = []
        accepted_confidences: list[float] = []
        for element, prediction in zip(elements, predictions, strict=True):
            label = prediction.label.lower()
            if label in DROP_LABELS:
                continue
            if label in MAIN_CONTENT_LABELS or label in STRUCTURED_LABELS:
                accepted.append((element, prediction))
                accepted_confidences.append(prediction.confidence)

        if not accepted:
            return self.fallback_extractor.extract_text(html, stop_words, language)

        mean_confidence = sum(accepted_confidences) / len(accepted_confidences)
        if mean_confidence < self.fallback_threshold:
            return self.fallback_extractor.extract_text(html, stop_words, language)

        rendered_blocks = [
            self._render_element(element, prediction.label.lower()) for element, prediction in accepted
        ]
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
                batch_size=self.batch_size,
                max_length=self.max_length,
                local_files_only=self.local_files_only,
                transformers_init_kwargs=self.transformers_init_kwargs,
            )
        return self.classifier

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
