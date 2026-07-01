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

import pytest
from bs4 import BeautifulSoup

from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.download.common_crawl.extract import CommonCrawlHTMLExtractor
from nemo_curator.stages.text.download.html_extractors import (
    HTMLElement,
    HTMLElementPrediction,
    ModelBasedHTMLExtractionStage,
)
from nemo_curator.stages.text.download.html_extractors.base import HTMLExtractorAlgorithm
from nemo_curator.stages.text.download.html_extractors.model_based import (
    extract_candidate_elements,
    render_blocks_from_predictions,
)


class LifecycleTrackingAlgorithm(HTMLExtractorAlgorithm):
    def __init__(self) -> None:
        self.resources = Resources(cpus=1.0, gpus=1.0)
        self.setup_on_node_called = False
        self.setup_called = False
        self.teardown_called = False

    def extract_text(self, html: str, stop_words: frozenset[str], language: str) -> list[str] | None:
        return [html]

    def setup_on_node(self, *args, **kwargs) -> None:
        self.setup_on_node_called = True

    def setup(self, *args, **kwargs) -> None:
        self.setup_called = True

    def teardown(self) -> None:
        self.teardown_called = True

    @staticmethod
    def ray_stage_spec() -> dict[str, bool]:
        return {"is_actor_stage": True}


def _predict_for_elements(
    elements: list[HTMLElement], predictions_by_tag: dict[str, HTMLElementPrediction]
) -> list[HTMLElementPrediction]:
    return [
        predictions_by_tag.get(element.tag_name, HTMLElementPrediction(label="boilerplate", confidence=0.99))
        for element in elements
    ]


def test_model_based_helpers_preserve_structured_markdown() -> None:
    html = """
    <html>
      <body>
        <nav>Home Pricing Login</nav>
        <main>
          <h1>Example Article</h1>
          <p>This is the main article body with useful words and context.</p>
          <pre><code class="language-python">print("hello")</code></pre>
          <math>x = y + 1</math>
          <table>
            <tr><th>Name</th><th>Value</th></tr>
            <tr><td>alpha</td><td>1</td></tr>
          </table>
        </main>
        <footer>Copyright 2026</footer>
      </body>
    </html>
    """
    elements = extract_candidate_elements(BeautifulSoup(html, "lxml"))
    predictions = _predict_for_elements(
        elements,
        {
            "h1": HTMLElementPrediction(label="main_content", confidence=0.99),
            "p": HTMLElementPrediction(label="main_content", confidence=0.99),
            "pre": HTMLElementPrediction(label="code_block", confidence=0.99),
            "math": HTMLElementPrediction(label="formula", confidence=0.99),
            "table": HTMLElementPrediction(label="table", confidence=0.99),
        }
    )
    result = render_blocks_from_predictions(
        elements=elements,
        predictions=predictions,
        output_format="markdown",
        fallback_threshold=0.5,
    )

    assert result == [
        "# Example Article",
        "This is the main article body with useful words and context.",
        '```python\nprint("hello")\n```',
        "$$\nx = y + 1\n$$",
        "| Name | Value |\n| --- | --- |\n| alpha | 1 |",
    ]


def test_model_based_helpers_return_none_when_confidence_is_low() -> None:
    html = "<html><body><p>Low confidence main content.</p></body></html>"
    elements = extract_candidate_elements(BeautifulSoup(html, "lxml"))
    predictions = _predict_for_elements(
        elements,
        {
            "p": HTMLElementPrediction(label="main_content", confidence=0.1),
        }
    )
    result = render_blocks_from_predictions(
        elements=elements,
        predictions=predictions,
        output_format="markdown",
        fallback_threshold=0.8,
    )

    assert result is None


def test_model_based_extractor_direct_use_is_not_supported() -> None:
    extractor = ModelBasedHTMLExtractionStage()

    with pytest.raises(NotImplementedError, match="configuration wrapper"):
        extractor.extract_text("<html><body><p>content</p></body></html>", frozenset(), "ENGLISH")


def test_common_crawl_direct_model_based_algorithm_string_is_not_supported() -> None:
    with pytest.raises(ValueError, match="only supported through CommonCrawlDownloadExtractStage"):
        CommonCrawlHTMLExtractor(
            algorithm="model",
            algorithm_kwargs={
                "fallback_threshold": 0.5,
            },
        )


def test_common_crawl_extractor_delegates_lifecycle_and_resources() -> None:
    algorithm = LifecycleTrackingAlgorithm()
    extractor = CommonCrawlHTMLExtractor(algorithm=algorithm)

    extractor.setup_on_node()
    extractor.setup()
    extractor.teardown()

    assert extractor.resources == algorithm.resources
    assert extractor.ray_stage_spec() == {"is_actor_stage": True}
    assert algorithm.setup_on_node_called
    assert algorithm.setup_called
    assert algorithm.teardown_called


def test_model_based_helpers_plain_text_table_drop_alignment_separator() -> None:
    html = """
        <html><body>
          <table>
            <tr><th>Name</th><th>Value</th></tr>
            <tr><td>alpha</td><td>1</td></tr>
          </table>
        </body></html>
        """
    elements = extract_candidate_elements(BeautifulSoup(html, "lxml"))
    predictions = _predict_for_elements(
        elements,
        {
            "table": HTMLElementPrediction(label="table", confidence=0.99),
        }
    )
    plain_text = render_blocks_from_predictions(
        elements=elements,
        predictions=predictions,
        output_format="plain_text",
        fallback_threshold=0.5,
    )

    assert plain_text == ["Name\tValue\nalpha\t1"]


def test_model_based_helpers_plain_text_code_preserve_backtick_lines() -> None:
    html = """
        <html><body>
          <pre><code>first line
```literal
last line</code></pre>
        </body></html>
        """
    elements = extract_candidate_elements(BeautifulSoup(html, "lxml"))
    predictions = _predict_for_elements(
        elements,
        {
            "pre": HTMLElementPrediction(label="code_block", confidence=0.99),
        }
    )
    plain_text = render_blocks_from_predictions(
        elements=elements,
        predictions=predictions,
        output_format="plain_text",
        fallback_threshold=0.5,
    )

    assert plain_text == ["first line\n```literal\nlast line"]
