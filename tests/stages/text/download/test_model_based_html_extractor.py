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

from nemo_curator.stages.text.download.common_crawl.extract import CommonCrawlHTMLExtractor
from nemo_curator.stages.text.download.html_extractors import (
    HTMLElement,
    HTMLElementPrediction,
    ModelBasedHTMLExtractionStage,
)
from nemo_curator.stages.text.download.html_extractors.base import HTMLExtractorAlgorithm
from nemo_curator.stages.resources import Resources


class FakeElementClassifier:
    def __init__(self, predictions_by_tag: dict[str, HTMLElementPrediction]):
        self.predictions_by_tag = predictions_by_tag

    def predict(self, elements: list[HTMLElement]) -> list[HTMLElementPrediction]:
        return [
            self.predictions_by_tag.get(element.tag_name, HTMLElementPrediction(label="boilerplate", confidence=0.99))
            for element in elements
        ]


class FakeFallbackExtractor(HTMLExtractorAlgorithm):
    def extract_text(self, _html: str, _stop_words: frozenset[str], _language: str) -> list[str] | None:
        return ["fallback text"]


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


def test_model_based_extractor_preserves_structured_markdown() -> None:
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
    classifier = FakeElementClassifier(
        {
            "h1": HTMLElementPrediction(label="main_content", confidence=0.99),
            "p": HTMLElementPrediction(label="main_content", confidence=0.99),
            "pre": HTMLElementPrediction(label="code_block", confidence=0.99),
            "math": HTMLElementPrediction(label="formula", confidence=0.99),
            "table": HTMLElementPrediction(label="table", confidence=0.99),
        }
    )
    extractor = ModelBasedHTMLExtractionStage(classifier=classifier, fallback_threshold=0.5)

    result = extractor.extract_text(html, frozenset({"the", "and", "with"}), "ENGLISH")

    assert result == [
        "# Example Article",
        "This is the main article body with useful words and context.",
        '```python\nprint("hello")\n```',
        "$$\nx = y + 1\n$$",
        "| Name | Value |\n| --- | --- |\n| alpha | 1 |",
    ]


def test_model_based_extractor_falls_back_when_confidence_is_low() -> None:
    classifier = FakeElementClassifier(
        {
            "p": HTMLElementPrediction(label="main_content", confidence=0.1),
        }
    )
    extractor = ModelBasedHTMLExtractionStage(
        classifier=classifier,
        fallback_extractor=FakeFallbackExtractor(),
        fallback_threshold=0.8,
    )

    result = extractor.extract_text(
        "<html><body><p>Low confidence main content.</p></body></html>",
        frozenset(),
        "ENGLISH",
    )

    assert result == ["fallback text"]


def test_common_crawl_accepts_model_based_algorithm_string() -> None:
    extractor = CommonCrawlHTMLExtractor(
        algorithm="model",
        algorithm_kwargs={
            "classifier": FakeElementClassifier(
                {"p": HTMLElementPrediction(label="main_content", confidence=0.99)}
            ),
            "fallback_threshold": 0.5,
        },
    )

    assert isinstance(extractor.algorithm, ModelBasedHTMLExtractionStage)


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


def test_model_based_extractor_plain_text_table_drops_alignment_separator() -> None:
    classifier = FakeElementClassifier(
        {
            "table": HTMLElementPrediction(label="table", confidence=0.99),
        }
    )
    extractor = ModelBasedHTMLExtractionStage(classifier=classifier, output_format="plain_text")

    plain_text = extractor.extract_text(
        """
        <html><body>
          <table>
            <tr><th>Name</th><th>Value</th></tr>
            <tr><td>alpha</td><td>1</td></tr>
          </table>
        </body></html>
        """,
        frozenset(),
        "ENGLISH",
    )

    assert plain_text == ["Name\tValue\nalpha\t1"]


def test_model_based_extractor_plain_text_code_preserves_backtick_lines() -> None:
    classifier = FakeElementClassifier(
        {
            "pre": HTMLElementPrediction(label="code_block", confidence=0.99),
        }
    )
    extractor = ModelBasedHTMLExtractionStage(classifier=classifier, output_format="plain_text")

    plain_text = extractor.extract_text(
        """
        <html><body>
          <pre><code>first line
```literal
last line</code></pre>
        </body></html>
        """,
        frozenset(),
        "ENGLISH",
    )

    assert plain_text == ["first line\n```literal\nlast line"]
