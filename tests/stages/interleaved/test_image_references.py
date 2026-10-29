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

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.interleaved import ImageReferenceExtractor
from nemo_curator.stages.interleaved.image_references import _resolve_url as resolve_url
from nemo_curator.stages.interleaved.image_references import extract_image_references as scan
from nemo_curator.stages.text.io.reader.parquet import ParquetReaderStage
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import DocumentBatch, FileGroupTask


def urls(rows: list[dict]) -> set[str]:
    return {r["url"] for r in rows if r["url"]}


def test_image_mentions():
    cases = [
        ("inline", "![x](https://e.test/a.jpg)", {"https://e.test/a.jpg"}),
        ("reference", "![x][pic]\n\n[pic]: https://e.test/a.jpg", {"https://e.test/a.jpg"}),
        ("escaped", r"\![x](https://e.test/a.jpg)", {"https://e.test/a.jpg"}),
        ("code", "```md\n![x](https://e.test/a.jpg)\n```", {"https://e.test/a.jpg"}),
        ("balanced", "![x](https://e.test/a_(b).jpg)", {"https://e.test/a_(b).jpg"}),
        ("title", '![x](https://e.test/a.jpg "caption")', {"https://e.test/a.jpg"}),
        ("html", '<img alt="A > B" src="https://e.test/a.jpg">', {"https://e.test/a.jpg"}),
        (
            "alternatives",
            '<img src="a.jpg" data-src="b.jpg" srcset="c.jpg 1x, d.jpg 2x">',
            {"https://e.test/a.jpg", "https://e.test/b.jpg", "https://e.test/c.jpg", "https://e.test/d.jpg"},
        ),
        (
            "srcset comma",
            '<img srcset="https://e.test/a,b.jpg 1x, https://e.test/c.jpg 2x">',
            {"https://e.test/a,b.jpg", "https://e.test/c.jpg"},
        ),
        ("extensionless", "![x](/image?id=123)", {"https://e.test/image?id=123"}),
        (
            "literal query ampersands",
            "![x](https://e.test/i?not=1&notebook=2&param=3&timescale=4&amp;y=5)",
            {"https://e.test/i?not=1&notebook=2&param=3&timescale=4&y=5"},
        ),
        ("nested alt", "![A [nested] label](https://e.test/a.jpg)", {"https://e.test/a.jpg"}),
        ("multiline alt", "![A\nmultiline label](https://e.test/a.jpg)", {"https://e.test/a.jpg"}),
        ("escaped title", r'![x](https://e.test/a.jpg "A \"quoted\" title")', {"https://e.test/a.jpg"}),
        ("blank-line alt mention", "![one\n\ntwo](https://e.test/a.jpg)", {"https://e.test/a.jpg"}),
        ("fenced HTML mention", '```html\n<img src="a.jpg">\n```', {"https://e.test/a.jpg"}),
        ("indented image mention", "    ![x](a.jpg)", {"https://e.test/a.jpg"}),
        ("video source", '<source src="clip.mp4" type="video/mp4">', set()),
        ("audio source", '<source src="sound.ogg" type="audio/ogg">', set()),
        ("image source", '<source src="a.webp" type="image/webp">', {"https://e.test/a.webp"}),
        ("unknown source", '<source src="media">', set()),
        ("typographic quotes", "<img src=”a.webp”>", set()),
        ("typographic quotes with spaces", "<img src=“a b.webp”>", set()),
        ("broken markup destination", "![](https://e.test/a.jpg<br%20/>error)", set()),
        ("template destination", "![wink](https://e.test/<#EMO_DIR#>/wink.png)", set()),
        ("srcset malformed markup", '<img srcset="a.jpg&lt;br&gt; 1x, b.jpg 2x">', {"https://e.test/b.jpg"}),
        ("srcset decode once", '<img srcset="a.jpg?x=&amp;amp;y 1x">', {"https://e.test/a.jpg?x=&amp;y"}),
        (
            "BBCode wrapped image",
            "[url=https://other.test/][img] https://e.test/a.jpg [/img][/url]",
            {"https://e.test/a.jpg"},
        ),
        ("BBCode case and relative URL", "[IMG]\n/a.jpg\n[/IMG]", {"https://e.test/a.jpg"}),
        ("BBCode literal query", "[img]https://e.test/a?x=&amp;y[/img]", {"https://e.test/a?x=&amp;y"}),
        ("BBCode incomplete", "[img]https://e.test/a.jpg", set()),
        ("deep URL parentheses", "![x](https://e.test/a_(b_(c_(d))).jpg)", {"https://e.test/a_(b_(c_(d))).jpg"}),
        ("escaped opening bracket in label", r"![x \[ y](a.jpg)", {"https://e.test/a.jpg"}),
        ("escaped closing bracket in label", r"![x \] y](a.jpg)", {"https://e.test/a.jpg"}),
        ("malformed HTML ending", '<img src="a.gif"/&gt ;\ntext<br>', {"https://e.test/a.gif"}),
        ("truncated HTML URL", '<img src="a.gif\ntext<br>', set()),
    ]
    for name, text, expected in cases:
        found, flags = scan(text, "https://e.test/page")
        assert urls(found) == expected, (name, found)
    difficult = "![x](https://e.test/a_(b_(c_(d))).jpg)"
    found, flags = scan(difficult, "https://e.test/page")
    assert urls(found) == {"https://e.test/a_(b_(c_(d))).jpg"}
    assert flags["uncovered_markers"] == 0
    found, flags = scan("<img src=”a.webp”>", "https://e.test/page")
    assert found[0]["raw"] == "”a.webp”"
    assert found[0]["url"] is None
    assert found[0]["candidate_url"] == "https://e.test/a.webp"
    assert found[0]["status"] == "ambiguous_typographic_quotes"
    assert flags["review_candidates"] == 1
    found, _ = scan("![wink](https://e.test/<#EMO_DIR#>/wink.png)", "https://e.test/page")
    assert found[0]["status"] == "invalid_destination_markup"
    assert "<#EMO_DIR#>" in found[0]["raw"]
    found, _ = scan('<img srcset="a.jpg&lt;br&gt; 1x, b.jpg 2x">', "https://e.test/page")
    assert found[0]["status"] == "invalid_destination_markup"
    assert found[0]["source_attribute_raw"] == "a.jpg&lt;br&gt; 1x, b.jpg 2x"
    found, flags = scan("[img]a.jpg[/img] [img]a.jpg[/img] [img]b.jpg", "https://e.test/page")
    assert len(found) == 2
    assert all(r["kind"] == "bbcode_img" for r in found)
    assert found[0]["start"] != found[1]["start"]
    assert flags["uncovered_bbcode_images"] == 1
    source = "é🙂 ![x](a.jpg) and ![x](a.jpg)"
    found, _ = scan(source, "https://e.test/page")
    assert len(found) == 2
    assert found[0]["start"] == 3
    assert all(source[r["start"] : r["end"]] == "![x](a.jpg)" for r in found)
    source = 'é🙂\n<img src="a.gif"/&gt ;\ntext<br>'
    found, _ = scan(source, "https://e.test/page")
    assert found[0]["extraction_method"] == "html_parser_fallback"
    assert found[0]["source_tag_start"] == source.index("<img")
    assert source[found[0]["start"] : found[0]["end"]] == 'src="a.gif"'
    source = '<![%\n\n;[\n\n<!--\n\n<!EN <img src="broken.gif"'
    found, flags = scan("![keep](a.jpg)\n" + source, "https://e.test/page")
    assert urls(found) == {"https://e.test/a.jpg"}
    # Python patch releases differ in whether this malformed declaration raises.
    assert flags["html_fallback_errors"] in {0, 1}
    assert flags["uncovered_html_tags"] == 1
    assert resolve_url("/a.jpg", None)[1] == "missing_page_base"
    assert resolve_url("https://example.org:bad/a.jpg", None)[1] == "invalid_url"
    assert resolve_url("javascript:bad", None)[1] == "unsupported_scheme"


@pytest.mark.parametrize("as_arrow", [False, True])
def test_stage_parquet_roundtrip(tmp_path: Path, as_arrow: bool):
    source = tmp_path / "source.parquet"
    frame = pd.DataFrame(
        {
            "text": ["é🙂 ![x](a.jpg) ![x](a.jpg)", None, "no images"],
            "url": ["https://example.test/page", None, "https://example.test/page"],
            "crawl_id": ["crawl"] * 3,
            "source_file": ["source.parquet"] * 3,
            "row_index": [0, 1, 2],
        }
    )
    frame.to_parquet(source, index=False)
    batch = ParquetReaderStage().process(FileGroupTask(dataset_name="test", data=[str(source)]))
    if as_arrow:
        batch.data = batch.to_pyarrow()
    before = batch.to_pyarrow()
    result = ImageReferenceExtractor().process(batch)
    assert result.data.select(before.column_names).equals(before)
    assert batch.get_columns() == before.column_names
    assert result.dataset_name == batch.dataset_name
    assert "image_reference_diagnostics" not in batch._metadata
    assert {k: result._metadata[k] for k in batch._metadata} == batch._metadata
    assert result.data.column_names == [*before.column_names, "image_references"]
    refs = result.data["image_references"].to_pylist()
    assert [len(r) for r in refs] == [2, 0, 0]
    assert refs[0][0]["url"] == refs[0][1]["url"] == "https://example.test/a.jpg"
    assert refs[0][0]["start"] == 3
    assert refs[0][0]["start"] != refs[0][1]["start"]
    writer = ParquetWriter(path=str(tmp_path / "out"))
    writer.setup()
    written = writer.process(result)
    assert written._metadata["image_reference_diagnostics"] == result._metadata["image_reference_diagnostics"]
    assert pq.read_table(written.data[0]).equals(result.data)


@pytest.mark.parametrize("texts", [[], [None], ["no images"]])
def test_stable_empty_schema(texts: list[str | None]):
    def run(values: list[str | None]) -> pa.Table:
        table = pa.table(
            {"text": pa.array(values, type=pa.string()), "url": pa.array([None] * len(values), type=pa.string())}
        )
        return ImageReferenceExtractor().process(DocumentBatch(dataset_name="test", data=table)).data

    assert run(texts).schema == run(["![x](https://example.test/a)"]).schema


def test_custom_fields_and_invalid_input():
    stage = ImageReferenceExtractor(text_field="markdown", url_field="page_url")
    batch = DocumentBatch(dataset_name="test", data=pa.table({"markdown": ["![x](/a)"], "page_url": [None]}))
    output = stage.process(batch)
    assert output.data["image_references"][0].as_py()[0]["status"] == "missing_page_base"
    with pytest.raises(ValueError, match="refusing to overwrite"):
        stage.process(output)
    with pytest.raises(TypeError, match="expects string or null"):
        stage.process(DocumentBatch(dataset_name="test", data=pa.table({"markdown": [123], "page_url": [None]})))


def test_batch_diagnostics():
    batch = DocumentBatch(
        dataset_name="test",
        data=pa.table(
            {
                "text": ['![x][missing] <img src="a.jpg">', '![y][missing] <img src="b.jpg">'],
                "url": ["https://example.test/page"] * 2,
            }
        ),
        _metadata={"source_files": ["input.parquet"]},
    )
    stage = ImageReferenceExtractor()
    result = stage.process(batch)
    diagnostics = result._metadata["image_reference_diagnostics"]
    assert diagnostics["uncovered_markers"] == 2
    assert diagnostics["unresolved_reference_usages"] == 2
    assert diagnostics["html_attribute_counts"] == {"src": 2}
    assert batch._metadata == {"source_files": ["input.parquet"]}
    empty = DocumentBatch(dataset_name="test", data=batch.data.slice(0, 0))
    assert stage.process(empty)._metadata["image_reference_diagnostics"] == {"html_attribute_counts": {}}


def test_fallback_exception_keeps_other_mentions(monkeypatch: pytest.MonkeyPatch):
    from nemo_curator.stages.interleaved.image_references import ImageTags

    def fail_feed(_self: object, _data: str) -> None:
        msg = "malformed declaration"
        raise ValueError(msg)

    monkeypatch.setattr(ImageTags, "feed", fail_feed)
    rows, flags = scan('![keep](a.jpg) <img src="broken.gif"', "https://e.test/page")
    assert urls(rows) == {"https://e.test/a.jpg"}
    assert flags["html_fallback_errors"] == 1
    assert flags["uncovered_html_tags"] == 1
