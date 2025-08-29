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

import pytest

from nemo_curator.stages.text.download.html_extractors import (
    JusTextExtractor,
    ResiliparseExtractor,
    TrafilaturaExtractor,
)
from nemo_curator.stages.text.download.html_extractors.utils import get_stop_list_dict


@pytest.fixture
def html_string() -> str:
    # Modified from https://github.com/chatnoir-eu/chatnoir-resiliparse/blob/abdf1966fb3cefe3e0790e510ab5cb1446f99a79/tests/resiliparse/extract/test_html2text.py
    return """<!doctype html>
        <head>
            <title>My Title</title>
            <meta charset="utf-8">
            <style>* { margin: 0; }</style>
        </head>
        <body>
            <section id="wrapper">
                <nav>
                    <ul>
                        <li>Nav 1</li>
                        <li>
                            <p>Nav 2</p>
                            <ul>
                                <li><p>Nav 3</p></li>
                            </ul>
                        </li>
                    </ul>
                </nav>
                <main>
                    This is a sample paragraph. In it we write words.
                    These are stopwords: because did than has near we almost while what still.
                    <a href="#foo" hidden>bar</a>

                    <p>
                    This paragraph doesn't have many stopwords. Remove it.
                    <br>Let's keep this paragraph: either came does last new took taken making became from.
                    </p>

                    <button aria-hidden="true">Click here</button>
                    <input type="hidden" value="foo">
                    <input type="text" value="Some text" placeholder="Insert text">
                    <input type="text" placeholder="Insert text">
                    <img src="" alt="Some image">
                    <object data="" class="some-class hidden">Cannot display object</object>
                </main>
                <script language="vbscript" type="text/vbscript">MsgBox("Hello World!")</script>
                <noscript>Sorry, your browser doesn't support VB Script!</noscript>
                <div><div><div><footer id="global-footer">
                    Copyright (C) 2021 Foo Bar
                </footer></div></div></div>
            </section>
        </body>
    </html>"""


class TestHTMLExtractors:
    """Test suite for individual HTML extraction algorithms."""

    @pytest.mark.parametrize("extraction_algorithm", ["justext", "resiliparse", "trafilatura"])
    def test_extract_english_text(
        self, html_string: str, extraction_algorithm: Literal["justext", "resiliparse", "trafilatura"]
    ) -> None:
        """Test English text extraction with different algorithms."""
        if extraction_algorithm == "justext":
            algorithm = JusTextExtractor(is_boilerplate=False)
            expected = [
                "Nav 1",
                "Nav 2",
                "Nav 3",
                "This is a sample paragraph. In it we write words.\nThese are stopwords: because did than has near we almost while what still.\nbar",
                "This paragraph doesn't have many stopwords. Remove it.\nLet's keep this paragraph: either came does last new took taken making became from.",
                "Cannot display object\nSorry, your browser doesn't support VB Script!",
                "Copyright (C) 2021 Foo Bar",
            ]
        elif extraction_algorithm == "resiliparse":
            algorithm = ResiliparseExtractor()
            expected = [
                "This is a sample paragraph. In it we write words. These are stopwords: because did than has near we almost while what still.",
                "Let's keep this paragraph: either came does last new took taken making became from.",
            ]
        elif extraction_algorithm == "trafilatura":
            algorithm = TrafilaturaExtractor(
                min_extracted_size=10,
                min_duplcheck_size=10,
                max_repetitions=1,
                deduplicate=True,
            )
            expected = [
                "Let's keep this paragraph: either came does last new took taken making became from.",
            ]

        stop_words = get_stop_list_dict()
        result = algorithm.extract_text(html_string, stop_words["ENGLISH"], "ENGLISH")

        assert result == expected

    @pytest.mark.parametrize("extraction_algorithm", ["justext", "resiliparse", "trafilatura"])
    def test_extract_simple_english_text(
        self, extraction_algorithm: Literal["justext", "resiliparse", "trafilatura"]
    ) -> None:
        """Test simple English text extraction with different algorithms."""
        simple_html = "<html><body><p>Common Crawl test paragraph for extraction. Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal.</p></body></html>"

        if extraction_algorithm == "justext":
            algorithm = JusTextExtractor()
            expected = [
                "Common Crawl test paragraph for extraction. Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal.",
            ]
        elif extraction_algorithm == "resiliparse":
            algorithm = ResiliparseExtractor()
            expected = [
                "Common Crawl test paragraph for extraction. Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal.",
            ]
        elif extraction_algorithm == "trafilatura":
            algorithm = TrafilaturaExtractor()
            expected = [
                "Common Crawl test paragraph for extraction. Four score and seven years ago our fathers brought forth on this continent a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal.",
            ]

        stop_words = get_stop_list_dict()
        result = algorithm.extract_text(simple_html, stop_words["ENGLISH"], "ENGLISH")

        assert result == expected

    @pytest.mark.parametrize("extraction_algorithm", ["justext", "resiliparse", "trafilatura"])
    def test_extract_thai_text(self, extraction_algorithm: Literal["justext", "resiliparse", "trafilatura"]) -> None:
        thai_html = """<!doctype html>
            <head>
                <title>ชื่อเรื่องของฉัน</title>
            </head>
            <body>
                    <main>
                        นี่คือตัวอย่างย่อหน้า ในนั้นเราเขียนคำต่างๆ
                        เหล่านี้เป็นคำหยุด: เพราะว่า ทำ กว่า มี ใกล้ เรา เกือบจะ ขณะที่ อะไร ยังคง

                        <p>
                        ย่อหน้านี้ไม่มีคำหยุดมากนัก ลบออก
                        <br>เรามาเก็บย่อหน้าไว้ดังนี้: ไม่ว่าจะมาทำอะไรใหม่ ๆ ก็เกิดขึ้น เกิดขึ้นจาก
                        </p>

                    </main>
            </body>
        </html>"""

        if extraction_algorithm == "justext":
            algorithm = JusTextExtractor()
            expected = [
                "นี่คือตัวอย่างย่อหน้า ในนั้นเราเขียนคำต่างๆ\nเหล่านี้เป็นคำหยุด: เพราะว่า ทำ กว่า มี ใกล้ เรา เกือบจะ ขณะที่ อะไร ยังคง",
                "ย่อหน้านี้ไม่มีคำหยุดมากนัก ลบออก\nเรามาเก็บย่อหน้าไว้ดังนี้: ไม่ว่าจะมาทำอะไรใหม่ ๆ ก็เกิดขึ้น เกิดขึ้นจาก",
            ]
        elif extraction_algorithm == "resiliparse":
            algorithm = ResiliparseExtractor()
            expected = [
                "นี่คือตัวอย่างย่อหน้า ในนั้นเราเขียนคำต่างๆ เหล่านี้เป็นคำหยุด: เพราะว่า ทำ กว่า มี ใกล้ เรา เกือบจะ ขณะที่ อะไร ยังคง",
                "ย่อหน้านี้ไม่มีคำหยุดมากนัก ลบออก",
                "เรามาเก็บย่อหน้าไว้ดังนี้: ไม่ว่าจะมาทำอะไรใหม่ ๆ ก็เกิดขึ้น เกิดขึ้นจาก",
            ]
        elif extraction_algorithm == "trafilatura":
            algorithm = TrafilaturaExtractor()
            expected = [
                "ย่อหน้านี้ไม่มีคำหยุดมากนัก ลบออก",
                "เรามาเก็บย่อหน้าไว้ดังนี้: ไม่ว่าจะมาทำอะไรใหม่ ๆ ก็เกิดขึ้น เกิดขึ้นจาก",
                "ย่อหน้านี้ไม่มีคำหยุดมากนัก ลบออก",
                "เรามาเก็บย่อหน้าไว้ดังนี้: ไม่ว่าจะมาทำอะไรใหม่ ๆ ก็เกิดขึ้น เกิดขึ้นจาก",
            ]

        stop_words = get_stop_list_dict()
        result = algorithm.extract_text(thai_html, stop_words["THAI"], "THAI")

        assert result == expected

    @pytest.mark.parametrize("extraction_algorithm", ["justext", "resiliparse", "trafilatura"])
    def test_extract_chinese_text(
        self, extraction_algorithm: Literal["justext", "resiliparse", "trafilatura"]
    ) -> None:
        chinese_html = """<!doctype html>
            <head>
                <title>我的标题</title>
            </head>
            <body>
                    <main>
                        这是一个示例段落。我们在其中写下单词。

                        <p>
                        本段落没有太多停用词。请将其删除。
                        <br>让我们保留这一段：要么来了，要么最后来了，要么新来了，要么采取了行动。
                        </p>

                    </main>
            </body>
        </html>"""  # noqa: RUF001

        if extraction_algorithm == "justext":
            algorithm = JusTextExtractor()
            expected = [
                "这是一个示例段落。我们在其中写下单词。",
                "本段落没有太多停用词。请将其删除。\n让我们保留这一段：要么来了，要么最后来了，要么新来了，要么采取了行动。",  # noqa: RUF001
            ]
        elif extraction_algorithm == "resiliparse":
            algorithm = ResiliparseExtractor()
            expected = [
                "这是一个示例段落。我们在其中写下单词。",
                "本段落没有太多停用词。请将其删除。",
                "让我们保留这一段：要么来了，要么最后来了，要么新来了，要么采取了行动。",  # noqa: RUF001
            ]
        elif extraction_algorithm == "trafilatura":
            algorithm = TrafilaturaExtractor()
            expected = [
                "这是一个示例段落。我们在其中写下单词。",
                "本段落没有太多停用词。请将其删除。",
                "让我们保留这一段：要么来了，要么最后来了，要么新来了，要么采取了行动。",  # noqa: RUF001
            ]

        stop_words = get_stop_list_dict()
        result = algorithm.extract_text(chinese_html, stop_words["CHINESE"], "CHINESE")

        assert result == expected

    @pytest.mark.parametrize("extraction_algorithm", ["justext", "resiliparse", "trafilatura"])
    def test_extract_japanese_text(
        self, extraction_algorithm: Literal["justext", "resiliparse", "trafilatura"]
    ) -> None:
        japanese_html = """<!doctype html>
            <head>
                <title>私のタイトル</title>
            </head>
            <body>
                    <main>
                        これはサンプルの段落です。ここに単語を書き込みます。

                        <p>
                        この段落にはストップワードがあまりありません。削除してください。
                        <br>この段落を維持しましょう: どちらかが来て、最後に新しいものを取って、作成し、なったのです。
                        </p>

                    </main>
            </body>
        </html>"""

        if extraction_algorithm == "justext":
            algorithm = JusTextExtractor()
            expected = [
                "これはサンプルの段落です。ここに単語を書き込みます。",
                "この段落にはストップワードがあまりありません。削除してください。\nこの段落を維持しましょう: どちらかが来て、最後に新しいものを取って、作成し、なったのです。",
            ]
        elif extraction_algorithm == "resiliparse":
            algorithm = ResiliparseExtractor()
            expected = [
                "これはサンプルの段落です。ここに単語を書き込みます。",
                "この段落にはストップワードがあまりありません。削除してください。",
                "この段落を維持しましょう: どちらかが来て、最後に新しいものを取って、作成し、なったのです。",
            ]
        elif extraction_algorithm == "trafilatura":
            algorithm = TrafilaturaExtractor()
            expected = [
                "この段落にはストップワードがあまりありません。削除してください。",
                "この段落を維持しましょう: どちらかが来て、最後に新しいものを取って、作成し、なったのです。",
                "この段落にはストップワードがあまりありません。削除してください。",
                "この段落を維持しましょう: どちらかが来て、最後に新しいものを取って、作成し、なったのです。",
            ]

        stop_words = get_stop_list_dict()
        result = algorithm.extract_text(japanese_html, stop_words["JAPANESE"], "JAPANESE")

        assert result == expected

    @pytest.mark.parametrize("extraction_algorithm", ["justext", "resiliparse", "trafilatura"])
    def test_extract_korean_text(self, extraction_algorithm: Literal["justext", "resiliparse", "trafilatura"]) -> None:
        korean_html = """<!doctype html>
            <head>
                <title>내 제목</title>
            </head>
            <body>
                    <main>
                        이것은 샘플 문단입니다. 여기에 단어를 적습니다.
                        이것들은 불용어입니다: 왜냐하면, 했으므로, 보다, 가까이에, 우리, 거의, 동안, 무엇, 아직도.

                        <p>
                        이 문단에는 불용어가 많지 않습니다. 제거하세요.
                        <br>이 문단을 유지해 보겠습니다: 왔거나 마지막이거나 새로운 것이거나 가져갔거나 만들어지거나 되었거나에서 왔습니다.
                        </p>

                    </main>
            </body>
        </html>"""

        if extraction_algorithm == "justext":
            algorithm = JusTextExtractor()
            expected = [
                "이것은 샘플 문단입니다. 여기에 단어를 적습니다.\n이것들은 불용어입니다: 왜냐하면, 했으므로, 보다, 가까이에, 우리, 거의, 동안, 무엇, 아직도.",
                "이 문단에는 불용어가 많지 않습니다. 제거하세요.\n이 문단을 유지해 보겠습니다: 왔거나 마지막이거나 새로운 것이거나 가져갔거나 만들어지거나 되었거나에서 왔습니다.",
            ]
        elif extraction_algorithm == "resiliparse":
            algorithm = ResiliparseExtractor()
            expected = [
                "이것은 샘플 문단입니다. 여기에 단어를 적습니다. 이것들은 불용어입니다: 왜냐하면, 했으므로, 보다, 가까이에, 우리, 거의, 동안, 무엇, 아직도.",
                "이 문단에는 불용어가 많지 않습니다. 제거하세요.",
                "이 문단을 유지해 보겠습니다: 왔거나 마지막이거나 새로운 것이거나 가져갔거나 만들어지거나 되었거나에서 왔습니다.",
            ]
        elif extraction_algorithm == "trafilatura":
            algorithm = TrafilaturaExtractor()
            expected = [
                "이 문단에는 불용어가 많지 않습니다. 제거하세요.",
                "이 문단을 유지해 보겠습니다: 왔거나 마지막이거나 새로운 것이거나 가져갔거나 만들어지거나 되었거나에서 왔습니다.",
                "이 문단에는 불용어가 많지 않습니다. 제거하세요.",
                "이 문단을 유지해 보겠습니다: 왔거나 마지막이거나 새로운 것이거나 가져갔거나 만들어지거나 되었거나에서 왔습니다.",
            ]

        stop_words = get_stop_list_dict()
        result = algorithm.extract_text(korean_html, stop_words["KOREAN"], "KOREAN")

        assert result == expected
