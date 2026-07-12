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

from nemo_curator.stages.text.download.wikipedia.extract import WikipediaExtractor


class TestWikipediaExtractor:
    """Test suite for WikipediaExtractor."""

    def test_init_default_language(self):
        """Test initialization with default language."""
        extractor = WikipediaExtractor()
        assert extractor._language == "en"

    def test_init_custom_language(self):
        """Test initialization with custom language."""
        extractor = WikipediaExtractor(language="es")
        assert extractor._language == "es"

    def test_extract_valid_wikipedia_article(self):
        """Test extraction of valid Wikipedia article with basic markup."""
        extractor = WikipediaExtractor(language="en")

        record = {
            "title": "Test Article",
            "id": "12345",
            "url": "https://en.wikipedia.org/wiki/Test_Article",
            "language": "en",
            "source_id": "enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "raw_content": """'''Test Article''' is a test article.

== Section 1 ==
This is the first section with some [[Wikipedia:Link|links]] and '''bold text'''.

== Section 2 ==
This section has ''italic text'' and some references.<ref>A reference</ref>

=== Subsection ===
More content here.

== External links ==
* [http://example.com External link]

[[Category:Test]]
[[es:Artículo de prueba]]
""",
        }

        result = extractor.extract(record)

        assert result is not None
        assert result["title"] == "Test Article"
        assert result["id"] == "12345"
        assert result["url"] == "https://en.wikipedia.org/wiki/Test_Article"
        assert result["language"] == "en"
        assert result["source_id"] == "enwiki-20230501-pages-articles-multistream1.xml.bz2"
        assert "text" in result
        assert "raw_content" not in result

        # Check that the text contains the main content
        assert "Test Article is a test article" in result["text"]
        assert "first section" in result["text"]
        assert "italic text" in result["text"]

        # Check that markup elements are removed or processed
        assert "[[Category:Test]]" not in result["text"]
        assert "[[es:Artículo de prueba]]" not in result["text"]

    def test_extract_article_with_infobox(self):
        """Test extraction of article with infobox."""
        extractor = WikipediaExtractor(language="en")

        record = {
            "title": "Person",
            "id": "67890",
            "url": "https://en.wikipedia.org/wiki/Person",
            "language": "en",
            "source_id": "enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "raw_content": """{{Infobox person
| name = Test Person
| birth_date = 1980
| occupation = Test
}}

'''Test Person''' (born 1980) is a test person.

== Early life ==
Test Person was born in 1980.

== Career ==
Test Person works in testing.
""",
        }

        result = extractor.extract(record)

        assert result is not None
        assert result["title"] == "Person"
        assert "Test Person" in result["text"]
        assert "Early life" in result["text"]
        assert "Career" in result["text"]

        # Infobox should be removed or processed
        assert "{{Infobox person" not in result["text"]

    def test_extract_article_with_templates(self):
        """Test extraction of article with various templates."""
        extractor = WikipediaExtractor(language="en")

        record = {
            "title": "Template Test",
            "id": "11111",
            "url": "https://en.wikipedia.org/wiki/Template_Test",
            "language": "en",
            "source_id": "enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "raw_content": """{{Disambiguation}}

'''Template Test''' may refer to:

* [[Template Test 1]] - First meaning
* [[Template Test 2]] - Second meaning

{{Main|Template Test 1}}

See also {{Template Test 3}}.
""",
        }

        result = extractor.extract(record)

        assert result is not None
        assert result["title"] == "Template Test"
        assert "Template Test may refer to" in result["text"]
        assert "First meaning" in result["text"]
        assert "Second meaning" in result["text"]

    def test_extract_article_with_references(self):
        """Test extraction of article with references and citations."""
        extractor = WikipediaExtractor(language="en")

        record = {
            "title": "Reference Test",
            "id": "22222",
            "url": "https://en.wikipedia.org/wiki/Reference_Test",
            "language": "en",
            "source_id": "enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "raw_content": """'''Reference Test''' is a test article.<ref>{{cite book|title=Test Book|author=Test Author}}</ref>

According to scholars,<ref name="test">Test citation</ref> this is important.

== References ==
<references/>
""",
        }

        result = extractor.extract(record)

        assert result is not None
        assert result["title"] == "Reference Test"
        assert "Reference Test is a test article" in result["text"]
        assert "According to scholars" in result["text"]

        # References section might be removed or processed
        assert "<references/>" not in result["text"]

    def test_extract_empty_content(self):
        """Test extraction with empty raw content."""
        extractor = WikipediaExtractor(language="en")

        record = {
            "title": "Empty Test",
            "id": "33333",
            "url": "https://en.wikipedia.org/wiki/Empty_Test",
            "language": "en",
            "source_id": "enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "raw_content": "",
        }

        result = extractor.extract(record)

        # Empty content should return None according to the implementation
        assert result is None

    def test_extract_missing_raw_content(self):
        """Test extraction with missing raw_content field."""
        extractor = WikipediaExtractor(language="en")

        record = {
            "title": "Missing Content",
            "id": "44444",
            "url": "https://en.wikipedia.org/wiki/Missing_Content",
            "language": "en",
            "source_id": "enwiki-20230501-pages-articles-multistream1.xml.bz2",
        }

        result = extractor.extract(record)

        assert result is None

    def test_extract_redirect_page(self):
        """Test extraction of redirect page."""
        extractor = WikipediaExtractor(language="en")

        record = {
            "title": "Redirect Test",
            "id": "55555",
            "url": "https://en.wikipedia.org/wiki/Redirect_Test",
            "language": "en",
            "source_id": "enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "raw_content": "#REDIRECT [[Target Article]]",
        }

        result = extractor.extract(record)

        # Redirect pages might be processed differently
        assert result is not None
        assert result["title"] == "Redirect Test"

    def test_extract_disambiguation_page(self):
        """Test extraction of disambiguation page."""
        extractor = WikipediaExtractor(language="en")

        record = {
            "title": "Disambiguation Test",
            "id": "66666",
            "url": "https://en.wikipedia.org/wiki/Disambiguation_Test",
            "language": "en",
            "source_id": "enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "raw_content": """{{Disambiguation}}

'''Disambiguation Test''' may refer to:

* [[Option 1]] - First option
* [[Option 2]] - Second option
* [[Option 3]] - Third option

{{disambig}}
""",
        }

        result = extractor.extract(record)

        assert result is not None
        assert result["title"] == "Disambiguation Test"
        assert "may refer to" in result["text"]
        assert "First option" in result["text"]

    def test_extract_different_languages(self):
        """Test extraction with different language settings."""
        # Test Spanish extractor
        extractor_es = WikipediaExtractor(language="es")

        record_es = {
            "title": "Artículo de prueba",
            "id": "77777",
            "url": "https://es.wikipedia.org/wiki/Artículo_de_prueba",
            "language": "es",
            "source_id": "eswiki-20230501-pages-articles-multistream1.xml.bz2",
            "raw_content": """'''Artículo de prueba''' es un artículo de prueba.

== Sección 1 ==
Esta es la primera sección con algunos [[Wikipedia:Enlace|enlaces]] y '''texto en negrita'''.

== Enlaces externos ==
* [http://ejemplo.com Enlace externo]

[[Categoría:Prueba]]
""",
        }

        result = extractor_es.extract(record_es)

        assert result is not None
        assert result["title"] == "Artículo de prueba"
        assert result["language"] == "es"
        assert "artículo de prueba" in result["text"]
        assert "primera sección" in result["text"]

    def test_input_columns(self):
        """Test that input_columns returns the expected column names."""
        extractor = WikipediaExtractor(language="en")
        columns = extractor.input_columns()

        expected_columns = ["title", "id", "url", "language", "source_id", "raw_content"]
        assert columns == expected_columns

    def test_output_columns(self):
        """Test that output_columns returns the expected column names."""
        extractor = WikipediaExtractor(language="en")
        columns = extractor.output_columns()

        expected_columns = ["text", "title", "id", "url", "language", "source_id"]
        assert columns == expected_columns

    def test_extract_preserves_metadata(self):
        """Test that extraction preserves all metadata fields."""
        extractor = WikipediaExtractor(language="en")

        record = {
            "title": "Metadata Test",
            "id": "88888",
            "url": "https://en.wikipedia.org/wiki/Metadata_Test",
            "language": "en",
            "source_id": "enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "raw_content": "'''Metadata Test''' is a test for metadata preservation.",
        }

        result = extractor.extract(record)

        assert result is not None
        assert result["title"] == record["title"]
        assert result["id"] == record["id"]
        assert result["url"] == record["url"]
        assert result["language"] == record["language"]
        assert result["source_id"] == record["source_id"]
        assert "text" in result
        assert "raw_content" not in result

    def test_extract_complex_markup(self):
        """Test extraction with complex Wikipedia markup."""
        extractor = WikipediaExtractor(language="en")

        record = {
            "title": "Complex Markup",
            "id": "99999",
            "url": "https://en.wikipedia.org/wiki/Complex_Markup",
            "language": "en",
            "source_id": "enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "raw_content": """'''Complex Markup''' is a test article.

{| class="wikitable"
|-
! Header 1 !! Header 2
|-
| Cell 1 || Cell 2
|-
| Cell 3 || Cell 4
|}

{{Quote|This is a quote|Author Name}}

<math>E = mc^2</math>

== Gallery ==
<gallery>
File:Image1.jpg|Caption 1
File:Image2.jpg|Caption 2
</gallery>

== Notes ==
{{reflist}}
""",
        }

        result = extractor.extract(record)

        assert result is not None
        assert result["title"] == "Complex Markup"
        assert "Complex Markup is a test article" in result["text"]

        # Tables, math, and galleries might be processed or removed
        assert "wikitable" not in result["text"]
        assert "<math>" not in result["text"]
        assert "<gallery>" not in result["text"]


class TestWikipediaExtractorEdgeCases:
    """Test edge cases and error conditions for WikipediaExtractor."""

    def test_extract_with_malformed_markup(self):
        """Test extraction with malformed Wikipedia markup."""
        extractor = WikipediaExtractor(language="en")

        record = {
            "title": "Malformed",
            "id": "101010",
            "url": "https://en.wikipedia.org/wiki/Malformed",
            "language": "en",
            "source_id": "enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "raw_content": """'''Malformed''' article with [[unclosed link

{{unclosed template

== Unclosed section

More content here.
""",
        }

        # Should not crash on malformed markup
        result = extractor.extract(record)

        assert result is not None
        assert result["title"] == "Malformed"
        assert "text" in result

    def test_extract_very_long_article(self):
        """Test extraction with very long article content."""
        extractor = WikipediaExtractor(language="en")

        # Create a very long article
        long_content = "'''Long Article''' is a test.\n\n" + "This is a very long paragraph. " * 1000

        record = {
            "title": "Long Article",
            "id": "111111",
            "url": "https://en.wikipedia.org/wiki/Long_Article",
            "language": "en",
            "source_id": "enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "raw_content": long_content,
        }

        result = extractor.extract(record)

        assert result is not None
        assert result["title"] == "Long Article"
        assert len(result["text"]) > 0

    def test_extract_with_unicode_content(self):
        """Test extraction with Unicode content."""
        extractor = WikipediaExtractor(language="en")

        record = {
            "title": "Unicode Test",
            "id": "121212",
            "url": "https://en.wikipedia.org/wiki/Unicode_Test",
            "language": "en",
            "source_id": "enwiki-20230501-pages-articles-multistream1.xml.bz2",
            "raw_content": """'''Unicode Test''' contains various Unicode characters.

== Symbols ==
* Mathematical: ∑, ∫, ∂, ∆, ∇
* Currency: €, £, ¥, ¢
* Arrows: ←, →, ↑, ↓
* Greek: alpha, beta, gamma, delta, epsilon, zeta, eta, theta

== Languages ==
* Chinese: 中文
* Japanese: 日本語
* Arabic: العربية
* Russian: Русский

== Emojis ==
Testing emojis: 🌍 🔬 📖 🎨 🎵
""",
        }

        result = extractor.extract(record)

        assert result is not None
        assert result["title"] == "Unicode Test"
        assert "∑" in result["text"]
        assert "中文" in result["text"]
        assert "العربية" in result["text"]
        assert "🌍" in result["text"]
