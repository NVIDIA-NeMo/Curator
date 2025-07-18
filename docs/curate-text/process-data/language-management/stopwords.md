---
description: "Manage high-frequency words to enhance text extraction and content detection with language-specific stop word lists"
categories: ["how-to-guides"]
tags: ["stop-words", "preprocessing", "filtering", "language-specific", "text-extraction", "content-detection"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-languages-stop-words)=
# Stop Words in Text Processing

Stop words are common words that are often filtered out in natural language processing (NLP) tasks because they typically don't carry significant meaning. NVIDIA NeMo Curator provides built-in stop word lists for several languages to support text analysis and extraction processes.

```{note}
Studies on stopword lists and their distribution in various text corpora have shown that typical English text contains 30–40% stop words.
```

## What Are Stop Words?

Stop words are high-frequency words that generally don't contribute much semantic value to text analysis. Examples in English include "the," "is," "at," "which," and "on." These words appear so frequently in language that they can distort text processing tasks if not properly managed.

Key characteristics of stop words:
- They appear with high frequency in text
- They typically serve grammatical rather than semantic functions
- They're language-specific (each language has its own set of stop words)
- Removing them can improve efficiency in many NLP tasks

## Why Stop Words Matter in NeMo Curator

In NeMo Curator, stop words play several important roles:

1. **Text Extraction**: The text extraction process (especially for Common Crawl data) uses stop word density as a key metric to identify meaningful content
2. **Language Detection**: Stop words help in language detection and processing
3. **Efficient Processing**: Filtering stop words reduces the amount of data that needs to be processed
4. **Boilerplate Removal**: Stop word density helps differentiate between main content and boilerplate in web pages

## Available Stop Word Lists

NeMo Curator leverages the extensive stop word collection from [JusText](https://github.com/miso-belica/jusText/tree/main/justext/stoplists) for most languages. In addition, NeMo Curator provides custom stop word lists for the following languages not covered by JusText:

| Language | File Name | Number of Stop Words |
|----------|-----------|---------------------|
| Chinese | `zh_stopwords.py` | 794 |
| Japanese | `ja_stopwords.py` | 134 |
| Thai | `th_stopwords.py` | 115 |

These stop word lists are implemented as Python frozen sets for efficient lookup and immutability.

### Chinese Stop Words

Chinese stop words in `zh_stopwords.py` form the largest list with 794 entries, including words like "一个" (one), "不是" (isn't), and "他们" (they).

```python
# Example from zh_stopwords.py (partial)
zh_stopwords = frozenset([
    "、", "。", "〈", "〉", "《", "》", "一", "一个",
    # ... many more words
])
```

### Japanese Stop Words

Japanese stop words in `ja_stopwords.py` include 134 common Japanese words like "あそこ" (there), "これ" (this), and "ます" (a polite verb ending).

```python
# Example from ja_stopwords.py
ja_stopwords = frozenset([
    "あそこ", "あっ", "あの", "あのかた", "あの人",
    # ... more words
    "私", "私達", "貴方", "貴方方",
])
```

### Thai Stop Words

Thai stop words are available in `th_stopwords.py`. The file contains 115 common Thai words like "กล่าว" (to say), "การ" (the), and "ของ" (of).

```python
# Example from th_stopwords.py
thai_stopwords = frozenset([
    "กล่าว", "กว่า", "กัน", "กับ", "การ", "ก็", "ก่อน",
    # ... more words
    "ไป", "ไม่", "ไว้",
])
```

## How Stop Words Are Used in Text Extraction

Stop words are a critical component in NeMo Curator's text extraction algorithms. Here's how they're used in different extractors:

### JusText Extractor

The JusText algorithm uses stop word density to classify text blocks as main content or boilerplate:

1. **Context-Free Classification**: Text blocks with a high density of stop words are classified as "good" (likely main content)
2. **Parameter Customization**: You can customize the stop word density thresholds via `stopwords_low` and `stopwords_high` parameters

```python
from nemo_curator.download import JusTextExtractor

# Customize stop word thresholds
extractor = JusTextExtractor(
    stopwords_low=0.30,   # Minimum stop word density
    stopwords_high=0.32,  # Maximum stop word density
)
```

### Resiliparse and Trafilatura Extractors

These extractors also use stop word density to filter extracted content:

```python
from nemo_curator.download import ResiliparseExtractor, TrafilaturaExtractor

# Resiliparse with custom stop word density
resiliparse = ResiliparseExtractor(
    required_stopword_density=0.32  # Only keep paragraphs with >= 32% stop words
)

# Trafilatura with custom stop word density
trafilatura = TrafilaturaExtractor(
    required_stopword_density=0.35  # Higher threshold for more selective extraction
)
```

## Special Handling for Non-Spaced Languages

Languages like Thai, Chinese, Japanese, and Korean don't use spaces between words, which affects how stop word density is calculated. NeMo Curator identifies these languages via the `NON_SPACED_LANGUAGES` constant:

```python
NON_SPACED_LANGUAGES = ["THAI", "CHINESE", "JAPANESE", "KOREAN"]
```

For these languages, special handling is applied:
- Stop word density calculations are disabled
- Boilerplate removal based on stop words is adjusted

## Creating Custom Stop Word Lists

You can create and use your own stop word lists when processing text with NeMo Curator:

```python
from nemo_curator.download import download_common_crawl

# Define custom stop words for multiple languages
custom_stop_lists = {
    "ENGLISH": frozenset(["the", "and", "is", "in", "for", "where", "when", "to", "at"]),
    "SPANISH": frozenset(["el", "la", "los", "las", "un", "una", "y", "o", "de", "en", "que"]),
}

# Use custom stop lists in download process
dataset = download_common_crawl(
    "/output/folder",
    "2023-06",
    "2023-10",
    stop_lists=custom_stop_lists
)
```

## Performance Considerations

- Stop word lists are implemented as frozen sets for fast lookups (O(1) complexity)
- Using appropriate stop word lists can significantly improve extraction quality
- For specialized domains, consider customizing the stop word lists

## Additional Resources

- [JusText Algorithm Overview](https://corpus.tools/wiki/Justext/Algorithm)
- [Resiliparse Documentation](https://resiliparse.chatnoir.eu/en/latest/man/extract/html2text.html)
- [Trafilatura Documentation](https://trafilatura.readthedocs.io/en/latest/) 