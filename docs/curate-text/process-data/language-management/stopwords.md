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

Stop words are common words that are often filtered out in natural language processing (NLP) tasks because they typically don't carry significant meaning. Curator provides built-in stop word lists for several languages to support text analysis and extraction processes.

```{note}
Studies on stopword lists and their distribution in various text corpora have shown that typical English text contains 30–40% stop words.
```

## What Are Stop Words?

Stop words are high-frequency words that generally don't contribute much semantic value to text analysis. Examples in English include "the," "is," "at," "which," and "on." These words appear so frequently in language that they can distort text processing tasks if not properly managed.

Key characteristics of stop words:

- They appear with high frequency in text
- They typically serve grammatical rather than semantic functions
- They're language-specific (each language has its own set of stop words)
- Removing them can improve efficiency in NLP tasks

## Why Stop Words Matter in Curator

In Curator, stop words play several important roles:

1. **Text Extraction**: The text extraction process (for Common Crawl data) uses stop word density as a key metric to identify meaningful content
2. **Efficient Processing**: Filtering stop words can reduce the amount of data to process
3. **Boilerplate Removal**: Stop word density helps differentiate between main content and boilerplate in web pages

## Available Stop Word Lists

Curator leverages the extensive stop word collection from [JusText](https://github.com/miso-belica/jusText/tree/main/justext/stoplists) for most languages. Curator also provides custom stop word lists for the following languages not covered by JusText:

| Language | File Name |
|----------|-----------|
| Chinese | `zh_stopwords.py` |
| Japanese | `ja_stopwords.py` |
| Thai | `th_stopwords.py` |

These stop word lists use Python frozen sets for fast membership checks and immutability.

### Chinese Stop Words

Chinese stop words in `zh_stopwords.py` include common function words and punctuation, for example "一个" (one), "不是" (isn't), and "他们" (they).

```python
# Example from zh_stopwords.py (partial)
zh_stopwords = frozenset([
    "、", "。", "〈", "〉", "《", "》", "一", "一个",
    # ... many more words
])
```

### Japanese Stop Words

Japanese stop words in `ja_stopwords.py` include common Japanese function words such as "あそこ" (there), "これ" (this), and "ます" (a polite verb ending).

```python
# Example from ja_stopwords.py
ja_stopwords = frozenset([
    "あそこ", "あっ", "あの", "あのかた", "あの人",
    # ... more words
    "私", "私達", "貴方", "貴方方",
])
```

### Thai Stop Words

Thai stop words are available in `th_stopwords.py`, including common Thai words like "กล่าว" (to say), "การ" (the), and "ของ" (of).

```python
# Example from th_stopwords.py
thai_stopwords = frozenset([
    "กล่าว", "กว่า", "กัน", "กับ", "การ", "ก็", "ก่อน",
    # ... more words
    "ไป", "ไม่", "ไว้",
])
```

## How Curator Uses Stop Words in Text Extraction

Stop words are a critical component in Curator's text extraction algorithms. Here's how they're used in different extractors:

### JusText Extractor

The JusText algorithm uses stop word density to classify text blocks as main content or boilerplate:

1. **Context-Free Classification**: The algorithm classifies text blocks with a high density of stop words as "good" (main content)
2. **Parameter Customization**: You can customize the stop word density thresholds via `stopwords_low` and `stopwords_high` parameters

```python
from nemo_curator.stages.text.download.html_extractors.justext import JusTextExtractor

# Customize stop word thresholds
extractor = JusTextExtractor(
    stopwords_low=0.30,   # Minimum stop word density
    stopwords_high=0.32,  # Maximum stop word density
)
```

### Other Extractors

The `ResiliparseExtractor` and `TrafilaturaExtractor` also use stop word density to filter extracted content:

```python
from nemo_curator.stages.text.download.html_extractors.resiliparse import ResiliparseExtractor
from nemo_curator.stages.text.download.html_extractors.trafilatura import TrafilaturaExtractor

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

Languages like Thai, Chinese, Japanese, and Korean don't use spaces between words, which affects how the system calculates stop word density. Curator identifies these languages via the `NON_SPACED_LANGUAGES` constant:

```python
NON_SPACED_LANGUAGES = frozenset(["THAI", "CHINESE", "JAPANESE", "KOREAN"])
```

For these languages, the extractor applies special handling:

- Stopword densities are still computed and used for paragraph classification.
- Disable the final `is_boilerplate` filter to avoid over-removal in non-spaced scripts.

## Creating Custom Stop Word Lists

You can create and use your own stop word lists when processing text with Curator:

```python
from nemo_curator.stages.text.download import CommonCrawlDownloadExtractStage
from nemo_curator.pipeline import Pipeline

# Define custom stop words for multiple languages
custom_stop_lists = {
    "ENGLISH": frozenset(["the", "and", "is", "in", "for", "where", "when", "to", "at"]),
    "SPANISH": frozenset(["el", "la", "los", "las", "un", "una", "y", "o", "de", "en", "que"]),
}

# Create Common Crawl processing stage with custom stop lists
cc_stage = CommonCrawlDownloadExtractStage(
    start_snapshot="2023-06",
    end_snapshot="2023-10", 
    download_dir="/output/folder",
    stop_lists=custom_stop_lists
)

# Create and run pipeline
pipeline = Pipeline(name="custom_stopwords_pipeline")
pipeline.add_stage(cc_stage)

# Execute pipeline
results = pipeline.run()
```

## Performance Considerations

- Stop word lists use frozen sets for fast membership checks (O(1) complexity)
- Using appropriate stop word lists can improve extraction quality
- For specialized domains, consider customizing the stop word lists

## More Resources

- [JusText algorithm overview](https://corpus.tools/wiki/Justext/Algorithm)
- [HTML2Text extraction docs](https://resiliparse.chatnoir.eu/en/latest/man/extract/html2text.html)
- [Web text extraction docs](https://trafilatura.readthedocs.io/en/latest/)
