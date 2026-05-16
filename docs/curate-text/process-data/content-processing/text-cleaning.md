---
description: "Remove undesirable text including improperly decoded Unicode characters, inconsistent spacing, and excessive URLs"
categories: ["how-to-guides"]
tags: ["text-cleaning", "unicode", "normalization", "url-removal", "preprocessing", "ftfy"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-format-text-cleaning)=
# Text Cleaning

Remove undesirable text such as improperly decoded Unicode characters, inconsistent line spacing, or excessive URLs from documents being pre-processed for your dataset using NeMo Curator.

One common issue in text datasets is improper Unicode character encoding, which can result in garbled or unreadable text, particularly with special characters like apostrophes, quotes, or diacritical marks. For example, the input sentence `"The Mona Lisa doesn't have eyebrows."` from a given document may not have included a properly encoded apostrophe (`'`), resulting in the sentence decoding as `"The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows."`.

NeMo Curator enables you to easily run this document through the default `UnicodeReformatter` module to detect and remove the unwanted text, or you can define your own custom Unicode text cleaner tailored to your needs.

## How it Works

NeMo Curator provides the following modules for cleaning text:

- `UnicodeReformatter`: Uses [ftfy](https://ftfy.readthedocs.io/en/latest/) to fix broken Unicode characters. Modifies the "text" field of the dataset by default. The module accepts extensive configuration options for fine-tuning Unicode repair behavior. Please see the [ftfy documentation](https://ftfy.readthedocs.io/en/latest/config.html) for more information about parameters used by the `UnicodeReformatter`.
- `NewlineNormalizer`: Uses regex to replace 3 or more consecutive newline characters in each document with only 2 newline characters.
- `UrlRemover`: Uses regex to remove all URLs in each document.

You can use these modules individually or sequentially in a cleaning pipeline.

---

## Usage

::::{tab-set}

:::{tab-item} Python

Consider the following example, which loads a dataset from a directory (`books/`), steps through each module in a cleaning pipeline, and outputs the processed dataset to `cleaned_books/`:

```python
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modifiers.string import UrlRemover, NewlineNormalizer
from nemo_curator.stages.text.modifiers.unicode import UnicodeReformatter
from nemo_curator.stages.text.modifiers import Modify

def main():
    # Initialize Ray client
    ray_client = RayClient()
    ray_client.start()

    # Create processing pipeline
    pipeline = Pipeline(
        name="text_cleaning_pipeline",
        description="Clean text data using Unicode reformatter, newline normalizer, and URL remover"
    )

    # Add reader stage
    pipeline.add_stage(JsonlReader(file_paths="books/"))

    # Add processing stages
    pipeline.add_stage(Modify(UnicodeReformatter()))
    pipeline.add_stage(Modify(NewlineNormalizer()))
    pipeline.add_stage(Modify(UrlRemover()))

    # Add writer stage
    pipeline.add_stage(JsonlWriter(path="cleaned_books/"))

    # Execute pipeline
    results = pipeline.run()

    # Stop Ray client
    ray_client.stop()

if __name__ == "__main__":
    main()
```
:::

::::

## Custom Text Cleaner

You can create your own custom text cleaner by extending the `DocumentModifier` class. The implementation of `UrlRemover` demonstrates this approach:

```python
import re

from nemo_curator.stages.text.modifiers import DocumentModifier

URL_REGEX = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)


class UrlRemover(DocumentModifier):
    """
    Removes all URLs in a document.
    """

    def __init__(self):
        super().__init__()

    def modify_document(self, text: str) -> str:
        return URL_REGEX.sub("", text)
```

To create a custom text cleaner, inherit from the `DocumentModifier` class and implement the constructor and `modify_document` method.
