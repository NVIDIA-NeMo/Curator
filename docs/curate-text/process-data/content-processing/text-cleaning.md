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

Consider the following example, which loads a dataset (`books.jsonl`), steps through each module in a cleaning pipeline, and outputs the processed dataset as `cleaned_books.jsonl`:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.datasets import DocumentDataset
from nemo_curator.stages.text.modifiers import UnicodeReformatter, UrlRemover, NewlineNormalizer
from nemo_curator.stages.text.modules import Modify

def main():
    # Create processing pipeline
    pipeline = Pipeline(
        name="text_cleaning_pipeline",
        description="Clean text data using Unicode reformatter, newline normalizer, and URL remover",
        stages=[
            Modify(UnicodeReformatter()),
            Modify(NewlineNormalizer()),
            Modify(UrlRemover()),
        ]
    )

    # Execute pipeline
    results = pipeline.run()
    
if __name__ == "__main__":
    main()
```
:::

:::{tab-item} CLI

You can also perform text cleaning operations using the CLI by running the `text_cleaning` command:

```bash
text_cleaning \
  --input-data-dir=/path/to/input/ \
  --output-clean-dir=/path/to/output/ \
  --normalize-newlines \
  --remove-urls
```

By default, the CLI will only perform Unicode reformatting. Appending the `--normalize-newlines` and `--remove-urls` options adds the other text cleaning options.
:::

::::

## Custom Text Cleaner

You can create your own custom text cleaner by extending the `DocumentModifier` class. The implementation of `UnicodeReformatter` demonstrates this approach:

```python
import ftfy

from nemo_curator.stages.text.modifiers import DocumentModifier


class UnicodeReformatter(DocumentModifier):
    def __init__(self):
        super().__init__()

    def modify_document(self, text: str) -> str:
        return ftfy.fix_text(text)
```

To create a custom text cleaner, inherit from the `DocumentModifier` class and implement the constructor and `modify_document` method. Also, like the `DocumentFilter` class, `modify_document` can be annotated with `batched` to take in a pandas Series of documents instead of a single document. See the {ref}`custom filters documentation <text-process-data-filter-custom>` for more information.
