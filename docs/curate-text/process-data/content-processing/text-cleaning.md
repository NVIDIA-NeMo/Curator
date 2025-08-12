---
description: "Create custom text processing stages to clean and normalize text using ray-curator's processing pipeline"
categories: ["how-to-guides"]
tags: ["text-cleaning", "unicode", "normalization", "url-removal", "preprocessing", "ray-curator", "processing-stages"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-format-text-cleaning)=

# Text Cleaning

Create custom text processing stages to clean and normalize text data in your ray-curator pipelines. Ray-curator uses a task-centric architecture where you build text cleaning operations as `ProcessingStage` components that transform `DocumentBatch` tasks.

Common text cleaning needs include removing improperly decoded Unicode characters, normalizing inconsistent line spacing, and filtering out excessive URLs. For example, corrupted encoding might turn `"The Mona Lisa doesn't have eyebrows."` into `"The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows."`.

## How it Works

Ray-curator provides a flexible stage-based architecture for text processing:

- **ProcessingStage**: Base class for all text transformation operations that accepts a `DocumentBatch` and returns a transformed `DocumentBatch`.
- **Pipeline**: Orchestrates several processing stages in sequence.
- **DocumentBatch**: Task containing a pandas DataFrame with text data that flows through the pipeline.
- **Text Processing Utilities**: Helper functions for common text operations like `remove_control_characters()`.

You create custom text cleaning stages by extending `ProcessingStage` and implementing the `process()` method.

## Usage

The following example shows how to create custom text cleaning stages and combine them in a ray-curator pipeline:

```python
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd
from ray_curator.pipeline import Pipeline
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.io.reader import JsonlReader
from ray_curator.stages.io.writer import JsonlWriter
from ray_curator.backends.experimental.ray_data import RayDataExecutor
from ray_curator.tasks import DocumentBatch

@dataclass
class UnicodeCleaningStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage that cleans Unicode control characters from text."""
    
    text_field: str = "text"
    _name: str = "unicode_cleaning"
    
    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]
    
    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]
    
    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Remove Unicode control characters from text."""
        df = batch.to_pandas()
        
        def clean_unicode(text: str) -> str:
            # Remove control characters (non-printable characters)
            return "".join(char for char in text if unicodedata.category(char)[0] != "C")
        
        df[self.text_field] = df[self.text_field].apply(clean_unicode)
        
        return DocumentBatch(
            task_id=f"{batch.task_id}_unicode_cleaned",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

@dataclass  
class NewlineNormalizationStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage that normalizes excessive newlines in text."""
    
    text_field: str = "text"
    _name: str = "newline_normalization"
    
    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]
    
    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]
    
    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Replace 3+ consecutive newlines with exactly 2 newlines."""
        df = batch.to_pandas()
        
        def normalize_newlines(text: str) -> str:
            # Replace 3 or more consecutive newlines with exactly 2
            return re.sub(r'\n{3,}', '\n\n', text)
        
        df[self.text_field] = df[self.text_field].apply(normalize_newlines)
        
        return DocumentBatch(
            task_id=f"{batch.task_id}_newlines_normalized",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

@dataclass
class UrlRemovalStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Stage that removes URLs from text."""
    
    text_field: str = "text"
    _name: str = "url_removal"
    
    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]
    
    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]
    
    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Remove URLs from text using regex."""
        df = batch.to_pandas()
        
        def remove_urls(text: str) -> str:
            # Remove HTTP/HTTPS URLs
            url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+[^\s<>"{}|\\^`[\].,;!?]'
            return re.sub(url_pattern, '', text)
        
        df[self.text_field] = df[self.text_field].apply(remove_urls)
        
        return DocumentBatch(
            task_id=f"{batch.task_id}_urls_removed",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

def main():
    """Create and run a text cleaning pipeline."""
    
    # Create pipeline
    pipeline = Pipeline(
        name="text_cleaning_pipeline",
        description="Clean and normalize text data"
    )
    
    # Add stages to pipeline
    pipeline.add_stage(JsonlReader(file_paths="books.jsonl", columns=["text"]))
    pipeline.add_stage(UnicodeCleaningStage(text_field="text"))
    pipeline.add_stage(NewlineNormalizationStage(text_field="text"))
    pipeline.add_stage(UrlRemovalStage(text_field="text"))
    pipeline.add_stage(JsonlWriter(output_dir="./cleaned_output"))
    
    # Create executor and run pipeline
    executor = RayDataExecutor()
    results = pipeline.run(executor)
    
    print(f"Text cleaning complete. Processed {len(results) if results else 0} batches.")

if __name__ == "__main__":
    main()
```

## Custom Text Processing Stages

You can create custom text processing stages by extending the `ProcessingStage` class. Here's a template that demonstrates the key patterns:

```python
from dataclasses import dataclass
from typing import Any
import pandas as pd

from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import DocumentBatch

@dataclass
class CustomTextStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Template for creating custom text processing stages."""
    
    text_field: str = "text"  # Field containing text to process
    _name: str = "custom_text_stage"
    
    def inputs(self) -> tuple[list[str], list[str]]:
        """Define input requirements - DocumentBatch with specified text field."""
        return ["data"], [self.text_field]
    
    def outputs(self) -> tuple[list[str], list[str]]:
        """Define output - DocumentBatch with processed text field.""" 
        return ["data"], [self.text_field]
    
    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Process the DocumentBatch and return transformed result."""
        df = batch.to_pandas()
        
        # Apply your custom text processing function
        def custom_text_function(text: str) -> str:
            # Insert your text processing logic here
            return text.strip().lower()  # Example: normalize whitespace and case
        
        df[self.text_field] = df[self.text_field].apply(custom_text_function)
        
        # Return new DocumentBatch with processed data
        return DocumentBatch(
            task_id=f"{batch.task_id}_{self.name}",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
```

### Key Implementation Guidelines

1. **Extend ProcessingStage**: Inherit from `ProcessingStage[DocumentBatch, DocumentBatch]` for text transformations
2. **Define Input/Output Requirements**: Add `inputs()` and `outputs()` methods to specify data field dependencies  
3. **Add process() method**: Transform the pandas DataFrame within the DocumentBatch and return a new DocumentBatch
4. **Preserve Metadata**: Always pass through `_metadata` and `_stage_perf` to maintain task lineage
5. **Use @dataclass decorator**: Leverage `@dataclass` decorator for easy configuration

### Advanced Text Processing

For more complex text processing operations, you can:

- **Use External Libraries**: Import libraries like `ftfy` for Unicode repair, `spacy` for NLP, or `regex` for advanced pattern matching
- **Batch Processing**: Override `process_batch()` for more efficient batch operations
- **Resource Management**: Specify GPU/CPU requirements using the `_resources` field
- **Setup Methods**: Use `setup_on_node()` or `setup()` for model loading or expensive initialization

Example with external library:

```python
@dataclass
class AdvancedUnicodeStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Unicode cleaning using ftfy library."""
    
    def setup_on_node(self, node_info=None, worker_metadata=None):
        """Install ftfy if needed (called once per node)."""
        try:
            import ftfy
        except ImportError:
            import subprocess
            subprocess.check_call(["pip", "install", "ftfy"])
    
    def process(self, batch: DocumentBatch) -> DocumentBatch:
        import ftfy
        
        df = batch.to_pandas()
        df[self.text_field] = df[self.text_field].apply(ftfy.fix_text)
        
        return DocumentBatch(
            task_id=f"{batch.task_id}_unicode_fixed",
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
```
