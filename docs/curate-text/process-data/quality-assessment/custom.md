---
description: "Create and combine custom filters using NeMo Curator's flexible framework for specialized data quality requirements"
categories: ["how-to-guides"]
tags: ["custom-filters", "extensible", "flexible", "advanced", "framework", "batched"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-filter-custom)=
# Custom Filters

NVIDIA NeMo Curator provides a flexible framework for implementing and combining custom filters to meet your specific data quality requirements. Whether you need to filter documents based on domain-specific criteria or optimize your pipeline's performance, custom filters give you complete control over the filtering process.

## Creating Custom Filters

Custom filters in NeMo Curator inherit from the `DocumentFilter` abstract base class, which requires implementing two key methods:

1. `score_document`: Analyzes a document and assigns it a quality score
2. `keep_document`: Determines whether to keep a document based on its score

Here's a simple example of a custom filter:

```python
from nemo_curator.filters import DocumentFilter

class CustomWordFilter(DocumentFilter):
    def __init__(self, target_words, min_occurrences=1):
        super().__init__()  # Call the parent constructor
        self._target_words = set(target_words)
        self._min_occurrences = min_occurrences
        self._name = 'custom_word_filter'
        
    def score_document(self, text: str):
        """Count occurrences of target words in the document."""
        words = text.lower().split()
        count = sum(1 for word in words if word in self._target_words)
        return count
        
    def keep_document(self, score: int):
        """Keep documents with enough target words."""
        return score >= self._min_occurrences
        
    @property
    def backend(self):
        """Specify which dataframe backend this filter supports."""
        return "pandas"  # Options are "pandas", "cudf", or "any"
```

By default, the `backend` property returns "pandas", but you can override it to support GPU-accelerated processing with "cudf" or specify "any" if your filter works with either backend.

## Using Custom Filters

Once you've defined your custom filter, you can use it with NeMo Curator's filtering framework:

```python
import nemo_curator as nc
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

# Load your dataset
pipeline = Pipeline(name="custom_filtering")
pipeline.add_stage(JsonlReader(file_paths="input_data/*.jsonl", fields=["text", "id"]))

# Create and configure your custom filter
my_filter = CustomWordFilter(
    target_words=["machine", "learning", "ai", "deep", "neural"],
    min_occurrences=3
)

# Apply the filter
from nemo_curator.stages.text.modules import ScoreFilter

filter_step = ScoreFilter(
    my_filter,
    text_field="text",
    score_field="target_word_count"
)

# Get filtered dataset
filtered_dataset = filter_step(dataset)

# Save results
filtered_dataset.to_json("filtered_output/", write_to_filename=True)
```

## Optimizing Performance with Batched Filters

For improved performance, especially with large datasets, you can implement batched versions of your filters using the `@batched` decorator:

```python
import pandas as pd
from nemo_curator.filters import DocumentFilter
from nemo_curator.utils.decorators import batched

class BatchedCustomFilter(DocumentFilter):
    def __init__(self, threshold=0.5):
        super().__init__()
        self._threshold = threshold
        self._name = 'batched_custom_filter'
    
    def score_document(self, text: str):
        # Single document scoring logic
        return compute_quality_score(text)
    
    @batched
    def keep_document(self, scores: pd.Series):
        """Process multiple documents at once.
        
        Args:
            scores: Pandas Series containing scores with document IDs as index
            
        Returns:
            Pandas Series of boolean values with same index as input
        """
        return scores >= self._threshold
```

When implementing batched methods, it's crucial to maintain the original index in the returned Series to ensure proper document tracking.

## Filter Composition Methods

NeMo Curator makes it easy to combine multiple filters using different composition approaches:

### Sequential

The `Sequential` class applies a series of filters in order:

```python
import nemo_curator as nc
from nemo_curator.filters import WordCountFilter, NonAlphaNumericFilter, UrlsFilter

# Create a pipeline of filters
filter_pipeline = nc.Sequential([
    nc.ScoreFilter(WordCountFilter(min_words=100)),
    nc.ScoreFilter(NonAlphaNumericFilter(max_symbol_ratio=0.3)),
    nc.ScoreFilter(UrlsFilter(max_urls=3))
])

# Apply the pipeline
high_quality_docs = filter_pipeline(dataset)
```

### Parallel with Voting (Custom Implementation)

You can implement a custom voting system where documents must pass a certain number of filters. This is not a built-in class but can be implemented as a utility function:

```python
import pandas as pd
import nemo_curator as nc

# Custom utility function for filter voting
def voting_filter(dataset, filters, min_passing=2):
    """
    Custom implementation of a voting filter system.
    
    Args:
        dataset: DocumentDataset to filter
        filters: List of filter modules
        min_passing: Minimum number of filters that must accept a document
        
    Returns:
        Filtered DocumentDataset
    """
    results = []
    for f in filters:
        results.append(f(dataset))
    
    # Create a mask where documents pass at least min_passing filters
    document_ids = dataset.df.index
    pass_counts = pd.Series(0, index=document_ids)
    
    for result in results:
        pass_counts[result.df.index] += 1
    
    passing_ids = pass_counts[pass_counts >= min_passing].index
    return nc.DocumentDataset(dataset.df.loc[passing_ids])
```

## Scoring Without Filtering

Sometimes you want to add quality scores to your documents without actually filtering them:

```python
import nemo_curator as nc
from nemo_curator.filters import WordCountFilter, NonAlphaNumericFilter

# Score documents without filtering them
scoring_step = nc.Score(
    WordCountFilter().score_document,
    text_field="text",
    score_field="word_count"
)

# Add multiple scores
symbol_scoring = nc.Score(
    NonAlphaNumericFilter().score_document,
    text_field="text",
    score_field="symbol_ratio"
)

# Apply scoring
scored_dataset = scoring_step(dataset)
scored_dataset = symbol_scoring(scored_dataset)

# Save the scored dataset
scored_dataset.to_json("scored_output/", write_to_filename=True)
```

## Filtering on Existing Metadata

If your dataset already contains quality metrics, you can filter directly on those:

```python
import nemo_curator as nc

# Filter based on existing metadata field
filter_step = nc.Filter(
    lambda score: score < 0.3,  # Keep only documents with toxicity < 0.3
    filter_field="toxicity_score"
)

safe_documents = filter_step(scored_dataset)
```

## Integrating with CLI

To make your custom filters available through the command-line interface, you can register them in a configuration file:

```yaml
# custom_filters.yaml
input_field: text
filters:
  - name: ScoreFilter
    filter:
      name: path.to.your.CustomWordFilter
      params:
        target_words: ["machine", "learning", "ai"]
        min_occurrences: 2
    text_field: text
    score_field: target_word_count
  
  # Add more filters as needed
```

Then use this configuration with the `filter_documents` CLI:

```bash
filter_documents \
  --input-data-dir=/path/to/input/data \
  --filter-config-file=./custom_filters.yaml \
  --output-retained-document-dir=/path/to/output \
  --log-dir=/path/to/logs
```

## Best Practices

When developing custom filters:

1. **Optimize for performance**: Implement batch processing for computationally intensive operations
2. **Add meaningful metadata**: Store scores that provide insight into why documents were kept or removed
3. **Start simple**: Begin with basic filters and incrementally add complexity
4. **Test on samples**: Validate your filters on small samples before processing large datasets
5. **Monitor filter impact**: Track how many documents each filter removes to identify potential issues
6. **Document behavior**: Add clear documentation about what your filter does and its parameters 