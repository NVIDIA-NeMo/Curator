---
description: "Combine synthetic data generation with NeMo Curator's filtering and processing capabilities for comprehensive data workflows"
categories: ["how-to-guides"]
tags: ["integration", "synthetic-data", "filtering", "deduplication", "pipeline", "workflow"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "text-only"
---

(text-generate-data-integration)=
# Integration with NeMo Curator

Synthetic data generation, unlike the rest of NeMo Curator, operates independently of Dask by default. This is due to the scale differences between typical use cases. Synthetic data is usually generated on the order of thousands to hundreds of thousands of samples, while pretraining datasets operate at the scale of millions to billions of samples. For smaller synthetic datasets, the overhead of starting up a Dask cluster is usually not justified.

However, you may want to deduplicate or filter your synthetic responses with NeMo Curator's powerful processing capabilities. For example, topics might end up getting duplicated during generation, and sending duplicate topics as queries to an LLM wastes valuable computational resources.

We recommend using `DocumentDataset.from_pandas` and `DocumentDataset.to_pandas` to transition between workflows that require the other NeMo Curator modules.

For example, you could do something like this:

```python
import pandas as pd
from nemo_curator.datasets import DocumentDataset

# Initialize client, etc.

model = "mistralai/mixtral-8x7b-instruct-v0.1"
macro_topic_responses = generator.generate_macro_topics(
    n_macro_topics=20, model=model
)
macro_topics_list = ... # Parse responses manually or with convert_response_to_yaml_list

subtopic_responses = generator.generate_subtopics(
    macro_topic=macro_topics_list[0], n_subtopics=5, model=model
)
subtopic_list = ... # Parse responses manually or with convert_response_to_yaml_list

# Convert to DocumentDataset for processing with NeMo Curator
df = pd.DataFrame({"topics": subtopic_list})
# Add required ID field for deduplication
df["id"] = range(len(df))
dataset = DocumentDataset.from_pandas(df)

# Deduplicate/filter with NeMo Curator
# For example, apply exact deduplication using the ExactDuplicates module
from nemo_curator.modules.exact_dedup import ExactDuplicates

exact_dups = ExactDuplicates(
    id_field="id",
    text_field="topics"
)

# Identify and remove exact duplicates
duplicates = exact_dups.identify_duplicates(dataset)
filtered_dataset = exact_dups.remove(dataset, duplicates)

# Convert back to a list for continued synthetic generation
filtered_topics = filtered_dataset.to_pandas()["topics"].to_list()

# Continue with synthetic data generation pipeline
question_responses = generator.generate_open_qa_from_topic(
    topic=filtered_topics[0], n_openlines=10, model=model
)
```

This approach allows you to leverage the powerful filtering and processing capabilities of NeMo Curator while still using the synthetic data generation tools for creating custom datasets.

## Alternative Filtering Approaches

Besides exact deduplication, you can apply various other filters available in NeMo Curator:

```python
# Using text-based filters for quality control
from nemo_curator.filters.heuristic_filter import LongWordFilter, RepeatedLinesFilter
from nemo_curator.modules.filter import Filter

# Filter out topics with very long words (likely corrupted text)
long_word_filter = Filter(LongWordFilter(max_word_length=100), text_field="topics")
quality_filtered = long_word_filter(dataset)

# Filter out topics with repeated content
repeated_lines_filter = Filter(RepeatedLinesFilter(), text_field="topics")
final_filtered = repeated_lines_filter(quality_filtered)
```

## Integration Workflow Recommendations

When integrating synthetic data generation with NeMo Curator processing:

1. **Generate synthetic data** using NeMo Curator's synthetic generation modules
2. **Convert to DocumentDataset** using `from_pandas()` for NeMo Curator processing
3. **Apply deduplication and filtering** to improve data quality and remove redundancy
4. **Convert back to pandas** using `to_pandas()` for continued synthetic generation
5. **Repeat the cycle** as needed for iterative data generation and refinement

This integration pattern is particularly useful for:
- Removing duplicate topics before expensive LLM generation
- Quality filtering of generated content
- Combining multiple synthetic datasets with consistent processing 