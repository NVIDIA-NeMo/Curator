(text-process-data-filter-code)=
# Code Filtering

NVIDIA NeMo Curator provides specialized filters for assessing and filtering code snippets and programming files. These filters help ensure that code included in your training dataset meets quality standards and doesn't contain problematic patterns. Code filtering addresses specific challenges related to programming content, including code quality assessment, detection of non-code content mislabeled as code, identification of embedded data structures or boilerplate, language-specific filtering considerations, and token efficiency for code. These filters are particularly important when preparing datasets for code language models or programming assistants.

## How It Works

Code filtering evaluates programming content based on measurable attributes that correlate with code quality and usability for model training. The filters analyze various aspects of code:

1. **Structure Analysis**: Examines lines of code, indentation patterns, and overall file organization
2. **Comment Analysis**: Measures the ratio of comments to executable code to identify well-documented code versus automatically generated or tutorial content
3. **Content Verification**: Ensures files actually contain code rather than data, configuration, or misclassified content 
4. **Language-Specific Patterns**: Applies different criteria based on programming language conventions
5. **Token Efficiency**: Evaluates how efficiently the code can be tokenized for model training

These filters can be applied individually or in combination to create comprehensive quality assessment pipelines. Each filter typically computes a score or makes a binary decision based on configurable thresholds that can be adjusted to match specific requirements.

---

## Usage

Here's an example of applying code filters to a dataset using `ray_curator`:

```python
from ray_curator.backends.xenna import XennaExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.filters.code import (
    PythonCommentToCodeFilter,
    NumberOfLinesOfCodeFilter,
    AlphaFilter
)
from ray_curator.stages.io.reader.jsonl import JsonlReader
from ray_curator.stages.io.writer.jsonl import JsonlWriter
from ray_curator.stages.modules.score_filter import ScoreFilter

def create_code_filtering_pipeline(data_dir: str, output_dir: str) -> Pipeline:
    """Create a pipeline for filtering code quality."""
    
    # Define pipeline
    pipeline = Pipeline(
        name="code_quality_filtering", 
        description="Filter code based on quality metrics"
    )
    
    # Add stages
    # 1. Reader stage - load JSONL files
    pipeline.add_stage(
        JsonlReader(
            file_paths=data_dir,
            files_per_partition=2,
            reader="pandas"
        )
    )
    
    # 2. Comment ratio filtering
    pipeline.add_stage(
        ScoreFilter(
            PythonCommentToCodeFilter(
                min_comment_to_code_ratio=0.01,
                max_comment_to_code_ratio=0.8
            ),
            text_field="content",
            score_field="comment_ratio"
        )
    )
    
    # 3. Line count filtering
    pipeline.add_stage(
        ScoreFilter(
            NumberOfLinesOfCodeFilter(min_lines=5, max_lines=1000),
            text_field="content", 
            score_field="line_count"
        )
    )
    
    # 4. Alphabetic content filtering
    pipeline.add_stage(
        ScoreFilter(
            AlphaFilter(min_alpha_ratio=0.3),
            text_field="content",
            score_field="alpha_ratio"
        )
    )
    
    # 5. Writer stage - save filtered results
    pipeline.add_stage(
        JsonlWriter(output_dir=output_dir)
    )
    
    return pipeline

def main():
    # Create and run pipeline
    pipeline = create_code_filtering_pipeline("./code_data", "./filtered_code")
    
    # Print pipeline description
    print(pipeline.describe())
    
    # Execute with distributed backend
    executor = XennaExecutor()
    results = pipeline.run(executor)
    
    # Process results
    print(f"Pipeline completed! Processed {len(results)} batches")

if __name__ == "__main__":
    main()
```

## Available Code Filters

NeMo Curator offers several specialized filters for code content:

### Comment Analysis Filters

| Filter | Description | Key Parameters | Default Values |
|--------|-------------|----------------|---------------|
| **PythonCommentToCodeFilter** | Filters Python files based on comment-to-code ratio | `min_comment_to_code_ratio`, `max_comment_to_code_ratio` | min=0.01, max=0.85 |
| **GeneralCommentToCodeFilter** | Similar filter for other languages | `language`, `min_comment_to_code_ratio`, `max_comment_to_code_ratio` | min=0.01, max=0.85 |

The comment-to-code ratio is an important metric for code quality. Too few comments may indicate poor documentation, while too many comments might suggest automatically generated code or tutorials:

```python
from ray_curator.stages.filters.code import PythonCommentToCodeFilter, GeneralCommentToCodeFilter
from ray_curator.stages.modules.score_filter import ScoreFilter

# For Python files with docstrings
python_filter = ScoreFilter(
    PythonCommentToCodeFilter(
        min_comment_to_code_ratio=0.05,  # At least 5% comments
        max_comment_to_code_ratio=0.7    # At most 70% comments
    ),
    text_field="content",
    score_field="comment_ratio"
)

# For other languages
cpp_filter = ScoreFilter(
    GeneralCommentToCodeFilter(
        language="text/x-c++",  # MIME type for C++
        min_comment_to_code_ratio=0.02,
        max_comment_to_code_ratio=0.6
    ),
    text_field="content",
    score_field="comment_ratio"
)
```

The `GeneralCommentToCodeFilter` supports various language MIME types:
- `text/x-c++` for C++
- `text/x-java` for Java
- `text/javascript` for JavaScript
- `text/x-ruby` for Ruby
- `text/x-csharp` for C#

### Code Structure Filters

| Filter | Description | Key Parameters | Default Values |
|--------|-------------|----------------|---------------|
| **NumberOfLinesOfCodeFilter** | Filters based on the number of lines | `min_lines`, `max_lines` | min=10, max=20000 |
| **AlphaFilter** | Ensures code has sufficient alphabetic content | `min_alpha_ratio` | 0.25 |
| **TokenizerFertilityFilter** | Measures token efficiency | `path_to_tokenizer` (required), `min_char_to_token_ratio` | ratio=2.5 |

Code structure filters help identify problematic patterns:

```python
from ray_curator.stages.filters.code import NumberOfLinesOfCodeFilter, AlphaFilter
from ray_curator.stages.modules.score_filter import ScoreFilter

# Filter for reasonable line counts
line_filter = ScoreFilter(
    NumberOfLinesOfCodeFilter(
        min_lines=5,     # Filter out tiny snippets
        max_lines=2000   # Filter out extremely long files
    ),
    text_field="content",
    score_field="line_count"
)

# Filter for alphabetic content (avoid large data blobs)
alpha_filter = ScoreFilter(
    AlphaFilter(min_alpha_ratio=0.3),  # At least 30% alphabetic chars
    text_field="content",
    score_field="alpha_ratio"
)
```

The `TokenizerFertilityFilter` helps ensure code is efficiently tokenized:

```python
from ray_curator.stages.filters.code import TokenizerFertilityFilter
from ray_curator.stages.modules.score_filter import ScoreFilter

# Filter for token efficiency
# Note: path_to_tokenizer is required
tokenization_filter = ScoreFilter(
    TokenizerFertilityFilter(
        path_to_tokenizer="/path/to/code_tokenizer.model",  # Required parameter
        min_char_to_token_ratio=2.5  # Each token encodes at least 2.5 chars on average
    ),
    text_field="content",
    score_field="token_ratio"
)
```

This filter helps avoid content that has poor token efficiency, which can impact model training.

### File Format Filters

| Filter | Description | Key Parameters | Default Values |
|--------|-------------|----------------|---------------|
| **XMLHeaderFilter** | Identifies files that are actually XML | `char_prefix_search_length` | 100 |
| **HTMLBoilerplateFilter** | Filters HTML with too much boilerplate | `min_lang_content_ratio`, `min_lang_content_num_chars` | ratio=0.2, chars=100 |
| **PerExtensionFilter** | Applies standards based on file extension | `lang`, `extension`, `metadata_file` | depends on metadata |

## Language-Specific Considerations

Different programming languages have different conventions and characteristics. The `PerExtensionFilter` applies customized filtering based on file extension:

```python
from ray_curator.stages.filters.code import PerExtensionFilter
from ray_curator.stages.modules.score_filter import ScoreFilter

# Apply language-specific filters
python_specific = ScoreFilter(
    PerExtensionFilter(
        lang="python",
        extension=".py",
        metadata_file="code_meta.csv"  # Contains language-specific thresholds
    ),
    text_field="content",
    score_field="extension_score"
)
```

The metadata file can specify different thresholds for metrics like:
- Average line length
- Comment ratio
- Empty line ratio
- Alphabetic content ratio

## Pipeline Configuration

Ray Curator uses programmatic pipeline configuration. Here's how to structure a comprehensive code filtering pipeline:

```python
from ray_curator.pipeline import Pipeline
from ray_curator.stages.filters.code import (
    PythonCommentToCodeFilter,
    NumberOfLinesOfCodeFilter,
    AlphaFilter,
    XMLHeaderFilter
)
from ray_curator.stages.modules.score_filter import ScoreFilter

def create_comprehensive_code_pipeline(data_dir: str, output_dir: str) -> Pipeline:
    """Create a comprehensive code filtering pipeline."""
    
    pipeline = Pipeline(
        name="comprehensive_code_filtering",
        description="Multi-stage code quality filtering"
    )
    
    # Add reader stage  
    from ray_curator.stages.io.reader.jsonl import JsonlReader
    from ray_curator.stages.io.writer.jsonl import JsonlWriter
    pipeline.add_stage(JsonlReader(file_paths=data_dir, files_per_partition=2))
    
    # Add multiple filtering stages
    filters = [
        (PythonCommentToCodeFilter(min_comment_to_code_ratio=0.01, max_comment_to_code_ratio=0.85), "comment_ratio"),
        (NumberOfLinesOfCodeFilter(min_lines=10, max_lines=5000), "line_count"),
        (AlphaFilter(min_alpha_ratio=0.25), "alpha_ratio"),
        (XMLHeaderFilter(), "xml_detected")
    ]
    
    for filter_obj, score_field in filters:
        pipeline.add_stage(
            ScoreFilter(
                filter_obj,
                text_field="content", 
                score_field=score_field
            )
        )
    
    # Add writer stage
    pipeline.add_stage(JsonlWriter(output_dir=output_dir))
    
    return pipeline
```

## Best Practices for Code Filtering

When filtering code datasets, consider these best practices:

1. **Language-specific configurations**: Adjust thresholds based on the programming language

   ```python
   from ray_curator.stages.filters.code import PythonCommentToCodeFilter, GeneralCommentToCodeFilter
   
   # Python tends to have more comments than C
   python_comment_filter = PythonCommentToCodeFilter(min_comment_to_code_ratio=0.05)
   c_comment_filter = GeneralCommentToCodeFilter(language="text/x-c", min_comment_to_code_ratio=0.02)
   ```

2. **Preserve code structure**: Ensure filters don't inadvertently remove valid coding patterns

   ```python
   # Some languages naturally have low comment ratios
   assembly_filter = GeneralCommentToCodeFilter(
       language="text/x-asm",
       min_comment_to_code_ratio=0.001  # Very low minimum for assembly
   )
   ```

3. **Combine with language detection**: Verify file extensions match content

   ```python
   # Language detection for code is available through ray_curator
   from ray_curator.stages.filters.fasttext_filter import FastTextLangId
   
   python_detection = FastTextLangId(
       model_path="/path/to/lid.176.bin",  # Download from fasttext.cc
       min_langid_score=0.8
   )
   
   # Implementation depends on ray_curator pipeline setup
   # Refer to ray_curator documentation for complete examples
   ```

   :::{note}
   The `FastTextLangId` filter requires downloading the FastText language identification model from [fasttext.cc](https://fasttext.cc/docs/en/language-identification.html).
   :::

4. **Avoid over-filtering**: Monitor rejection rates and adjust thresholds as needed

   ```python
   # Track filter statistics in ray_curator pipeline
   def monitor_pipeline_results(results):
       total_input_docs = 0
       total_output_docs = 0
       
       for batch in results:
           if batch is not None:
               total_output_docs += batch.num_items
               # Input count available from metadata
               if 'original_count' in batch._metadata:
                   total_input_docs += batch._metadata['original_count']
       
       if total_input_docs > 0:
           retention_rate = total_output_docs / total_input_docs
           print(f"Pipeline retention rate: {retention_rate:.2%}")
           print(f"Filtered out: {total_input_docs - total_output_docs} documents")
   ```

## Use Cases

::::{tab-set}

:::{tab-item} Cleaning Open Source Code Datasets

```python
from ray_curator.pipeline import Pipeline
from ray_curator.stages.filters.code import NumberOfLinesOfCodeFilter, XMLHeaderFilter, GeneralCommentToCodeFilter
from ray_curator.stages.modules.score_filter import ScoreFilter
from ray_curator.stages.io.reader.jsonl import JsonlReader
from ray_curator.stages.io.writer.jsonl import JsonlWriter

def create_repo_cleaning_pipeline(data_dir: str, output_dir: str) -> Pipeline:
    """Filter to remove non-functional code snippets."""
    
    pipeline = Pipeline(name="repo_cleaning", description="Clean open source repositories")
    
    pipeline.add_stage(JsonlReader(file_paths=data_dir, files_per_partition=2))
    
    # Remove extremely short files
    pipeline.add_stage(ScoreFilter(NumberOfLinesOfCodeFilter(min_lines=3), text_field="content"))
    
    # Remove files with XML preamble (misidentified as code)
    pipeline.add_stage(ScoreFilter(XMLHeaderFilter(), text_field="content"))
    
    # Ensure reasonable comment-to-code ratio
    pipeline.add_stage(ScoreFilter(GeneralCommentToCodeFilter(language="text/x-c++"), text_field="content"))
    
    pipeline.add_stage(JsonlWriter(output_dir=output_dir))
    
    return pipeline
```

:::

:::{tab-item} Training Data Preparation

```python
from ray_curator.pipeline import Pipeline
from ray_curator.stages.filters.code import AlphaFilter, TokenizerFertilityFilter, HTMLBoilerplateFilter
from ray_curator.stages.modules.score_filter import ScoreFilter
from ray_curator.stages.io.reader.jsonl import JsonlReader
from ray_curator.stages.io.writer.jsonl import JsonlWriter

def create_training_data_pipeline(data_dir: str, output_dir: str) -> Pipeline:
    """Prepare high-quality training data."""
    
    pipeline = Pipeline(name="training_prep", description="Prepare training data")
    
    pipeline.add_stage(JsonlReader(file_paths=data_dir, files_per_partition=2))
    
    # Ensure enough alphabetic content (not just symbols or data)
    pipeline.add_stage(ScoreFilter(AlphaFilter(min_alpha_ratio=0.3), text_field="content"))
    
    # Check token efficiency
    pipeline.add_stage(ScoreFilter(TokenizerFertilityFilter(path_to_tokenizer="tokenizer.model"), text_field="content"))
    
    # Remove HTML with mostly boilerplate
    pipeline.add_stage(ScoreFilter(HTMLBoilerplateFilter(min_lang_content_ratio=0.3), text_field="content"))
    
    pipeline.add_stage(JsonlWriter(output_dir=output_dir))
    
    return pipeline
```

:::

::::

By applying these specialized code filters, you can significantly improve the quality of code in your training datasets, leading to better model performance for code-related tasks.
