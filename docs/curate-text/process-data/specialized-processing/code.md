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

Here's an example of applying code filters to a dataset:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import (
    PythonCommentToCodeFilter,
    NumberOfLinesOfCodeFilter,
    AlphaFilter
)

# Create pipeline
pipeline = Pipeline(name="code_quality_filtering")

# Load your code dataset
reader = JsonlReader(
    file_paths="code_data/*.jsonl",
    fields=["content", "id"]  # Specify fields to read
)
pipeline.add_stage(reader)

# Add filter stages for code quality
pipeline.add_stage(ScoreFilter(
    score_fn=PythonCommentToCodeFilter(
        min_comment_to_code_ratio=0.01,
        max_comment_to_code_ratio=0.8
    ),
    text_field="content",
    score_field="comment_ratio"
))
pipeline.add_stage(ScoreFilter(
    score_fn=NumberOfLinesOfCodeFilter(min_lines=5, max_lines=1000),
    text_field="content",
    score_field="line_count"
))
pipeline.add_stage(ScoreFilter(
    score_fn=AlphaFilter(min_alpha_ratio=0.3),
    text_field="content",
    score_field="alpha_ratio"
))

# Add output stage
writer = JsonlWriter(path="filtered_code/")
pipeline.add_stage(writer)

# Execute pipeline
results = pipeline.run()
```

## Available Code Filters

NeMo Curator offers several specialized filters for code content:

### Comment Analysis Filters

| Filter | Description | Key Parameters | Default Values |
|--------|-------------|----------------|---------------|
| **PythonCommentToCodeFilter** | Filters Python files based on comment-to-code ratio | `min_comment_to_code_ratio`, `max_comment_to_code_ratio` | min=0.01, max=0.85 |
| **GeneralCommentToCodeFilter** | Similar filter for other languages | `language`, `min_comment_to_code_ratio`, `max_comment_to_code_ratio` | min=0.01, max=0.85 |

The comment-to-code ratio is an important metric for code quality. Low comment ratios may indicate poor documentation, while high comment ratios might suggest automatically generated code or tutorials:

```python
# For Python files with docstrings
python_filter = ScoreFilter(
    score_fn=PythonCommentToCodeFilter(
        min_comment_to_code_ratio=0.05,  # At least 5% comments
        max_comment_to_code_ratio=0.7    # At most 70% comments
    ),
    text_field="content"
)

# For other languages
cpp_filter = ScoreFilter(
    score_fn=GeneralCommentToCodeFilter(
        language="text/x-c++",  # MIME type for C++
        min_comment_to_code_ratio=0.02,
        max_comment_to_code_ratio=0.6
    ),
    text_field="content"
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
# Filter for reasonable line counts
line_filter = ScoreFilter(
    score_fn=NumberOfLinesOfCodeFilter(
        min_lines=5,     # Filter out tiny snippets
        max_lines=2000   # Filter out extremely long files
    ),
    text_field="content"
)

# Filter for alphabetic content (avoid large data blobs)
alpha_filter = ScoreFilter(
    score_fn=AlphaFilter(min_alpha_ratio=0.3),  # At least 30% alphabetic chars
    text_field="content"
)
```

The `TokenizerFertilityFilter` helps ensure code has efficient token encoding:

```python
# Filter for token efficiency
# Note: path_to_tokenizer is required
tokenization_filter = ScoreFilter(
    score_fn=TokenizerFertilityFilter(
        path_to_tokenizer="/path/to/code_tokenizer.model",  # Required parameter
        min_char_to_token_ratio=2.5  # Each token encodes at least 2.5 chars on average
    ),
    text_field="content"
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
# Apply language-specific filters
python_specific = ScoreFilter(
    score_fn=PerExtensionFilter(
        lang="python",
        extension=".py",
        metadata_file="code_meta.csv"  # Contains language-specific thresholds
    ),
    text_field="content"
)
```

The metadata file can specify different thresholds for metrics like:

- Average line length
- Comment ratio
- Empty line ratio
- Alphabetic content ratio

## Filter Configuration

A typical configuration for code filtering in YAML format:

```yaml
stages:
  - name: JsonlReader
    file_paths: "code_data/*.jsonl"
    fields: ["content", "id"]
  
  - name: ScoreFilter
    score_fn:
      name: PythonCommentToCodeFilter
      min_comment_to_code_ratio: 0.01
      max_comment_to_code_ratio: 0.85
    text_field: content
    score_field: comment_ratio
  
  - name: ScoreFilter
    score_fn:
      name: NumberOfLinesOfCodeFilter
      min_lines: 10
      max_lines: 5000
    text_field: content
    score_field: line_count
  
  - name: ScoreFilter
    score_fn:
      name: AlphaFilter
      min_alpha_ratio: 0.25
    text_field: content
    score_field: alpha_ratio
  
  - name: ScoreFilter
    score_fn:
      name: XMLHeaderFilter
    text_field: content
    score_field: xml_detected
  
  - name: JsonlWriter
    path: "filtered_code/"
```

## Best Practices for Code Filtering

When filtering code datasets, consider these best practices:

1. **Language-specific configurations**: Adjust thresholds based on the programming language

   ```python
   # Python tends to have more comments than C
   python_comment_filter = ScoreFilter(
       score_fn=PythonCommentToCodeFilter(min_comment_to_code_ratio=0.05),
       text_field="content"
   )
   c_comment_filter = ScoreFilter(
       score_fn=GeneralCommentToCodeFilter(language="text/x-c", min_comment_to_code_ratio=0.02),
       text_field="content"
   )
   ```

2. **Preserve code structure**: Ensure filters don't inadvertently remove valid coding patterns

   ```python
   # Some languages naturally have low comment ratios
   assembly_filter = ScoreFilter(
       score_fn=GeneralCommentToCodeFilter(
           language="text/x-asm",
           min_comment_to_code_ratio=0.001  # Very low minimum for assembly
       ),
       text_field="content"
   )
   ```

3. **Combine with language detection**: Verify file extensions match content

   ```python
   # First check if the content is actually Python using FastText language ID
   from nemo_curator.stages.text.filters import FastTextLangId
   from nemo_curator.pipeline import Pipeline
   
   # Create pipeline for Python code filtering with language detection
   pipeline = Pipeline(name="python_code_filtering")
   
   # Add language detection stage
   pipeline.add_stage(ScoreFilter(
       score_fn=FastTextLangId(
           model_path="/path/to/lid.176.bin",  # Download from fasttext.cc
           min_langid_score=0.8
       ),
       text_field="content",
       score_field="language"
   ))
   
   # Then apply Python-specific filters
   pipeline.add_stage(ScoreFilter(
       score_fn=PythonCommentToCodeFilter(),
       text_field="content"
   ))
   ```

   :::{note}
   The `FastTextLangId` filter requires downloading the FastText language identification model from [fasttext.cc](https://fasttext.cc/docs/en/language-identification.html).
   :::

4. **Avoid over-filtering**: Track rejection rates and adjust thresholds as needed

   ```python
   # Track filter statistics by running individual filters and measuring results
   from nemo_curator.stages.text.io.reader import JsonlReader
   
   # Load dataset for testing
   reader = JsonlReader(file_paths="test_data/*.jsonl")
   
   # Test individual filters to measure rejection rates
   filters_to_test = {
       "python_comment": PythonCommentToCodeFilter(),
       "line_count": NumberOfLinesOfCodeFilter(min_lines=5, max_lines=1000),
       "alpha_content": AlphaFilter(min_alpha_ratio=0.3)
   }
   
   # Note: Actual statistics collection would require running the pipeline
   # and analyzing the results to determine optimal thresholds
   ```

## Use Cases

::::{tab-set}

:::{tab-item} Cleaning Open Source Code Datasets

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.modules import ScoreFilter

# Create pipeline to filter non-functional code snippets
pipeline = Pipeline(name="code_cleaning")

# Remove extremely short files
pipeline.add_stage(ScoreFilter(
    score_fn=NumberOfLinesOfCodeFilter(min_lines=3),
    text_field="content"
))

# Remove files with XML preamble (misidentified as code)
pipeline.add_stage(ScoreFilter(
    score_fn=XMLHeaderFilter(),
    text_field="content"
))

# Ensure reasonable comment-to-code ratio
pipeline.add_stage(ScoreFilter(
    score_fn=GeneralCommentToCodeFilter(language="text/x-c++"),
    text_field="content"
))
```

:::

:::{tab-item} Training Data Preparation

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.modules import ScoreFilter

# Create pipeline for training data preparation
pipeline = Pipeline(name="training_data_prep")

# Ensure enough alphabetic content (not just symbols or data)
pipeline.add_stage(ScoreFilter(
    score_fn=AlphaFilter(min_alpha_ratio=0.3),
    text_field="content"
))

# Check token efficiency
pipeline.add_stage(ScoreFilter(
    score_fn=TokenizerFertilityFilter(path_to_tokenizer="tokenizer.model"),
    text_field="content"
))

# Remove HTML with mostly boilerplate
pipeline.add_stage(ScoreFilter(
    score_fn=HTMLBoilerplateFilter(min_lang_content_ratio=0.3),
    text_field="content"
))
```

:::

::::

By applying these specialized code filters, you can improve the quality of code in your training datasets, leading to better model performance for code-related tasks.
