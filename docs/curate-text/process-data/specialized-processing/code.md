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
import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import (
    PythonCommentToCodeFilter,
    NumberOfLinesOfCodeFilter,
    AlphaFilter
)

# Load your code dataset
dataset = DocumentDataset.read_json("code_data/*.jsonl")

# Create a filter chain for code quality
filter_step = nc.Sequential([
    nc.ScoreFilter(
        PythonCommentToCodeFilter(
            min_comment_to_code_ratio=0.01,
            max_comment_to_code_ratio=0.8
        ),
        text_field="content",
        score_field="comment_ratio"
    ),
    nc.ScoreFilter(
        NumberOfLinesOfCodeFilter(min_lines=5, max_lines=1000),
        text_field="content",
        score_field="line_count"
    ),
    nc.ScoreFilter(
        AlphaFilter(min_alpha_ratio=0.3),
        text_field="content",
        score_field="alpha_ratio"
    )
])

# Apply the filters
quality_code = filter_step(dataset)

# Save the results
quality_code.to_json("filtered_code/", write_to_filename=True)
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
# For Python files with docstrings
python_filter = nc.ScoreFilter(
    PythonCommentToCodeFilter(
        min_comment_to_code_ratio=0.05,  # At least 5% comments
        max_comment_to_code_ratio=0.7    # At most 70% comments
    ),
    text_field="content"
)

# For other languages
cpp_filter = nc.ScoreFilter(
    GeneralCommentToCodeFilter(
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
line_filter = nc.ScoreFilter(
    NumberOfLinesOfCodeFilter(
        min_lines=5,     # Filter out tiny snippets
        max_lines=2000   # Filter out extremely long files
    ),
    text_field="content"
)

# Filter for alphabetic content (avoid large data blobs)
alpha_filter = nc.ScoreFilter(
    AlphaFilter(min_alpha_ratio=0.3),  # At least 30% alphabetic chars
    text_field="content"
)
```

The `TokenizerFertilityFilter` helps ensure code is efficiently tokenizable:

```python
# Filter for token efficiency
# Note: path_to_tokenizer is required
tokenization_filter = nc.ScoreFilter(
    TokenizerFertilityFilter(
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
python_specific = nc.ScoreFilter(
    PerExtensionFilter(
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
filters:
  - name: ScoreFilter
    filter:
      name: PythonCommentToCodeFilter
      min_comment_to_code_ratio: 0.01
      max_comment_to_code_ratio: 0.85
    text_field: content
    score_field: comment_ratio
  
  - name: ScoreFilter
    filter:
      name: NumberOfLinesOfCodeFilter
      min_lines: 10
      max_lines: 5000
    text_field: content
    score_field: line_count
  
  - name: ScoreFilter
    filter:
      name: AlphaFilter
      min_alpha_ratio: 0.25
    text_field: content
    score_field: alpha_ratio
  
  - name: ScoreFilter
    filter:
      name: XMLHeaderFilter
    text_field: content
    score_field: xml_detected
```

## Best Practices for Code Filtering

When filtering code datasets, consider these best practices:

1. **Language-specific configurations**: Adjust thresholds based on the programming language
   ```python
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
   # First check if the content is actually Python using FastText language ID
   from nemo_curator.filters import FastTextLangId
   
   python_detection = nc.ScoreFilter(
       FastTextLangId(
           model_path="/path/to/lid.176.bin",  # Download from fasttext.cc
           min_langid_score=0.8
       ),
       score_field="language"
   )
   # Then apply Python-specific filters
   python_filters = nc.Sequential([
       python_detection,
       nc.ScoreFilter(PythonCommentToCodeFilter())
   ])
   ```
   
   :::{note}
   The `FastTextLangId` filter requires downloading the fastText language identification model from [fasttext.cc](https://fasttext.cc/docs/en/language-identification.html).
   :::

4. **Avoid over-filtering**: Monitor rejection rates and adjust thresholds as needed
   ```python
   # Track filter statistics
   rejection_stats = {}
   for filter_name, filter_obj in filters.items():
       filter_step = nc.ScoreFilter(filter_obj, text_field="content")
       before_count = len(dataset)
       filtered = filter_step(dataset)
       after_count = len(filtered)
       rejection_stats[filter_name] = (before_count - after_count) / before_count
   ```

## Use Cases

::::{tab-set}

:::{tab-item} Cleaning Open Source Code Datasets
```python
# Filter to remove non-functional code snippets
repo_filter = nc.Sequential([
    # Remove extremely short files
    nc.ScoreFilter(NumberOfLinesOfCodeFilter(min_lines=3)),
    
    # Remove files with XML preamble (misidentified as code)
    nc.ScoreFilter(XMLHeaderFilter()),
    
    # Ensure reasonable comment-to-code ratio
    nc.ScoreFilter(GeneralCommentToCodeFilter(language="text/x-c++"))
])
```
:::

:::{tab-item} Training Data Preparation
```python
training_filter = nc.Sequential([
    # Ensure enough alphabetic content (not just symbols or data)
    nc.ScoreFilter(AlphaFilter(min_alpha_ratio=0.3)),
    
    # Check token efficiency
    nc.ScoreFilter(TokenizerFertilityFilter(path_to_tokenizer="tokenizer.model")),
    
    # Remove HTML with mostly boilerplate
    nc.ScoreFilter(HTMLBoilerplateFilter(min_lang_content_ratio=0.3))
])
```
:::

::::

By applying these specialized code filters, you can significantly improve the quality of code in your training datasets, leading to better model performance for code-related tasks. 