---
description: "Remove downstream task data from training datasets to prevent evaluation contamination and ensure valid benchmarking"
categories: ["how-to-guides"]
tags: ["task-decontamination", "benchmarks", "contamination", "evaluation", "n-grams", "downstream-tasks"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-filter-task-decontamination)=
# Downstream Task Decontamination

## Background

After training, large language models are usually evaluated by their performance on downstream tasks consisting of unseen test data. When dealing with large datasets there is a potential for leakage of this test data into the model's training dataset. Therefore, NVIDIA NeMo Curator follows the approach of [OpenAI GPT3](https://arxiv.org/pdf/2005.14165.pdf) and [Microsoft Turing NLG 530B](https://arxiv.org/abs/2201.11990) to remove sections of documents in your dataset that are present in downstream tasks.

## Usage

The `TaskDecontamination` module provides the central functionality in NVIDIA NeMo Curator. Let's examine this small example:

```python
import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.tasks import Winogrande, Squad, TriviaQA

files = get_all_files_paths_under("books_dataset/", keep_extensions="jsonl")
books = DocumentDataset.read_json(files, add_filename=True)

downstream_tasks = [
    Winogrande(),
    Squad(),
    TriviaQA(),
]

task_decontaminate = nc.TaskDecontamination(downstream_tasks)

decontaminated_books = task_decontaminate(books)

decontaminated_books.to_json("decontaminated_books/", write_to_filename=True)
```

### Parameters

The `TaskDecontamination` class accepts several parameters to control the decontamination process:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tasks` | Required | A single task or list of `DownstreamTask` objects |
| `text_field` | `"text"` | Field in dataset containing document text |
| `max_ngram_size` | `13` | Maximum size of n-grams to check for contamination |
| `max_matches` | `10` | If an n-gram appears more than this many times, it's considered too common and not removed |
| `min_document_length` | `200` | Minimum character length for split documents to be kept |
| `remove_char_each_side` | `200` | Number of characters to remove on either side of matching n-gram |
| `max_splits` | `10` | Maximum number of splits allowed before discarding document entirely |
| `removed_dir` | `None` | Optional directory to save discarded documents |

For example, to use more aggressive removal settings:

```python
task_decontaminate = nc.TaskDecontamination(
    tasks=downstream_tasks,
    max_ngram_size=10,               # Use smaller n-grams for matching
    max_matches=5,                   # Remove n-grams that appear in fewer documents
    remove_char_each_side=300,       # Remove more context around matches
    min_document_length=500          # Keep only longer document fragments
)
```

### Available Downstream Tasks

NVIDIA NeMo Curator provides implementations for many common benchmark tasks. Here's a comprehensive list of the supported tasks:

| Task Category | Available Tasks |
|---------------|----------------|
| **Question Answering** | `Squad`, `TriviaQA`, `Quac`, `WebQA`, `COQA`, `Drop` |
| **Reading Comprehension** | `Race`, `MultiRC`, `Record` |
| **Commonsense Reasoning** | `PIQA`, `Copa`, `Winogrande`, `StoryCloze` |
| **Natural Language Inference** | `ANLI`, `RTE`, `CB`, `WiC` |
| **Knowledge Tasks** | `ArcEasy`, `ArcChallenge`, `OpenBookQA`, `BoolQ`, `Lambada` |
| **Multi-task Benchmarks** | `MMLU`, `BigBenchHard`, `BigBenchLight` |
| **Specialized Tasks** | `WSC`, `NumDasc`, `Multilingual` |

You can import these tasks directly from the `nemo_curator.tasks` module:

```python
from nemo_curator.tasks import Squad, TriviaQA, MMLU, Winogrande, ANLI
```

## Task Decontamination Process

If you'd like more fine-grained control over the task decontamination process, NVIDIA NeMo Curator provides several CLI tools you can manually apply. You can use the `prepare_task_data`, `find_matching_ngrams` and `remove_matching_ngrams` scripts to remove any task data that might be contained (that's "contaminate") within your training data. You'll need a list of your downstream tasks to modify the [task configuration file (lm_tasks.yaml)](../../../../config/lm_tasks.yaml). If your task doesn't already exist as a class, you'll need to construct a class that extends `nemo_curator.tasks.DownstreamTask`.

### 1. Prepare Task N-grams

First, construct the n-grams from task documents using the `prepare_task_data` module:

```bash
prepare_task_data \
    --task-config-file=./config/lm_tasks.yaml \
    --output-task-ngrams=./data/task_ngrams.pkl
```

This module requires a configuration file that specifies how to form n-grams from the task data. An example configuration is provided in `config/lm_tasks.yaml`. This step only needs to be done once per set of tasks, and the resulting pickle file can be reused across datasets.

The n-gram generation process:
1. Extracts text from each task's test examples
2. Tokenizes the text into words
3. Creates n-grams of varying sizes (up to `max_ngram_size`)
4. Stores these n-grams in a dictionary

### 2. Find Matching N-grams

Next, use the `find_matching_ngrams` module to search for matches within your corpus:

```bash
find_matching_ngrams \
    --input-data-dir=<Path to the input directory containing jsonl files> \
    --input-task-ngrams=./data/task_ngrams.pkl \
    --output-matched-ngram-data=./data/matched_ngrams.pkl
```

This module:
1. Loads the precomputed task n-grams
2. Searches each document in your dataset for these n-grams
3. Counts occurrences of each n-gram across the entire corpus
4. Outputs a dictionary of n-grams and their frequencies

### 3. Remove Matching N-grams

Finally, use the `remove_matching_ngrams` module to remove contaminated content:

```bash
remove_matching_ngrams \
    --input-data-dir=<Path to the input directory containing jsonl files> \
    --input-matched-ngrams=./data/matched_ngrams.pkl \
    --output-task-deduped-dir=<Output directory containing task-deduped jsonl files>
```

This module:
1. Loads the matched n-grams and their frequencies
2. Identifies n-grams that appear fewer than `max_matches` times (rare enough to be actual task contamination)
3. For each document containing these n-grams:
   - Removes the n-gram and surrounding characters (up to `remove_char_each_side` on each side)
   - Splits the document at removal points
   - Keeps only the split fragments longer than `min_document_length`
   - Discards documents that require more than `max_splits`

## Creating Custom Downstream Tasks

If you need to decontaminate against a task not included in NeMo Curator, you can create your own task class:

```python
from nemo_curator.tasks import DownstreamTask

class MyCustomTask(DownstreamTask):
    def __init__(self):
        super().__init__()
        self._task_name = "my_custom_task"
        
    def generate_ngrams(self):
        # Load your task's test data
        test_examples = load_my_test_data()
        
        # Process each example and update ngrams
        for example in test_examples:
            # If your task has multiple text fields, process each one
            self._update_ngrams(example["question"])
            self._update_ngrams(example["context"])
            
        return self._ngrams
```

You can then use this custom task with the `TaskDecontamination` module:

```python
task_decontaminate = nc.TaskDecontamination([MyCustomTask()])
```

## Performance Considerations

Task decontamination can be computationally intensive for large datasets. Consider these optimization strategies:

1. **Prioritize important tasks**: Start with the most critical benchmark tasks for your application
2. **Process in batches**: Decontaminate your dataset in manageable chunks
3. **Save intermediate results**: Store the results from each step of the CLI workflow
4. **Adjust n-gram size**: Smaller values of `max_ngram_size` reduce computation but may increase false positives

## References

- [Language Models are Few-Shot Learners (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
- [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B (Smith et al., 2021)](https://arxiv.org/abs/2201.11990) 