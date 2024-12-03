# NeMo Curator CLI Scripts

The following Python scripts are designed to be executed from the command line (terminal) only.

Here, we list all of the Python scripts and their terminal commands:

| Python Command                           | CLI Command                    |
|------------------------------------------|--------------------------------|
| python add_id.py                         | add_id                         |
| python blend_datasets.py                 | blend_datasets                 |
| python download_and_extract.py           | download_and_extract           |
| python filter_documents.py               | filter_documents               |
| python find_exact_duplicates.py          | gpu_exact_dups                 |
| python find_matching_ngrams.py           | find_matching_ngrams           |
| python find_pii_and_deidentify.py        | deidentify                     |
| python get_common_crawl_urls.py          | get_common_crawl_urls          |
| python get_wikipedia_urls.py             | get_wikipedia_urls             |
| python make_data_shards.py               | make_data_shards               |
| python prepare_fasttext_training_data.py | prepare_fasttext_training_data |
| python prepare_task_data.py              | prepare_task_data              |
| python remove_matching_ngrams.py         | remove_matching_ngrams         |
| python separate_by_metadata.py           | separate_by_metadata           |
| python text_cleaning.py                  | text_cleaning                  |
| python train_fasttext.py                 | train_fasttext                 |
| python verify_classification_results.py  | verify_classification_results  |

For more information about the arguments needed for each script, you can use `add_id --help`, etc.

More scripts can be found in the `classifiers`, `fuzzy_deduplication`, and `semdedup` subdirectories.
