# NeMo Curator Python API examples

This directory contains multiple Python scripts with examples of how to use various NeMo Curator classes and functions.
The goal of these examples is to give the user an overview of many of the ways your text data can be curated.
These include:

| Python Script                         | Description                                                                                                   |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------|
| blend_and_shuffle.py                  | Combine multiple datasets into one with different amounts of each dataset, then randomly permute the dataset. |
| classifier_filtering.py               | Train a fastText classifier, then use it to filter high and low quality data.                                 |
| download_arxiv.py                     | Download Arxiv tar files and extract them.                                                                    |
| download_common_crawl.py              | Download Common Crawl WARC snapshots and extract them.                                                        |
| download_wikipedia.py                 | Download the latest Wikipedia dumps and extract them.                                                         |
| exact_deduplication.py                | Use the `ExactDuplicates` class to perform exact deduplication on text data.                                  |
| find_pii_and_deidentify.py            | Use the `PiiModifier` and `Modify` classes to remove personally identifiable information from text data.      |
| fuzzy_deduplication.py                | Use the `FuzzyDuplicatesConfig` and `FuzzyDuplicates` classes to perform fuzzy deduplication on text data.    |
| identify_languages.py                 | Use `FastTextLangId` to filter data by language                                                               |
| raw_download_common_crawl.py          | Download the raw compressed WARC files from Common Crawl without extracting them.                             |
| semdedup_example.py                   | Use the `SemDedup` class to perform semantic deduplication on text data.                                      |
| task_decontamination.py               | Remove segments of downstream evaluation tasks from a dataset.                                                |
| translation_example.py                | Create and use an `IndicTranslation` model for language translation.                                          |

Before running any of these scripts, we strongly recommend displaying `python <script name>.py --help` to ensure that any needed or relevant arguments are specified.

The `classifiers`, `k8s`, `nemo_run`, and `slurm` subdirectories contain even more examples of NeMo Curator's capabilities.
