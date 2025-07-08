# Multilingual Data Curation Pipeline with NeMo Curator

The following files demonstrate how to curate a multilingual Common Crawl dataset using NeMo Curator.
The pipeline is as follows:

- *Step 1*: Download and extract Common Crawl data. By default, the pipeline is set to extract the April 2025 Common Crawl snapshot with the jusText text extraction algorithm and its default settings.
- *Step 2*: Language identification. The [FastText language identification model](https://fasttext.cc/docs/en/language-identification.html) can be downloaded with:

```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

- *Step 3*: Add ID field. In order to run deduplication with NeMo Curator, text datasets must have an ID column.
- *Step 4*: Exact deduplication.
- *Step 5*: Fuzzy deduplication.
- *Step 6*: Exact substring deduplication.

Above, steps 1-2 and step 6 are run on the CPU. Meanwhile, steps 3-5 are run on the GPU.

Steps 2 and 4 (language identification and exact deduplication) are optional but recommended for this pipeline. The `download_common_crawl` function already creates a `language` field using [CLD2](https://github.com/CLD2Owners/cld2), however we recommend using the FastText model for more accurate language detection results. Running fuzzy deduplication will identify both exact and fuzzy duplicates, but running exact deduplication before fuzzy deduplication is recommended to help lower the dataset size and hence lessen the compute requirements for fuzzy deduplication, which is expensive.

For fuzzy deduplication, we recommend using the `5_fuzzy_deduplication.py` script.
If there are limiting factors such as dataset size, compute resources, and/or time, we recommend running each stage of fuzzy deduplication as its own job on Slurm.
Please refer to the `fuzzy_deduplication_stages/` directory for more information on how to do this.

To start a job, please edit `scripts/start-slurm.sh` to match the expectations and parameters of the current job.
We suggest several places to customize parameters, marked by `TODO`s.
You can submit your Slurm job with `sbatch start-slurm.sh`.

To perform exact substring deduplication, please refer to: https://github.com/google-research/deduplicate-text-datasets/tree/master
