# Zyda2
This tutorial demonstrates how to reproduce Zyda2 dataset, that was curated by Zyphra in collaboration with Nvidia using NeMo Curator.

- Download Zyda2 dataset from HuggingFace: https://huggingface.co/datasets/Zyphra/Zyda2
- Zyphra blog: https://www.zyphra.com/post/building-zyda-2
- Nvidia blog: https://developer.nvidia.com/blog/train-highly-accurate-llms-with-the-zyda-2-open-5t-token-dataset-processed-with-nvidia-nemo-curator/

## Tutorial structure
Tutorial is split into separate folders each containing scripts for running corresponding steps:
- `0_processing`: scripts for preprocessing individual datasets into the format optimal for NeMo Curator
- `1_fuzzy_dedup`: scripts for running fuzzy deduplication pipeline
- `2_dupes_removal`: scripts for post processing results of fuzzy dedup and removing duplicated documents
- `3_quality_model`: scripts for running inference of Nvidia's quality model
- `4_filtering`: scripts for filtering

## NeMo Curator setup
Before running this tutorial one needs to set up a Dask cluster, which involves starting one Dask scheduler process on the head node and Dask workers on every compute node.
Below is an example of how things could be configured for running NeMo Curator on multiple nodes of GPUs using Infiniband between the nodes and NVLink between GPUs on the same node.
1. Set the following flags according to you cluster configuration on every node:
```
export DASK_DISTRIBUTED_UCX__CUDA_COPY=True
export DASK_DISTRIBUTED_UCX__TCP=True
export DASK_DISTRIBUTED_UCX__NVLINK=True
export DASK_DISTRIBUTED_UCX__INFINIBAND=True
export DASK_DISTRIBUTED_UCX__RDMACM=True
export DASK_DISTRIBUTED_RMM__POOL_SIZE=1GB
export PROTOCOL=ucx
```
2. Set the location of the scheduler file at `SCHEDULER_FILE`
3. Set the network interface you want to use at `INTERFACE` (if unsure, ask your network administrator for what works with your Infiniband setup)
3. Start Dask scheduler on your head node with `DASK_DISTRIBUTED__COMM__UCX__CREATE_CUDA_CONTEXT=True DASK_DISTRIBUTED__RMM__POOL_SIZE=$RMM_SCHEDULER_POOL_SIZE dask scheduler --scheduler-file $SCHEDULER_FILE --protocol $PROTOCOL --interface $INTERFACE`
4. Start Dask workers on every compute node with `dask-cuda-worker --enable-tcp-over-ucx --enable-nvlink --enable-infiniband --enable-rdmacm --scheduler-file /shared/yury/nemo_scheduler.json --interface $INTERFACE`. [Optional] To help with potential out-of-memory memory issues due to fragmentation, one can set flags `--rmm-async --rmm-release-threshold <threshold>`, which will force RMM to release cache when memory usage is higher than specified threshold (this comes with a performance hit). In addition, Dask supports spilling into CPU RAM, it should allow running workloads when there is not enough VRAM, but it comes with a big performance hit; to enable spilling specify `--enable-cudf-spill` flag.

To comfortably reproduce Zyda2 in 2 days we recommend using a cluster with 8 nodes of H100s (or A100s with 80GB of VRAM, but it will take longer). It could be run with less, but it will run into memory issues and will require spilling into CPU RAM, slowing down processing. Scripts in this tutorial assume that all the data is being stored at a shared storage accessible to all the nodes. However, Dask supports cloud storage (like GCS or AWS S3), so with minor modifications to the scripts one can read and write to the cloud.

## How to reproduce Zyda2 dataset
Below are step-by-step instructions on how to reproduce Zyda2 dataset.
Before running the scripts, please, start Dask cluster as in the instructions above, and make sure to set the following environment variables:
- `DATA_BASE` - base location in your filesystem to save results of processing steps
- `SCHEDULER_FILE` - file created by Dask scheduler when creating Dask cluster
- `CPU_WORKERS` - number of CPU workers for steps that don't require GPUs


### 1. Downloading component datasets
Most source datasets can be simply downloaded by cloning their respective HuggingFace repositories:
- DCLM: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet
- Fineweb-edu-score-2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2
- Zyda: https://huggingface.co/datasets/Zyphra/Zyda
Simply clone them inside the `$DATA_BASE/raw` folder using any of the ways HuggingFace recommends for doing that (e.g., using Git LFS or huggingface-cli: https://huggingface.co/docs/huggingface_hub/en/guides/cli#download-a-dataset-or-a-space). All of the above datasets are already in parquet format, which is suitable for processing with NeMo Curator/Dask.

However, Dolma-CC v1.7 requires special handling, since the Dolma repository only contains links to raw files. One can do the following:
1. Filter Dolma v1.7 file links to only contain the CC component of Dolma. Links can be found here: https://huggingface.co/datasets/allenai/dolma/blob/main/urls/v1_7.txt. Relevant links only contain `cc_en_head`, `cc_en_middle`, or `cc_en_tail` in their names.
2. Download those files, e.g. using wget.
3. Convert those files to parquet format and save to `$DATA_BASE/raw/dolma-v1_7-cc-parquet`

The whole raw dataset contains roughly 12 billion documents with roughly 11.6 trillion `gpt-neox-20b` tokens.

### 2. Preprocessing
NeMo Curator is based on Dask. Dask works best when datasets are split into partitions of small size: https://docs.dask.org/en/stable/dataframe-parquet.html#partition-size. This step includes repartitioning parquet shards to make sure they have optimal size. After some experimentation we decided to limit a partition in-memory size to 512MB.

In addition, we add unique IDs to every document in this step, so that we can easily identify documents at later stages.

This step can be run on CPUs or GPUs.

Run all the Python scripts in the `0_processing` folder and it will create folders in `$DATA_BASE/processed` for all the component datasets.

### 3. Global fuzzy deduplication
NeMo Curator implements minhash LSH fuzzy deduplication technique. The steps involve computing minhashes, identifying duplicate pairs within minhash bands, and then clustering duplicated documents using connected components computation. Minhash LSH does produce false positives and false negatives, and NeMo Curator supports explicitly checking for Jaccard similarity with anchor documents within buckets to filter out false positives. However, this is computationally expensive, and for Zyda2 we did not perform such a check. Given the parameters of minhash LSH, it is possible to theoretically estimate the rate of false positives/negatives, and in our case it is up to 2-3%, which we find acceptable.

The whole procedure is split into four stages organized in the following scripts in folder `1_fuzzy_dedup`:
1. `0_minhash.py` - computing minhashes of every document;
2. `1_lsh.py` - splitting minhashes into bands/buckets and then shuffling the dataset by the band/bucket id;
3. `2_buckets_to_edges.py` - generating duplicate pairs, which serve as edges for connected components computation;
4. `3_connected_components.py` - computing connected components, which are essentially the clusters of duplicates.

Every step produces its own artifacts, which are used by the subsequent steps.

Fuzzy deduplication can only be run on GPUs in NeMo Curator.

#### 1. Computing minhashes
The script for computing minhash signatures is located at `1_fuzzy_dedup/0_minhash.py`.

This stage performs the following operations:
1. Generates 25-grams based on characters
2. Computes minhash signatures with of the size of 128
3. Saves results in `$DATA_BASE/fuzzy/minhash`

This is the most time-consuming step, though it is embarrassingly parallel and doesn't require to use much VRAM.

#### 2. Generating LSH buckets
The script for computing LSH buckets is located at `1_fuzzy_dedup/1_lsh.py`.

For building LSH buckets, we split minhash signatures into 8 bands (each having range 16). This gives us a theoretical 85% Jaccard similarity threshold (meaning that documents that have at least 85% similarity are deemed duplicates).

This step performs the following operation:
1. Splits ID's into dataset_id and doc_id and converts them to integers. This step is no longer necessary, since recent releases of NeMo Curator support long strings on GPUs, but when we started our project this wasn't the default.
2. Splits minhashes of all documents into bands
3. Groups documents into buckets, that correspond to identical values of bands
4. Shuffles the resultant dataset by buckets, so that documents within the same bucket are in the same Dask partition
5. Saves results in `$DATA_BASE/fuzzy/lsh`

This is a memory intensive step and we recommend running it on as many GPU nodes as possible to avoid spilling into CPU RAM.

#### 3. Generating duplicate pairs
The script for computing duplicate pairs located at `1_fuzzy_dedup/2_buckets_to_edges.py`.

This step takes the buckets of duplicated documents from the LSH step and generates a list of duplicate pairs that are subsequently used as edges in the connected components computation.

This step assumes that the results of LSH computation are shuffled by bucket id, hence, it is very important to set the flag `split_row_groups=False` when reading the LSH buckets dataframe.

Results of this stage are saved in `$DATA_BASE/fuzzy/buckets_to_edges`. This step doesn't consume much resources and can be run on one node.

#### 4. Clustering duplicates using connected components
The script for clustering duplicates using connected is located at `1_fuzzy_dedup/3_connected_components.py`.

This stage performs clustering of identified duplicated documents by identifying connected components in a graph, where the nodes are documents and the edges are duplicate pairs.

The results of this stage are saved in `$DATA_BASE/fuzzy/cc`. This stage is memory intensive and we recommend running it on as many GPU nodes as possible.

### 4. Identification of documents to remove
The result of the previous stage is essentially a collection of clusters of duplicated documents. Now we need to decide which of them to actually remove. Scripts in this stage can be run on CPUs.

Since the source datasets were deduplicated with their own strategies, we decided to only remove duplicates found across datasets. We perform this in several steps.

#### 1. Conversion of ID's back to strings
First we need to convert ID's back to the original strings, so that we are able to find documents in the datasets (if you don't perform the id conversion during the LSH step, this can be skipped). This is done in two steps:
1. Generate ID mapping using `2_dupes_removal/1_id_conversion.py` script. This must be run on GPUs (could be even just 1 GPU), since it requires running a hashing function from the `cudf` package.
2. Apply the mapping to the results of connected components, converting IDs into their original form.

After the conversion we are ready to generate a list of documents to remove.

#### 2. Identifying documents to remove
For simplicity we explicitly group all the duplicates by their cluster ID, then compute counts of sources of duplicated documents in every cluster and save the results to disk. This is done using the script at `2_dupes_removal/2_compute_counts.py`.

Then we identify cross duplicates that we need to remove in the script `2_dupes_removal/3_prep_dupes.py`. There we use the following ranking of the sources (from highest to lowest): Fineweb-edu-score-2 -> DCLM -> Zyda1 -> Dolma-CC. We only identify duplicates that appear in several datasets, while preserving internal duplicates intact. Because Fineweb-edu-score-2 has the top ranking, we don't remove any dupes from it.

Then we convert identified document ID's into a format most suitable for easy removal of documents. The scripts `2_dupes_removal/4_get_dupes_*.py` perform this operation for every component. Every ID generated in the preprocessing step actually encodes the folder and the partition the document is coming from and also the explicit row in that partition. So once we decode this information, it is straightforward to remove duplicates.

The removal of duplicates is actually performed by bash scripts `2_dupes_removal/run_remove_dupes*.sh`, which runs Python script `2_dupes_removal/remove_dupes.py` for every component.

The deduplicated datasets are saved explicitly in the `$DATA_BASE/deduped` folder (except for DCLM, which we save in the folder `$DATA_BASE/zyda2` as it is in its final version).

### 5. Running quality model predictions
We ran a quality model classifier on Zyda1 and Dolma-CC v1.7 portions of our dataset. To run the prediction, use bash script `3_quality_model/run_quality_classifier.sh`. It calls the Python script `3_quality_model/run_quality_classifier.py` for all the components. All the results are saved in `$DATA_BASE/deduped-with-quality`. This step must be run on GPUs.

### 6. Filtering
As the final step we perform filtering on some components of our dataset.

We convert Fineweb-edu-score-2 into Fineweb-edu by keeping only the documents with edu score >=3. In principle, this dataset should be the same as the official version of Fineweb-edu. However, to be consistent we performed our own filtering in the script `4_filtering/filter_fwe.py`.

We only keep documents marked as High quality in Zyda1 and Dolma-CC. To perform filtering of those datasets, run scripts `4_filtering/run_filter_zyda.sh` and `4_filtering/run_filter_dolma.sh`.

The results are saved in the `$DATA_BASE/zyda2` folder.

### 7. Final dataset
The final datasets can be found in `$DATA_BASE/zyda2`, organized in folders corresponding to different components.
