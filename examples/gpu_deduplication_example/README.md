### Deduplication Steps

> [!CAUTION]
> The examples references here are outdated and will be replaced with an example using the Python API directly. For more details on the scripts refer to [nemo_curator/scripts/fuzzy_deduplication](/nemo_curator/scripts/fuzzy_deduplication)

1. Exact dedup
    1. Input: Data directories
    2. Output: exact_duplicates.parquet. List of exact duplicates and the document hash.

Fuzzy Dedup
1. Minhashes (Compute minhashes)
    1. Input: Data Directories
    2. Output: minhashes.parquet for each data dir.
2. Buckets (Minhash Buckets)
    1. Input: Minhash directories
    2. Output: Buckets.parquet
3. Jaccard Map Buckets + Jaccard shuffle
    1. Input: Buckets.parquet + Data Dir
    2. Output: Shuffled docs.parquet
4. Jaccard compute
    1. Input: Shuffled docs.parquet
    2. Output: dedup_final_results.parquet
5. Connected Components
    1. Input: Dedup_final_Results.parquet
    2. Output: connected_components.parquet


While calling the main `run-workflow.sh` script that points to these runscripts users can also set the relevant `LIBCUDF_CUFILE_POLICY`.
It is reccomended to set `LIBCUDF_CUFILE_POLICY=OFF` for all runs calling the script.
