# SemDeDup Pipeline

This pipeline is used to cluster and deduplicate data points based on their embeddings.
Please edit `config/sem_dedup_config.yaml` to configure the pipeline and run it using the following commands.


## Pipeline Steps

1) Modify `config/sem_dedup_config.yaml`

2) Compute embeddings:
    ```sh
    semdedup_extract_embeddings --input-data-dir "$INPUT_DATA_DIR" --input-file-type "jsonl" --input-file-extension "json" --input-text-field "text" --config-file "$CONFIG_FILE"
    ```
    **Input:** `input_data_dir/*.jsonl` and YAML file from step (1)

    **Output:** Embedding Parquet files in the `{config.cache_dir}/{config.embeddings_save_loc}` directory

3) Clustering
    ```sh
    semdedup_clustering --id-column "my_id" --config-file "$CONFIG_FILE"
    ```
    **Input:** Output from step (2) and YAML file from step (1)

    **Output:** Under `{config.cache_dir}/{config.clustering_save_loc}` directory, including:

        - `kmeans_centroids.npy`
        - `embs_by_nearest_center` directory, containing `nearest_cent={x}` where x ranges from 0 to `num_clusters - 1`
        - Parquet files within `embs_by_nearest_center/nearest_cent={x}` containing the data points in each cluster

4) Extract duplicate data IDs to remove
    ```sh
    semdedup_extract_duplicate_ids --id-column "my_id" --config-file "$CONFIG_FILE"
    ```
    **Input:** Output from step (3) and YAML file from step (1)

    **Output:** `{config.cache_dir}/{config.clustering_save_loc}/duplicate_ids_{}.parquet`
