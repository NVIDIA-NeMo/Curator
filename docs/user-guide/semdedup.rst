.. _data-curator-semdedup:

#######################################################
Semantic Deduplication
#######################################################

-----------------------------------------
Background
-----------------------------------------

Semantic deduplication is an advanced technique for removing redundant data from large datasets by identifying and eliminating semantically similar data points.
Unlike exact or fuzzy deduplication, which focus on textual similarity, semantic deduplication leverages the semantic meaning of the content to identify duplicates.

As outlined in the paper `SemDeDup: Data-efficient learning at web-scale through semantic deduplication <https://arxiv.org/pdf/2303.09540>`_ by Abbas et al.,
this method can significantly reduce dataset size while maintaining or even improving model performance.
 Semantic deduplication is particularly effective for large, uncurated web-scale datasets, where it can remove up to 50% of the data with minimal performance loss.
The semantic deduplication module in NeMo Curator uses embeddings from to identify and remove "semantic duplicates" - data pairs that are semantically similar but not exactly identical.
While this documentation primarily focuses on text-based deduplication, the underlying principles can be extended to other modalities with appropriate embedding models.

-----------------------------------------
How It Works
-----------------------------------------

The SemDeDup algorithm consists of the following main steps:

1. Embedding Generation: Each data point is embedded using a pre-trained model.
2. Clustering: The embeddings are clustered into k clusters using k-means clustering.
3. Similarity Computation: Within each cluster, pairwise cosine similarities are computed.
4. Duplicate Identification: Data pairs with cosine similarity above a threshold are considered semantic duplicates.
5. Duplicate Removal: From each group of semantic duplicates within a cluster, one representative datapoint is kept (typically the one with the lowest cosine similarity to the cluster centroid) and the rest are removed.

-----------------------------------------
Configuration
-----------------------------------------

Semantic deduplication in NeMo Curator can be configured using a YAML file. Here's an example `sem_dedup_config.yaml`:

.. code-block:: yaml

    # Configuration file for semantic dedup
    cache_dir: "semdedup_cache"
    num_files: -1
    id_col_name: "id"
    id_col_type: "int"
    input_column: "text"

    # Embeddings configuration
    embeddings_save_loc: "embeddings"
    embedding_model_name_or_path: "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: 128
    embedding_max_mem_gb: 25

    # Clustering configuration
    clustering_save_loc: "clustering_results"
    n_clusters: 1000
    seed: 1234
    max_iter: 100
    kmeans_with_cos_dist: false

    # Semdedup configuration
    which_to_keep: "hard"
    largest_cluster_size_to_process: 100000
    sim_metric: "cosine"

    # Extract dedup configuration
    eps_thresholds:
      - 0.01
      - 0.001

    # Which threshold to use for extracting deduped data
    eps_to_extract: 0.01

You can customize this configuration file to suit your specific needs and dataset characteristics.

-----------------------------------------
Changing Embedding Models
-----------------------------------------

One of the key advantages of the semantic deduplication module is its flexibility in using different pre-trained models for embedding generation.
You can easily change the embedding model by modifying the `embedding_model_name_or_path` parameter in the configuration file.

For example, to use a different sentence transformer model, you could change:

.. code-block:: yaml

    embedding_model_name_or_path: "sentence-transformers/all-MiniLM-L6-v2"

to:

.. code-block:: yaml

    embedding_model_name_or_path: "facebook/opt-125m"

The module supports various types of models, including:

1. Sentence Transformers: Ideal for text-based semantic similarity tasks.
2. Custom models: You can use your own pre-trained models by specifying the path to the model.

When changing the model, ensure that:

1. The model is compatible with the data type you're working with (primarily text for this module).
2. You adjust the `embedding_batch_size` and `embedding_max_mem_gb` parameters as needed, as different models may have different memory requirements.
3. The chosen model is appropriate for the language or domain of your dataset.

By selecting an appropriate embedding model, you can optimize the semantic deduplication process for your specific use case and potentially improve the quality of the deduplicated dataset.

-----------------------------------------
Deduplication Thresholds
-----------------------------------------

The semantic deduplication process is controlled by two key threshold parameters:

.. code-block:: yaml

    eps_thresholds:
      - 0.01
      - 0.001

    eps_to_extract: 0.01

1. `eps_thresholds`: A list of similarity thresholds used to compute semantic matches. Each threshold represents a different level of strictness in determining duplicates.
                     Lower values are more strict, requiring higher similarity for documents to be considered duplicates.

2. `eps_to_extract`: The specific threshold used for the final extraction of deduplicated data.
                     This value must be one of the thresholds listed in `eps_thresholds`.

This two-step approach offers several advantages:
- Flexibility to compute matches at multiple thresholds without rerunning the entire process.
- Ability to analyze the impact of different thresholds on your dataset.
- Option to fine-tune the final threshold based on specific needs without recomputing all matches.

Choosing appropriate thresholds:
- Lower thresholds (e.g., 0.001): More strict, resulting in less deduplication but higher confidence in the identified duplicates.
- Higher thresholds (e.g., 0.1): Less strict, leading to more aggressive deduplication but potentially removing documents that are only somewhat similar.

It's recommended to experiment with different threshold values to find the optimal balance between data reduction and maintaining dataset diversity and quality.
The impact of these thresholds can vary depending on the nature and size of your dataset.

Remember, if you want to extract data using a threshold that's not in `eps_thresholds`, you'll need to recompute the semantic matches with the new threshold included in the list.

-----------------------------------------
Usage
-----------------------------------------

Before running semantic deduplication, ensure that each document/datapoint in your dataset has a unique identifier.
You can use the `add_id` module from NeMo Curator if needed:

.. code-block:: python

    from nemo_curator import AddId
    from nemo_curator.datasets import DocumentDataset

    add_id = AddId(id_field="doc_id")
    dataset = DocumentDataset.read_json("input_file_path", add_filename=True)
    id_dataset = add_id(dataset)
    id_dataset.to_json("output_file_path", write_to_filename=True)


To perform semantic deduplication, you can either use individual components or the SemDedup class with a configuration file:

Using individual components:

1. Embedding Creation:

.. code-block:: python

    from nemo_curator import EmbeddingCreator

    # Step 1: Embedding Creation
    embedding_creator = EmbeddingCreator(
        embedding_model_name_or_path="path/to/pretrained/model",
        embedding_max_mem_gb=32,
        embedding_batch_size=128,
        embedding_output_dir="path/to/output/embeddings",
        input_column="text",
        logger="path/to/log/dir"
    )
    embeddings_dataset = embedding_creator(dataset)


2. Clustering:

.. code-block:: python

    from nemo_curator import ClusteringModel

    # Step 2: Clustering
    clustering_model = ClusteringModel(
        id_col="doc_id",
        max_iter=100,
        n_clusters=50000,
        clustering_output_dir="path/to/output/clusters",
        logger="path/to/log/dir"
    )
    clustered_dataset = clustering_model(embeddings_dataset)

1. Semantic Deduplication:

.. code-block:: python

    from nemo_curator import SemanticClusterLevelDedup

    # Step 3: Semantic Deduplication
    semantic_dedup = SemanticClusterLevelDedup(
        n_clusters=50000,
        emb_by_clust_dir="path/to/embeddings/by/cluster",
        sorted_clusters_dir="path/to/sorted/clusters",
        id_col="doc_id",
        id_col_type="str",
        which_to_keep="hard",
        output_dir="path/to/output/deduped",
        logger="path/to/log/dir"
    )
    semantic_dedup.compute_semantic_match_dfs()
    deduplicated_dataset_ids = semantic_dedup.extract_dedup_data(eps_to_extract=0.07)

1. Alternatively, you can use the SemDedup class to perform all steps:

.. code-block:: python

    from nemo_curator import SemDedup, SemDedupConfig
    import yaml

    # Load configuration from YAML file
    with open("sem_dedup_config.yaml", "r") as config_file:
        config_dict = yaml.safe_load(config_file)

    # Create SemDedupConfig object
    config = SemDedupConfig(**config_dict)

    # Initialize SemDedup with the configuration
    sem_dedup = SemDedup(config, logger="path/to/log/dir")

    # Perform semantic deduplication
    deduplicated_dataset_ids = sem_dedup(dataset)

This approach allows for easy experimentation with different configurations and models without changing the core code.

-----------------------------------------
Parameters
-----------------------------------------

Key parameters in the configuration file include:

- `embedding_model_name_or_path`: Path or identifier for the pre-trained model used for embedding generation.
- `embedding_max_mem_gb`: Maximum memory usage for the embedding process.
- `embedding_batch_size`: Number of samples to process in each embedding batch.
- `n_clusters`: Number of clusters for k-means clustering.
- `eps_to_extract`: Deduplication threshold. Higher values result in more aggressive deduplication.
- `which_to_keep`: Strategy for choosing which duplicate to keep ("hard" or "soft").

-----------------------------------------
Output
-----------------------------------------

The semantic deduplication process produces a deduplicated dataset, typically reducing the dataset size by 20-50% while maintaining or improving model performance. The output includes:

1. Embeddings for each datapoint
2. Cluster assignments for each datapoint
3. A list of semantic duplicates
4. The final deduplicated dataset

-----------------------------------------
Performance Considerations
-----------------------------------------

Semantic deduplication is computationally intensive, especially for large datasets. However, the benefits in terms of reduced training time and improved model performance often outweigh the upfront cost. Consider the following:

- Use GPU acceleration for faster embedding generation and clustering.
- Adjust the number of clusters (`n_clusters`) based on your dataset size and available computational resources.
- The `eps_to_extract` parameter allows you to control the trade-off between dataset size reduction and potential information loss.

For more details on the algorithm and its performance implications, refer to the original paper: `SemDeDup: Data-efficient learning at web-scale through semantic deduplication <https://arxiv.org/pdf/2303.09540>`_ by Abbas et al.
