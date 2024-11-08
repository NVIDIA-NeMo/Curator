
.. _data-curator-gpu-deduplication:

#######################################################
GPU Accelerated Exact and Fuzzy Deduplication
#######################################################

=========================================
Background
=========================================

The exact and fuzzy document-level deduplication modules in NeMo Curator aim to reduce the occurrence of duplicate and
near-duplicate documents in a dataset. Both functionalities are supported in NeMo Curator and accelerated using `RAPIDS <https://rapids.ai>`_.

The main motivation for this is that training on randomly selected documents for many epochs can be sub-optimal to downstream performance for language models.
For more information on when this is harmful, please see `Muennighoff et al., 2023 <https://arxiv.org/abs/2305.16264>`_ and `Tirumala et al., 2023 <https://arxiv.org/abs/2308.12284>`_.

=========================================
Exact Deduplication
=========================================

Exact deduplication refers to removing identical documents (i.e., document strings that are equal) from the dataset.

As exact deduplication requires significantly less compute, we typically will run exact deduplication before fuzzy deduplication.
Also, from our experience in deduplicating Common Crawl snapshots, a significant portion (as high as ~40%) of the duplicates can be exact duplicates.

-----------------------------------------
How It Works
-----------------------------------------

Exact dedpulication works by hashing each document and only keeping one document per hash.
Running exact deduplication works on both CPU- and GPU-based backends.

-----------------------------------------
Usage
-----------------------------------------

.. _exactdup_pyapi:

""""""""""""
Python API
""""""""""""

.. note::
    Before running exact deduplication, you need to ensure that the dataset contains a unique ID for each document.
    If needed, you can use the :code:`add_id` module within NeMo Curator to accomplish this.

    .. code-block:: python

      from nemo_curator import AddId
      from nemo_curator.datasets import DocumentDataset

      add_id = AddId(id_field="my_id", id_prefix="doc_prefix")
      dataset = DocumentDataset.read_json("input_file_path")
      id_dataset = add_id(dataset)
      id_dataset.to_parquet("/path/to/parquet/data")

After ensuring your dataset has a unique ID field (or creating one with the code above), you can perform exact deduplication as follows:

.. code-block:: python

    from nemo_curator import ExactDuplicates
    from nemo_curator.datasets import DocumentDataset

    # Initialize the deduplication object
    ExactDups = ExactDuplicates(id_field="my_id", text_field="text")

    dataset = DocumentDataset.read_parquet(
        input_files="/path/to/parquet/data",
        backend="cudf",  # or "pandas" for CPU
    )

    duplicate_docs = ExactDups(dataset)

    """
    Sample output:
    my_id                  _hashes
    22   doc_prefix-37820  e7cb1e88a7a30ea101d33e0c4c8857ef
    70   doc_prefix-56261  bfce4501b9caa93cb3daccd6db1f13af
    75   doc_prefix-56271  bfce4501b9caa93cb3daccd6db1f13af
    84   doc_prefix-52261  0f763a2937d57b9d96bf9f220e55f2bd
    107  doc_prefix-52271  0f763a2937d57b9d96bf9f220e55f2bd
    """

.. tip::
  A more comprehensive example, including how to remove documents from a corpus using the list of
  duplicate IDs generated from the exact deduplication step above, can be found in `examples/exact_deduplication.py <https://github.com/NVIDIA/NeMo-Curator/blob/main/examples/exact_deduplication.py>`_.

""""""""""""
CLI Utility
""""""""""""
Assuming that a unique ID has been added to each document, users can proceed with finding exact duplicates
as follows:

* Find Exact Duplicates
    1. Input: Data directories
    2. Output: ``_exact_duplicates.parquet``. List of exact duplicates and the document hash.

.. code-block:: bash

        # same as `python nemo_curator/scripts/find_exact_duplicates.py`
         gpu_exact_dups \
           --input-data-dirs /path/to/jsonl/dir1 /path/to/jsonl/dir2 \
           --output-dir /path/to/output_dir \
           --input-json-text-field text_column_name \
           --input-json-id-field id_column_name \
           --log-dir ./
           # --scheduler-file /path/to/file.json

All CLI scripts are included in the :code:`nemo_curator/scripts/` subdirectory.

.. caution::
    The CLI utilities are limited to JSONL datasets and only work with GPU-based backends.
    For different dataset formats or backends use the :ref:`exactdup_pyapi`.

=========================================
Fuzzy Deduplication
=========================================

When removing near-duplicates within the corpus, we perform fuzzy deduplication at the document level in order to remove documents with
high Jaccard similarity scores. Our approach closely resembles the approach described in `Smith et al., 2020 <https://arxiv.org/abs/2201.11990>`_.

-----------------------------------------
How It Works
-----------------------------------------

This approach can essentially be split into the following stages:

1. **Compute Minhashes**: The first stage involves computing `MinHash <https://en.wikipedia.org/wiki/MinHash>`_ Signatures on documents.
   NeMo Curator currently only supports character-based n-grams for MinHashing. An approximate metric of ~4.5 characters per word can be used to determine the n-gram size for users familiar with word-based ngrams.
2. **LSH** *(Locality Sensitive Hashing)*: Perform `LSH <https://en.wikipedia.org/wiki/Locality-sensitive_hashing>`_
   to find candidate duplicates.

3. **Buckets to Edgelist**: If not using the false positive check, we directly convert the LSH buckets to edges for the connected components computation.

3. **False Positive Check** *(optional alternative to Buckets to Edgelist)*: Due to the approximate nature of the bucketing via MinHash + LSH
   (`Leskovec et al., 2020 <http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf>`_), NeMo Curator provides the option to further
   process each of the buckets by computing some pairwise Jaccard similarity scores between documents in each bucket and filter out false positives that might have been hashed into the same bucket.

  a. **Jaccard Map Buckets:** Since buckets generated by LSH can have high cardinality, we map multiple LSH buckets to larger batches for
     efficient processing. Aditionally we assign a few documents (controlled via :code:`num_anchor_docs`) for each bucket to be candidate documents
     for pairwise Jaccard similarity computations within that bucket.
  b. **Jaccard Shuffle**: Store documents from the original dataset into new directories and files such that all documents in the same batch (bucket)
     are stored together. This allows parallelizing pairwise Jaccard similarity computations across different buckets.
  c. **Jaccard Compute**: Compute Jaccard similarity scores between all pairs of documents in each bucket to the candidate anchor docs.

4. **Connected Components**: Due to the approximate nature of LSH, documents that are near duplicates may be assigned into different buckets with a few overlapping documents
   between these buckets. We use a GPU accelerated connected components algorithm to find all connected components in the graph formed by the edges between documents in the same bucket.

The result from the connected components step is a list of document IDs and the group they belong to.
All documents in the same group are considered near duplicates.
These results can be used to remove the near duplicates from the corpus.

-----------------------------------------
Usage
-----------------------------------------

.. _fuzzydup_pyapi:

""""""""""""
Python API
""""""""""""

.. note::
    Before running fuzzy deduplication, you need to ensure that the dataset contains a unique ID for each document.
    If needed, you can use the ``add_id`` module within NeMo Curator to accomplish this.

    .. code-block:: python

      from nemo_curator import AddId
      from nemo_curator.datasets import DocumentDataset

      add_id = AddId(id_field="my_id", id_prefix="doc_prefix")
      dataset = DocumentDataset.read_json("input_file_path")
      id_dataset = add_id(dataset)
      id_dataset.to_json("/path/to/jsonl/data")

1. Configuration

  a. Using the API Directlty

  .. code-block:: python

    from nemo_curator import FuzzyDuplicatesConfig

    config = FuzzyDuplicatesConfig(
        cache_dir="/path/to/dedup_outputs", # must be cleared between runs
        id_field="my_id",
        text_field="text",
        seed=42,
        char_ngrams=24,
        num_buckets=20,
        hashes_per_bucket=13,
        use_64_bit_hash=False,
        buckets_per_shuffle=2,
        false_positive_check=False,
    )

  b. Using a YAML file

  .. code-block:: yaml

    cache_dir: /path/to/dedup_outputs
    id_field: my_id
    text_field: text
    seed: 42
    char_ngrams: 24
    num_buckets: 20
    hashes_per_bucket: 13
    use_64_bit_hash: False
    buckets_per_shuffle: 2
    false_positive_check: False

  .. code-block:: python

      from nemo_curator import FuzzyDuplicatesConfig

      config = FuzzyDuplicatesConfig.from_yaml("/path/to/config.yaml")


2. Usage Post Configuration

.. code-block:: python

    from nemo_curator import FuzzyDuplicates
    from nemo_curator.datasets import DocumentDataset

    # Initialize the deduplication object
    FuzzyDups = FuzzyDuplicates(config=config, logger="./")

    dataset = DocumentDataset.read_json(
        input_files="/path/to/jsonl/data",
        backend="cudf", # FuzzyDuplicates only supports datasets with the cuDF backend.
    )

    duplicate_docs = FuzzyDups(dataset)
    """
    Sample output:
                  my_id  group
    0  doc_prefix-56151     32
    1  doc_prefix-47071    590
    2  doc_prefix-06840    305
    3  doc_prefix-20910    305
    4  doc_prefix-42050    154
    """

.. tip::

  - A more comprehensive example for the above, including how to remove documents from a corpus using the list of
    duplicate IDs generated from fuzzy deduplication, can be found in `examples/fuzzy_deduplication.py <https://github.com/NVIDIA/NeMo-Curator/blob/main/examples/fuzzy_deduplication.py>`_.
  - The default values of ``num_buckets`` and ``hashes_per_bucket`` are set to find documents with an approximately Jaccard similarity of 0.8 or above.
  - Higher ``buckets_per_shuffle`` values can lead to better performance but might lead to out of memory errors.
  - Setting the ``false_positive_check`` flag to ``False`` is ideal for optimal performance.
  - When setting the ``false_positive_check`` flag to ``True`` ensure ``cache_dir`` between runs is emptied to avoid data from previous runs interfering with the current run's results.

""""""""""""
CLI Utility
""""""""""""

.. caution::
  Fuzzy deduplication CLI scripts only work with the specific ID format generated by the :code:`add_id` script. If the
  dataset does not contain IDs in this format, it is recommended to create them with the :code:`add_id` script as follows:

  .. code-block:: bash

    add_id \
      --id-field-name="my_id" \
      --input-data-dir=<Path to directory containing jsonl files> \
      --id-prefix="doc_prefix" \
      --log-dir=./log/add_id

  This will create a new field named :code:`my_id` within each JSON document which will have the form "doc_prefix-000001".
  If the dataset already has a unique ID this step can be skipped.

Once a unique ID has been added to each document, users can proceed with fuzzy deduplication, which roughly require the following
steps (all scripts are included in the `nemo_curator/scripts/fuzzy_deduplication <https://github.com/NVIDIA/NeMo-Curator/blob/main/nemo_curator/scripts/fuzzy_deduplication>`_ subdirectory):

1. Compute Minhashes
  - Input: Data directories
  - Output: ``minhashes.parquet`` for each data directory
  - Example call:

       .. code-block:: bash

               # same as `python compute_minhashes.py`
               gpu_compute_minhashes \
                 --input-data-dirs /path/to/jsonl/dir1 /path/to/jsonl/dir2 \
                 --output-minhash-dir /path/to/output_minhashes \
                 --input-json-text-field text_column_name \
                 --input-json-id-field id_column_name \
                 --minhash-length number_of_hashes \
                 --char-ngram char_ngram_size \
                 --hash-bytes 4 `#or 8 byte hashes` \
                 --seed 42 \
                 --log-dir ./
                 # --scheduler-file /path/to/file.json

.. _fuzzydup_lsh:

2. Buckets (Minhash Buckets)
  - Input: Minhash directories
  - Output: ``_buckets.parquet``
  - Example call:

       .. code-block:: bash

               # same as `python minhash_lsh.py`
               minhash_buckets \
                 --input-data-dirs /path/to/output_minhashes/dir1 /path/to/output_minhashes/dir2 \
                 --output-bucket-dir /path/to/dedup_output \
                 --input-minhash-field _minhash_signature \
                 --input-json-id-field id_column_name \
                 --minhash-length number_of_hashes \
                 --num-bands num_bands \
                 --buckets-per-shuffle 1 `#Value between [1-num_bands]. Higher is better but might lead to OOM` \
                 --log-dir ./
                 # --scheduler-file /path/to/file.json

3. False Positive Check (optional): If skipping this step, proceed to the :ref:`skip fp check section <fuzzydup_nofp>`.

  a. Jaccard Map Buckets
    - Input: ``_buckets.parquet`` and data directories
    - Output: ``anchor_docs_with_bk.parquet``
    - Example call:

       .. code-block:: bash

               # same as `python map_buckets.py`
               jaccard_map_buckets \
                 --input-data-dirs /path/to/jsonl/dir1 /path/to/jsonl/dir2 \
                 --input-bucket-dir /path/to/dedup_output/_buckets.parquet \
                 --output-dir /path/to/dedup_output \
                 --input-json-text-field text_column_name \
                 --input-json-id-field id_column_name
                 # --scheduler-file /path/to/file.json

  b. Jaccard Shuffle
    - Input: ``anchor_docs_with_bk.parquet`` and data directories
    - Output: ``shuffled_docs.parquet``
    - Example call:

       .. code-block:: bash

               # same as `python jaccard_shuffle.py`
               jaccard_shuffle \
                 --input-data-dirs /path/to/jsonl/dir1 /path/to/jsonl/dir2 \
                 --input-bucket-mapping-dir /path/to/dedup_output/anchor_docs_with_bk.parquet \
                 --output-dir /path/to/dedup_output \
                 --input-json-text-field text_column_name \
                 --input-json-id-field id_column_name
                 # --scheduler-file /path/to/file.json

  c. Jaccard Compute
    - Input: ``shuffled_docs.parquet``
    - Output: ``jaccard_similarity_results.parquet``
    - Example call:

       .. code-block:: bash

               # same as `python jaccard_compute.py`
               jaccard_compute \
                 --shuffled-docs-path /path/to/dedup_output/shuffled_docs.parquet \
                 --output-dir /path/to/dedup_output \
                 --ngram-size char_ngram_size_for_similarity \
                 --input-json-id-field id_column_name
                 # --scheduler-file /path/to/file.json

.. _fuzzydup_nofp:

3. Skipping the false positive check (more performant). This step is not needed if the false positive check was performed.

  a. Buckets to Edgelist
    - Input: ``_buckets.parquet``
    - Output: ``_edges.parquet``
    - Example call:

       .. code-block:: bash

               # same as `python buckets_to_edges.py`
               buckets_to_edges \
                 --input-bucket-dir /path/to/dedup_output/_buckets.parquet \
                 --output-dir /path/to/dedup_output \
                 --input-json-id-field id_column_name
                 # --scheduler-file /path/to/file.json

4. Connected Components
  - Input: ``jaccard_similarity_results.parquet`` (if you ran the false positive check) or ``_edges.parquet`` (if you skipped the false positive check)
  - Output: ``connected_components.parquet``
  - Example call:

       .. code-block:: bash

               # same as `python connected_components.py`
               gpu_connected_component \
                 --jaccard-pairs-path /path/to/dedup_output/jaccard_similarity_results.parquet `#Or /path/to/dedup_output/_edges.parquet` \
                 --output-dir /path/to/dedup_output \
                 --cache-dir /path/to/cc_cache \
                 --jaccard-threshold 0.8 \
                 --input-json-id-field id_column_name
                 # --scheduler-file /path/to/file.json

.. caution::
  The CLI utilities are limited to JSONL datasets and only work with specific ID formats.
  For different dataset or ID formats, use the :ref:`fuzzydup_pyapi`.

------------------------
Incremental Fuzzy Deduplication
------------------------

* If any new data is added to the corpus, you will need to perform deduplication incrementally. To incrementally perform fuzzy deduplication, we do not need to recompute minhashes for datasets where minhashes were already computed.
  Instead, you can organize your incremental datasets into separate directories and pass a list of all new directories to :code:`gpu_compute_minhashes`.

    - Input (assuming incremental snapshots are all under :code:`/input/`):

         .. code-block:: bash

                 /input/cc-2020-40
                 /input/cc-2021-42
                 /input/cc-2022-60
    - Output (assuming :code:`--output-minhash-dir=/output`):

         .. code-block:: bash

                 /output/cc-2020-40/minhashes.parquet
                 /output/cc-2021-42/minhashes.parquet
                 /output/cc-2022-60/minhashes.parquet
    - Example call:

         .. code-block:: bash

                 # same as `python compute_minhashes.py`
                 gpu_compute_minhashes \
                   --input-data-dirs /input/cc-2020-40 /input/cc-2020-42 /input/cc-2020-60 \
                   --output-minhash-dir /output/ \
                   --input-json-text-field text_column_name \
                   --input-json-id-field id_column_name \
                   --minhash-length number_of_hashes \
                   --char-ngram char_ngram_size \
                   --hash-bytes 4(or 8 byte hashes) \
                   --seed 42 \
                   --log-dir ./
                   # --scheduler-file /path/to/file.json

All subsequent steps, starting with :ref:`Buckets <fuzzydup_lsh>`, can be executed on all the data
(old and new) as described above without modification.
