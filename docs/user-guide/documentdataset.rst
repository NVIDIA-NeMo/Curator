
.. _data-curator-documentdataset:

======================================
Working with DocumentDataset
======================================
-----------------------------------------
Background
-----------------------------------------
Text datasets are responsible for storing metadata along with the core text/document.
``jsonl``` files are common for their ease of processing and inspecting.
``parquet`` files are also a common format.
In both cases, a single dataset is often represented with multiple underlying files (called shards).
For example, if you have a large dataset named "books" it is likely you will store it in shards with each shard being named something like ``books_00.jsonl``, ``books_01.jsonl``, ``books_02.jsonl``, etc.

How you store your dataset in memory is just as important as how you store it on disk.
If you have a large dataset that is too big to fit directly into memory, you will have to somehow distribute it across multiple machines/nodes.
Furthermore, if curating your dataset takes a long time, it is likely to get interrupted due to some unforseen failure or another.
NeMo Curator's ``DocumentDataset`` employs `Dask's distributed dataframes <https://docs.dask.org/en/stable/dataframe.html>`_ to mangage large datasets across multiple nodes and allow for easy restarting of interrupted curation.
``DocumentDataset`` supports reading and writing to sharded ``jsonl`` and ``parquet`` files both on local disk and from remote sources directly like S3.

-----------------------------------------
Usage
-----------------------------------------
############################
Reading and Writing
############################
``DocumentDataset`` is the standard format for text datasets in NeMo Curator.
Imagine we have a "books" dataset stored in the following structure:
::

    books_dataset/
        books_00.jsonl
        books_01.jsonl
        books_02.jsonl

You could read, filter the dataset, and write it using the following methods

.. code-block:: python

    import nemo_curator as nc
    from nemo_curator.datasets import DocumentDataset
    from nemo_curator.utils.file_utils import get_all_files_paths_under
    from nemo_curator.filters import WordCountFilter

    files = get_all_files_paths_under("books_dataset/")
    books = DocumentDataset.read_json(files, add_filename=True)

    filter_step = nc.ScoreFilter(
                    WordCountFilter(min_words=80),
                    text_field="text",
                    score_field="word_count",
                )

    long_books = filter_step(books)

    long_books.to_json("long_books/", write_to_filename=True)

Let's walk through this code line by line.

* ``files = get_all_files_paths_under("books_dataset/")`` This retrieves a list of all files in the given directory.
  In our case, this is equivalent to writing

  .. code-block:: python

    files = ["books_dataset/books_00.jsonl",
             "books_dataset/books_01.jsonl",
             "books_dataset/books_02.jsonl"]

* ``books = DocumentDataset.read_json(files, add_filename=True)`` This will read the files listed into memory.
  The ``add_filename=True`` option preserves the name of the shard (``books_00.jsonl``, ``books_01.jsonl``, etc.) as an additional ``filename`` field.
  When the dataset is written back to disk, this option (in conjunction with the ``write_to_filename`` option) ensure that documents stay in their original shard.
  This can be useful for manually inspecting the results of filtering shard by shard.
* ``filter_step = ...`` This constructs and applies a heuristic filter for the length of the document.
  More information is provided in the filtering page of the documentation.
* ``long_books.to_json("long_books/", write_to_filename=True)`` This writes the filtered dataset to a new directory.
  As mentioned above, the ``write_to_filename=True`` preserves the sharding of the dataset.
  If the dataset was not read in with ``add_filename=True``, setting ``write_to_filename=True`` will throw an error.

``DocumentDataset`` is just a wrapper around a `Dask dataframe <https://docs.dask.org/en/stable/dataframe.html>`_.
The underlying dataframe can be accessed with the ``DocumentDataset.df`` member variable.
It is important to understand how Dask handles computation.
To quote from their `documentation <https://docs.dask.org/en/stable/10-minutes-to-dask.html#computation>`_:

    Dask is lazily evaluated. The result from a computation isn't computed until you ask for it. Instead, a Dask task graph for the computation is produced.

Because of this, the call to ``DocumentDataset.read_json`` will not execute immediately.
Instead, tasks that read each shard of the dataset will be placed on the task graph.
The task graph is only executed when a call to ``DocumentDataset.df.compute()`` is made, or some operation that depends on ``DocumentDataset.df`` calls ``.compute()``.
This allows us to avoid reading massive datasets into memory.
In our case, ``long_books.to_json()`` internally calls ``.compute()``, so the task graph will be executed then.

############################
Resuming from Interruptions
############################
It can be helpful to track which documents in a dataset have already been processed so that long curation jobs can be resumed if they are interrupted.
NeMo Curator provides a utility for easily tracking which dataset shards have already been processed.
Consider a modified version of the code above:

.. code-block:: python

    from nemo_curator.utils.file_utils import get_remaining_files

    files = get_remaining_files("books_dataset/", "long_books/", "jsonl")
    books = DocumentDataset.read_json(files, add_filename=True)

    filter_step = nc.ScoreFilter(
                    WordCountFilter(min_words=80),
                    text_field="text",
                    score_field="word_count",
                )

    long_books = filter_step(books)

    long_books.to_json("long_books/", write_to_filename=True)

``get_remaining_files`` compares the input directory (``"books_dataset/"``) and the output directory (``"long_books"``) and returns a list of all the shards in the input directory that have not yet been written to the output directory.



While Dask provides an easy way to avoid reading too much data into memory, there are times when we may need to call ``persist()`` or a similar operation that forces the dataset into memory.
In these cases, we recommend processing the input dataset in batches using a simple wrapper function around ``get_remaining_files`` as shown below.

.. code-block:: python

    from nemo_curator.utils.file_utils import get_batched_files

    for files in get_batched_files("books_dataset/", "long_books/", "jsonl", batch_size=64):
        books = DocumentDataset.read_json(files, add_filename=True)

        filter_step = nc.ScoreFilter(
                        WordCountFilter(min_words=80),
                        text_field="text",
                        score_field="word_count",
                    )

        long_books = filter_step(books)

        long_books.to_json("long_books/", write_to_filename=True)

This will read in 64 shards at a time, process them, and write them back to disk.
Like ``get_remaining_files``, it only includes files that are in the input directory and not in the output directory.

############################
Blending and Shuffling
############################

Blending data from multiple sources can be a great way of improving downstream model performance.
This blending can be done during model training itself (i.e., *online* blending) or it can be done before training (i.e., *offline* blending).
Online blending is useful for rapidly iterating in the training process.
Meanwhile, offline blending is useful if you want to distribute the dataset.
Online blending is currently possible in `NeMo via NVIDIA Megatron Core <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/blended_dataset.py>`_, and NeMo Curator offers a way to perform blending offline.

Let's take a look at how datasets can be combined using ``nc.blend_datasets``

.. code-block:: python

  import nemo_curator as nc

  books = DocumentDataset.read_json("books_dataset/")
  articles = DocumentDataset.read_json("articles_dataset/")
  journals = DocumentDataset.read_json("journals_dataset/")

  datasets = [books, articles, journals]
  target_samples = 1000
  weights = [5.0, 2.0, 1.0]

  blended_dataset = nc.blend_datasets(target_samples, datasets, weights)

  blended_dataset.to_json("blended_dataset/")


* ``datasets = [books, articles, journals]`` Here, we are choosing to blend three different datasets.
  These datasets do not have to be in the same file format, or similar in size.
  So long as they can be read in as a DocumentDataset, they will be fine.
  The samples from each dataset are always drawn "in order".
  The precise order depends on the format.
  For sharded jsonl files, the entries at the beginning of the file with the first name in sorted order will be chosen first.
* ``target_samples = 1000`` This is the desired number of samples in the resulting dataset.
  By sample, we mean document or just generally a single datapoint.
  There may end up being more samples in the dataset depending on the weights.
* ``weights = [5.0, 2.0, 1.0]`` The relative number of samples that should be taken from each dataset.
  Given these weights, the blended dataset will have five times as many samples from books as there are samples from journals.
  Similarly, there will be two times as many samples from articles when compared to samples from journals.
  Weights can be a list of non-negative real numbers.
  ``nc.blend_datasets`` will do the normalization and combine the normalized weights with the target samples to determine
  how many samples should be taken from each dataset.
  In the case of the books dataset, the following would be the calculation.

  .. math::

    \lceil target\_samples \cdot w_i\rceil=\lceil 1000\cdot \frac{5}{8}\rceil=625
  If any datasets have fewer samples than the calculated weight, they will be oversampled to meet the quota.
  For example, if the books dataset only had 500 documents in it, the first 125 would be repeated to achieve
  the 625 samples.
* ``blended_dataset = nc.blend_datasets(target_samples, datasets, weights)`` We now call the function itself.
  Afterwards, we are left with a blended dataset that we can operate on like any other dataset.
  We can apply filters, deduplicate, or classify the documents.

Because blending datasets involves combining data from multiple sources, the sharding of the original datasets
cannot be preserved. The options ``add_filename=True`` and ``write_to_filename=True`` for reading and writing
datasets are therefore incompatible with ``nc.blend_datasets``.


Shuffling can be another important aspect of dataset management.
NeMo Curator's ``nc.Shuffle`` allows users to reorder all entries in the dataset.

Here is a small example on how this can be done:

.. code-block:: python

  import nemo_curator as nc

  books = DocumentDataset.read_json("books_dataset/")

  shuffle = nc.Shuffle(seed=42)

  shuffled_books = shuffle(books)

  shuffled_books.to_json("shuffled_books/")

* ``shuffle = nc.Shuffle(seed=42)`` This creates a shuffle operation that can be chained with
  the various other modules in NeMo Curator. In this example, we fix the seed to be 42.
  Setting the seed will guarantee determinism, but may be slightly slower (20-30% slower)
  depending on the dataset size.
* ``shuffled_books = shuffle(books)`` The dataset has now been shuffled, and we can save it to the filesystem.
