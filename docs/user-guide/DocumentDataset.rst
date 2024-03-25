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
