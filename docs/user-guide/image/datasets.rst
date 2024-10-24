.. _data-curator-image-datasets:

=========================
Image-Text Pair Datasets
=========================

Image-text pair datasets are commonly used for training generative text to image models or CLIP models.
NeMo Curator supports reading and writing datasets based on the `WebDataset <https://github.com/webdataset/webdataset>`_ file format.
This format allows NeMo Curator to annotate the dataset with metadata including embeddings and classifier scores.
Its sharded format also makes it easier to distribute work to different workers processing the dataset.

------------
File Format
------------

Here is an example of what a dataset directory that is in the WebDataset format should look like.

::

    dataset/
    ├── 00000.tar
    │   ├── 000000000.jpg
    │   ├── 000000000.json
    │   ├── 000000000.txt
    │   ├── 000000001.jpg
    │   ├── 000000001.json
    │   ├── 000000001.txt
    │   └── ...
    ├── 00001.tar
    │   ├── 000010000.jpg
    │   ├── 000010000.json
    │   ├── 000010000.txt
    │   ├── 000010001.jpg
    │   ├── 000010001.json
    │   ├── 000010001.txt
    │   └── ...
    ├── 00002.tar
    │   └── ...
    ├── 00000.parquet
    ├── 00001.parquet
    └── 00002.parquet


The exact format assumes a single directory with sharded ``.tar``, ``.parquet``, and (optionally)
``.idx`` files. Each tar file should have a unique integer ID as its name (``00000.tar``,
``00001.tar``, ``00002.tar``, etc.). The tar files should contain images in ``.jpg`` files, text captions
in ``.txt`` files, and metadata in ``.json`` files. Each record of the dataset is identified by
a unique ID that is a mix of the shard ID along with the offset of the record within a shard.
For example, the 32nd record of the 43rd shard would be in ``00042.tar`` and have image ``000420031.jpg``,
caption ``000420031.txt``, and metadata ``000420031.json`` (assuming zero indexing).

In addition to the collection of tar files, NeMo Curator's ``ImageTextPairDataset`` expects there to be .parquet files
in the root directory that follow the same naming convention as the shards (``00042.tar`` -> ``00042.parquet``).
Each Parquet file should contain an aggregated tabular form of the metadata for each record, with
each row in the Parquet file corresponding to a record in that shard. The metadata, both in the Parquet
files and the JSON files, must contain a unique ID column that is the same as its record ID (000420031
in our examples).

-------
Reading
-------

Datasets can be read in using ``ImageTextPairDataset.from_webdataset()``

.. code-block:: python
    from nemo_curator.datasets import ImageTextPairDataset

    dataset = ImageTextPairDataset.from_webdataset(path="/path/to/dataset", id_col="key")

* ``path="/path/to/dataset"`` should point to the root directory of the WebDataset.
* ``id_col="key"`` lets us know that the unique ID column in the dataset is named "key".

A more thorough list of parameters can be found in the `API Reference <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/api/datasets.html>`_.

-------
Writing
-------

There are two ways to write an image dataset. The first way only saves the metadata, while the second way will reshard the tar files.
Both trigger the computation of all the tasks you have set to run beforehand.

.. code-block:: python
    from nemo_curator.datasets import ImageTextPairDataset

    dataset = ImageTextPairDataset.from_webdataset(path="/path/to/dataset", id_col="key")

    # Perform your operations (embedding creation, classifiers, etc.)

    dataset.save_metadata()

``save_metadata()`` will only save sharded Parquet files to the target directory. It does not modify the tar files.
There are two optional parameters:

* ``path`` allows you to change the location of where the dataset is saved. By default, it will overwrite the original Parquet files.
* ``columns`` allows you to only save a subset of metadata. By default, all metadata will be saved.


.. code-block:: python
    from nemo_curator.datasets import ImageTextPairDataset

    dataset = ImageTextPairDataset.from_webdataset(path="/path/to/dataset", id_col="key")

    # Perform your operations (embedding creation, classifiers, etc.)

    dataset.to_webdataset(path="/path/to/output", filter_column="passes_curation")

``to_webdataset()`` will reshard the WebDataset to only include elements that have a value of ``True`` in the ``filter_column``.
Resharding can take a while, so this should typically only be done at the end of your curation pipeline when you are ready to export the dataset for training.


A more thorough list of parameters can be found in the `API Reference <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/api/datasets.html>`_.

-------------
Index Files
-------------

NeMo Curator uses `DALI <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/dataloading_webdataset.html>`_ for image data loading from the tar files.
In order to speed up the data loading, you can supply ``.idx`` files in your dataset.
The index files must be generated by DALI's wds2idx tool.
See the `DALI documentation <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/dataloading_webdataset.html#Creating-an-index>`_ for more information.
Each index file must follow the same naming convention as the tar files (00042.tar -> 00042.idx).