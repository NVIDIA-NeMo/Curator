
.. _data-curator-pii:

======================================
PII Identification and Removal
======================================

--------------------------------------
Background
--------------------------------------

The purpose of the personal identifiable information (PII) de-identification tool is to help scrub sensitive data
out of datasets. The following is the list of sensitive data types that
are currently supported by the tool:

- Name
- Email Address
- Physical Address
- Phone Numbers
- IP Address
- Credit/Debit Card Numbers
- US Social Security Numbers
- Dates

The tool utilizes `Dask <https://dask.org>`_ to parallelize tasks and hence it can be
used to scale up to terabytes of data easily. Although Dask can be deployed on various
distributed compute environments such as HPC clusters, Kubernetes and other cloud
offerings such as AWS EKS, Google cloud etc, the current implementation only supports
Dask on HPC clusters that use SLURM as the resource manager.

-----------------------------------------
Usage
-----------------------------------------
########################################################
Reading documents and de-identifying
########################################################

Imagine we have a "books" dataset stored in the following structure:
::

    books_dataset/
        books_00.jsonl
        books_01.jsonl
        books_02.jsonl

You could read, de-identify the dataset, and write it to an output directory using the following approach

.. code-block:: python

    from nemo_curator.datasets import DocumentDataset
    from nemo_curator.utils.distributed_utils import read_data, write_to_disk, get_client
    from nemo_curator.utils.file_utils import get_batched_files
    from nemo_curator.modules.modify import Modify
    from nemo_curator.modifiers.pii_modifier import PiiModifier

    modifier = PiiModifier(
        language="en",
        supported_entities=["PERSON", "EMAIL_ADDRESS"],
        anonymize_action="replace",
        batch_size=1000,
        device="gpu")

    for file_names in get_batched_files(
            "book_dataset,
            "output_directory",
            "jsonl",
            32
    ):
        source_data = read_data(file_names, file_type="jsonl", backend='pandas', add_filename=True)
        dataset = DocumentDataset(source_data)
        print(f"Dataset has {source_data.npartitions} partitions")

        modify = Modify(modifier)
        modified_dataset = modify(dataset)
        write_to_disk(modified_dataset.df,
                      "output_directory",
                      write_to_filename=True,
                      output_type="jsonl"
                      )

Let's walk through this code line by line.

* ``modifier = PiiModifier`` creates an instance of ``PiiModifier`` class that is responsible for PII de-identification
* ``for file_names in get_batched_files`` retrieves a batch of 32 documents from the `book_dataset`
* ``source_data = read_data(file_names, file_type="jsonl", backend='pandas', add_filename=True)`` reads the data from all the files using Dask using Pandas as the backend. The ``add_filename`` argument ensures that the output files have the same filename as the input files.
* ``dataset = DocumentDataset(source_data)``  creates an instance of ``DocumentDataset`` using the batch files. ``DocumentDataset`` is the standard format for text datasets in NeMo Curator.
* ``modify = Modify(modifier)`` creates an instance of the ``Modify`` class. This class can take any modifier as an argument
* ``modified_dataset = modify(dataset)`` modifies the data in the dataset by performing the PII de-identification based upon the passed parameters.
* ``write_to_disk(modified_dataset.df ....`` writes the de-identified documents to disk.

The PII redaction module can also be invoked via ``script/find_pii_and_deidentify.py`` script which provides a CLI based interface. To see a complete list of options supported by the script just execute

``python nemo_curator/scripts/find_pii_and_deidentify.py``

To launch the script from within a SLURM environment, the script ``examples/slurm/start-slurm.sh`` can be modified and used.


############################
Resuming from Interruptions
############################
It can be helpful to track which documents in a dataset have already been processed so that long curation jobs can be resumed if they are interrupted.
NeMo Curator provides a utility for easily tracking which dataset shards have already been processed. A call to ``get_batched_files`` will return an iterator over the files that have yet to be processed by a modifier such as ``PiiModifierBatched``
When you re-run the code example provided above, NeMo Curator ensures that only unprocessed files are processed by the PII module.
