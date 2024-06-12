
.. _data-curator-downstream:

#######################################################
Downstream Task Decontamination/Deduplication
#######################################################

-----------------------------------------
Background
-----------------------------------------

After training, large language models are usually evaluated by their performance on downstream tasks consisting of unseen test data.
When dealing with large datasets there is a potential for leakage of this test data into the model's training dataset.
Therefore, NeMo Curator follows the approach of `OpenAI GPT3 <https://arxiv.org/pdf/2005.14165.pdf>`_ and `Microsoft Turing NLG 530B <https://arxiv.org/abs/2201.11990>`_
to remove sections of documents in your dataset that are present in downstream tasks.

-----------------------------------------
Usage
-----------------------------------------

The ``TaskDecontamination`` modules provides the central functionality in NeMo Curator.
Let's examine this small example:

.. code-block:: python

    import nemo_curator as nc
    from nemo_curator.datasets import DocumentDataset
    from nemo_curator.utils.file_utils import get_all_files_paths_under
    from nemo_curator.tasks import Winogrande, Squad, TriviaQA,

    files = get_all_files_paths_under("books_dataset/")
    books = DocumentDataset.read_json(files, add_filename=True)

    downstream_tasks = [
        Winogrande(),
        Squad(),
        TriviaQA(),
    ]

    task_decontaminate = nc.TaskDecontamination(downstream_tasks)

    decontaminated_books = task_decontaminate(books)

    decontaminated_books.to_json("decontaminated_books/", write_to_filename=True)

If you would like more fine-grained control over the task decontamination process, NeMo Curator provides several CLI tools you can manually apply.
You can use the :code:`prepare_task_data`, :code:`find_matching_ngrams` and :code:`remove_matching_ngrams`
scripts in order to remove any task data that might be contained (i.e., "contaminate") within your training data.
You will need a list of your downstream tasks to modify the `task config (lm_tasks.yaml) <../../config/lm_tasks.yaml>`_.
If your task does not already exist as a class, you will need to construct a class that extends :code:`nemo_curator.tasks.DownstreamTask`.

Then, you can start by constructing the n-grams from the task documents using the :code:`prepare_task_data` module.
This module requires an input configuration file that contains the different modules that describe how to form N-grams from the task data of interest.
An example of a configuration file is provided in :code:`config/lm_tasks.yaml`. A number of tasks are already implemented within the NeMo Curator
and can be found within :code:`nemo_curator.tasks`. Should users desire to add their own tasks, they can prescribe their own class similar
to those defined in :code:`nemo_curator.tasks`. Once all N-grams have been computed, they are written as keys of a dictionary that is written to a pickle file.
This step only needs to be done once per set of tasks. This pickle file can be reused across datasets that share the same downstream tasks.

.. code-block:: bash

    prepare_task_data \
        --task-config-file=./config/lm_tasks.yaml \
        --output-task-ngrams=./data/task_ngrams.pkl



Once users have computed the task N-grams, they can use the :code:`find_matching_ngrams` module in order to search for matches within their corpus.
This module task as input the path to the users dataset consisting of JSONL files as well as precomputed task N-grams, and as output provides a pickle
file consisting of the count of how many times a specific task N-gram ocurred within the training set. This N-gram count will be used in the final
step to determine if an a document should be split and the N-gram removed.

.. code-block:: bash

    find_matching_ngrams \
        --input-data-dir=<Path to the input directory containing jsonl files> \
        --input-task-ngrams=./data/task_ngrams.pkl \
        --output-matched-ngram-data=./data/matched_ngrams.pkl

As a final step in the task decontamination procedure, the counts associated with the matched N-grams are used to determine if a particular N-gram
should be removed from the training corpus. If the N-gram has a count that is higher than a user-defined threshold, it is not considered. Otherwise,
it is considered and will be removed from the corpus. When an N-gram is removed from the corpus, a user-defined character window that extends from
the N-gram in both directions is also removed from the corpus. Additionally, the document will be split into two separate documents. If the split
document is too short after splitting, it will be removed. Additionally, documents that are split more than a user-defined number of times are also
removed from the corpus. For more information on the task decontamination procedure, please see `Brown et al., 2020 <https://arxiv.org/abs/2005.14165>`_ and `Smith et al., 2021 <https://arxiv.org/abs/2201.11990>`_

.. code-block:: bash

    remove_matching_ngrams \
        --input-data-dir=<Path to the input directory containing jsonl files> \
        --input-matched-ngrams=./data/matched_ngrams.pkl \
        --output-task-deduped-dir=<Output directory containing task-deduped jsonl files>
