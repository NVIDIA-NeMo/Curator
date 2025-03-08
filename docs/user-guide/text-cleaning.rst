.. _data-curator-text-cleaning:

=========================
Text Cleaning
=========================

--------------------
Overview
--------------------
Use NeMo Curator's text cleaning modules to remove undesirable text such as improperly decoded Unicode characters, inconsistent line spacing, or excessive URLs from documents being pre-processed for dataset.

For example, the input sentence ``"The Mona Lisa doesn't have eyebrows."`` from a given document may not have included a properly encoded apostrophe (``'``), resulting in the sentence decoding as ``"The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows."``. NeMo Curator enables you to easily run this document through the default ``UnicodeReformatter`` module to detect and remove the unwanted text, or you can define your own custom Unicode text cleaner tailored to your needs.

--------------------
Use Cases
--------------------
* Fix improperly decoded Unicode characters from webpages.
* Standardize document layout by removing excessive newlines.
* Remove URLs in documents.

--------------------
Modules
--------------------
NeMo Curator provides the following modules for cleaning text:

- ``UnicodeReformatter``: Uses `ftfy <https://ftfy.readthedocs.io/en/latest/>`_ to fix broken Unicode characters. Modifies the "text" field of the dataset by default. Please see the `ftfy documentation <https://ftfy.readthedocs.io/en/latest/config.html>`_ for more information about parameters used by the ``UnicodeReformatter``.
- ``NewlineNormalizer``: Uses regex to replace 3 or more consecutive newline characters in each document with only 2 newline characters.
- ``UrlRemover``: Uses regex to remove all URLs in each document.

You can use these modules individually or sequentially in a cleaning pipeline.

Consider the following example, which loads a dataset (``books.jsonl``), steps through each module in a cleaning pipeline, and outputs the processed dataset as ``cleaned_books.jsonl``:


.. code-block:: python

    from nemo_curator import Sequential, Modify, get_client
    from nemo_curator.datasets import DocumentDataset
    from nemo_curator.modifiers import UnicodeReformatter, UrlRemover, NewlineNormalizer

    def main():
        client = get_client(cluster_type="cpu")

        dataset = DocumentDataset.read_json("books.jsonl")
        cleaning_pipeline = Sequential([
            Modify(UnicodeReformatter()),
            Modify(NewlineNormalizer()),
            Modify(UrlRemover()),
        ])

        cleaned_dataset = cleaning_pipeline(dataset)

        cleaned_dataset.to_json("cleaned_books.jsonl")

    if __name__ == "__main__":
        main()

You can also perform text cleaning operations using the CLI by running the ``text_cleaning`` command:

.. code-block:: bash

    text_cleaning \
      --input-data-dir=/path/to/input/ \
      --output-clean-dir=/path/to/output/ \
      --normalize-newlines \
      --remove-urls

By default, the CLI will only perform Unicode reformatting. Appending the ``--normalize-newlines`` and ``--remove-urls`` options adds the other text cleaning options.

------------------------
Custom Text Cleaner
------------------------
It's easy to write your own custom text cleaner. The implementation of ``UnicodeReformatter`` can be used as an example.

.. code-block:: python

    import ftfy

    from nemo_curator.modifiers import DocumentModifier


    class UnicodeReformatter(DocumentModifier):
        def __init__(self):
            super().__init__()

        def modify_document(self, text: str) -> str:
            return ftfy.fix_text(text)

Simply define a new class that inherits from ``DocumentModifier`` and define the constructor and ``modify_text`` method.
Also, like the ``DocumentFilter`` class, ``modify_document`` can be annotated with ``batched`` to take in a Pandas Series of documents instead of a single document.
See the :ref:`document filtering page <data-curator-qualityfiltering>` for more information.

---------------------------
Additional Resources
---------------------------
* `Single GPU Tutorial <https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/single_node_tutorial/single_gpu_tutorial.ipynb>`_
* `ftfy <https://ftfy.readthedocs.io/en/latest/>`_
* `Refined Web Paper <https://arxiv.org/abs/2306.01116>`_
* `Nemotron-CC Paper <https://arxiv.org/abs/2412.02595>`_
