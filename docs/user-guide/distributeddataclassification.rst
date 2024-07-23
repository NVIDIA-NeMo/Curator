============================================
Distributed Data Classification
============================================

-----------------------------------------
Background
-----------------------------------------

When preparing text data to be used in training a large language model (LLM), it is useful to classify
text documents in various ways, to enhance the LLM's performance by making it able to produce more
contextually appropriate and accurate language across various subjects. NeMo Curator provides this module to
help a user run inference with pre-trained models on large amounts of text documents. We achieve
this by chunking the datasets across multiple computing nodes, each equipped with multiple GPUs, to
accelerate the classification task in a distributed way. In other words, because the classification of
a single text document is independent of other documents within a dataset, we can distribute the
workload across multiple nodes and multiple GPUs to perform parallel processing.

Domain classification and quality classification are two tasks we include as examples within our module.
Here, we summarize why each is useful for training an LLM.

Domain classification is useful because it helps the LLM understand the context and specific domain of
the input text. Because different domains have different linguistic characteristics and terminologies,
an LLM's ability to generate contextually relevant responses can be improved by tailoring training data
to a specific domain. Overall, this helps provide more accurate and specialized information.

Quality classification is useful for filtering out noisy or low quality data. This allows the model to
focus on learning from high quality and informative examples, which contributes to the LLM's robustness
and enhances its ability to generate reliable and meaningful outputs. Additionally, quality
classification helps mitigate biases and inaccuracies that may arise from poorly curated training data.

-----------------------------------------
Usage
-----------------------------------------

NeMo Curator provides a base class ``DistributedDataClassifier`` that can be extended to fit your specific model.
The only requirement is that the model can fit on a single GPU.
We have also provided two subclasses that focus on domain and quality classification.
Let's see how ``DomainClassifier`` works in a small excerpt taken from ``examples/domain_classifier_example.py``:

.. code-block:: python

    files = get_all_files_paths_under("books_dataset/")
    input_dataset = DocumentDataset.read_json(files, backend="cudf", add_filename=True)

    domain_classifier = DomainClassifier(filter_by=["Games", "Sports"])
    result_dataset = domain_classifier(dataset=input_dataset)

    result_dataset.to_json("games_and_sports/", write_to_filename=True)

In the above excerpt, the domain classifier is obtained directly from `HuggingFace <https://huggingface.co/nvidia/domain-classifier>`_.

This module functions very similarly to the ``ScoreFilter`` module.
The key differences is that it operates on the GPU instead of the CPU.
Therefore, the Dask cluster must be started as a GPU one.
And, ``DomainClassifier`` requires ``DocumentDataset`` to be on the GPU (i.e., have ``backend=cudf``).
It is easy to extend ``DistributedDataClassifier`` to your own model.
Check out ``nemo_curator.modules.distributed_data_classifier.py`` for reference.
