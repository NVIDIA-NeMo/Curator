============================================
Distributed Data Classification
============================================

-----------------------------------------
Background
-----------------------------------------

When preparing text data for training a large language model (LLM), it is useful to classify text documents in various ways.
This enhances the LLM’s performance by enabling it to produce more contextually appropriate and accurate language across different subjects.

NeMo Curator provides a module to help users run inference with pre-trained models on large volumes of text documents.
This is achieved by chunking the datasets across multiple computing nodes, each equipped with multiple GPUs, to accelerate the classification task in a distributed manner.
Since the classification of a single text document is independent of other documents within the dataset, we can distribute the workload across multiple nodes and GPUs to perform parallel processing.

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
Let's see how ``DomainClassifier`` works in a small excerpt taken from ``examples/classifiers/domain_example.py``:

.. code-block:: python

    files = get_all_files_paths_under("books_dataset/")
    input_dataset = DocumentDataset.read_json(files, backend="cudf")

    domain_classifier = DomainClassifier(filter_by=["Games", "Sports"])
    result_dataset = domain_classifier(dataset=input_dataset)

    result_dataset.to_json("games_and_sports/")

In the above excerpt, the domain classifier is obtained directly from `Hugging Face <https://huggingface.co/nvidia/domain-classifier>`_.

This module functions similarly to the ``ScoreFilter`` module.
The key differences is that it operates on the GPU instead of the CPU.
Therefore, the Dask cluster must be started as a GPU cluster.
Additionally, ``DomainClassifier`` requires ``DocumentDataset`` to be on the GPU with ``backend="cudf"``.
It is easy to extend ``DistributedDataClassifier`` to your own model.
Check out ``nemo_curator.classifiers.base.py`` for reference.

AEGIS Safety Model
#####################
Aegis is a family of content-safety LLMs used for detecting if a piece of text contains content that is a part of 13 critical risk categories.
There are two variants, `defensive <https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0>`_ and `permissive <https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0>`_, that are useful for filtering harmful data out of your training set.
The models are parameter-efficient instruction-tuned versions of Llama Guard based on Llama2-7B trained on the NVIDIA content-safety dataset `Aegis Content Safety Dataset <https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0>`_.
More details on training and the model can be found `here <https://arxiv.org/abs/2404.05993>`_.

To use this AEGIS classifiers, you must get access to
Llama Guard on Hugging Face here: https://huggingface.co/meta-llama/LlamaGuard-7b
Afterwards, you should set up a `user access token <https://huggingface.co/docs/hub/en/security-tokens>`_ and pass that token into
the constructor of this classifier.

NeMo Curator provides an easy way to annotate and filter your data using the safety models through our distributed data classfication framework.

.. code-block:: python
  files = get_all_files_paths_under("unsafe_documents/")
  input_dataset = DocumentDataset.read_json(files, backend="cudf")

  token = "hf_1234"  # Replace with your user access token
  safety_classifier = AegisClassifier(
    aegis_variant="nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
    token=token,
    filter_by=["safe", "O13"]
  )
  result_dataset = safety_classifier(dataset=input_dataset)

  result_dataset.to_json("safe_documents/")

This example filters out all documents except those that AEGIS classifies as safe or O13 (the category for "Needs caution").
The possible labels are as follows: ``"safe", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10", "O11", "O12", "O13", or "unknown"``.

* "safe" means that the document is considered safe by the model.
* "O1" through "O13" mean the document is unsafe according to the model. Each number corresponds to a different category of safety from the safety taxonomy defined in the `paper <https://arxiv.org/pdf/2404.05993>`_ and listed on the `model cards <https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0>`_.
* "unknown" means that the LLM output a non-standard response. To view the raw response of the LLM, you can set ``keep_raw_pred=True`` and ``raw_pred_column="raw_predictions"`` like this:

  .. code-block:: python

    safety_classifier = AegisClassifier(
      aegis_variant="nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0",
      filter_by=["safe", "O13"],
      keep_raw_pred=True,
      raw_pred_column="raw_predictions",
    )

  This will create a column in the dataframe with the raw output of the LLM. You can choose to parse this response however you want.


CrossFit Integration
####################

CrossFit is an open-source library by RAPIDS AI for fast offline inference scaled to Multi-Node Multi-GPU (MNMG) environments.
It accelerates NeMo Curator's classifiers described above.

The key features include:

- PyTorch integration for model inference.
- Efficient I/O and tokenization with cuDF.
- Smart batching/chunking for optimized processing.
- 1.4x-4x performance improvement over Dask + PyTorch baselines.


Sorted Sequence Data Loader
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The key feature of CrossFit used in NeMo Curator is the sorted sequence data loader, which optimizes throughput for offline processing.

- Sorts input sequences by length.
- Groups sorted sequences into optimized batches.
- Efficiently allocates batches to the the provided GPU memories by estimating the memory footprint for each sequence length and batch size.

.. image:: images/sorted_sequence_dataloader.png
   :alt: Sorted Sequence Data Loader

Check out the `rapidsai/crossfit`_ repository for more information.

.. _rapidsai/crossfit: https://github.com/rapidsai/crossfit
