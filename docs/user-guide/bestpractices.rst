.. _data-curator-best-practices:

======================================
Best Practices
======================================

-------------------------------------------
Choosing the Right Quality Model Type
-------------------------------------------
NeMo Curator offers a variety of methods for determining the quality of a piece of text.
Here are the methods in increasing order of compute required for them.

#. `fastText <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/qualityfiltering.html#classifier-filtering>`_ is an n-gram based bag-of-words classifier. It is typically trained on a high quality reference corpus and a low quality corpus (typically unfiltered Common Crawl dumps). While NeMo Curator does not provide pretrained versions of the classifier, training it is incredibly fast and easy. It only requires 100,000 - 1,000,000 text samples to train on, and can complete training in mere seconds. Its small size also allows it to train and run inference on the CPU. Due to these factors, we recommend using fastText classifiers on large scale pretraining datasets where you don't have the compute budget for more sophisticated methods.

#. `BERT-style classifiers <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/distributeddataclassification.html>`_ - NeMo Curator's distributed data classification modules work with many BERT-style classifiers for `domain classification <https://huggingface.co/nvidia/domain-classifier>`_, quality classification, and more. For this comparison, we'll focus on just the text quality classifier. NeMo Curator provides a pretrained version of the classifier on HuggingFace and NGC that can be immediately used. We recommend using these classifiers towards the end of your data filtering pipeline for pretraining.

#. `Language model labelling <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/syntheticdata.html>`_ - Language models can be used to label text as high quality or low quality. NeMo Curator allows you to connect to arbitrary LLM inference endpoints which you can use to label your data. One example of such an endpoint would be Nemotron-4 340B Instruct on `build.nvidia.com <https://build.nvidia.com/explore/discover#nemotron-4-340b-instruct>`_. Due to their size, these models can require a lot of compute and are usually infeasible to run across an entire pretraining dataset. We recommend using these large models on very little amounts of data. Fine-tuning datasets can make good use of them.

#. `Reward model labelling <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/syntheticdata.html>`_ - Unlike the previous methods, reward models label the quality of conversations between a user and an assistant instead of labelling the quality of a document. In addition, models (like `Nemotron-4 340B Reward <https://huggingface.co/nvidia/Nemotron-4-340B-Reward>`_) may output multiple scores covering different categories. Like LLM labelling, NeMo Curator can connect to arbitrary reward models hosted as an external service. Due to these differences and their large size, we recommend using reward models when filtering fine-tuning data. In particular, synthetic data filtering is a good use of them.

-------------------------------------------
Handling GPU Out-of-Memory (OOM) Errors
-------------------------------------------
NeMo Curator is designed to be scalable with large amounts of text data, but OOM errors occur when the available GPU memory is insufficient for a given task.
To help avoid these issues and ensure efficient processing, here are some strategies for managing memory usage and mitigating OOM challenges.

Utilize RMM Options
~~~~~~~~~~~~~~~~~~~
`RAPIDS Memory Manager (RMM) <https://github.com/rapidsai/rmm>`_ is a package that enables you to allocate device memory in a highly configurable way.
The NeMo Curator team has found several of its features to be especially useful for fuzzy deduplication, notably the connected components step.
Here are some features which can help optimize memory usage:

* Enable asynchronous memory allocation: Use the ``--rmm-async`` flag to allow RMM to handle memory allocation more efficiently, by allocating and deallocating GPU memory asynchronously.
* Set a memory release threshold: For example, ``--rmm-release-threshold 50GB`` can help prevent holding onto excess memory, releasing unused memory when a certain limit is reached. Please keep in mind that using this flag may degrade performance slightly as RMM is busy releasing the unused memory.

You can set these flags while calling your Python script directly, for example:

.. code-block:: bash

  python your_script.py --rmm-async --rmm-release-threshold 50GB

Fuzzy Deduplication Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fuzzy deduplication is one of the most computationally expensive algorithms within the NeMo Curator pipeline.
Here are some suggestions for managing memory use during fuzzy deduplication:

* Reduce bucket counts: During deduplication, the data is grouped into buckets to compare and identify near-duplicate documents. Increasing the number of buckets increases the probability that two documents within a given Jaccard similarity score are marked as duplicates. This leads to more accurate results as it reduces the number of false negatives. However, increasing the number of buckets also increases the memory requirements from the increased number of hashes it needs to store. Thus, it is important to find an optimal balance between memory usage and deduplication accuracy. You can experiment with this by using the ``num_buckets`` parameter when initializing your ``FuzzyDuplicatesConfig``.
* Reduce buckets per shuffle: Because duplicates are still considered bucket by bucket, reducing the ``buckets_per_shuffle`` parameter in the ``FuzzyDuplicatesConfig`` does not affect accuracy. Instead, reducing the buckets per shuffle helps lower the amount of data being transferred between GPUs. However, using a lower ``buckets_per_shuffle`` will increase the time it takes to process the data.
* Adjust files per partition: Processing large datasets in smaller chunks can help reduce the memory load. When reading data with ``DocumentDataset.read_json`` or ``DocumentDataset.read_parquet``, start with a smaller ``files_per_partition`` value and increase as needed.

Add More GPUs
~~~~~~~~~~~~~
If possible, scale your system by adding more GPUs.
This provides additional VRAM (Video Random Access Memory), which is crucial for holding datasets and intermediate computations.
Thus, adding more GPUs allows you to distribute the workload, reducing the memory load on each GPU.
