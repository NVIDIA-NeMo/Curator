.. include:: datacuration.rsts

-------------------
Text Curation
-------------------

:ref:`Downloading and Extracting Text <data-curator-download>`
   Downloading a massive public dataset is usually the first step in data curation, and it can be cumbersome due to the dataset’s massive size and hosting method. This section describes how to download and extract large corpora efficiently.

:ref:`Working with DocumentDataset <data-curator-documentdataset>`
   DocumentDataset is the standard format for datasets in NeMo Curator. This section describes how to get datasets in and out of this format, as well as how DocumentDataset interacts with the modules.

:ref:`CPU and GPU Modules with Dask <data-curator-cpuvsgpu>`
   NeMo Curator provides both CPU based modules and GPU based modules and supports methods for creating compatible Dask clusters and managing the dataset transfer between CPU and GPU.

:ref:`Document Filtering <data-curator-qualityfiltering>`
   This section describes how to use the 30+ heuristic and classifier filters available within the NeMo Curator and implement custom filters to apply to the documents within the corpora.

:ref:`Language Identification and Unicode Fixing <data-curator-languageidentification>`
   Large, unlabeled text corpora often contain a variety of languages. The NeMo Curator provides utilities to identify languages and fix improperly decoded Unicode characters.

:ref:`GPU Accelerated Exact and Fuzzy Deduplication <data-curator-gpu-deduplication>`
   Both exact and fuzzy deduplication functionalities are supported in NeMo Curator and accelerated using RAPIDS cuDF.

:ref:`GPU Accelerated Semantic Deduplication <data-curator-semdedup>`
   NeMo Curator provides scalable and GPU accelerated semantic deduplication functionality using RAPIDS cuML, cuDF, crossfit and PyTorch.

:ref:`Distributed Data Classification <data-curator-distributeddataclassifer>`
   NeMo-Curator provides a scalable and GPU accelerated module to help users run inference with pre-trained models on large volumes of text documents.

:ref:`Synthetic Data Generation <data-curator-syntheticdata>`
   Synthetic data generation tools and example piplines are available within NeMo Curator.

:ref:`Downstream Task Decontamination <data-curator-downstream>`
   After training, large language models are usually evaluated by their performance on downstream tasks consisting of unseen test data. When dealing with large datasets, there is a potential for leakage of this test data into the model’s training dataset. NeMo Curator allows you to remove sections of documents in your dataset that are present in downstream tasks.

:ref:`Personally Identifiable Information Identification and Removal <data-curator-pii>`
   The purpose of the personally identifiable information (PII) redaction tool is to help scrub sensitive data out of training datasets

.. toctree::
   :maxdepth: 4
   :titlesonly:


   download.rst
   documentdataset.rst
   cpuvsgpu.rst
   qualityfiltering.rst
   languageidentificationunicodeformatting.rst
   gpudeduplication.rst
   semdedup.rst
   syntheticdata.rst
   taskdecontamination.rst
   personalidentifiableinformationidentificationandremoval.rst
   distributeddataclassification.rst

-------------------
Image Curation
-------------------

:ref:`Get Started <data-curator-image-getting-started>`
   Install NeMo Curator's image curation modules.

:ref:`Image-Text Pair Datasets <data-curator-image-datasets>`
   Image-text pair datasets are commonly used as the basis for training multimodal generative models. NeMo Curator interfaces with the standardized WebDataset format for curating such datasets.

:ref:`Image Embedding Creation <data-curator-image-embedding>`
   Image embeddings are the backbone to many data curation operations in NeMo Curator. This section describes how to efficiently create embeddings for massive datasets.

:ref:`Classifiers <data-curator-image-classifiers>`
   NeMo Curator provides several ways to use common classifiers like aesthetic scoring and not-safe-for-work (NSFW) scoring.

:ref:`Semantic Deduplication <data-curator-semdedup>`
   Semantic deduplication with image datasets has been shown to drastically improve model performance. NeMo Curator has a semantic deduplication module that can work with any modality.

.. toctree::
   :maxdepth: 4
   :titlesonly:

   image/gettingstarted.rst
   image/datasets.rst
   image/classifiers/index.rst
   semdedup.rst


-------------------
Reference
-------------------

:ref:`NeMo Curator on Kubernetes <data-curator-kubernetes>`
   Demonstration of how to run the NeMo Curator on a Dask Cluster deployed on top of Kubernetes

:ref:`NeMo Curator and Apache Spark <data-curator-sparkother>`
   Demonstration of how to read and write datasets when using Apache Spark and NeMo Curator

:ref:`Best Practices <data-curator-best-practices>`
   A collection of suggestions on how to best use NeMo Curator to curate your dataset

:ref:`Next Steps <data-curator-next-steps>`
   Now that you've curated your data, let's discuss where to go next in the NeMo Framework to put it to good use.

`Tutorials <https://github.com/NVIDIA/NeMo-Curator/tree/main/tutorials>`__
   To get started, you can explore the NeMo Curator GitHub repository and follow the available tutorials and notebooks. These resources cover various aspects of data curation, including training from scratch and Parameter-Efficient Fine-Tuning (PEFT).

:ref:`API Docs <data-curator-api>`
   API Documentation for all the modules in NeMo Curator

.. toctree::
   :maxdepth: 4
   :titlesonly:


   kubernetescurator.rst
   sparkother.rst
   bestpractices.rst
   nextsteps.rst
   api/index.rst