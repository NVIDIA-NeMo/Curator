=============
Text Curation
=============
:ref:`Downloading and Extracting Text <data-curator-download>`
   Downloading a massive public dataset is usually the first step in data curation, and it can be cumbersome due to the dataset’s massive size and hosting method. This section describes how to download and extract large corpora efficiently.

:ref:`Working with DocumentDataset <data-curator-documentdataset>`
   DocumentDataset is the standard format for datasets in NeMo Curator. This section describes how to get datasets in and out of this format, as well as how DocumentDataset interacts with the modules.

:ref:`CPU and GPU Modules with Dask <data-curator-cpuvsgpu>`
   NeMo Curator provides both CPU based modules and GPU based modules and supports methods for creating compatible Dask clusters and managing the dataset transfer between CPU and GPU.

:ref:`Document Filtering <data-curator-qualityfiltering>`
   This section describes how to use the 30+ heuristic and classifier filters available within the NeMo Curator and implement custom filters to apply to the documents within the corpora.

:ref:`Language Identification <data-curator-languageidentification>`
   Large, unlabeled text corpora often contain a variety of languages. NeMo Curator provides utilities to identify languages.

:ref:`Text Cleaning <data-curator-text-cleaning>`
   Many parts of the Internet contained malformed or poorly formatted text. NeMo Curator can fix many of these issues with text.

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
   languageidentification.rst
   textcleaning.rst
   gpudeduplication.rst
   semdedup.rst
   syntheticdata.rst
   taskdecontamination.rst
   personalidentifiableinformationidentificationandremoval.rst
   distributeddataclassification.rst