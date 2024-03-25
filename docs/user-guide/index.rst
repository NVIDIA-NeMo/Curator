.. include:: DataCuration.rsts

:ref:`Downloading and Extracting Text <data-curator-download>`
   Downloading a massive public dataset is usually the first step in data curation, and it can be cumbersome due to the dataset’s massive size and hosting method. This section describes how to download and extract large corpora efficiently.

:ref:`Working with DocumentDataset <data-curator-documentdataset>`
   DocumentDataset is the standard format for datasets in NeMo Curator. This section describes how to get datasets in and out of this format, as well as how DocumentDataset interacts with the modules.

:ref:`CPU and GPU Modules with Dask <data-curator-cpuvsgpu>`
   NeMo Curator provides both CPU based modules and GPU based modules and supports methods for creating compatible Dask clusters and managing the dataset transfer between CPU and GPU.

:ref:`Document Filtering <data-curator-documentfiltering>`
   This section describes how to use the 30+ filters available within the NeMo Curator and implement custom filters to apply to the documents within the corpora.

:ref:`Language Identification and Unicode Fixing <data-curator-languageidentification>`
   Large, unlabeled text corpora often contain a variety of languages. The NeMo Curator provides utilities to identify languages and fix improperly decoded Unicode characters.

:ref:`GPU Accelerated Exact and Fuzzy Deduplication <data-curator-gpu-deduplication>`
   Both exact and fuzzy deduplication functionalities are supported in NeMo Curator and accelerated using RAPIDS cuDF.

:ref:`Classifier and Heuristic Quality Filtering <data-curator-qualityfiltering>`
   Classifier-based filtering involves training a small text classifer to label a document as either high quality or low quality. Heuristic-based filtering uses simple rules (e.g. Are there too many punctuation marks?) to score a document. NeMo Curator offers both classifier and heuristic-based quality filtering of documents.

:ref:`Downstream Task Decontamination/Deduplication <data-curator-downstream>`
   After training, large language models are usually evaluated by their performance on downstream tasks consisting of unseen test data. When dealing with large datasets, there is a potential for leakage of this test data into the model’s training dataset. NeMo Curator allows you to remove sections of documents in your dataset that are present in downstream tasks.

:ref:`Personally Identifiable Information Identification and Removal <data-curator-pii>`
   The purpose of the personally identifiable information (PII) redaction tool is to help scrub sensitive data out of training datasets

.. toctree::
   :maxdepth: 4
   :titlesonly:


   Download.rst
   DocumentDataset.rst
   CPUvsGPU.rst
   QualityFiltering.rst
   LanguageIdentificationUnicodeFormatting.rst
   GpuDeduplication.rst
   TaskDecontamination.rst
   PersonalIdentifiableInformationIdentificationAndRemoval.rst
   DistributedDataClassification.rst
