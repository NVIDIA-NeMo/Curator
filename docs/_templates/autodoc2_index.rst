API Reference
=============

NeMo Curator's API reference provides comprehensive technical documentation for all modules, classes, and functions. Use these references to understand the technical foundation of NeMo Curator and integrate it with your data curation workflows.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`database;1.5em;sd-mr-1` Core Data Handling
      :link: datasets/datasets
      :link-type: doc
      :class-card: sd-border-0

      **Datasets & Download**

      Essential classes for loading, managing, and downloading training data from various sources.
      
      :bdg-secondary:`doc-dataset` :bdg-secondary:`parallel-dataset` :bdg-secondary:`arxiv` :bdg-secondary:`commoncrawl`

   .. grid-item-card:: :octicon:`filter;1.5em;sd-mr-1` Data Processing  
      :link: filters/filters
      :link-type: doc
      :class-card: sd-border-0

      **Filters & Modifiers**

      Tools for cleaning, filtering, and transforming text data to improve quality and remove unwanted content.
      
      :bdg-secondary:`classifier-filter` :bdg-secondary:`heuristic-filter` :bdg-secondary:`pii-modifier`

   .. grid-item-card:: :octicon:`code;1.5em;sd-mr-1` Classification & Analysis
      :link: classifiers/classifiers  
      :link-type: doc
      :class-card: sd-border-0

      **AI-Powered Analysis**

      Advanced classification tools and image processing capabilities for content analysis and quality assessment.
      
      :bdg-secondary:`aegis` :bdg-secondary:`content-type` :bdg-secondary:`domain-classifier`

   .. grid-item-card:: :octicon:`shield-check;1.5em;sd-mr-1` Privacy & Security
      :link: pii/pii
      :link-type: doc
      :class-card: sd-border-0

      **PII Detection & Redaction**

      Identify and handle personally identifiable information in datasets with advanced recognition algorithms.
      
      :bdg-secondary:`recognizers` :bdg-secondary:`algorithms` :bdg-secondary:`redaction`

   .. grid-item-card:: :octicon:`zap;1.5em;sd-mr-1` Synthetic Data
      :link: synthetic/synthetic
      :link-type: doc
      :class-card: sd-border-0

      **Data Generation**

      Create high-quality synthetic training data using advanced language models and generation techniques.
      
      :bdg-secondary:`generator` :bdg-secondary:`nemotron` :bdg-secondary:`mixtral`

   .. grid-item-card:: :octicon:`tools;1.5em;sd-mr-1` Advanced Processing
      :link: modules/modules
      :link-type: doc
      :class-card: sd-border-0

      **Deduplication & Modules**

      Advanced processing modules including semantic deduplication, fuzzy matching, and data pipeline components.
      
      :bdg-secondary:`semantic-dedup` :bdg-secondary:`fuzzy-dedup` :bdg-secondary:`add-id`

.. toctree::
   :maxdepth: 1
   :caption: API Modules
   :hidden:

   datasets/datasets
   download/download
   filters/filters
   modifiers/modifiers
   modules/modules
   classifiers/classifiers
   image/image
   pii/pii
   synthetic/synthetic
   services/services
   nemo_run/nemo_run
   tasks/tasks
   utils/utils
