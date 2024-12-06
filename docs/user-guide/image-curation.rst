==============
Image Curation
==============

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