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