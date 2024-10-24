=========================
NSFW Classifier
=========================

--------------------
Overview
--------------------
Not-safe-for-work (NSFW) classifiers determine the likelihood of an image containing sexually explicity material.
NeMo Curator integrates with `CLIP-based-NSFW-Detector <https://github.com/LAION-AI/CLIP-based-NSFW-Detector>`_ that outputs a value between 0 and 1 where 1 means the content is NSFW.

--------------------
Use Cases
--------------------
Removing unsafe content is common in most data processing pipelines to prevent your generative AI model from learning to produce unsafe material.
For example, `Data Comp <https://arxiv.org/abs/2304.14108>`_ filter out NSFW content before conducting their experiments.

--------------------
Prerequisites
--------------------
Make sure you check out the `image curation getting started page <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/image/gettingstarted.html>`_ to install everything you will need.

--------------------
Usage
--------------------

The NSFW classifier is a small MLP classifier that takes OpenAI CLIP ViT-L/14 image embeddings as input.
This model is available through the ``vit_large_patch14_clip_quickgelu_224.openai`` identifier in ``TimmImageEmbedder``.
First, we can compute these embeddings, then we can perform the classification.

.. code-block:: python

    from nemo_curator import get_client
    from nemo_curator.datasets import ImageTextPairDataset
    from nemo_curator.image.embedders import TimmImageEmbedder
    from nemo_curator.image.classifiers import NsfwClassifier

    client = get_client(cluster_type="gpu")

    dataset = ImageTextPairDataset.from_webdataset(path="/path/to/dataset", id_col="key")

    embedding_model = TimmImageEmbedder(
        "vit_large_patch14_clip_quickgelu_224.openai",
        pretrained=True,
        batch_size=1024,
        num_threads_per_worker=16,
        normalize_embeddings=True,
    )
    safety_classifier = NsfwClassifier()

    dataset_with_embeddings = embedding_model(dataset)
    dataset_with_nsfw_scores = safety_classifier(dataset_with_embeddings)

    # Metadata will have a new column named "nsfw_score"
    dataset_with_nsfw_scores.save_metadata()

--------------------
Key Parameters
--------------------
* ``batch_size=-1`` is the optional batch size parameter. By default, it will process all the embeddings in a shard at once. Since the NSFW classifier is a small model, this is usually fine.

---------------------------
Performance Considerations
---------------------------
Since the NSFW model is so small, you can load it onto the GPU at the same time as the embedding model and perform inference directly after computing the embeddings.
Check out this example:

.. code-block:: python

    from nemo_curator import get_client
    from nemo_curator.datasets import ImageTextPairDataset
    from nemo_curator.image.embedders import TimmImageEmbedder
    from nemo_curator.image.classifiers import NsfwClassifier

    client = get_client(cluster_type="gpu")

    dataset = ImageTextPairDataset.from_webdataset(path="/path/to/dataset", id_col="key")

    embedding_model = TimmImageEmbedder(
        "vit_large_patch14_clip_quickgelu_224.openai",
        pretrained=True,
        batch_size=1024,
        num_threads_per_worker=16,
        normalize_embeddings=True,
        classifiers=[NsfwClassifier()],
    )

    dataset_with_nsfw_scores = embedding_model(dataset)

    # Metadata will have a new column named "nsfw_score"
    dataset_with_nsfw_scores.save_metadata()


---------------------------
Additional Resources
---------------------------
* `Image Curation Tutorial <https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/image-curation/image-curation.ipynb>`_
* `API Reference <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/api/image/classifiers.html>`_