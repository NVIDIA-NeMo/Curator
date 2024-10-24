=========================
Aesthetic Classifier
=========================

--------------------
Overview
--------------------
Aesthetic classifiers can be used to assess the subjective quality of an image.
NeMo Curator integrates the `improved aesthetic predictor <https://github.com/christophschuhmann/improved-aesthetic-predictor>`_ that outputs a score from 0-10 where 10 is aesthetically pleasing.

--------------------
Use Cases
--------------------
Filtering by aesthetic quality is common in generative image pipelines.
For example, `Stable Diffusion <https://github.com/CompVis/stable-diffusion?tab=readme-ov-file#weights>`_ progressively filtered by aesthetic score during training.


--------------------
Prerequisites
--------------------
Make sure you check out the `image curation getting started page <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/image/gettingstarted.html>`_ to install everything you will need.

--------------------
Usage
--------------------

The aesthetic classifier is a linear classifier that takes OpenAI CLIP ViT-L/14 image embeddings as input.
This model is available through the ``vit_large_patch14_clip_quickgelu_224.openai`` identifier in ``TimmImageEmbedder``.
First, we can compute these embeddings, then we can perform the classification.

.. code-block:: python

    from nemo_curator import get_client
    from nemo_curator.datasets import ImageTextPairDataset
    from nemo_curator.image.embedders import TimmImageEmbedder
    from nemo_curator.image.classifiers import AestheticClassifier

    client = get_client(cluster_type="gpu")

    dataset = ImageTextPairDataset.from_webdataset(path="/path/to/dataset", id_col="key")

    embedding_model = TimmImageEmbedder(
        "vit_large_patch14_clip_quickgelu_224.openai",
        pretrained=True,
        batch_size=1024,
        num_threads_per_worker=16,
        normalize_embeddings=True,
    )
    aesthetic_classifier = AestheticClassifier()

    dataset_with_embeddings = embedding_model(dataset)
    dataset_with_aesthetic_scores = aesthetic_classifier(dataset_with_embeddings)

    # Metadata will have a new column named "aesthetic_score"
    dataset_with_aesthetic_scores.save_metadata()

--------------------
Key Parameters
--------------------
* ``batch_size=-1`` is the optional batch size parameter. By default, it will process all the embeddings in a shard at once. Since the aesthetic classifier is a linear model, this is usually fine.

---------------------------
Performance Considerations
---------------------------
Since the aesthetic model is so small, you can load it onto the GPU at the same time as the embedding model and perform inference directly after computing the embeddings.
Check out this example:

.. code-block:: python

    from nemo_curator import get_client
    from nemo_curator.datasets import ImageTextPairDataset
    from nemo_curator.image.embedders import TimmImageEmbedder
    from nemo_curator.image.classifiers import AestheticClassifier

    client = get_client(cluster_type="gpu")

    dataset = ImageTextPairDataset.from_webdataset(path="/path/to/dataset", id_col="key")

    embedding_model = TimmImageEmbedder(
        "vit_large_patch14_clip_quickgelu_224.openai",
        pretrained=True,
        batch_size=1024,
        num_threads_per_worker=16,
        normalize_embeddings=True,
        classifiers=[AestheticClassifier()],
    )

    dataset_with_aesthetic_scores = embedding_model(dataset)

    # Metadata will have a new column named "aesthetic_score"
    dataset_with_aesthetic_scores.save_metadata()

---------------------------
Additional Resources
---------------------------
* `Image Curation Tutorial <https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/image-curation/image-curation.ipynb>`_
* `API Reference <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/api/image/classifiers.html>`_