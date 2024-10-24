.. _data-curator-image-embedding:

=========================
Image Embedders
=========================

--------------------
Overview
--------------------
Many image curation features in NeMo Curator operate on image embeddings instead of images directly.
Image embedders provide a scalable way of generating embeddings for each image in the dataset.

--------------------
Use Cases
--------------------
* Aesthetic and NSFW classification both use image embeddings generated from OpenAI's CLIP ViT-L variant.
* Semantic deduplication computes the similarity of datapoints.

--------------------
Prerequisites
--------------------
Make sure you check out the `image curation getting started page <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/image/gettingstarted.html>`_ to install everything you will need.

--------------------
Timm Image Embedder
--------------------

`PyTorch Image Models (timm) <https://github.com/huggingface/pytorch-image-models>`_ is a library containing SOTA computer vision models.
Many of these models are useful in generating image embeddings for modules in NeMo Curator.

.. code-block:: python

    from nemo_curator import get_client
    from nemo_curator.datasets import ImageTextPairDataset
    from nemo_curator.image.embedders import TimmImageEmbedder

    client = get_client(cluster_type="gpu")

    dataset = ImageTextPairDataset.from_webdataset(path="/path/to/dataset", id_col="key")

    embedding_model = TimmImageEmbedder(
        "vit_large_patch14_clip_quickgelu_224.openai",
        pretrained=True,
        batch_size=1024,
        num_threads_per_worker=16,
        normalize_embeddings=True,
    )

    dataset_with_embeddings = embedding_model(dataset)

    # Metadata will have a new column named "image_embedding"
    dataset_with_embeddings.save_metadata()

Here, we load a dataset in and compute the image embeddings using ``vit_large_patch14_clip_quickgelu_224.openai``.
At the end of the process, our metadata files have a new column named "image_embedding" that contains the image embedddings for each datapoint.

--------------------
Key Parameters
--------------------
* ``pretrained=True`` ensures you download the pretrained weights of the model.
* ``batch_size=1024`` determines the number of images processed on each individual GPU at once.
* ``num_threads_per_worker=16`` determines the number of threads used by DALI for dataloading.
* ``normalize_embeddings=True`` will normalize each embedding. NeMo Curator's classifiers expect normalized embeddings as input.

---------------------------
Performance Considerations
---------------------------

Under the hood, the image embedding model performs the following operations:

1. Download the weights of the model.
2. Download the PyTorch image transformations (resize and center-crop for example).
3. Convert the PyTorch image transformations to DALI transformations.
4. Load a shard of metadata (a ``.parquet`` file) onto each GPU you have available using Dask-cuDF.
5. Load a copy of the model onto each GPU.
6. Repeatedly load images into batches of size ``batch_size`` onto each GPU with a given threads per worker (``num_threads_per_worker``) using DALI.
7. The model is run on the batch (without ``torch.autocast()`` since ``autocast=False``).
8. The output embeddings of the model are normalized since ``normalize_embeddings=True``.

There are a couple of key performance considerations from this flow.

* You must have an NVIDIA GPU that mets the `requirements <https://github.com/NVIDIA/NeMo-Curator?tab=readme-ov-file#requirements>`_.
* You can create ``.idx`` files in the same directory of the tar files to speed up dataloading times. See the `DALI documentation <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/dataloading_webdataset.html#Using-readers.webdataset-operator>`_ for more information.

------------------------
Custom Image Embedder
------------------------

To write your own custom embedder, you inherit from ``nemo_curator.image.embedders.ImageEmbedder`` and override two methods as shown below:

.. code-block:: python

    from nemo_curator.image.embedders import ImageEmbedder

    class MyCustomEmbedder(ImageEmbedder):

        def load_dataset_shard(self, tar_path: str) -> Iterable:
            # Implement me!
            pass

        def load_embedding_model(self, device: str) -> Callable:
            # Implement me!
            pass


* ``load_dataset_shard()`` will take in a path to a tar file and return an iterable over the shard. The iterable should return a tuple of ``(a batch of data, metadata)``.
  The batch of data can be of any form. It will be directly passed to the model returned by ``load_embedding_model()``.
  The metadata should be a dictionary of metadata, with a field corresponding to the ``id_col`` of the dataset.
  In our example, the metadata should include a value for ``"key"``.
* ``load_embedding_model()`` will take a device and return a callable object.
  This callable will take as input a batch of data produced by ``load_dataset_shard()``.

---------------------------
Additional Resources
---------------------------

* `Aesthetic Classifier <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/image/classifiers/aesthetic.html>`_
* `NSFW Classifier <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/image/classifiers/nsfw.html>`_
* `Semantic Deduplication <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/semdedup.html>`_
* `Image Curation Tutorial <https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/image-curation/image-curation.ipynb>`_
* `API Reference <https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/api/image/embedders.html>`_