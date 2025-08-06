---
description: "Implement custom image embedding logic by subclassing the ImageEmbedder base class for specialized models"
categories: ["how-to-guides"]
tags: ["embedding", "custom", "advanced", "subclassing", "research"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "how-to"
modality: "image-only"
---

(image-process-data-embeddings-custom)=
# Custom Image Embedders

Advanced users can implement their own image embedding logic by subclassing the `ImageEmbedder` base class. This is useful if you need to use a model not available in `timm`, integrate a proprietary or research model, or customize data loading or preprocessing.

## How It Works

To create a custom image embedder, follow these steps:

1. **Subclass the Base Class**: Create a new class that inherits from `nemo_curator.image.embedders.ImageEmbedder`.
2. **Implement Data Loading**: Define the `load_dataset_shard(self, tar_path: str)` method to load and yield batches of images and metadata from a WebDataset tar shard.
3. **Implement Model Loading**: Define the `load_embedding_model(self, device: str)` method to load your embedding model on the specified device and return a callable for inference.
4. **Return Results**: Ensure that the batch of data returned by `load_dataset_shard` matches the input expected by your model, and that the metadata is a list of dictionaries, each with the dataset's `id_col`.
5. **Use Your Embedder**: Instantiate and use your custom embedder in the same way as the built-in embedders.

## Example

```python
from nemo_curator.image.embedders import ImageEmbedder

class MyCustomEmbedder(ImageEmbedder):
    def load_dataset_shard(self, tar_path: str):
        # Implement your custom data loading logic here
        pass

    def load_embedding_model(self, device: str):
        # Load and return your model here
        pass
```

## When to Use

- You need a model or pipeline not supported by `TimmImageEmbedder`
- You want to experiment with new architectures or preprocessing

## Resources

- [ImageEmbedder base class (source)](https://github.com/NVIDIA/NeMo-Curator/blob/main/nemo_curator/image/embedders/base.py)