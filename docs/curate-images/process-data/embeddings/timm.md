---
description: "Generate image embeddings using PyTorch Image Models (timm) library with GPU acceleration and distributed processing"
categories: ["how-to-guides"]
tags: ["embedding", "timm", "clip", "vit", "gpu-accelerated", "dali"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "image-only"
---

(image-process-data-embeddings-timm)=
# TimmImageEmbedder

The `TimmImageEmbedder` is the primary image embedding option in NeMo Curator. It leverages the [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) library to provide access to a wide range of state-of-the-art computer vision models for generating image embeddings. These embeddings are used for downstream tasks such as classification, filtering, and duplicate removal.

## How it Works

The `TimmImageEmbedder` generates image embeddings in the following steps:

1. **Model Selection**: You specify a model name from the `timm` library (for example, CLIP, ViT, ResNet). The embedder loads the model, optionally with pretrained weights.

2. **Data Loading and Preprocessing**: The embedder uses NVIDIA DALI for efficient, GPU-accelerated loading and preprocessing of images from WebDataset tar files. DALI-generated `.idx` files can be used to further speed up loading.

3. **Batch Processing**: Images are processed in batches, with the batch size and number of data loading threads configurable for optimal GPU utilization.

4. **Embedding Generation**: Each batch of images is passed through the selected model to generate embeddings. Optionally, embeddings are normalized (recommended for most downstream tasks).

5. **Distributed Execution**: The process can be distributed across multiple GPUs or nodes, enabling large-scale embedding generation.

6. **Output**: The resulting embeddings are added as a new column (default: `image_embedding`) in the dataset metadata, which can then be saved for downstream tasks such as classification, filtering, or duplicate removal.

## Usage

### Python Example

```python
from nemo_curator import get_client
from nemo_curator.datasets import ImageTextPairDataset
from nemo_curator.image.embedders import TimmImageEmbedder

client = get_client(cluster_type="gpu")
dataset = ImageTextPairDataset.from_webdataset(path="/path/to/dataset", id_col="key")

embedding_model = TimmImageEmbedder(
    "vit_large_patch14_clip_quickgelu_224.openai",  # Any timm model name
    pretrained=True,
    batch_size=1024,
    num_threads_per_worker=16,
    normalize_embeddings=True,
)

dataset_with_embeddings = embedding_model(dataset)
# Metadata will have a new column named "image_embedding"
dataset_with_embeddings.save_metadata()
```

## Key Parameters

- `model_name`: Name of the timm model to use (see `timm.list_models()`)
- `pretrained`: Whether to use pretrained weights (recommended)
- `batch_size`: Number of images processed per GPU per batch
- `num_threads_per_worker`: Number of threads for DALI data loading
- `normalize_embeddings`: Normalize output embeddings (required for most classifiers)
- `autocast`: Use mixed precision for faster inference (default: True)
- `use_index_files`: Use DALI-generated `.idx` files for faster loading

## Performance Tips

- Use `.idx` files in your dataset directory to speed up DALI data loading ([DALI docs](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/dataloading_webdataset.html#Creating-an-index))
- Choose a batch size that fits your GPU memory for optimal throughput
- Use multiple GPUs for distributed embedding generation

## Best Practices

- Always use pretrained weights unless you have a specific reason to train from scratch
- Normalize embeddings if you plan to use them for classification or similarity search
- Monitor GPU memory usage and adjust `batch_size` accordingly
- For large datasets, generate and use `.idx` files to accelerate data loading
- Review the output embeddings for expected dimensionality and normalization

## Additional Resources

- [timm model documentation](https://huggingface.co/docs/timm/index)