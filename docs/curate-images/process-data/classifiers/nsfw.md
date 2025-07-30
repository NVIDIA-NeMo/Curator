---
description: "NSFW classifier for detecting inappropriate content in images using CLIP embeddings and MLP architecture"
categories: ["how-to-guides"]
tags: ["nsfw", "classification", "clip", "safety", "content-filtering"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "image-only"
---

(image-process-data-classifiers-nsfw)=
# NSFW Classifier

The NSFW (Not Safe For Work) Classifier detects the likelihood that an image contains explicit or unsafe content. It outputs a probability score from 0 (safe) to 1 (NSFW), helping you filter or flag images in your datasets.

## Model Details

- **Architecture:** MLP trained on OpenAI CLIP ViT-L/14 image embeddings
- **Source:** [CLIP-based NSFW Detector](https://github.com/LAION-AI/CLIP-based-NSFW-Detector)
- **Output Field:** `nsfw_score`
- **Score Range:** 0–1 (higher is more likely NSFW)
- **Embedding Requirement:** CLIP ViT-L/14 (see {ref}`image-process-data-embeddings`)

## How It Works

The classifier takes normalized image embeddings and predicts the probability of NSFW content. It is lightweight and can be run on the GPU alongside embedding computation for efficient batch processing.

## Usage

:::: {tab-set}

::: {tab-item} Python

```python
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

dataset_with_nsfw_scores.save_metadata()
```
:::

::::

## Key Parameters

| Parameter         | Default         | Description                                                                 |
|-------------------|-----------------|-----------------------------------------------------------------------------|
| `embedding_column`| `image_embedding`| Name of the column with image embeddings                                    |
| `pred_column`     | `nsfw_score`     | Name of the output column for scores                                        |
| `batch_size`      | `-1`            | Batch size for inference; `-1` processes all at once                        |
| `model_path`      | *auto*          | Path to model weights; downloads if not provided                            |

## Performance Notes

- The model is small and can be loaded onto the GPU with the embedding model for fast, in-place scoring.
- Batch size can be increased for faster throughput if memory allows.

## Best Practices

- Use normalized CLIP ViT-L/14 embeddings for best results.
- Run the classifier immediately after embedding to avoid extra I/O.
- Review a sample of scores to calibrate thresholds for your use case.

## Additional Resources

- [Image Curation Tutorial](https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/image-curation/image-curation.ipynb)
- [API Reference](https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/api/image/classifiers.html) 