---
description: "Aesthetic classifier for predicting visual quality of images using CLIP embeddings and human preference training"
categories: ["how-to-guides"]
tags: ["aesthetic", "classification", "clip", "quality-filtering", "mlp"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "image-only"
---

(image-process-data-classifiers-aesthetic)=
# Aesthetic Classifier

The Aesthetic Classifier predicts the subjective visual quality of images using a model trained on human aesthetic preferences. It outputs a score from 0 (least aesthetic) to 10 (most aesthetic), making it useful for filtering or ranking images in generative pipelines and dataset curation.

## Model Details

- **Architecture:** Linear MLP trained on OpenAI CLIP ViT-L/14 image embeddings
- **Source:** [Improved Aesthetic Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- **Output Field:** `aesthetic_score`
- **Score Range:** 0–10 (higher is more aesthetic)
- **Embedding Requirement:** CLIP ViT-L/14 (see {ref}`image-process-data-embeddings`)

## How It Works

The classifier takes normalized image embeddings and predicts an aesthetic score. It is lightweight and can be run on the GPU alongside embedding computation for efficient batch processing.

## Usage

:::: {tab-set}

::: {tab-item} Python

```python
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

dataset_with_aesthetic_scores.save_metadata()
```
:::

::::

## Key Parameters

| Parameter         | Default         | Description                                                                 |
|-------------------|-----------------|-----------------------------------------------------------------------------|
| `embedding_column`| `image_embedding`| Name of the column with image embeddings                                    |
| `pred_column`     | `aesthetic_score`| Name of the output column for scores                                        |
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