---
description: "Implement custom image embedding logic by subclassing ProcessingStage for specialized models and preprocessing"
categories: ["how-to-guides"]
tags: ["embedding", "custom", "advanced", "subclassing", "research"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "how-to"
modality: "image-only"
---

(image-process-data-embeddings-custom)=

# Custom Embedding Stages

Advanced users can create their own image embedding logic by extending the `ProcessingStage` base class. This is useful if you need to use a model not available in the built-in `ImageEmbeddingStage`, integrate a proprietary or research model, or customize preprocessing.

## How It Works

To create a custom embedding stage, follow these steps:

1. **Subclass ProcessingStage**: Create a new class that inherits from `ProcessingStage[ImageBatch, ImageBatch]`.
2. **Define Setup**: Define the `setup()` method to load your embedding model.
3. **Define Process**: Define the `process()` method to generate embeddings for `ImageBatch` objects.
4. **Handle Batching**: Add batch processing to process several images at once.
5. **Use Your Stage**: Add your custom stage to a pipeline like any built-in stage.

## Example

```python
from dataclasses import dataclass
import numpy as np
import torch
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import ImageBatch

@dataclass
class MyCustomEmbeddingStage(ProcessingStage[ImageBatch, ImageBatch]):
    model_path: str
    batch_size: int = 32
    _name: str = "my_custom_embedding"

    def setup(self, worker_metadata=None):
        # Load your custom model
        self.model = self.load_my_model(self.model_path)
        
    def load_my_model(self, model_path):
        # Implement your model loading logic here
        # This is a placeholder - replace with your actual model loading
        # Example: return torch.jit.load(model_path)
        class DummyModel:
            def __call__(self, images):
                # Return dummy embeddings - replace with real model inference
                return np.random.randn(len(images), 512).astype(np.float32)
        return DummyModel()
        
    def process(self, task: ImageBatch) -> ImageBatch:
        # Process images in batches
        for i in range(0, len(task.data), self.batch_size):
            batch = task.data[i:i + self.batch_size]
            
            # Extract image data from ImageObjects
            images = []
            for img_obj in batch:
                if img_obj.image_data is not None:
                    images.append(img_obj.image_data)
            
            if not images:
                continue
                
            # Generate embeddings using your model
            embeddings = self.model(images)
            
            # Store embeddings in ImageObjects
            for j, img_obj in enumerate(batch):
                if img_obj.image_data is not None:
                    img_obj.embedding = embeddings[j]
                
        return task

# Usage in pipeline
pipeline.add_stage(MyCustomEmbeddingStage(
    model_path="/path/to/my/model",
    batch_size=16,
))
```

## When to Use

- You need a model or architecture not supported by `ImageEmbeddingStage`
- You want to experiment with new preprocessing techniques
- You have a proprietary or research model to integrate
- You need specific batch processing or memory management approaches

## Best Practices

- Follow the same patterns as `ImageEmbeddingStage` for consistency
- Add proper error handling and logging
- Consider GPU acceleration if your model supports it
- Handle different image sizes and formats properly
- Consider memory usage with large batches

## Resources

- [ProcessingStage base class](https://github.com/NVIDIA/NeMo-Curator/blob/main/nemo_curator/stages/base.py)
- [ImageEmbeddingStage implementation](https://github.com/NVIDIA/NeMo-Curator/blob/main/nemo_curator/stages/image/embedders/clip_embedder.py)
