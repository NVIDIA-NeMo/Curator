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

Advanced users can implement their own image embedding logic by subclassing the `ProcessingStage` base class. This is useful if you need to use a model not available in the built-in `ImageEmbeddingStage`, integrate a proprietary or research model, or customize preprocessing.

## How It Works

To create a custom embedding stage, follow these steps:

1. **Subclass ProcessingStage**: Create a new class that inherits from `ProcessingStage[ImageBatch, ImageBatch]`.
2. **Implement Setup**: Define the `setup()` method to load your embedding model.
3. **Implement Process**: Define the `process()` method to generate embeddings for `ImageBatch` objects.
4. **Handle Batching**: Implement batch processing for efficient GPU utilization.
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
    num_gpus_per_worker: float = 0.25
    _name: str = "my_custom_embedding"

    def setup(self, worker_metadata=None):
        # Load your custom model
        self.model = self.load_my_model(self.model_path)
        
    def load_my_model(self, model_path):
        # Implement your model loading logic
        # Return a callable that takes image data and returns embeddings
        pass
        
    def process(self, task: ImageBatch) -> ImageBatch:
        # Process images in batches
        for i in range(0, len(task.data), self.batch_size):
            batch = task.data[i:i + self.batch_size]
            
            # Extract image data
            images = [img.image_data for img in batch]
            
            # Generate embeddings using your model
            embeddings = self.model(images)
            
            # Store embeddings in ImageObjects
            for j, img_obj in enumerate(batch):
                img_obj.embedding = embeddings[j]
                
        return task

# Usage in pipeline
pipeline.add_stage(MyCustomEmbeddingStage(
    model_path="/path/to/my/model",
    batch_size=16,
    num_gpus_per_worker=0.5,
))
```

## When to Use

- You need a model or architecture not supported by `ImageEmbeddingStage`
- You want to experiment with new preprocessing techniques
- You have a proprietary or research model to integrate
- You need custom batch processing or memory management

## Best Practices

- Follow the same patterns as `ImageEmbeddingStage` for consistency
- Implement proper error handling and logging
- Use GPU acceleration when available
- Handle different image sizes and formats gracefully
- Consider memory usage with large batches

## Resources

- [ProcessingStage base class](https://github.com/NVIDIA/NeMo-Curator/blob/main/nemo_curator/stages/base.py)
- [ImageEmbeddingStage implementation](https://github.com/NVIDIA/NeMo-Curator/blob/main/nemo_curator/stages/image/embedders/clip_embedder.py)