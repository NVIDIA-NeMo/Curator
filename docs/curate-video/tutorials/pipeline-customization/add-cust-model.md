---
description: "Tutorial for integrating custom models into video curation pipelines for specialized captioning, embedding, or filtering tasks"
categories: ["video-curation"]
tags: ["customization", "models", "machine-learning", "pipeline", "captioning", "embedding", "advanced"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "advanced"
content_type: "tutorial"
modality: "video-only"

---

(video-tutorials-pipeline-cust-add-model)=
# Adding Custom Models

Learn how to integrate custom models into NeMo Curator stages.

The NeMo Curator container includes a robust set of default models, but you can add your own for specialized tasks.

## Before You Start

Before you begin adding a custom model, make sure that you have:

* Reviewed the [pipeline concepts and diagrams](about-concepts-video).  
* A working NeMo Curator development environment.  
* Optionally prepared a container image that includes your model dependencies.  
* Optionally [created a custom environment](video-tutorials-pipeline-cust-env) to support your new custom model.

---

## How to Add a Custom Model

### Review Model Interface

In NeMo Curator, models inherit from `nemo_curator.models.base.ModelInterface` and must implement `model_id_names` and `setup`:

```py
class ModelInterface(abc.ABC):
    """Abstract base class for models used inside stages."""

    @property
    @abc.abstractmethod
    def model_id_names(self) -> list[str]:
        """Return a list of model IDs associated with this model (for example, Hugging Face IDs)."""

    @abc.abstractmethod
    def setup(self) -> None:
        """Set up the model (load weights, allocate resources)."""
```

### Create New Model

For this tutorial, we'll sketch a minimal model for demonstration.

```py
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch

from nemo_curator.models.base import ModelInterface

WEIGHTS_MODEL_ID = "example/my-model"


class MyCore(torch.nn.Module):
    def __init__(self, resolution: int = 224):
        super().__init__()
        self.resolution = resolution
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize your network here
        self.net = torch.nn.Identity().to(self.device)

    @torch.no_grad()
    def __call__(self, x: npt.NDArray[np.float32]) -> torch.Tensor:
        tensor = torch.from_numpy(x).to(self.device).float()
        return self.net(tensor)


class MyModel(ModelInterface):
    def __init__(self, model_dir: str, resolution: int = 224) -> None:
        self.model_dir = model_dir
        self.resolution = resolution
        self._model: Optional[MyCore] = None

    def model_id_names(self) -> list[str]:
        return [WEIGHTS_MODEL_ID]

    def setup(self) -> None:
        # Load weights from self.model_dir/WEIGHTS_MODEL_ID if needed
        self._model = MyCore(self.resolution)
        self._model.eval()
```

Let's go through each part of the code piece by piece.

#### Define the PyTorch Model

```py
WEIGHTS_MODEL_ID = "example/my-model"  # your huggingface (or other) model id

class MyCore(torch.nn.Module):
    def __init__(self, resolution: int = 224):
        super().__init__()
        # Initialize network and load weights from a local path derived from model_dir and WEIGHTS_MODEL_ID
```

Provide a model ID (for example, a HuggingFace ID) if you plan to cache or fetch weights. The pipeline can download weights prior to `setup()` via your model class method if you provide one (see `InternVideo2MultiModality.download_weights_on_node`).

#### Implement the Model Interface

```py
class MyModel(ModelInterface):
	...
```

Your model implements the interface. It defines methods to declare weight identifiers and to initialize the underlying core network.

```py
    def setup(self) -> None:
        self._model = MyCore(self.resolution)
        self._model.eval()
```

The setup method initializes the underlying `MyCore` class that performs the model inference.

```py
    def model_id_names(self) -> list[str]:
        return [WEIGHTS_MODEL_ID]
```

The `model_id_names` property returns a list of weight IDs. These typically correspond to model repository names but do not have to.

If your stage requires a specific environment, manage that in the stage’s `resources` (for example, `gpu_memory_gb`, `nvdecs`, `nvencs`) and container image, rather than on the model. GPU allocation is managed at the stage level using `Resources`, not on the model.

### Manage model weights

Provide your model with a `model_dir` where weights are stored. Your stage should ensure that any required weights are available at runtime (for example, by mounting them into the container or downloading them prior to execution). See existing models such as `InternVideo2MultiModality` for reference: `nemo_curator/models/internvideo2_mm.py`.

## Next Steps

Now that you have created a custom model, you can [create a custom stage](video-tutorials-pipeline-cust-add-stage) that uses your code.
