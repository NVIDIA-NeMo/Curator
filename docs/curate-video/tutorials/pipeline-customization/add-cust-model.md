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

Learn how to integrate custom models into Ray Curator stages.

The NeMo Video Curator container includes a robust set of default models, however, they may not always meet your pipeline requirements. In this tutorial, we'll demonstrate how to add a custom x model used in a new pipeline stage.

## Before You Start

Before you begin adding a custom model, make sure that you have:

* Reviewed the [pipeline concepts and diagrams](about-concepts-video).  
* A working Ray Curator development environment.  
* Optionally prepared a container image that includes your model dependencies.  
* Optionally [created a custom environment](video-tutorials-pipeline-cust-env) to support your new custom model.

---

## How to Add a Custom Model

### Review Model Interface

In Ray Curator, models are defined by classes inheriting from `ray_curator.models.base.ModelInterface`. The interface looks like this:

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

from ray_curator.models.base import ModelInterface

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

#### Define Guard Imports

If your model has optional imports tied to specific environments, guard them at import time in your stage or during setup to avoid import errors on workers that do not require that environment.

This condition checks to make sure we are running in a Conda environment that has our model's dependencies.

#### Define the PyTorch Model

```py
WEIGHTS_MODEL_ID = "example/my-model"  # your huggingface (or other) model id

class MyCore(torch.nn.Module):
    def __init__(self, resolution: int = 224):
        super().__init__()
        # Initialize network and load weights from a local path derived from model_dir and WEIGHTS_MODEL_ID
```

`WEIGHTS_NAME = "nvidia/C-RADIO"` is used to define where the model is on HuggingFace, and where it will be in our local model weights cache. During the execution of the pipelines the model weights for all models will be downloaded before `CRadio.setup()` is called (and therefore before `CRadioCore` is constructed), so you may assume the weights will be at `model_utils.get_local_dir_for_weights_name(WEIGHTS_NAME)`

#### Define the Model Interface

```py
class MyModel(ModelInterface):
	...
```

Our `CRadio` class implements the model interface. It defines a collection of methods that ensure the model weights can be downloaded and that it is initialized properly.

```py
    def setup(self) -> None:
        self._model = MyCore(self.resolution)
        self._model.eval()
```

The setup method initializes the underlying `CRadioCore` class that performs the model inference.

```py
    def model_id_names(self) -> list[str]:
        return [WEIGHTS_MODEL_ID]
```

The `weights_names` property returns a list of weights. These are used to identify the model weights that are cached locally or on S3. They typically correspond to the name of the HuggingFace models, but they don't have to.

Remove references to `id_file_mapping` and similar utilities; Ray Curator does not expose this in the base interface.

`id_file_mapping` is used when running the model download/upload script to selectively choose which files from the model repository should be downloaded. We don't need to make such a selection here, so we mark it as `None`.

If your stage requires a specific environment, manage that in the stage’s `resources` (for example, `gpu_memory_gb`, `nvdecs`, `nvencs`) and container image, rather than on the model.

`conda_env_name` defines the conda environment that should be used when running this model in a pipeline.

GPU allocation is managed at the stage level using `Resources`, not on the model.

`num_gpus` specifies the fractional amount of GPU resources that this model will need. This value is only used for scheduling workers across GPUs or on the same GPU, and it does nothing to actually limit the usage of the GPU(s) allocated to this model.

### Manage model weights

Provide your model with a `model_dir` where weights are stored. Your stage should ensure that any required weights are available at runtime (for example, by mounting them into the container or downloading them prior to execution). See existing models such as `InternVideo2MultiModality` for reference: `ray-curator/ray_curator/models/internvideo2_mm.py`.

## Next Steps

Now that you have created a custom model, you can [create a custom stage](video-tutorials-pipeline-cust-add-stage) that uses your code.