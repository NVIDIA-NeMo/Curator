---
description: "Advanced tutorial for adding custom processing stages to video curation pipelines for specialized workflow requirements"
categories: ["video-curation"]
tags: ["customization", "pipeline", "stages", "workflow", "advanced", "development"]
personas: ["mle-focused"]
difficulty: "advanced"
content_type: "tutorial"
modality: "video-only"

---

(video-tutorials-pipeline-cust-add-stage)=
# Adding Custom Stages

Learn how to customize NeMo Curator by adding new pipeline stages.

NeMo Curator includes a series of pipelines with default stages; however, they might not always meet your pipeline requirements. This tutorial demonstrates how to add a new pipeline stage and integrate it into a pipeline.

## Before You Start

Before you begin adding a new pipeline stage, make sure that you have:

* Reviewed the [pipeline concepts and diagrams](about-concepts-video).  
* Downloaded the NeMo Curator container.  
* Reviewed the [container environments](reference-infrastructure-container-environments) available.  
* Optionally [created custom code](video-tutorials-pipeline-cust-add-code) that defines your new requirements.  
* Optionally [created a custom environment](video-tutorials-pipeline-cust-env) to support your new custom code.  
* Optionally [created a custom model](video-tutorials-pipeline-cust-add-model).  

## How to Add a Custom Pipeline Stage

### 1. Define the Stage Class

```py
from typing import List

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import VideoTask


class MyCustomStage(ProcessingStage[VideoTask, VideoTask]):
    """Example stage that reads and writes to the VideoTask."""

    _name = "my_custom_stage"
    _resources = Resources(cpus=2.0, gpu_memory_gb=8.0)

    def setup(self, worker_metadata=None) -> None:
        # Initialize models or allocate resources here
        pass

    def process(self, task: VideoTask) -> VideoTask | list[VideoTask]:
        # Implement your processing and return the modified task (or list of tasks)
        return task
```

### 2. Specify Resource Requirements

```py
# You can override resources at construction time using with_()
from nemo_curator.stages.resources import Resources

stage = MyCustomStage().with_(
    resources=Resources(cpus=4.0, gpu_memory_gb=16.0, nvdecs=1, nvencs=1)
)
```

### 3. Implement Core Methods

Required methods for every stage:

#### Setup Method

```py
def setup(self, worker_metadata=None) -> None:
    # Load models, warm up caches, etc.
    pass
```

#### Process Data Method

```py
def process(self, task: VideoTask) -> VideoTask | list[VideoTask]:
    # Process implementation
    return task
```

### 4. Update Data Model

Modify the pipeline's data model to include your stage's outputs:

```py
# In Ray Curator, video data lives in VideoTask.data (a Video) which contains Clips.
# You can attach new information to existing structures (for example, store derived
# arrays in clip.egomotion or add keys to dictionaries), or maintain your own
# data alongside and write it in a custom writer stage.
```

### 5. Modify Pipeline Output Handling

Update the ClipWriterStage to handle your stage's output:

1. Create a writer method:

   ```py
   def _write_custom_output(self, clip: Clip) -> None:
       # writing implementation
   ```

2. Add to the main process:

   ```py
   def process(self, task: VideoTask) -> VideoTask | list[VideoTask]:
       # existing processing
       self._write_custom_output(clip)
       # continue processing
       return task
   ```

## Integration Steps

### 1. Build and Run a Pipeline in Python

```py
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.video.io.video_reader import VideoReader
from nemo_curator.stages.video.io.clip_writer import ClipWriterStage

pipeline = (
    Pipeline(name="custom-video-pipeline")
    .add_stage(VideoReader(input_video_path="/path/to/videos", video_limit=10))
    .add_stage(MyCustomStage())
    .add_stage(
        ClipWriterStage(
            output_path="/path/to/output",
            input_path="/path/to/videos",
            upload_clips=True,
            dry_run=False,
            generate_embeddings=False,
            generate_previews=False,
            generate_captions=False,
            embedding_algorithm="cosmos-embed1-224p",
            caption_models=["qwen"],
            enhanced_caption_models=["qwen_lm"],
        )
    )
)

# Optionally provide an executor; defaults to XennaExecutor
pipeline.run()
```

### 2. Refer to Examples

For end-to-end usage, review and adapt the example:

- `examples/video/video_split_clip_example.py`

### 3. (Optional) Containerize Your Changes

If you need a container image, extend your base image using a Dockerfile and include your code and dependencies. Then build and run with your preferred container tooling.

```{note}
Ray Data backends do not support `nvdecs`/`nvencs` resource keys. Xenna does. If you plan to use `nvdecs`/`nvencs`, prefer the default Xenna executor.
```
