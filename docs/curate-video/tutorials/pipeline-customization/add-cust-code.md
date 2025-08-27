---
description: "Tutorial for adding custom code to video curation pipelines for specialized processing requirements"
categories: ["video-curation"]
tags: ["customization", "custom-code", "pipeline", "advanced", "development"]
personas: ["mle-focused"]
difficulty: "advanced"
content_type: "tutorial"
modality: "video-only"

---

(video-tutorials-pipeline-cust-add-code)=
# Adding Custom Code

Learn how to extend Ray Curator by adding custom code to a new or existing stage.

The NeMo Video Curator container includes a robust set of default pipelines with commonly used stages, however, they may not always meet your pipeline requirements. In this tutorial, we'll demonstrate how to extend the default pipeline.

## Before You Start

Before you begin adding custom code, make sure that you have:

* Reviewed the [pipeline concepts and diagrams](about-concepts-video).  
* A working Ray Curator development environment.  
* Optionally prepared a container image that includes your dependencies.  
* Optionally [created a custom environment](video-tutorials-pipeline-cust-env) to support your new custom code.

---

## How to Add Custom Code

### Define New Functionality

1. Create a `custom_code` directory anywhere on your system to organize your custom pipeline code.  
2. Create a new folder for your environment, for example: `new_stage/`.
3. Create a new file, for example `my_file.py`. This file must define a class (`MyClass`) made available for import.

   ```py
   # your code here
   ```

4. Import the class in your stage or pipeline code to use it.

   ```py
   from my_code.my_file import MyClass

   ...
   ```

5. Save the files.

### Use your code in a pipeline

Create or edit a stage to use your code, then assemble a pipeline and run it in Python:

```py
from ray_curator.pipeline.pipeline import Pipeline
from ray_curator.stages.video.io.video_reader import VideoReaderStage
from ray_curator.stages.video.io.clip_writer import ClipWriterStage

from my_code.my_file import MyClass

class MyStage(ProcessingStage[VideoTask, VideoTask]):
    def process(self, task: VideoTask) -> VideoTask | list[VideoTask]:
        helper = MyClass()
        # use helper with task.data (Video/Clips)
        return task

pipeline = (
    Pipeline(name="my-pipeline")
    .add_stage(VideoReaderStage())
    .add_stage(MyStage())
    .add_stage(ClipWriterStage())
)

pipeline.run()
```

To containerize, use a Dockerfile to copy your code and install dependencies, then build and run with your preferred tooling.

## Next Steps

Now that you have created custom code, you can [create a custom stage](video-tutorials-pipeline-cust-add-stage) that uses your code.
