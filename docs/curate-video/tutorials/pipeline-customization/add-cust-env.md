---
description: "Guide to creating and managing custom Docker environments for specialized video curation pipeline requirements"
categories: ["video-curation"]
tags: ["customization", "docker", "environments", "pipeline", "deployment", "advanced"]
personas: ["mle-focused", "devops-focused"]
difficulty: "advanced"
content_type: "tutorial"
modality: "video-only"

---

(video-tutorials-pipeline-cust-env)=
# Add Custom Environment

Learn how to package dependencies for Ray Curator using a container image.

The NeMo Video Curator container includes a series of [default environments](reference-infrastructure-container-environments) with pre-installed dependencies, however, they may not always meet your pipeline requirements. In this tutorial, we'll demonstrate how to add a new custom environment to support a custom x stage.

## Before You Start

Before you begin, make sure that you have:

* Reviewed the [pipeline concepts and diagrams](about-concepts-video).  
* A base container image suitable for Ray Curator.  
* Optionally [created custom code](video-tutorials-pipeline-cust-add-code) that defines your new requirements.  

---

## How to Add Dependencies with a Dockerfile

### Define Build Steps

1. Create an `environments` directory anywhere on your system to organize your custom pipeline stage environments.  
2. Create a new folder for your environment, for example: `my-env/`.
3. Create a `Dockerfile` that installs your environment's dependencies on top of the base image.

   ```dockerfile
   FROM <your-base-image>

   # System deps
   RUN apt-get update && apt-get install -y --no-install-recommends \
       wget git && rm -rf /var/lib/apt/lists/*

   # Python deps
   RUN pip install --no-cache-dir \
       accelerate==0.32.1 av==13.1.0 packaging==24.0 ninja==1.11.1.1 \
       torch==2.2.1 torchvision torchaudio transformers==4.42.3 \
       scenedetect==0.6.4 PyNvVideoCodec==1.0.2 timm==0.9.12

   # Optional: CV-CUDA wheel
   RUN wget -P /tmp/ https://github.com/CVCUDA/CV-CUDA/releases/download/v0.11.0-beta/cvcuda_cu12-0.11.0b0-cp310-cp310-linux_x86_64.whl \
       && pip install /tmp/cvcuda_cu12-0.11.0b0-cp310-cp310-linux_x86_64.whl \
       && rm -f /tmp/cvcuda_cu12-0.11.0b0-cp310-cp310-linux_x86_64.whl

   # Optional: FlashAttention
   RUN pip install flash-attn==2.6.1 --no-build-isolation

   # Copy your code if needed
   COPY . /workspace
   WORKDIR /workspace
   ```

4. Save the file.

### Build the Container

Build and tag your image using Docker or your preferred tool:

```bash
docker build -t my-ray-curator:latest .
```

## Next Steps

Now that you have created a custom environment, you can [create custom code](video-tutorials-pipeline-cust-add-code) for that environment.
