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

Learn how to package dependencies for NeMo Curator using a container image.

The NeMo Curator container provides a primary `curator` conda environment with pre-installed dependencies. If your pipeline needs additional system or Python packages, create a custom image. Refer to the [container environments](reference-infrastructure-container-environments) reference for defaults and build arguments.

## Before You Start

Before you begin, make sure that you have:

* Reviewed the [pipeline concepts and diagrams](about-concepts-video).  
* A base container image suitable for NeMo Curator.  
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

   # Python deps (prefer aligning with pyproject optional extras)
   # Example: install video stacks used by NeMo Curator
   RUN pip install --no-cache-dir \
       av==13.1.0 opencv-python einops easydict

   # Optional: CV-CUDA (choose wheel that matches your CUDA/Python)
   # RUN pip install cvcuda_cu12

   # Optional: vLLM / FlashAttention (see platform markers in pyproject)
   # RUN pip install vllm==0.9.2 flash-attn<=2.8.3

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

```{note}
Ray Data backends do not support `nvdecs`/`nvencs` resource keys. Xenna does. If you plan to use `nvdecs`/`nvencs`, prefer the default Xenna executor.
```
