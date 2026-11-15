# Granary v2 full pipeline

One container image for every Granary v2 tutorial, plus the end-to-end pipelines
that chain them. Each language gets its own folder; the shared [Dockerfile](Dockerfile)
lives here.

| Path | Contents |
| --- | --- |
| [Dockerfile](Dockerfile) | The single image: metadata, Indic ASR, Qwen Omni ASR, text pipeline and `en/` |
| [en/](en/README.md) | English pipeline: recordings to enriched English transcripts in one Ray cluster |

## What runs in the image

The image extends the Granary image built from `docker/audio/Dockerfile.trt_llm
--target granary`, so the metadata and Indic ASR environments are unchanged. It adds
what the other tutorials need:

| Tutorial | Stages and runtime | Provided by |
| --- | --- | --- |
| [Metadata](../metadata/README.md) | Sortformer, VAD, sound events, language ID on TensorRT/NeMo/Torch 2.9.1 | Granary base |
| [Indic ASR](../indic_asr/README.md) | Canary and Parakeet TensorRT engines | Granary base |
| [Qwen Omni ASR](../qwen_omni_asr/README.md) | In-process vLLM 0.15.1 with Transformers 4.57.6; Qwen3-ASR recovery | `vllm`, `qwen-omni-utils`, two Qwen3-ASR fixes |
| [Text pipeline](../text_pipeline/README.md) | CPU HTTP stages plus a Dynamo-managed text model | `ai-dynamo`, `nixl`, etcd/NATS; model workers resolve their own environment at startup |
| [English](en/README.md) | All of the above in one Ray cluster | the whole image |

The base image already carries Torch 2.9.1, Transformers 4.57.6 and the NeMo fork, and
vLLM 0.15.1 uses the same Torch and Transformers versions, so these stacks share one
interpreter. Dynamo's text-model workers need newer vLLM, Torch and Transformers builds;
Curator creates them as isolated Ray worker environments, so the image never mixes them
with the audio packages.

## Build

```bash
docker buildx build --builder default --load \
  -f tutorials/audio/granary_v2/full_pipeline/Dockerfile \
  -t curator-granary-full .
```

Build `curator-granary-metadata` first, as described in the
[metadata tutorial](../metadata/README.md#build-the-image).

## Runtime settings

| Setting | Value | Reason |
| --- | --- | --- |
| `PYTORCH_CUDA_ALLOC_CONF` | image default `expandable_segments:True`; vLLM entrypoints set `False` | The TensorRT stages keep the base default. vLLM tensor-parallel CUDA graphs need CUDA-IPC-compatible allocations, so `en/main.py` sets `False` before starting Ray. Set it yourself when running the standalone Qwen tutorial in this image. |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` (image default) | Required by in-process vLLM with CUDA |
| `CUDA_HOME` for serving workers | `/usr/local/cuda-12.9` (set in each pipeline's `runtime_env`) | FlashInfer compiles with the toolkit that matches the worker's Torch; audio stages stay on CUDA 12.8 |

## What stays outside the image

- **Indic Canary engine conversion.** It needs a different NeMo fork than the Granary
  image's, so run the converter in its own environment as the
  [Indic tutorial](../indic_asr/README.md#canary) describes. The converted engines are inputs.
- **Model weights and recordings.** Mount them as each tutorial describes.
- **Dynamo worker packages.** They resolve at first startup and need network access
  unless you pre-populate the package and model caches.

## Limits

The English pipeline's dependency set was exercised on one H100 node (see its
[validation](en/README.md#5-validation)). The other tutorials were not re-run inside
this image. The inherited audio environment has conflicting package declarations:
`uv pip check` is not clean. The standalone Qwen tutorial documents its own virtual
environment (for example Ray 2.57.0, TorchCodec 0.9.0), which differs from this image's
resolved versions.
