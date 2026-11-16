# Builds the Granary base and the full-pipeline images in one command, from the repository root:
#   docker buildx bake -f tutorials/audio/granary_v2/full_pipeline/docker-bake.hcl         # both images
#   docker buildx bake -f tutorials/audio/granary_v2/full_pipeline/docker-bake.hcl full    # English + European
#   docker buildx bake -f tutorials/audio/granary_v2/full_pipeline/docker-bake.hcl indic   # Indic

group "default" {
  targets = ["full", "indic"]
}

# Repository docker/Dockerfile: public CUDA 12.9 devel base, Torch 2.11, vLLM 0.22, Dynamo, etcd/NATS.
target "curator" {
  context    = "."
  dockerfile = "docker/Dockerfile"
  target     = "nemo_curator"
  args       = { CURATOR_EXTRA = "audio_tensorrt --extra inference_server", CURATOR_GROUPS = "" }
}

# docker/audio/Dockerfile.trt_llm --target granary: Torch 2.9.1 + TensorRT-LLM for the Indic engines.
target "granary" {
  context    = "."
  dockerfile = "docker/audio/Dockerfile.trt_llm"
  target     = "granary"
}

target "full" {
  context    = "."
  dockerfile = "tutorials/audio/granary_v2/full_pipeline/Dockerfile"
  contexts   = { curator = "target:curator" }
  tags       = ["curator-granary-full"]
}

target "indic" {
  context    = "."
  dockerfile = "tutorials/audio/granary_v2/full_pipeline/Dockerfile.trt_llm_granary"
  contexts   = { granary = "target:granary" }
  tags       = ["curator-granary-indic"]
}
