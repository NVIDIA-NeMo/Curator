# See https://github.com/rapidsai/ci-imgs for ARG options
# NeMo Curator requires Python 3.12, Ubuntu 22.04/20.04, and CUDA 12 (or above)
ARG CUDA_VER=12.8.1
ARG LINUX_VER=ubuntu24.04

FROM nvidia/cuda:${CUDA_VER}-cudnn-devel-ubuntu${LINUX_VER}

COPY . .

RUN cd ray-curator && \
    pip install -e ".[all]"
