# See https://github.com/rapidsai/ci-imgs for ARG options
# NeMo Curator requires Python 3.12, Ubuntu 22.04/20.04, and CUDA 12 (or above)
ARG CUDA_VER=12.5.1
ARG LINUX_VER=ubuntu22.04
ARG PYTHON_VER=3.12
ARG IMAGE_LABEL

FROM rapidsai/ci-conda:cuda${CUDA_VER}-${LINUX_VER}-py${PYTHON_VER} AS curator-update

FROM rapidsai/ci-conda:cuda${CUDA_VER}-${LINUX_VER}-py${PYTHON_VER} AS deps
LABEL "nemo.library"=${IMAGE_LABEL}
WORKDIR /opt

# Re-declare ARGs after new FROM to make them available in this stage
ARG CUDA_VER

# Install the minimal libcu* libraries needed by NeMo Curator
RUN conda create -y --name curator -c nvidia/label/cuda-${CUDA_VER} -c conda-forge \
  python=3.12 \
  cuda-cudart \
  libcufft \
  libcublas \
  libcurand \
  libcusparse \
  libcusolver \
  cuda-nvvm && \
  source activate curator && \
  pip install --upgrade pytest pip pytest-coverage

WORKDIR /tmp/Curator
RUN \
  --mount=type=bind,source=nemo_curator/__init__.py,target=/tmp/Curator/nemo_curator/__init__.py \
  --mount=type=bind,source=nemo_curator/package_info.py,target=/tmp/Curator/nemo_curator/package_info.py \
  --mount=type=bind,source=pyproject.toml,target=/tmp/Curator/pyproject.toml \
  source activate curator && \
  pip install --extra-index-url https://pypi.nvidia.com -e ".[all]"


FROM rapidsai/ci-conda:cuda${CUDA_VER}-${LINUX_VER}-py${PYTHON_VER} AS final

ENV PATH /opt/conda/envs/curator/bin:$PATH
LABEL "nemo.library"=${IMAGE_LABEL}
WORKDIR /workspace
COPY --from=deps /opt/conda/envs/curator /opt/conda/envs/curator
