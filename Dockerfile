# See https://github.com/rapidsai/ci-imgs for ARG options
# NeMo Curator requires Python 3.10, Ubuntu 22.04/20.04, and CUDA 12 (or above)
ARG CUDA_VER=12.5.1
ARG LINUX_VER=ubuntu22.04
ARG PYTHON_VER=3.10
FROM rapidsai/ci-conda:cuda${CUDA_VER}-${LINUX_VER}-py${PYTHON_VER}

WORKDIR /opt

# Install the minimal libcu* libraries needed by NeMo Curator
RUN conda create -y --name curator -c conda-forge -c nvidia \
  python=3.10 \
  cuda-cudart \
  libcufft \
  libcublas \
  libcurand \
  libcusparse \
  libcusolver

# Needed to navigate to and pull the forked repository's changes
ARG FORKED_REPO_URL
ARG CURATOR_COMMIT

# Clone the user's repository, find the relevant commit, and install everything we need
RUN bash -exu <<EOF
  git clone $FORKED_REPO_URL
  cd NeMo-Curator
  git fetch origin $CURATOR_COMMIT --depth=1
  git checkout $CURATOR_COMMIT
  source activate curator
  pip install --upgrade cython pytest pip
  pip install --extra-index-url https://pypi.nvidia.com ".[all]"
EOF
