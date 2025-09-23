#!/bin/bash

# =============================================================================
# InternVideo2 Installation Script
# =============================================================================
# This script installs the InternVideo2 dependency for the Curator project.
# It clones the official InternVideo repository, applies necessary patches,
# and integrates it into the project's dependency management system using uv.

# Verify that the script is being run from the correct directory
# This ensures that relative paths in the script work correctly
if [ "$(basename "$PWD")" != "Curator" ]; then
  echo "Please run this script from the Curator/ directory."
  exit 1
fi

# Clone the official InternVideo repository from OpenGVLab
# This is the source repository for the InternVideo2 model implementation
git clone https://github.com/OpenGVLab/InternVideo.git;
cd InternVideo; git checkout 09d872e5093296c6f36b8b3a91fc511b76433bf7;

# Apply a custom patch to the InternVideo2 codebase
# This patch contains modifications needed for integration with NeMo Curator
patch -p1 < ../external/intern_video2_multimodal.patch; cd ../

# Synchronize all project dependencies using uv
uv sync --all-extras --all-groups

# Add the InternVideo2 multi-modality module as a local dependency
# This makes the patched InternVideo2 code available to the project
uv add InternVideo/InternVideo2/multi_modality