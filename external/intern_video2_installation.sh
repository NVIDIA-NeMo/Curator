#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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