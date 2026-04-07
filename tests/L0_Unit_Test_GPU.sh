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

# Custom HF model cache for video caption integration tests.
# Points to the shared HF cache in TestData; read in tests/stages/video/caption/conftest.py.
# Global HF_HOME is intentionally left unset to avoid affecting other GPU tests.
export CUSTOM_HF_DATASET=/home/TestData/HF_HOME
CUDA_VISIBLE_DEVICES="0,1" coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo_curator -m pytest -m gpu --rootdir /workspace tests
