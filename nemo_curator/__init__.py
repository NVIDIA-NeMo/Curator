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

import os

from .package_info import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __shortversion__,
    __version__,
)

os.environ["RAPIDS_NO_INITIALIZE"] = "1"

from cosmos_xenna.ray_utils.cluster import API_LIMIT

# We set these incase a user ever starts a ray cluster with nemo_curator, we need these for Xenna to work
os.environ["RAY_MAX_LIMIT_FROM_API_SERVER"] = str(API_LIMIT)
os.environ["RAY_MAX_LIMIT_FROM_DATA_SOURCE"] = str(API_LIMIT)
