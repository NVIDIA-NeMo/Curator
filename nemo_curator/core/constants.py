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

DEFAULT_RAY_PORT = 6379
DEFAULT_RAY_DASHBOARD_PORT = 8265
DEFAULT_RAY_TEMP_DIR = "/tmp/ray"  # noqa: S108
DEFAULT_RAY_METRICS_PORT = 8080
DEFAULT_RAY_DASHBOARD_HOST = "127.0.0.1"
DEFAULT_RAY_CLIENT_SERVER_PORT = 10001
DEFAULT_RAY_AUTOSCALER_METRIC_PORT = 44217
DEFAULT_RAY_DASHBOARD_METRIC_PORT = 44227

# We cannot use a free port between 10000 and 19999 as it is used by Ray.
DEFAULT_RAY_MIN_WORKER_PORT = 10002
DEFAULT_RAY_MAX_WORKER_PORT = 19999
