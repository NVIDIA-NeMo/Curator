# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any

DEFAULT_SERVE_PORT = 8000
DEFAULT_SERVE_HEALTH_TIMEOUT_S = 300

# --- Subprocess / placement-group tunables used by subprocess_mgr.py --------

SIGTERM_WAIT_S = 10
"""Seconds to wait for SIGTERM to reap a subprocess before escalating to SIGKILL."""

SIGKILL_WAIT_S = 5
"""Seconds to wait after SIGKILL before giving up on a subprocess."""

PLACEMENT_GROUP_READY_TIMEOUT_S = 180
"""Default timeout for ``pg.ready()`` on a freshly-created placement group."""

WORKER_NODE_LABEL = {"ray.io/node-type": "worker"}
"""Bundle label selector applied when ``CURATOR_IGNORE_RAY_HEAD_NODE=1``.

Anyscale auto-labels head/worker nodes. OSS Ray users must start worker
nodes with ``ray start --labels ray.io/node-type=worker`` for this to take
effect.
"""

NOSET_CUDA_RUNTIME_ENV: dict[str, Any] = {
    "env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"},
}
"""Runtime-env fragment telling Ray not to overwrite the worker's CUDA_VISIBLE_DEVICES.

We explicitly set ``CUDA_VISIBLE_DEVICES`` in ``subprocess_env`` from
``ray.get_accelerator_ids()``, so for the subprocess this flag is largely
redundant -- it's kept defensively because the canonical vLLM+Ray pattern
(vLLM issues #7890/#30016/#35848) relies on it.
"""
