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
import sys

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


def _ensure_ray_dashboard_frontend() -> None:
    """Stub Ray's dashboard frontend dir once (nightly ray only), before any cluster starts.

    Ray *nightly* wheels omit the prebuilt dashboard frontend (``dashboard/client/build``
    is an npm artifact built only for releases), so the dashboard process dies with
    ``FrontendNotFoundError`` and its state API server never registers — which breaks
    every ``ray.util.state`` call (Xenna drives pipelines through it) with "Could not
    read 'dashboard' from GCS". Creating the dir (relative to the installed ``ray``, so
    it works in any venv) lets the dashboard start; the web UI itself is unused.

    Gated to dev/nightly builds so published releases (which ship the frontend) are
    untouched. Best-effort: a read-only install must not break ``import``.
    """
    import contextlib
    from pathlib import Path

    import ray
    from packaging.version import Version

    if not Version(ray.__version__).is_devrelease:
        return
    # Best-effort: a read-only install must not break ``import nemo_curator``.
    with contextlib.suppress(OSError):
        (Path(ray.__file__).parent / "dashboard" / "client" / "build" / "static").mkdir(parents=True, exist_ok=True)


_ensure_ray_dashboard_frontend()

# Raise an informative error early to users on unsupported systems
if sys.platform != "linux":
    _msg = (
        "NeMo-Curator currently only supports Linux systems, "
        f"while the current machine has a {sys.platform} system. \n"
        "For more information on installation and system requirements, see "
        "https://docs.nvidia.com/nemo/curator/latest/admin/installation.html"
    )
    raise ValueError(_msg)
