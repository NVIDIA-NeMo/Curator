# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


MAJOR = 1
MINOR = 2
PATCH = 0
PRE_RELEASE = ""
DEV = ""

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE, DEV)

__shortversion__ = ".".join(map(str, VERSION[:3]))
__version__ = __shortversion__

if VERSION[3] != "":
    __version__ = __version__ + VERSION[3]

if VERSION[4] != "":
    __version__ = __version__ + "." + ".".join(VERSION[4:])

import os as _os  # noqa: E402, I001
import subprocess as _subprocess  # noqa: E402


if not int(_os.getenv("NO_VCS_VERSION", "0")):
    try:
        _git = _subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            cwd=_os.path.dirname(_os.path.abspath(__file__)),
            check=True,
            text=True,
        )
    except (_subprocess.CalledProcessError, OSError):
        pass
    else:
        __version__ += f"+{_git.stdout.strip()}"

__package_name__ = "nemo_curator"
__contact_names__ = "NVIDIA"
__contact_emails__ = "nemo-toolkit@nvidia.com"
__homepage__ = "https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/index.html"
__repository_url__ = "https://github.com/NVIDIA-NeMo/Curator"
__download_url__ = "https://github.com/NVIDIA-NeMo/Curator/releases"
__description__ = "NeMo Curator - Scalable Data Preprocessing Tool for Training Large Language Models"
__license__ = "Apache2"
__keywords__ = "deep learning, machine learning, gpu, NLP, NeMo, nvidia, pytorch, torch, language, preprocessing, LLM, large language model"
