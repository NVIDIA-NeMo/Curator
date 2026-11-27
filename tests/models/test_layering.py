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

"""``nemo_curator.models`` must not depend on ``nemo_curator.stages``."""

import pathlib

import nemo_curator.models


def test_models_do_not_reference_stages() -> None:
    root = pathlib.Path(nemo_curator.models.__file__).parent
    offenders = [
        str(path.relative_to(root)) for path in root.rglob("*.py") if "nemo_curator.stages" in path.read_text()
    ]

    assert offenders == []
