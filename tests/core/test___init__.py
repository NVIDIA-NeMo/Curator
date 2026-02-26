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

import sys

import pytest


def test_raises_system_error(monkeypatch: pytest.MonkeyPatch):
    dummy_platform = "asdfasdf"
    monkeypatch.setattr(sys, "platform", dummy_platform)

    # Remove module if already imported
    if "nemo_curator" in sys.modules:
        del sys.modules["nemo_curator"]

    with pytest.raises(ValueError, match="only supports Linux systems") as excinfo:
        import nemo_curator  # noqa

    assert dummy_platform in str(excinfo.value)
