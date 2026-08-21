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

import importlib
import os
import sys

import pytest


@pytest.mark.parametrize(
    ("configured_value", "expected_value"),
    [(None, "0"), ("1", "1")],
)
def test_kvikio_auto_direct_io_write_default(
    monkeypatch: pytest.MonkeyPatch,
    configured_value: str | None,
    expected_value: str,
):
    if configured_value is None:
        monkeypatch.delenv("KVIKIO_AUTO_DIRECT_IO_WRITE", raising=False)
    else:
        monkeypatch.setenv("KVIKIO_AUTO_DIRECT_IO_WRITE", configured_value)

    saved_module = sys.modules.pop("nemo_curator", None)
    try:
        importlib.import_module("nemo_curator")
        assert os.environ["KVIKIO_AUTO_DIRECT_IO_WRITE"] == expected_value
    finally:
        if saved_module is not None:
            sys.modules["nemo_curator"] = saved_module
        else:
            sys.modules.pop("nemo_curator", None)


def test_raises_system_error(monkeypatch: pytest.MonkeyPatch):
    dummy_platform = "asdfasdf"
    monkeypatch.setattr(sys, "platform", dummy_platform)

    # Remove module if already imported
    if "nemo_curator" in sys.modules:
        del sys.modules["nemo_curator"]

    with pytest.raises(ValueError, match="only supports Linux systems") as excinfo:
        import nemo_curator  # noqa

    assert dummy_platform in str(excinfo.value)
