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

import importlib.util
from pathlib import Path

import pytest

from nemo_curator.stages.resources import Resources

_MAIN_PATH = (
    Path(__file__).resolve().parents[4]
    / "tutorials"
    / "audio"
    / "qwen_omni_inprocess"
    / "main.py"
)


def _load_main_module():
    spec = importlib.util.spec_from_file_location("qwen_omni_main", _MAIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_instantiate_resources_accepts_canonical_hydra_target() -> None:
    main = _load_main_module()
    resources = main._instantiate_resources(
        {
            "_target_": "nemo_curator.stages.resources.Resources",
            "cpus": 2.0,
            "gpus": 1.0,
        }
    )
    assert isinstance(resources, Resources)
    assert resources.cpus == 2.0
    assert resources.gpus == 1.0


def test_instantiate_resources_rejects_arbitrary_target() -> None:
    main = _load_main_module()
    with pytest.raises(ValueError, match="may only target"):
        main._instantiate_resources(
            {
                "_target_": "os.system",
                "cpus": 1.0,
            }
        )


def test_instantiate_resources_plain_dict_without_target() -> None:
    main = _load_main_module()
    resources = main._instantiate_resources({"cpus": 1.5, "gpus": 0.0})
    assert resources.cpus == 1.5
    assert resources.gpus == 0.0
