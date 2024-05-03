# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass

import pytest
import yaml

from nemo_curator.modules.config import BaseConfig


@dataclass
class CustomConfig(BaseConfig):
    a: str
    b: int
    c: bool
    d: float = 3.0

    def __post_init__(self):
        if self.d <= 0:
            raise ValueError("d must be positive")


class TestConfig:
    @pytest.fixture(autouse=True)
    def config_params(self):
        self.config_dict = {"a": "a", "b": 1, "c": True, "d": 4.0}

    def test_init(self):
        config = CustomConfig(a="a", b=1, c=True)
        assert config.a == "a"
        assert config.b == 1
        assert config.c is True
        assert config.d == 3.0

    def test_from_yaml(self, tmpdir):
        with open(tmpdir / "test_config.yaml", "w") as file:
            yaml.dump(self.config_dict, file)

        config = CustomConfig.from_yaml(tmpdir / "test_config.yaml")
        for key, value in self.config_dict.items():
            assert getattr(config, key) == value

    def test_from_yaml_raises(self, tmpdir):
        config_dict = self.config_dict.copy()
        config_dict["d"] = -1.0
        with open(tmpdir / "test_config.yaml", "w") as file:
            yaml.dump(config_dict, file)
        with pytest.raises(ValueError):
            CustomConfig.from_yaml(tmpdir / "test_config.yaml")

    def test_from_yaml_missing_key(self, tmpdir):
        config_dict = self.config_dict.copy()
        del config_dict["a"]
        with open(tmpdir / "test_config.yaml", "w") as file:
            yaml.dump(config_dict, file)
        with pytest.raises(TypeError):
            CustomConfig.from_yaml(tmpdir / "test_config.yaml")

    def test_from_yaml_extra_key(self, tmpdir):
        config_dict = self.config_dict.copy()
        config_dict["e"] = "e"
        with open(tmpdir / "test_config.yaml", "w") as file:
            yaml.dump(config_dict, file)
        with pytest.raises(TypeError):
            CustomConfig.from_yaml(tmpdir / "test_config.yaml")

    def test_post_init_raises(self):
        with pytest.raises(ValueError):
            CustomConfig(a="a", b=1, c=True, d=-1.0)
