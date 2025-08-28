from dataclasses import dataclass

import yaml


@dataclass
class BaseConfig:
    @classmethod
    def from_yaml(cls, file_path: str) -> "BaseConfig":
        with open(file_path) as file:
            yaml_dict = yaml.safe_load(file)
        return cls(**yaml_dict)
