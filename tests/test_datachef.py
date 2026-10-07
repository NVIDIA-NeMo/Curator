import sys
from unittest.mock import MagicMock

# Mock out dependencies that fail on Windows
sys.modules['ray'] = MagicMock()
sys.modules['ray.util'] = MagicMock()
sys.modules['ray.data'] = MagicMock()
sys.modules['cosmos_xenna'] = MagicMock()
sys.modules['cosmos_xenna.ray_utils'] = MagicMock()
sys.modules['cosmos_xenna.ray_utils.cluster'] = MagicMock()

original_platform = sys.platform
sys.platform = "linux"
import nemo_curator
sys.platform = original_platform

import pytest
from pytest_httpserver import HTTPServer
from omegaconf import OmegaConf

from nemo_curator.recipe import DataChefRecipeGenerator


@pytest.fixture
def generator(httpserver: HTTPServer):
    return DataChefRecipeGenerator(
        target_benchmark="AIME_25",
        base_model_id="Qwen3-1.7B",
        available_data_sources=["finemath-v1"],
        compute_budget_tokens=1_000_000,
        api_url=httpserver.url_for("/v1/generate")
    )


def test_datachef_generate_success(generator, httpserver: HTTPServer):
    valid_yaml = '''
stages:
  - _target_: nemo_curator.stages.math.classifiers.finemath.FineMathClassifier
'''
    httpserver.expect_request("/v1/generate").respond_with_json({"config_yaml": valid_yaml})

    yaml_config = generator.generate()
    assert yaml_config.strip() == valid_yaml.strip()


def test_datachef_fallback(generator, httpserver: HTTPServer):
    # API fails, should fallback to a valid YAML
    httpserver.expect_request("/v1/generate").respond_with_data("Internal Server Error", status=500)

    yaml_config = generator.generate()
    # Ensure fallback uses the best-practice template for math
    assert "FineMathClassifier" in yaml_config
