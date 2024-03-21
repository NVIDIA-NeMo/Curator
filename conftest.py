import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--cpu", action="store_true", default=False, help="Run tests without gpu marker"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--cpu"):
        skip_gpu = pytest.mark.skip(reason="Skipping GPU tests")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
