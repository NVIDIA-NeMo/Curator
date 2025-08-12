from collections.abc import Generator

import pytest
from dask.distributed import Client

from nemo_curator.utils.import_utils import gpu_only_import, gpu_only_import_from

cudf = gpu_only_import("cudf")
dask_cudf = gpu_only_import("dask_cudf")
LocalCUDACluster = gpu_only_import_from("dask_cuda", "LocalCUDACluster")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--cpu", action="store_true", default=False, help="Run tests without gpu marker")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--cpu"):
        skip_gpu = pytest.mark.skip(reason="Skipping GPU tests")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture(autouse=True, scope="session")
def gpu_client(request: pytest.FixtureRequest) -> Generator[Client, None, None]:
    if not request.config.getoption("--cpu"):
        with LocalCUDACluster(n_workers=1) as cluster, Client(cluster) as client:
            request.session.client = client
            request.session.cluster = cluster
            yield client
            client.close()
            cluster.close()
    else:
        yield None
