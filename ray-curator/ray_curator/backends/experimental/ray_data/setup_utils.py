from collections.abc import Callable

from loguru import logger

from ray_curator.backends.base import NodeInfo, WorkerMetadata
from ray_curator.backends.experimental.utils import get_worker_metadata_and_node_id
from ray_curator.stages.base import ProcessingStage


def setup_stage(
    stage: ProcessingStage,  # noqa: ARG001
    setup_fn: Callable[[WorkerMetadata], None],
    setup_on_node_fn: Callable[[NodeInfo | None, WorkerMetadata | None], None],
) -> None:
    """Setup the stage on the worker and the node.
    Currently it is a barebone implementation, but ideally we improve it such that setup_on_node_fn is called only once per node.
    TODO: We need to improve this implementation to better support setup_on_node.
    Args:
        stage (ProcessingStage): Processing stage to setup
        setup_fn (Callable): Function to call for per-worker setup
        setup_on_node_fn (Callable): Function to call for per-node setup
    """
    logger.warning(
        "Ray Data setup / setup_on_node_fn is ineffecient due to unable to determine if setup has already been called."
        "Consider putting setup logic in init method of the stage."
    )
    # For Ray Data, we call setup directly since each worker processes independently
    # We create basic WorkerMetadata since some stages may need it
    node_info, worker_metadata = get_worker_metadata_and_node_id()

    # Call setup_on_node first (some stages may need node-level initialization)
    setup_on_node_fn(node_info, worker_metadata)

    # Then call per-worker setup
    setup_fn(worker_metadata)
