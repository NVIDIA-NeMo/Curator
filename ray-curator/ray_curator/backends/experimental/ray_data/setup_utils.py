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
    This is now called once per Ray Data actor (stateful transform) rather than per batch.
    Args:
        stage (ProcessingStage): Processing stage to setup
        setup_fn (Callable): Function to call for per-worker setup
        setup_on_node_fn (Callable): Function to call for per-node setup
    """
    # For Ray Data with stateful transforms, we call setup once per actor
    # We create basic WorkerMetadata since some stages may need it
    node_info, worker_metadata = get_worker_metadata_and_node_id()

    # Call setup_on_node first (some stages may need node-level initialization)
    setup_on_node_fn(node_info, worker_metadata)

    # Then call per-worker setup
    setup_fn(worker_metadata)
